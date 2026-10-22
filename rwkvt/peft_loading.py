"""Model construction: weight loading, PEFT wrapping and LightningModule setup."""

import json
import os

import torch
import torch.nn as nn
from peft import (
    AdaLoraConfig,
    LoraConfig,
    MissConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
)

from rwkvt.rwkv5.model import RWKV5
from rwkvt.rwkv6.model import RWKV6
from rwkvt.rwkv7.model import RWKV7

MODEL_CLASSES = {"5": RWKV5, "6": RWKV6, "7": RWKV7}

PEFT_CONFIGS = {
    "lora": LoraConfig,
    "miss": MissConfig,
    "adalora": AdaLoraConfig,
    "prefix": PrefixTuningConfig,
}

DEFAULT_TARGET_MODULES = ["receptance", "key", "value", "output"]


class RWKVConfig:
    """Minimal ``config`` object required by huggingface/PEFT."""

    model_type = "rwkv"
    tie_word_embeddings = False

    def __init__(self, n_embd: int, n_layer: int):
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return {"model_type": self.model_type, "n_embd": self.n_embd,
                "n_layer": self.n_layer}


def get_model_class(version: str) -> type:
    """Resolve the model class from ``RWKV_MY_TESTING`` (e.g. ``x070``)."""
    for key, cls in MODEL_CLASSES.items():
        if key in version:
            return cls
    raise ValueError(
        f"unsupported RWKV version {version!r}; expected one of "
        f"{sorted(MODEL_CLASSES)} (e.g. --my_testing x070)")


def build_model_from_state_dict(args, state_dict: dict, allow_missing: bool = False,
                                warn_unexpected: bool = True) -> nn.Module:
    """Instantiate the model on the meta device and assign ``state_dict``."""
    model_cls = get_model_class(os.environ["RWKV_MY_TESTING"])
    with torch.device("meta"):
        model = model_cls(args)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # ``state``/``fullstate`` parameters are created by the training mode itself
    # and are intentionally absent from the checkpoint.
    missing = [k for k in missing if "state" not in k]
    if missing and not allow_missing and not getattr(args, "load_partial", 0):
        raise RuntimeError(
            f"{args.load_model} is missing {len(missing)} parameter(s), e.g. "
            f"{missing[:5]}; pass --load_partial 1 to load anyway")
    if unexpected and warn_unexpected:
        print(f"[load] ignoring {len(unexpected)} unexpected key(s), e.g. {unexpected[:5]}")
    _materialize_meta_params(model)

    precision = str(getattr(args, "precision", ""))
    src_dtype = model.emb.weight.dtype
    if precision == "bf16-true" and src_dtype != torch.bfloat16:
        # cast once, before any FSDP2 sharding: the WKV kernels need bf16 and
        # `--precision bf16` (autocast) would leave the params in fp32
        model = model.to(torch.bfloat16)
        print(f"[load] cast weights {src_dtype} -> bfloat16 (--precision bf16-true)")
    elif "bf16" in precision and src_dtype != torch.bfloat16:
        print(f"[load] WARNING: checkpoint weights are {src_dtype} but the WKV "
              f"kernels need bf16; rerun with --precision bf16-true")
    return model


def build_model(args) -> nn.Module:
    """Instantiate the model on the meta device (no memory) and load weights."""
    if not args.load_model:
        raise ValueError("--load_model is required: fine-tuning starts from a checkpoint")

    state_dict = torch.load(args.load_model, map_location="cpu",
                            weights_only=True, mmap=True)
    model = build_model_from_state_dict(args, state_dict)
    del state_dict
    return model


def _materialize_meta_params(model: nn.Module) -> None:
    """Replace parameters that stayed on the meta device (absent from ckpt).

    ``param.data = ...`` cannot move a meta tensor to a real device, the
    parameter object has to be swapped out on its parent module.
    """
    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, nn.Parameter(
            torch.zeros(param.shape, dtype=torch.float32),
            requires_grad=param.requires_grad))


def apply_peft(model: nn.Module, args) -> nn.Module:
    """Wrap the model with the requested PEFT tuner."""
    config_cls = PEFT_CONFIGS.get(args.peft)
    if config_cls is None:
        raise ValueError(f"unsupported --peft {args.peft!r}; expected 'state', "
                         f"'none' or one of {sorted(PEFT_CONFIGS)}")

    peft_kwargs = args.peft_config
    if isinstance(peft_kwargs, str):
        peft_kwargs = json.loads(peft_kwargs)
    peft_kwargs = dict(peft_kwargs)
    peft_kwargs.setdefault("task_type", TaskType.CAUSAL_LM)
    peft_kwargs.setdefault("target_modules", DEFAULT_TARGET_MODULES)

    model.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)
    model = get_peft_model(model, config_cls(**peft_kwargs))
    model.print_trainable_parameters()
    return model


def freeze_except_state(model: nn.Module) -> None:
    """State tuning: only the ``*_state`` parameters stay trainable."""
    model.requires_grad_(False)
    trainable = []
    for name, param in model.named_parameters():
        if "state" in name:
            param.requires_grad = True
            trainable.append(name)
    if not trainable:
        raise RuntimeError("state tuning requested but the model has no state "
                           "parameters; is --my_testing correct?")
    print(f"[state] {len(trainable)} trainable state parameter tensors")


def load_peft_model(args):
    """Build the model, apply the tuning method and wrap it for Lightning."""
    if args.peft not in ("none", "state", "mtp") and args.peft not in PEFT_CONFIGS:
        raise ValueError(f"unsupported --peft {args.peft!r}")

    model = build_model(args)

    if args.peft == "mtp":
        model.requires_grad_(False)  # only the MTP heads (in the module) train
    elif args.peft == "state":
        if os.environ.get("RWKV_TRAIN_TYPE") != "state":
            raise RuntimeError("state tuning requires RWKV_TRAIN_TYPE=state "
                               "(set automatically by --peft state)")
        freeze_except_state(model)
    elif args.peft != "none":
        model = apply_peft(model, args)

    from rwkvt.lightning_train.light_rwkv import FSDP2RWKV, RWKV  # needs lightning

    module_cls = FSDP2RWKV if getattr(args, "fsdp2", False) else RWKV
    return args, module_cls(args, model=model)
