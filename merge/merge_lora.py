"""Merge a PEFT adapter checkpoint into the base weights.

Companion to ``--merge 0`` runs (e.g. ``--strategy fsdp2``): training saves a
small ``rwkv-N-adapter.pth``, this script turns it into a normal
``rwkv-N.pth`` that can be loaded with ``--load_model``.

Example:
    python merge/merge_lora.py \
        --base_model models/rwkv7-g1k-13.3b.pth \
        --adapter /home/rwkv/JL/out_model/rwkv7-13b-lora/rwkv-0-adapter.pth \
        --output  /home/rwkv/JL/out_model/rwkv7-13b-lora/rwkv-0.pth \
        --n_layer 61 --n_embd 4096 --vocab_size 65536 --my_testing x070 \
        --peft lora --peft_config '{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
"""

import json
import os
from argparse import ArgumentParser

import torch


def merge_adapter(base_model: str, adapter: str, output: str, args) -> None:
    """Load ``base_model``, apply ``adapter`` and save the merged weights."""
    from rwkvt.peft_loading import apply_peft, build_model
    from rwkvt.utils import strip_prefix

    if not os.path.exists(base_model):
        raise SystemExit(f"base model not found: {base_model}")
    if not os.path.exists(adapter):
        raise SystemExit(f"adapter not found: {adapter}")

    args.load_model = base_model
    model = build_model(args)
    model = apply_peft(model, args)

    from rwkvt.fsdp import is_adapter_key

    state = torch.load(adapter, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # base weights are expected to be missing from an adapter-only checkpoint
    missing = [k for k in missing if is_adapter_key(k)]
    if missing or unexpected:
        raise SystemExit(
            f"adapter does not match the model: {len(missing)} missing / "
            f"{len(unexpected)} unexpected adapter key(s), e.g. "
            f"{(missing + unexpected)[:3]}")

    merged = model.merge_and_unload()
    torch.save(strip_prefix(merged.state_dict(), "base_model.model."), output)
    print(f"merged {len(state)} adapter tensors -> {output}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--my_testing", default="x070")
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_embd", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--dim_att", type=int, default=0)
    parser.add_argument("--dim_ffn", type=int, default=0)
    parser.add_argument("--head_size_a", type=int, default=64)
    parser.add_argument("--head_size_divisor", type=int, default=8)
    parser.add_argument("--peft", default="lora")
    parser.add_argument("--peft_config", default='{"r":8,"lora_alpha":32}')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ.setdefault("RWKV_TRAIN_TYPE", "")
    os.environ.setdefault("WKV", "cuda")
    os.environ.setdefault("FUSED_KERNEL", "0")
    os.environ.setdefault("RWKV_HEAD_SIZE_A", str(args.head_size_a))
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    if isinstance(args.peft_config, str):
        args.peft_config = json.loads(args.peft_config)
    args.load_partial = 0

    merge_adapter(args.base_model, args.adapter, args.output, args)


if __name__ == "__main__":
    main()
