"""FSDP2 (``torch.distributed.fsdp.fully_shard``) support.

Three pieces:

* :func:`build_fsdp2_strategy` - ``--strategy fsdp2`` -> Lightning's
  experimental ``ModelParallelStrategy`` (pure data parallel: ``tp=1``).
* :class:`FSDP2RWKV` / :func:`apply_fsdp2` - shard every ``Block`` and then the
  root module inside Lightning's ``configure_model()`` hook.
* :func:`gather_model_state_dict` - collect a full (non-DTensor) state dict so
  the existing checkpoint code keeps working.

The shardable unit is the transformer block: it is structurally identical across
layers and holds every parameter except ``emb``/``head``.
"""

from __future__ import annotations

import torch

SHARDED_MODULE_ATTR = "blocks"


# --------------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------------- #
def is_fsdp2(strategy) -> bool:
    """True when ``strategy`` is Lightning's FSDP2 strategy."""
    if strategy is None:
        return False
    try:
        from lightning.pytorch.strategies import ModelParallelStrategy
    except ImportError:  # pragma: no cover - lightning is a hard dependency
        return False
    return isinstance(strategy, ModelParallelStrategy)


def build_fsdp2_strategy(devices: int, num_nodes: int = 1,
                         save_distributed_checkpoint: bool = False):
    """Create the Lightning strategy for pure FSDP2 data parallelism.

    ``ModelParallelStrategy`` requires ``data_parallel_size * tensor_parallel_size
    == world_size`` and defaults to tensor parallelism over all local GPUs, so
    both sizes have to be passed explicitly.
    """
    try:
        from lightning.pytorch.strategies import ModelParallelStrategy
    except ImportError as exc:  # pragma: no cover - version guard
        raise RuntimeError(
            "--strategy fsdp2 requires lightning>=2.5 (ModelParallelStrategy); "
            "upgrade lightning or use a deepspeed strategy") from exc

    if devices in (-1, "auto", None):
        devices = torch.cuda.device_count()
    world_size = int(devices) * int(num_nodes)
    if world_size < 1:
        raise RuntimeError("--strategy fsdp2 needs at least one device")
    return ModelParallelStrategy(
        data_parallel_size=world_size,
        tensor_parallel_size=1,
        save_distributed_checkpoint=save_distributed_checkpoint,
    )


# --------------------------------------------------------------------------- #
# sharding
# --------------------------------------------------------------------------- #
def unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the RWKV module, unwrapping a PEFT ``PeftModel`` if present."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def build_mixed_precision_policy():
    """FSDP2 mixed precision: keep params as loaded, reduce grads in fp32.

    The checkpoints are bf16, and the WKV CUDA kernels assert bf16 inputs, so
    ``param_dtype`` must stay ``None`` (no cast). Gradient reduction in fp32 is
    the standard FSDP2 recommendation.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)


def _move_unsharded(root: torch.nn.Module, device: torch.device) -> None:
    """Move every parameter/buffer that is not already a DTensor to ``device``."""
    from torch.distributed.tensor import DTensor

    for module in root.modules():
        for param in module.parameters(recurse=False):
            if not isinstance(param, DTensor) and param.device != device:
                param.data = param.data.to(device)
        for buffer in module.buffers(recurse=False):
            if not isinstance(buffer, DTensor) and buffer.device != device:
                buffer.data = buffer.data.to(device)


def apply_fsdp2(model: torch.nn.Module, mesh, mp_policy=None,
                reshard_after_forward: bool = True,
                device: torch.device | None = None) -> torch.nn.Module:
    """Shard ``model`` with FSDP2, one shard group per transformer block.

    Children must be sharded before the root so that the root only owns the
    parameters that live outside the blocks (``emb``/``head``/adapters).

    Parameters are moved to ``device`` **one block at a time** right before that
    block is sharded, so the peak device memory is a single block instead of the
    whole model (``fully_shard`` requires the params to be on the mesh device).
    """
    from torch.distributed.fsdp import fully_shard

    if mp_policy is None:
        mp_policy = build_mixed_precision_policy()
    if device is None:
        device = torch.device(mesh.device_type)

    base = unwrap_base_model(model)
    blocks = getattr(base, SHARDED_MODULE_ATTR, None)
    if blocks is None:
        raise TypeError(f"{type(base).__name__} has no .{SHARDED_MODULE_ATTR} to shard")

    for block in blocks:
        block.to(device)
        fully_shard(block, mesh=mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)

    # emb / head / ln_out / any adapter that sits outside the blocks
    _move_unsharded(model, device)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward)
    return model


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def gather_model_state_dict(model: torch.nn.Module) -> dict:
    """Full state dict (plain CPU tensors) of a possibly sharded model.

    This is a *collective*: every rank has to call it, only rank 0 needs the
    result.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_model_state_dict(model, options=options)


def gather_adapter_state_dict(model: torch.nn.Module) -> dict:
    """Full adapter tensors only (cheap: ~MBs instead of the whole model).

    Each adapter parameter lives in a sharded block, so reading it needs one
    small all-gather per tensor. This avoids the ``full_state_dict=True`` path,
    which materialises the entire model on *every* rank (26 GB x 8 for a 13B).
    """
    from torch.distributed.tensor import DTensor

    state = {}
    for name, param in model.named_parameters():
        if not is_adapter_key(name):
            continue
        if isinstance(param, DTensor):
            state[name] = param.full_tensor().detach().cpu()
        else:
            state[name] = param.detach().cpu()
    for name, buffer in model.named_buffers():
        if is_adapter_key(name):
            state[name] = buffer.detach().cpu()
    return state


def is_adapter_key(key: str) -> bool:
    """True for PEFT adapter parameters (LoRA/MiSS/AdaLoRA)."""
    markers = ("lora_", "lora_magnitude_vector", "ranknum", ".bone_",
               "bone_", "dora_scale", "modules_to_save")
    return any(m in key for m in markers)


def adapter_state_dict(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if is_adapter_key(k)}


def merge_peft_state_dict(args, state_dict: dict) -> dict:
    """Merge PEFT weights into a full checkpoint on CPU.

    Used under FSDP2 where the live model only holds a shard and ``merge_and_unload``
    cannot be applied in place. A throw-away CPU copy is built from the gathered
    weights, merged, and returned as the final state dict.
    """
    from rwkvt.peft_loading import apply_peft, build_model_from_state_dict
    from rwkvt.utils import strip_prefix

    base = build_model_from_state_dict(args, state_dict, allow_missing=True,
                                       warn_unexpected=False)
    base = apply_peft(base, args)
    base.load_state_dict(state_dict, strict=False)
    merged = base.merge_and_unload()
    return strip_prefix(merged.state_dict(), "base_model.model.")
