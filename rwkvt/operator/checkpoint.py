"""Gradient-checkpointing helper shared by the model versions.

DeepSpeed's checkpoint implementation is only used for plain full/LoRA runs;
PEFT / state-tuning runs use ``torch.utils.checkpoint`` because the extra
arguments (``v_first``, states) are not supported by DeepSpeed's variant.
"""

from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint_block(args, block, *inputs, **kwargs):
    """Checkpoint ``block(*inputs, **kwargs)`` with the right implementation.

    DeepSpeed's variant is used for plain full fine-tuning when it is available;
    everything else (extra arguments, state tuning, PEFT, no deepspeed) uses
    ``torch.utils.checkpoint``, which also keeps FSDP2 compatible.
    """
    needs_torch = (bool(kwargs) or args.peft != "none"
                   or args.train_type in ("state", "fullstate"))
    if not needs_torch:
        try:
            import deepspeed

            return deepspeed.checkpointing.checkpoint(block, *inputs)
        except ImportError:
            pass
    return torch_checkpoint(block, *inputs, use_reentrant=False, **kwargs)
