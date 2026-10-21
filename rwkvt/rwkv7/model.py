"""RWKV-7 model (x070)."""

import os

import torch
import torch.nn as nn

from rwkvt.hf_compat import HFCompatibleModel
from rwkvt.infctx_module import BlockStateList
from rwkvt.utils import drop_unsupported_kwargs

from .block import Block

_USE_INFCTX = os.environ.get("RWKV_TRAIN_TYPE") == "infctx"


class RWKV7(HFCompatibleModel, nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def _checkpoint(self, block, *args, **kwargs):
        """Gradient checkpointing: torch for peft/state runs, deepspeed otherwise."""
        from rwkvt.operator.checkpoint import checkpoint_block  # local import: optional dep

        return checkpoint_block(self.args, block, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if _USE_INFCTX:
            return self.forward_infctx(*args, **kwargs)
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, input_ids, attention_mask=None, **kwargs):
        drop_unsupported_kwargs("RWKV7", kwargs)
        args = self.args
        B, T = input_ids.size()
        assert args.ctx_len >= T, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(input_ids)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = self._checkpoint(block, x, v_first, attention_mask)
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        return self.head(x)

    def forward_infctx(self, input_ids, last_shift_states: torch.Tensor,
                       last_wkv_states: torch.Tensor, attention_mask=None,
                       **kwargs):
        drop_unsupported_kwargs("RWKV7", kwargs)
        args = self.args
        B, T = input_ids.size()
        assert args.chunk_ctx >= T, "Cannot forward, chunk_ctx is exhausted."
        H = args.dim_att // args.head_size_a
        assert args.n_embd == H * args.head_size_a

        x = self.emb(input_ids)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd, H,
                                          x.device, x.dtype)

        v_first = torch.empty_like(x)
        for i, (block, block_state) in enumerate(zip(
                self.blocks,
                      BlockStateList(last_shift_states, last_wkv_states),
                      strict=True)):
            if args.grad_cp == 1 and i > 0:
                out = self._checkpoint(block, x, v_first, attention_mask,
                                       last_state=block_state)
            else:
                out = block(x, v_first, attention_mask, last_state=block_state)
            x, v_first, new_states[i] = out

        x = self.ln_out(x)
        return self.head(x), new_states.shift_states, new_states.wkv_states
