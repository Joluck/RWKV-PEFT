"""RWKV-7 transformer block."""

import os

import torch.nn as nn

from rwkvt.infctx_module import BlockState

from .att import RWKV_Tmix_v7
from .ffn import RWKV_Cmix_v7


class Block(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # The training mode is fixed for the whole run, resolve the dispatch once.
        self._infctx = os.environ.get("RWKV_TRAIN_TYPE") == "infctx"

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v7(args, layer_id)
        self.ffn = RWKV_Cmix_v7(args, layer_id)

    def forward(self, x, v_first=None, attention_mask=None,
                last_state: BlockState = None):
        if self.layer_id == 0:
            x = self.ln0(x)

        if self._infctx:
            x_attn, v_first, att_state = self.att(
                self.ln1(x), v_first, last_state=last_state.time_mix_state,
                attention_mask=attention_mask)
            x = x + x_attn
            ffn_out, ffn_state = self.ffn(
                self.ln2(x), last_state=last_state.channel_mix_state,
                attention_mask=attention_mask)
            return x + ffn_out, v_first, BlockState(att_state, ffn_state)

        x_attn, v_first = self.att(self.ln1(x), v_first,
                                   attention_mask=attention_mask)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x), attention_mask=attention_mask)
        return x, v_first
