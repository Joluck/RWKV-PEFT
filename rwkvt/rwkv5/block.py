"""RWKV-5 transformer block."""

import torch.nn as nn

from .att import RWKV_Tmix_v5
from .ffn import RWKV_Cmix_v5


class Block(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v5(args, layer_id)
        self.ffn = RWKV_Cmix_v5(args, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        return x + self.ffn(self.ln2(x))
