########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import torch.nn as nn
from torch.nn import functional as F
import deepspeed
from .block import Block

class RWKV5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft !='none':
                    x = torch_checkpoint(block, x, use_reentrant=False)
                else:
                    x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x