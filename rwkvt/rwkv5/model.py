########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn as nn

from rwkvt.hf_compat import HFCompatibleModel
from rwkvt.operator.checkpoint import checkpoint_block
from rwkvt.utils import drop_unsupported_kwargs

from .block import Block


class RWKV5(HFCompatibleModel, nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, return_hidden=False,
                **kwargs):
        # RWKV-5 has no masked time-mix; the argument keeps the caller interface
        # uniform with RWKV-6/7 (see light_rwkv.RWKV.forward).
        drop_unsupported_kwargs("RWKV5", kwargs)
        args = self.args
        B, T = input_ids.size()
        assert args.ctx_len >= T, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(input_ids)
        token_embeds = x

        for block in self.blocks:
            x = checkpoint_block(args, block, x) if args.grad_cp == 1 else block(x)

        x = self.ln_out(x)
        logits = self.head(x)
        return (logits, x, token_embeds) if return_hidden else logits
