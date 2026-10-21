"""RWKV-6 channel-mixing (FFN) layers."""

import os

import torch
import torch.nn as nn

from rwkvt.infctx_module import ChannelMixState


def RWKV_Cmix_v6(*args, **kwargs):
    """Build the channel-mixing layer matching the current training mode."""
    if os.environ.get("RWKV_TRAIN_TYPE") == "infctx":
        return RWKV_CMix_x060_infctx(*args, **kwargs)
    return RWKV_CMix_x060(*args, **kwargs)


class RWKV_CMix_x060(nn.Module):
    variant = "base"

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def _shift(self, x, last_state=None):
        if self.variant == "infctx":
            return torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                                dim=1) - x
        return self.time_shift(x) - x

    def forward(self, x, attention_mask=None, last_state=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self._shift(x, last_state)
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = torch.relu(self.key(xk)) ** 2
        out = torch.sigmoid(self.receptance(xr)) * self.value(k)
        if self.variant == "infctx":
            return out, ChannelMixState(x[:, -1])
        return out


class RWKV_CMix_x060_infctx(RWKV_CMix_x060):
    """Infinite context channel mixing, carries the shift state."""

    variant = "infctx"
