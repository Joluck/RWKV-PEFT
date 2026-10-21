"""RWKV-7 channel-mixing (FFN) layers."""

import os

import torch
import torch.nn as nn

from rwkvt.infctx_module import ChannelMixState

_USE_FUSED_KERNEL = os.environ.get("FUSED_KERNEL", "0") == "1"

if _USE_FUSED_KERNEL:
    from rwkvfla.ops.rwkv7 import channel_mixing_rwkv7
else:  # pragma: no cover - import shim
    channel_mixing_rwkv7 = None


def RWKV_Cmix_v7(*args, **kwargs):
    """Build the channel-mixing layer matching the current training mode."""
    train_type = os.environ.get("RWKV_TRAIN_TYPE", "")
    if train_type == "infctx":
        return RWKV_CMix_x070_infctx(*args, **kwargs)
    if train_type == "fullstate":
        return RWKV_CMix_x070_FullState(*args, **kwargs)
    if _USE_FUSED_KERNEL:
        return RWKV_CMix_x070_fla(*args, **kwargs)
    return RWKV_CMix_x070(*args, **kwargs)


class RWKV_CMix_x070(nn.Module):
    variant = "base"

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

    def _shifted(self, x, attention_mask=None, last_state=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        if self.variant == "infctx":
            return torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xx = self.time_shift(x) - x
        if self.variant == "fullstate":
            xx[:, 0, :] += self.ts_state
        return xx

    def _mix(self, x, xx):
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

    def forward(self, x, attention_mask=None, last_state=None):
        xx = self._shifted(x, attention_mask, last_state)
        if self.variant == "infctx":
            return self._mix(x, xx), ChannelMixState(x[:, -1])
        return self._mix(x, xx)


class RWKV_CMix_x070_fla(RWKV_CMix_x070):
    """rwkv-fla fused channel mixing (no time shift state needed)."""

    variant = "fla"

    def forward(self, x, attention_mask=None, last_state=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        x_prev = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        output, _ = channel_mixing_rwkv7(x, x_prev, self.x_k,
                                         self.key.weight.t(), self.value.weight.t())
        return output


class RWKV_CMix_x070_infctx(RWKV_CMix_x070):
    """Infinite context channel mixing, carries the shift state."""

    variant = "infctx"


class RWKV_CMix_x070_FullState(RWKV_CMix_x070):
    """Channel mixing with an extra learned bias for the first shift vector."""

    variant = "fullstate"

    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        self.ts_state = nn.Parameter(torch.zeros(args.n_embd))
