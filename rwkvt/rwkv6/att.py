"""RWKV-6 time-mixing (attention) layers.

``base`` / ``state`` / ``infctx`` share the same math; only the WKV call and the
state plumbing differ. The initialisation of the mix/decay LoRAs intentionally
differs between ``base`` and the state variants and is kept byte-identical.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkvt.infctx_module import TimeMixState
from rwkvt.operator.rwkvop import RUN_CUDA_RWKV6, RUN_CUDA_RWKV6_STATE


def RWKV_Tmix_v6(*args, **kwargs):
    """Build the time-mixing layer matching the current training mode."""
    train_type = os.environ.get("RWKV_TRAIN_TYPE", "")
    if train_type == "state":
        return RWKV_Tmix_x060_state(*args, **kwargs)
    if train_type == "infctx":
        return RWKV_Tmix_x060_infctx(*args, **kwargs)
    return RWKV_Tmix_x060(*args, **kwargs)


class RWKV_Tmix_x060(nn.Module):
    variant = "base"

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size

        C = args.n_embd
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            time_mix_extra_dim = 64 if C == 4096 else 32
            time_decay_extra_dim = 128 if C == 4096 else 64

            if self.variant == "base":
                self.time_maa_w1 = nn.Parameter(
                    torch.zeros(C, time_mix_extra_dim * 5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(
                    torch.zeros(5, time_mix_extra_dim, C).uniform_(-1e-4, 1e-4))
            else:  # state / infctx start from a zero LoRA
                self.time_maa_w1 = nn.Parameter(torch.zeros(C, time_mix_extra_dim * 5))
                self.time_maa_w2 = nn.Parameter(
                    torch.zeros(5, time_mix_extra_dim, C).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            if self.variant == "base":
                self.time_decay_w1 = nn.Parameter(
                    torch.zeros(C, time_decay_extra_dim).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(
                    torch.zeros(time_decay_extra_dim, args.dim_att).uniform_(-1e-4, 1e-4))
            else:
                self.time_decay_w1 = nn.Parameter(torch.zeros(C, time_decay_extra_dim))
                self.time_decay_w2 = nn.Parameter(
                    torch.zeros(time_decay_extra_dim, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            if self.variant == "state":
                self.time_state = nn.Parameter(
                    torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, args.dim_att, bias=False)
        self.key = nn.Linear(C, args.dim_att, bias=False)
        self.value = nn.Linear(C, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, C, bias=False)
        self.gate = nn.Linear(C, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att,
                                 eps=(1e-5) * (args.head_size_divisor ** 2))

    def _shift(self, x, last_state=None):
        if self.variant == "infctx":
            return torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                                dim=1) - x
        return self.time_shift(x) - x

    def mix(self, x, xx):
        """Time-mix projections: returns (r, k, v, g, w)."""
        B, T, _ = x.size()
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        return r, k, v, g, w

    def _apply_output(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        return self.output(x * g)

    def forward(self, x, attention_mask=None, last_state=None):
        B, T, C = x.size()
        H = self.n_head
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        shift_state = x[:, -1, :]

        r, k, v, g, w = self.mix(x, self._shift(x, last_state))
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]

        if self.variant == "base":
            mixed = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)
            return self._apply_output(mixed, g)

        if self.variant == "infctx":
            state = last_state.wkv_state.clone().contiguous()
            mixed, wkv_state = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w,
                                                    u=self.time_faaaa, s=state)
            return self._apply_output(mixed, g), TimeMixState(shift_state, wkv_state)

        mixed = RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w,
                                     u=self.time_faaaa, s=self.time_state)
        return self._apply_output(mixed, g)

class RWKV_Tmix_x060_state(RWKV_Tmix_x060):
    """State tuning: adds a learned per-head state fed to the kernel."""

    variant = "state"


class RWKV_Tmix_x060_infctx(RWKV_Tmix_x060):
    """Infinite context: carries shift/wkv state across chunks."""

    variant = "infctx"
