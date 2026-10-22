"""RWKV-7 time-mixing (attention) layers.

Four execution modes share the same math and only differ in how the WKV scan is
invoked:

* ``base``      - plain training, stateless kernel
* ``state``     - state tuning, kernel takes a learned ``time_state``
* ``infctx``    - infinite context, kernel takes/returns the recurrent state
* ``fullstate`` - state tuning with an extra shift-state bias

Keeping them in one class avoids four near-identical copies of the forward pass.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkvt.infctx_module import TimeMixState
from rwkvt.operator.rwkvop import (
    RUN_RWKV7_INFCTX,
    RUN_RWKV7_STATE,
    RUN_CUDA_RWKV7g,
)

_USE_FUSED_KERNEL = os.environ.get("FUSED_KERNEL", "0") == "1"

if _USE_FUSED_KERNEL:
    from rwkvfla.modules.layernorm import GroupNorm as FusedGroupNorm
    from rwkvfla.ops.rwkv7 import fused_addcmul_rwkv7
else:  # pragma: no cover - import shim
    FusedGroupNorm = None
    fused_addcmul_rwkv7 = None


def RWKV_Tmix_v7(*args, **kwargs):
    """Build the time-mixing layer matching the current training mode."""
    train_type = os.environ.get("RWKV_TRAIN_TYPE", "")
    if train_type == "state":
        return RWKV_Tmix_x070_State(*args, **kwargs)
    if train_type == "infctx":
        return RWKV_Tmix_x070_infctx(*args, **kwargs)
    if train_type == "fullstate":
        return RWKV_Tmix_x070_FullState(*args, **kwargs)
    return RWKV_Tmix_x070(*args, **kwargs)


def _ortho_init(x: torch.Tensor, scale: float) -> torch.Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(f"unsupported init shape {shape}")
    return x


class RWKV_Tmix_x070(nn.Module):
    variant = "base"

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        H, N, C = self.n_head, self.head_size, args.n_embd

        if _USE_FUSED_KERNEL:
            self.addcmul_kernel = self.fused_addcmul
        else:
            self.addcmul_kernel = self.torch_addcmul

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            # decay LoRA (w0 + tanh(xw @ w1) @ w2)
            D_DECAY_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32))  # suggestion
            if args.n_layer == 61:
                D_DECAY_LORA = 192
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5)  # 0.5 comes from F.softplus

            # "a" (in-context learning rate) LoRA
            D_AAA_LORA = max(32, int(round((1.8 * (C ** 0.5)) / 32) * 32))  # suggestion
            if args.n_layer == 61:
                D_AAA_LORA = 192
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            # value residual LoRA
            D_MV_LORA = max(32, int(round((1.3 * (C ** 0.5)) / 32) * 32))  # suggestion
            if args.n_layer == 61:
                D_MV_LORA = 128
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # output gate LoRA (can be shrunk or removed for some data)
            D_GATE_LORA = max(32, int(round((0.6 * (C ** 0.8)) / 32) * 32))  # suggestion
            if C == 1024:
                D_GATE_LORA = 128
            if args.n_layer == 61:
                D_GATE_LORA = 384
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            eps = (1e-5) * (args.head_size_divisor ** 2)  # !!! notice eps value !!!
            if _USE_FUSED_KERNEL:
                self.ln_x = FusedGroupNorm(H, C, eps=eps, bias=True)
            else:
                self.ln_x = nn.GroupNorm(H, C, eps=eps)

    # -- helpers ---------------------------------------------------------- #
    def torch_addcmul(self, x, xx):
        return (x + xx * self.x_r, x + xx * self.x_w, x + xx * self.x_k,
                x + xx * self.x_v, x + xx * self.x_a, x + xx * self.x_g)

    def fused_addcmul(self, x, xx):
        return fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k,
                                   self.x_v, self.x_a, self.x_g)

    def _time_shift(self, x, attention_mask=None, last_state=None):
        """Return (masked x, x_shifted - x)."""
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        if self.variant == "infctx":
            shift = last_state.shift_state.unsqueeze(1)
            xx = torch.concat((shift, x[:, :-1]), dim=1) - x
        else:
            xx = self.time_shift(x) - x
            if self.variant == "fullstate":
                xx[:, 0, :] += self.ts_state
        return x, xx

    def _decay(self, xw):
        """log decay before the kernel (cuda kernels expect raw log values)."""
        log_w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        if os.environ.get("WKV") == "cuda":
            return log_w
        # soft-clamp to (-inf, -0.5) for the fla/triton path
        return -F.softplus(-log_w) - 0.5

    def _project(self, x, xx, v_first):
        """Shared projection half of the block: returns (r, w, k, v, a, g, kk)."""
        B, T, C = x.size()
        H = self.n_head
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        r = self.receptance(xr)
        w = self._decay(xw)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # in-context learning rate
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        return r, w, k, v, g, kk, a, v_first

    def _apply_kernel(self, r, w, k, v, kk, a, last_state=None):
        # NOTE: the infctx/state ops take (r, k, v, w, ...) while the stateless
        # op takes (r, w, k, v, ...); the argument order is part of the ABI.
        if self.variant == "infctx":
            return RUN_RWKV7_INFCTX(r, k, v, w, -kk, kk * a,
                                    last_state.wkv_state.clone().contiguous())
        if self.variant in ("state", "fullstate"):
            x, _ = RUN_RWKV7_STATE(r, k, v, w, -kk, kk * a, self.time_state)
            return x
        return RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)

    def _output(self, x, r, k, v, g):
        # x is (B, T, C) for the CUDA kernel but (B, T, H, C/H) for rwkv-fla,
        # so flatten with reshape instead of unpacking x.size()
        B, T, C = r.size()
        H = self.n_head
        x = self.ln_x(x.reshape(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
                 .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        return self.output(x * g)

    # -- main ------------------------------------------------------------- #
    def forward(self, x, v_first, attention_mask=None, last_state=None):
        x, xx = self._time_shift(x, attention_mask, last_state)
        shift_state = x[:, -1, :] if self.variant == "infctx" else None

        r, w, k, v, g, kk, a, v_first = self._project(x, xx, v_first)
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]

        if self.variant == "infctx":
            mixed, wkv_state = self._apply_kernel(r, w, k, v, kk, a, last_state)
            out = self._output(mixed, r, k, v, g)
            return out, v_first, TimeMixState(shift_state, wkv_state)

        mixed = self._apply_kernel(r, w, k, v, kk, a, last_state)
        return self._output(mixed, r, k, v, g), v_first


class RWKV_Tmix_x070_State(RWKV_Tmix_x070):
    """State tuning: adds a learned per-head state fed to the kernel."""

    variant = "state"

    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        with torch.no_grad():
            self.time_state = nn.Parameter(
                torch.zeros(self.n_head, self.head_size, self.head_size))


class RWKV_Tmix_x070_infctx(RWKV_Tmix_x070):
    """Infinite context: carries shift/wkv state across chunks."""

    variant = "infctx"


class RWKV_Tmix_x070_FullState(RWKV_Tmix_x070):
    """State tuning on both the recurrent state and the first shift vector."""

    variant = "fullstate"

    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        with torch.no_grad():
            self.time_state = nn.Parameter(
                torch.zeros(self.n_head, self.head_size, self.head_size))
            self.ts_state = nn.Parameter(torch.zeros(args.n_embd))
