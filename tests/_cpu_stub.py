"""CPU-only stubs used by the tests and by ``cpu_parity.py``.

The real WKV kernels need a CUDA compiler (or rwkv-fla / triton); these stubs
replace them with deterministic element-wise functions that keep the same
signatures and shapes so that the model plumbing can be exercised on CPU.

This is a *plumbing* harness: it verifies shapes, state handling and gradient
flow, not the numerical correctness of the kernels.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import torch


# --------------------------------------------------------------------------- #
# environment / module stubs
# --------------------------------------------------------------------------- #
def install_env(version: str = "x070", train_type: str = "",
                fused_kernel: bool = False) -> None:
    os.environ["WKV"] = "fla"
    os.environ["RWKV_MY_TESTING"] = version
    os.environ["RWKV_TRAIN_TYPE"] = train_type
    os.environ["FUSED_KERNEL"] = "1" if fused_kernel else "0"
    os.environ["RWKV_HEAD_SIZE_A"] = "8"
    os.environ["RWKV_CTXLEN"] = "16"
    os.environ["RWKV_FLOAT_MODE"] = "fp32"


def _fake_mix(*tensors):
    """Linear combination of the inputs so every argument reaches the graph."""
    out = None
    scale = 1.0
    for t in tensors:
        term = t * scale
        out = term if out is None else out + term
        scale *= 0.5
    return out


def install_rwkvfla_stub() -> None:
    if "rwkvfla" in sys.modules:
        return

    modules = {
        "rwkvfla": types.ModuleType("rwkvfla"),
        "rwkvfla.ops": types.ModuleType("rwkvfla.ops"),
        "rwkvfla.ops.rwkv7": types.ModuleType("rwkvfla.ops.rwkv7"),
        "rwkvfla.ops.rwkv6": types.ModuleType("rwkvfla.ops.rwkv6"),
        "rwkvfla.modules": types.ModuleType("rwkvfla.modules"),
        "rwkvfla.modules.layernorm": types.ModuleType("rwkvfla.modules.layernorm"),
    }

    def chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None,
                    output_final_state=False, head_first=False):
        out = _fake_mix(r, w, k, v, a, b)
        state = None
        if output_final_state:
            bsz, _, heads, head_dim = r.shape
            state = torch.zeros(bsz, heads, head_dim, head_dim, dtype=r.dtype)
        return out, state

    def chunk_rwkv6(r, k, v, w, u, scale=1.0, initial_state=None,
                    output_final_state=False):
        out = _fake_mix(r, k, v, w) + u.mean() * 0.0
        state = None
        if output_final_state:
            bsz, heads, _, head_dim = r.shape
            state = torch.zeros(bsz, heads, head_dim, head_dim, dtype=r.dtype)
        return out, state

    def fused_addcmul_rwkv7(x, xx, x_r, x_w, x_k, x_v, x_a, x_g):
        return (x + xx * x_r, x + xx * x_w, x + xx * x_k,
                x + xx * x_v, x + xx * x_a, x + xx * x_g)

    def channel_mixing_rwkv7(x, x_prev, x_k, key_weight, value_weight):
        k = torch.relu(x + (x_prev - x) * x_k) ** 2
        hidden = torch.relu(k @ key_weight.t()) ** 2
        return hidden @ value_weight.t(), None

    modules["rwkvfla.ops.rwkv7"].chunk_rwkv7 = chunk_rwkv7
    modules["rwkvfla.ops.rwkv7"].fused_addcmul_rwkv7 = fused_addcmul_rwkv7
    modules["rwkvfla.ops.rwkv7"].channel_mixing_rwkv7 = channel_mixing_rwkv7
    modules["rwkvfla.ops.rwkv6"].chunk_rwkv6 = chunk_rwkv6
    modules["rwkvfla.modules"].FusedCrossEntropyLoss = torch.nn.CrossEntropyLoss
    modules["rwkvfla.modules.layernorm"].GroupNorm = torch.nn.GroupNorm
    sys.modules.update(modules)


# --------------------------------------------------------------------------- #
# fake kernels (same call ABI as the CUDA/FLA implementations)
# --------------------------------------------------------------------------- #
def fake_rwkv7g(r, w, k, v, a, b, HEAD_SIZE=8):
    return _fake_mix(r, w, k, v, a, b)


def fake_rwkv7_state(r, k, v, w, a, b, s, HEAD_SIZE=8):
    return _fake_mix(r, k, v, w, a, b), s * 0.5


def fake_rwkv7_infctx(r, k, v, w, a, b, s, HEAD_SIZE=8):
    return _fake_mix(r, k, v, w, a, b), s * 0.5


def fake_rwkv6(B, T, C, H, r, k, v, w, u):
    return _fake_mix(r, k, v, w)


def fake_rwkv6_state(B, T, C, H, r, k, v, w, u, s):
    out = _fake_mix(r, k, v, w)
    if os.environ.get("RWKV_TRAIN_TYPE") == "infctx":
        return out, s * 0.5
    return out


def fake_rwkv5(B, T, C, H, r, k, v, w, u):
    return _fake_mix(r, k, v)


def patch_ops() -> None:
    """Override the backend chosen by rwkvop with the CPU fakes (import order!)."""
    import rwkvt.operator.rwkvop as ops

    ops.RUN_CUDA_RWKV7g = fake_rwkv7g
    ops.RUN_RWKV7_STATE = fake_rwkv7_state
    ops.RUN_RWKV7_INFCTX = fake_rwkv7_infctx
    ops.RUN_CUDA_RWKV6 = fake_rwkv6
    ops.RUN_CUDA_RWKV6_STATE = fake_rwkv6_state
    ops.RUN_CUDA_RWKV5 = fake_rwkv5


# --------------------------------------------------------------------------- #
# tiny model factory
# --------------------------------------------------------------------------- #
def build_args(version: str, grad_cp: int = 0, train_type: str = "none"):
    return SimpleNamespace(
        vocab_size=64, n_embd=32, dim_att=32, dim_ffn=64, n_layer=3,
        head_size_a=8, head_size_divisor=8, ctx_len=16, chunk_ctx=8,
        grad_cp=grad_cp, peft="lora", train_type=train_type,
        my_testing=f"x0{version}0", my_pile_stage=0, layerwise_lr=1,
        weight_decay=0.0,
    )


def load_model(version: str = "7", mode: str = "normal", grad_cp: int = 0,
               seed: int = 0):
    """(Re)build a tiny model for ``version``/``mode`` and return (model, args).

    The operator module branches on env vars at import time, so every reload
    has to drop the previously imported ``rwkvt.*`` modules first.
    """
    train_type = "" if mode == "normal" else mode
    install_env(version=f"x0{version}0", train_type=train_type)
    install_rwkvfla_stub()
    for name in [m for m in sys.modules if m.startswith("rwkvt.")]:
        del sys.modules[name]
    patch_ops()

    module = importlib.import_module(f"rwkvt.rwkv{version}.model")
    args = build_args(version, grad_cp=grad_cp,
                      train_type=train_type or "none")
    torch.manual_seed(seed)
    model = getattr(module, f"RWKV{version}")(args)
    return model, args
