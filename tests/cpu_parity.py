"""Dump / compare model outputs while refactoring the model layer.

Usage:
    python tests/cpu_parity.py dump  --version 7 --mode normal --out /tmp/a.pt
    python tests/cpu_parity.py compare /tmp/a.pt /tmp/b.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _cpu_stub as stub


def build_batch(version: str, mode: str):
    args = stub.build_args(version, train_type="none" if mode == "normal" else mode)
    t = args.chunk_ctx if mode == "infctx" else args.ctx_len
    torch.manual_seed(1)
    idx = torch.randint(0, args.vocab_size, (2, t))
    mask = torch.ones(2, t, dtype=torch.bool)
    return args, idx, mask


def dump(version: str, mode: str, grad_cp: int, out_path: str) -> None:
    model, args = stub.load_model(version=version, mode=mode, grad_cp=grad_cp,
                                  seed=0)
    args.grad_cp = grad_cp
    model.eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
            else:
                p.fill_(0.01)

    _, idx, mask = build_batch(version, mode)
    out = {}
    with torch.no_grad():
        if mode == "infctx":
            from rwkvt.infctx_module import BlockStateList

            h = args.dim_att // args.head_size_a
            states = BlockStateList.create(args.n_layer, idx.shape[0], args.n_embd,
                                           h, idx.device, torch.float32)
            y, shift, wkv = model.forward_infctx(idx, states.shift_states,
                                                 states.wkv_states, mask)
            out.update(y=y, shift=shift.float(), wkv=wkv.float())
        elif version == "5":
            out["y"] = model(idx)
        elif version == "7":
            out["y"] = model(idx, attention_mask=mask)
        else:
            out["y"] = model(idx, mask)

    torch.save(out, out_path)
    print(f"saved {sorted(out)} -> {out_path}")


def compare(path_a: str, path_b: str) -> int:
    a, b = torch.load(path_a), torch.load(path_b)
    assert set(a) == set(b), set(a) ^ set(b)
    bad = [(k, (a[k] - b[k]).abs().max().item()) for k in a
           if not torch.equal(a[k], b[k])]
    if bad:
        print("MISMATCH:", bad)
        return 1
    print(f"OK identical: {len(a)} tensors")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--version", default="7", choices=["5", "6", "7"])
    d.add_argument("--mode", default="normal",
                   choices=["normal", "state", "infctx", "fullstate"])
    d.add_argument("--grad-cp", type=int, default=0)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    ns = parser.parse_args()

    if ns.cmd == "compare":
        return compare(ns.a, ns.b)
    dump(ns.version, ns.mode, ns.grad_cp, ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
