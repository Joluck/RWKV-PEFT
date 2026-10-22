"""Wall-clock benchmark: MTP speculative decoding vs plain greedy decoding.

No theory, no ``1 + alpha``: both loops are timed on the same GPU with the same
(non-state-cached) forward, and the reported speedup is measured seconds.

The absolute tokens/s is low because this harness re-runs the whole prefix on
every step (there is no KV/state cache yet) - but that cost applies to *both*
loops, so the ratio is a real measurement of the algorithm.

    python scripts/bench_spec_speed.py \
        --load_model /path/to/rwkv7-13b.pth \
        --mtp_path   /path/to/out/rwkv-2-mtp.pth \
        --n_layer 61 --n_embd 4096 --vocab_size 65536 \
        --prompts 8 --max_new_tokens 48 --depth 1
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_model", required=True)
    parser.add_argument("--mtp_path", required=True)
    parser.add_argument("--tokenizer_path", default="RWKV/rwkv-5-world-3b")
    parser.add_argument("--n_layer", type=int, default=61)
    parser.add_argument("--n_embd", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--head_size_a", type=int, default=64)
    parser.add_argument("--ctx_len", type=int, default=512)
    parser.add_argument("--my_testing", default="x070")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    os.environ["RWKV_MY_TESTING"] = cli.my_testing
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["WKV"] = "cuda"
    os.environ["RWKV_HEAD_SIZE_A"] = str(cli.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = "none"

    import torch

    import train
    from rwkvt.mtp import MTPHead
    from rwkvt.peft_loading import build_model
    from rwkvt.spec_decode import SpecStats, greedy_decode, mtp_greedy_decode

    args = train.finalize_args(train.build_parser().parse_args([
        "--load_model", cli.load_model,
        "--proj_dir", "/tmp/spec_speed", "--data_file", cli.load_model,
        "--data_type", "binidx",
        "--n_layer", str(cli.n_layer), "--n_embd", str(cli.n_embd),
        "--vocab_size", str(cli.vocab_size), "--head_size_a", str(cli.head_size_a),
        "--ctx_len", str(cli.ctx_len), "--my_testing", cli.my_testing,
        "--precision", "bf16-true", "--op", "cuda", "--peft", "mtp",
        "--mtp_depth", str(cli.depth),
    ]))
    model = build_model(args).to(cli.device).eval()
    dtype = next(model.parameters()).dtype

    mtp = MTPHead(args, depth=cli.depth)
    state = torch.load(cli.mtp_path, map_location="cpu")
    mtp.load_state_dict({k.removeprefix("mtp."): v for k, v in state.items()},
                        strict=True)
    mtp = mtp.to(device=cli.device, dtype=dtype).eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer_path,
                                              trust_remote_code=True)

    from measure_mtp_acceptance import PROMPTS

    prompts = PROMPTS[:cli.prompts]
    prompts = [tokenizer(p).input_ids for p in prompts]

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - start

    # warmup (kernel autotune / cudnn caches / allocator)
    with torch.no_grad():
        for ids in prompts[:cli.warmup]:
            tokens = torch.tensor([ids], device=cli.device)
            greedy_decode(model, tokens, 4)
            mtp_greedy_decode(model, mtp, tokens, 4, depth=cli.depth)

    rows = []
    with torch.no_grad():
        for ids in prompts:
            tokens = torch.tensor([ids], device=cli.device)
            n_new = cli.max_new_tokens
            _, base_s = timed(functools.partial(greedy_decode, model, tokens, n_new))
            stats = SpecStats()
            out, spec_s = timed(functools.partial(
                mtp_greedy_decode, model, mtp, tokens, n_new,
                depth=cli.depth, stats=stats))
            rows.append((base_s, spec_s, stats))
            print(f"  base {base_s:6.3f}s  spec {spec_s:6.3f}s  "
                  f"speedup {base_s / spec_s:5.2f}x  "
                  f"alpha {stats.accept_rate:.3f}  tokens {out.size(1) - tokens.size(1)}")

    total_base = sum(r[0] for r in rows)
    total_spec = sum(r[1] for r in rows)
    drafted = sum(r[2].drafted for r in rows)
    accepted = sum(r[2].accepted for r in rows)
    generated = cli.max_new_tokens * len(rows)
    print("\n=== measured ===")
    print(f"prompts        : {len(rows)} x {cli.max_new_tokens} tokens")
    print(f"baseline       : {total_base:7.3f} s  ({generated / total_base:6.1f} tok/s, "
          f"{generated} forwards)")
    print(f"spec (depth {cli.depth})   : {total_spec:7.3f} s  "
          f"({generated / total_spec:6.1f} tok/s, {drafted} forwards)")
    print(f"speedup        : {total_base / total_spec:.2f}x   (measured wall clock)")
    print(f"accept rate    : {accepted / drafted if drafted else 0:.3f} "
          f"({accepted}/{drafted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
