"""Measure whether an MTP head actually speeds up decoding.

The MTP *loss* says nothing about speedup: the speedup depends on the
**acceptance rate** (how often the drafter's token survives the verifier).

This script runs the greedy draft/verify loop from :mod:`rwkvt.spec_decode` on
real prompts and reports ``accept_rate`` and ``tokens per target forward``
(``1 + alpha`` for depth 1, matching DeepSeek's accounting). RWKV re-runs the
whole prefix here (no KV cache), so the wall-clock is not representative - the
acceptance rate is the hardware-independent number.

Usage::

    python scripts/measure_mtp_acceptance.py \
        --load_model /path/to/rwkv7-13b.pth \
        --mtp_path   /path/to/out/rwkv-0-mtp.pth \
        --n_layer 61 --n_embd 4096 --vocab_size 65536 \
        --tokenizer_path RWKV/rwkv-5-world-3b \
        --max_new_tokens 32 --depth 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "Question: If x + 3 = 10, what is x?\nAnswer:",
    "Question: A train travels 60 km in 1.5 hours. What is its average speed?\nAnswer:",
    "Question: What is 15% of 240?\nAnswer:",
    "Question: Solve for y: 2y - 7 = 11.\nAnswer:",
    "Question: Compute 7! / 5!.\nAnswer:",
    "Question: A rectangle has sides 4 and 9. What is its area?\nAnswer:",
    "Question: Simplify (3x^2)(4x^3).\nAnswer:",
    "Question: What is the next prime after 17?\nAnswer:",
]


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
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--trace", action="store_true",
                        help="print the per-round draft/verify table")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)

    # same env-var contract train.py uses (must be set before importing rwkvt)
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
        "--proj_dir", "/tmp/mtp_acceptance",
        "--data_file", cli.load_model, "--data_type", "binidx",
        "--n_layer", str(cli.n_layer), "--n_embd", str(cli.n_embd),
        "--vocab_size", str(cli.vocab_size), "--head_size_a", str(cli.head_size_a),
        "--ctx_len", str(cli.ctx_len), "--my_testing", cli.my_testing,
        "--precision", "bf16-true", "--op", "cuda", "--peft", "mtp",
        "--mtp_depth", str(cli.depth),
    ]))

    print(f"[load] {cli.load_model}")
    model = build_model(args)
    model = model.to(cli.device).eval()

    mtp = MTPHead(args, depth=cli.depth)
    state = torch.load(cli.mtp_path, map_location="cpu")
    state = {k.removeprefix("mtp."): v for k, v in state.items()}
    missing, unexpected = mtp.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[mtp] missing={missing[:3]} unexpected={unexpected[:3]}")
    mtp = mtp.to(device=cli.device, dtype=next(model.parameters()).dtype).eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer_path,
                                              trust_remote_code=True)
    torch.manual_seed(0)
    total = SpecStats()
    with torch.no_grad():
        for prompt in PROMPTS[: max(1, cli.prompts)]:
            ids = tokenizer(prompt).input_ids
            tokens = torch.tensor([ids], device=cli.device)
            stats = SpecStats()
            trace = [] if cli.trace else None
            out = mtp_greedy_decode(model, mtp, tokens, cli.max_new_tokens,
                                    depth=cli.depth, stats=stats, trace=trace)
            if trace:
                for row in trace:
                    print("    round {round:2d}: drafted {drafted} -> accepted {accepted}"
                          "  emits {emitted} token(s)".format(**row))
            baseline = greedy_decode(model, tokens, min(8, cli.max_new_tokens))
            same = torch.equal(out[:, :baseline.size(1)], baseline)
            total.rounds += stats.rounds
            total.drafted += stats.drafted
            total.accepted += stats.accepted
            print(f"  prompt={prompt[:40]!r} drafted={stats.drafted} "
                  f"accepted={stats.accepted} rate={stats.accept_rate:.3f} "
                  f"max_exact_match={same}")

    print("\n=== summary ===")
    print(f"depth          : {cli.depth}")
    print(f"accept_rate    : {total.accept_rate:.3f}  "
          f"({total.accepted}/{total.drafted})")
    print(f"tokens/forward : {total.tokens_per_round:.3f}  (1 + alpha)")
    print(f"est. speedup   : {total.tokens_per_round:.2f}x "
          "(ignoring verify overhead; RWKV verify is one full forward)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
