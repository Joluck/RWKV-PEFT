"""Measure the MTP acceptance rate for every epoch checkpoint as it appears.

The base model is loaded once; each new ``rwkv-<n>-mtp.pth`` only needs the
151M head to be swapped in, so watching a full run costs ~10s per epoch.

    python scripts/mtp_epoch_bench.py \
        --load_model /path/to/rwkv7-13b.pth \
        --ckpt_dir   /path/to/out \
        --tokenizer_path RWKV/rwkv-5-world-3b \
        --n_layer 61 --n_embd 4096 --vocab_size 65536 \
        --max_new_tokens 32 --prompts 8 --watch

Results are appended to ``<ckpt_dir>/mtp_bench.jsonl``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_model", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--tokenizer_path", default="RWKV/rwkv-5-world-3b")
    parser.add_argument("--n_layer", type=int, default=61)
    parser.add_argument("--n_embd", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--head_size_a", type=int, default=64)
    parser.add_argument("--ctx_len", type=int, default=512)
    parser.add_argument("--my_testing", default="x070")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--domains", default="math,prose,wiki,arc",
                        help="comma separated domains (empty = built-in prompts)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max_epoch", type=int, default=-1)
    return parser.parse_args(argv)


def build_base(cli):
    os.environ["RWKV_MY_TESTING"] = cli.my_testing
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["WKV"] = "cuda"
    os.environ["RWKV_HEAD_SIZE_A"] = str(cli.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = "none"

    import torch

    import train
    from rwkvt.peft_loading import build_model

    args = train.finalize_args(train.build_parser().parse_args([
        "--load_model", cli.load_model,
        "--proj_dir", "/tmp/mtp_epoch_bench",
        "--data_file", cli.load_model, "--data_type", "binidx",
        "--n_layer", str(cli.n_layer), "--n_embd", str(cli.n_embd),
        "--vocab_size", str(cli.vocab_size), "--head_size_a", str(cli.head_size_a),
        "--ctx_len", str(cli.ctx_len), "--my_testing", cli.my_testing,
        "--precision", "bf16-true", "--op", "cuda", "--peft", "mtp",
        "--mtp_depth", str(cli.depth),
    ]))
    model = build_model(args).to(cli.device).eval()
    return model, args, torch


def load_head(args, path, cli, torch, dtype):
    from rwkvt.mtp import MTPHead

    mtp = MTPHead(args, depth=cli.depth)
    state = torch.load(path, map_location="cpu")
    state = {k.removeprefix("mtp."): v for k, v in state.items()}
    mtp.load_state_dict(state, strict=True)
    return mtp.to(device=cli.device, dtype=dtype).eval()


def measure(model, mtp, tokenizer, cli, prompts, torch):
    from rwkvt.spec_decode import SpecStats, mtp_greedy_decode

    total = SpecStats()
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt).input_ids
            tokens = torch.tensor([ids], device=cli.device)
            mtp_greedy_decode(model, mtp, tokens, cli.max_new_tokens,
                              depth=cli.depth, stats=total)
    return {"drafted": total.drafted, "accepted": total.accepted,
            "accept_rate": round(total.accept_rate, 4),
            "tokens_per_forward": round(total.tokens_per_round, 4)}


#: prompt sources. ``sft`` = the MetaMathQA training template (in domain),
#: ``raw`` = plain continuation, ``qa`` = short answer-style question.
DOMAINS = {
    "math": {"dataset": "meta-math/MetaMathQA", "config": None,
             "split": "train", "field": "query", "template": "sft",
             "offset": 100000},
    "prose": {"dataset": "baber/paul_graham_essays", "config": None,
              "split": "train", "field": "text", "template": "raw", "offset": 0},
    "wiki": {"dataset": "EleutherAI/wikitext_document_level",
             "config": "wikitext-103-raw-v1", "split": "train",
             "field": "page", "template": "raw", "offset": 0},
    "arc": {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge",
            "split": "test", "field": "question", "template": "qa", "offset": 0},
}


def build_prompts(domain, cli):
    from datasets import load_dataset

    from rwkvt.dataset.SFTdataset import PROMPT

    spec = DOMAINS[domain]
    split = f"{spec['split']}[{spec['offset']}:{spec['offset'] + cli.prompts}]"
    rows = load_dataset(spec["dataset"], spec["config"], split=split)
    prompts = []
    for row in rows:
        text = row[spec["field"]]
        if spec["template"] == "sft":
            prompts.append(PROMPT.format_map({"instruction": text}))
        elif spec["template"] == "qa":
            prompts.append(f"Question: {text}\nAnswer:")
        else:
            prompts.append(text[:400])
    return prompts


def main(argv=None):
    cli = parse_args(argv)
    domains = [d for d in cli.domains.split(",") if d]
    if not domains:
        from measure_mtp_acceptance import PROMPTS

        prompts_by_domain = {"builtin": PROMPTS[: max(1, cli.prompts)]}
    else:
        prompts_by_domain = {d: build_prompts(d, cli) for d in domains}
    print(f"[bench] domains={list(prompts_by_domain)} "
          f"({cli.prompts} prompts x {cli.max_new_tokens} tokens)", flush=True)
    print(f"[bench] loading {cli.load_model} on {cli.device} ...", flush=True)
    model, args, torch = build_base(cli)
    dtype = next(model.parameters()).dtype

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer_path,
                                              trust_remote_code=True)
    out_path = os.path.join(cli.ckpt_dir, "mtp_bench.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as handle:
            done = {json.loads(line)["epoch"] for line in handle if line.strip()}
    print(f"[bench] already measured: {sorted(done)}", flush=True)

    while True:
        for path in sorted(glob.glob(os.path.join(cli.ckpt_dir, "rwkv-*-mtp.pth"))):
            match = re.search(r"rwkv-(\d+)-mtp\.pth$", path)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch in done:
                continue
            if os.path.getsize(path) < 1024:
                continue  # still being written
            mtp = load_head(args, path, cli, torch, dtype)
            for domain, prompts in prompts_by_domain.items():
                result = measure(model, mtp, tokenizer, cli, prompts, torch)
                result.update({"epoch": epoch, "domain": domain})
                with open(out_path, "a") as handle:
                    handle.write(json.dumps(result) + "\n")
                print(f"[bench] epoch {epoch} {domain:5s}: "
                      f"alpha={result['accept_rate']:.3f} "
                      f"tokens/forward={result['tokens_per_forward']:.3f} "
                      f"({result['accepted']}/{result['drafted']})", flush=True)
            done.add(epoch)
            del mtp
            torch.cuda.empty_cache()

        if not cli.watch:
            break
        if cli.max_epoch >= 0 and done and max(done) >= cli.max_epoch:
            print("[bench] reached --max_epoch, done", flush=True)
            break
        time.sleep(cli.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
