"""Download ``allenai/tulu-3-sft-mixture`` and convert it to RWKV binidx.

The dataset is a ready-mixed SFT blend (math / code / chat / instruction /
multilingual, ~939k rows). For an MTP head we only need plain text, so each row's
``messages`` are flattened with role markers and one EOS is appended.

    python scripts/prepare_tulu_binidx.py \
        --output-prefix /data/binidx/tulu \
        --workers 16

Output: ``<prefix>.bin`` (uint16 tokens), ``<prefix>.idx`` (Megatron format) and
``<prefix>.json`` (doc/token counts). Re-running is a no-op once ``.json`` exists
unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_parquet_to_binidx import DTYPE, merge_shards, write_idx

ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### Instruction:\n",
    "assistant": "### Response:\n",
}


def format_row(messages) -> str:
    """Flatten a chat transcript into the plain text the MTP head is trained on."""
    parts = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        role = message.get("role", "")
        parts.append(ROLE_PREFIX.get(role, "") + content)
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n"


def build_tokenizer(impl, tokenizer_path, vocab_path):
    if impl == "rwkv":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "json2binidx_tool", "tools"))
        from rwkv_tokenizer import RWKV_TOKENIZER
        return RWKV_TOKENIZER(vocab_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,
                                         use_fast=True)


def tokenize_chunk(task):
    """Worker: tokenise one chunk of documents into its own shard."""
    index, texts, prefix, tokenizer_path, vocab_path, impl, eos = task
    tokenizer = build_tokenizer(impl, tokenizer_path, vocab_path)
    sizes = []
    t0 = time.time()
    with open(prefix + ".bin", "wb") as out:
        for offset in range(0, len(texts), 512):
            batch = [t for t in texts[offset:offset + 512] if t]
            if not batch:
                continue
            if impl == "rwkv":
                encoded = [tokenizer.encode(text) for text in batch]
            else:
                encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                ids = [*ids, 0] if eos else ids
                out.write(np.asarray(ids, dtype=DTYPE).tobytes())
                sizes.append(len(ids))
    write_idx(prefix + ".idx", np.asarray(sizes, dtype=np.int32))
    return {"index": index, "doc": len(sizes), "tokens": int(sum(sizes)),
            "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True,
                        help="e.g. /data/binidx/tulu -> tulu.bin / tulu.idx / tulu.json")
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="RWKV/rwkv-5-world-3b")
    parser.add_argument("--tokenizer-impl", default="hf", choices=("hf", "rwkv"),
                        help="hf = transformers tokenizer (fast, default); "
                             "rwkv = json2binidx_tool's greedy tokenizer (exact "
                             "RWKV training pipeline, ~10x slower)")
    parser.add_argument("--vocab-file", default="", help="vocab for --tokenizer-impl rwkv")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--docs-per-shard", type=int, default=20000)
    parser.add_argument("--max-docs", type=int, default=0, help="0 = whole dataset")
    parser.add_argument("--eos", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    prefix = args.output_prefix
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    if os.path.exists(prefix + ".json") and not args.force:
        print(f"[tulu] {prefix}.json exists, nothing to do "
              f"(use --force to rebuild)")
        return 0

    from datasets import load_dataset

    print(f"[tulu] loading {args.dataset} ({args.split}) ...", flush=True)
    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset) if not args.max_docs else min(len(dataset), args.max_docs)
    print(f"[tulu] {total} rows, formatting + tokenising with {args.workers} workers",
          flush=True)

    vocab = args.vocab_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "json2binidx_tool",
        "rwkv_vocab_v20230424.txt")
    chunks = []
    for start in range(0, total, args.docs_per_shard):
        stop = min(start + args.docs_per_shard, total)
        rows = dataset.select(range(start, stop))
        texts = [format_row(messages) for messages in rows["messages"]]
        chunks.append((len(chunks), texts, f"{prefix}.part{len(chunks):04d}",
                       args.tokenizer, vocab, args.tokenizer_impl,
                       bool(args.eos)))

    results, t0 = [], time.time()
    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(tokenize_chunk, chunks), 1):
            results.append(result)
            done = sum(r["tokens"] for r in results)
            elapsed = time.time() - t0
            print(f"[tulu] {i}/{len(chunks)} shards  {done / 1e6:.0f}M tokens  "
                  f"{done / elapsed / 1e3:.0f} ktok/s", flush=True)

    ordered = [f"{prefix}.part{index:04d}" for index in range(len(chunks))]
    docs, tokens = merge_shards(ordered, prefix)
    for name in ordered:
        for suffix in (".bin", ".idx"):
            path = name + suffix
            if os.path.exists(path):
                os.remove(path)
    with open(prefix + ".json", "w") as handle:
        json.dump({"dataset": args.dataset, "docs": docs, "tokens": tokens,
                   "seqs@8192": tokens // 8192}, handle, indent=2)
    print(f"[tulu] done: {docs} docs, {tokens / 1e9:.2f}B tokens, "
          f"{tokens // 8192} seqs of 8192, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
