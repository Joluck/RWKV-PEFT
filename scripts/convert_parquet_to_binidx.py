"""Convert a parquet text corpus to RWKV ``.bin``/``.idx`` (binidx) shards.

The RWKV tokenizer is a slow Python tokenizer, so the work is split into row
ranges, tokenised in parallel processes, and the shards are merged at the end.

    python scripts/convert_parquet_to_binidx.py \
        --input-dir /home/rwkv/data/fineweb-10BT \
        --output-prefix /home/rwkv/data/fineweb-10BT/binidx/fineweb \
        --workers 32 --rows-per-shard 200000

Output format is Megatron binidx with ``uint16`` tokens and one EOS token
appended per document.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import time

import numpy as np

DTYPE = np.uint16
DTYPE_CODE = 8
MAGIC = b"MMIDIDX\x00\x00"
HEADER = 34  # magic(9) + version(8) + dtype(1) + ndocs(8) + doc_count(8)


def write_idx(path: str, sizes: np.ndarray) -> None:
    sizes = sizes.astype(np.int64)
    pointers = np.zeros(len(sizes), dtype=np.int64)
    if len(sizes):
        pointers[1:] = np.cumsum(sizes)[:-1] * np.dtype(DTYPE).itemsize
    # layout copied byte-for-byte from json2binidx_tool/tools/indexed_dataset.py
    # (one item per document, so doc_idx is 0..n and doc_count is n + 1)
    doc_idx = np.arange(len(sizes) + 1, dtype=np.int64)
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", DTYPE_CODE))
        handle.write(struct.pack("<Q", len(sizes)))
        handle.write(struct.pack("<Q", len(doc_idx)))
        handle.write(sizes.astype(np.int32).tobytes())
        handle.write(pointers.tobytes())
        handle.write(doc_idx.tobytes())


def read_idx_sizes(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        header = handle.read(HEADER)
        assert header[:9] == MAGIC, path
        ndocs = struct.unpack("<Q", header[18:26])[0]
        return np.frombuffer(handle.read(4 * ndocs), dtype=np.int32)


def tokenize_shard(task):
    """Worker: tokenise rows ``[start, end)`` of one parquet file into a shard."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    path, rg_start, rg_end, out_prefix, tokenizer_path, eos = task
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,
                                              use_fast=True)
    parquet = pq.ParquetFile(path)
    sizes = []
    total = 0
    t0 = time.time()
    with open(out_prefix + ".bin", "wb") as out:
        for rg in range(rg_start, rg_end):
            table = parquet.read_row_group(rg, columns=["text"])
            for offset in range(0, table.num_rows, 2000):
                texts = table.slice(offset, 2000).column("text").to_pylist()
                for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
                    ids = [*ids, 0] if eos else ids
                    out.write(np.asarray(ids, dtype=DTYPE).tobytes())
                    sizes.append(len(ids))
                    total += len(ids)
    write_idx(out_prefix + ".idx", np.asarray(sizes, dtype=np.int32))
    return {"shard": out_prefix, "docs": len(sizes), "tokens": total,
            "seconds": round(time.time() - t0, 1)}


def merge_shards(shards, prefix):
    sizes = []
    with open(prefix + ".bin", "wb") as out:
        for shard in shards:
            with open(shard + ".bin", "rb") as handle:
                while chunk := handle.read(1 << 26):
                    out.write(chunk)
            sizes.append(read_idx_sizes(shard + ".idx"))
    sizes = (np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int32))
    write_idx(prefix + ".idx", sizes)
    return len(sizes), int(sizes.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tokenizer", default="RWKV/rwkv-5-world-3b")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--rows-per-shard", type=int, default=50000)
    parser.add_argument("--eos", type=int, default=1, help="append EOS (token 0) per doc")
    parser.add_argument("--limit-files", type=int, default=0,
                        help="only convert the first N parquet files (debug)")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {args.input_dir}")
    if args.limit_files:
        files = files[: args.limit_files]

    import pyarrow.parquet as pq

    tasks = []
    for path in files:
        meta = pq.ParquetFile(path).metadata
        # fineweb row groups are tiny (~1.3k rows): group them by row count
        rg_rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        start_rg, rows = 0, 0
        for i, count in enumerate(rg_rows):
            rows += count
            if rows >= args.rows_per_shard or i == len(rg_rows) - 1:
                index = len(tasks)
                tasks.append((path, start_rg, i + 1,
                              f"{args.output_prefix}.part{index:03d}",
                              args.tokenizer, bool(args.eos)))
                start_rg, rows = i + 1, 0
    print(f"[convert] {len(files)} files -> {len(tasks)} shards, "
          f"{args.workers} workers", flush=True)

    import multiprocessing as mp

    results = []
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(tokenize_shard, tasks), 1):
            results.append(result)
            elapsed = time.time() - t0
            done = sum(r["tokens"] for r in results)
            print(f"[convert] {i}/{len(tasks)} shards  {done / 1e6:.0f}M tokens  "
                  f"{done / elapsed / 1e3:.0f} ktok/s  eta "
                  f"{(len(tasks) - i) * elapsed / i / 60:.0f} min", flush=True)

    ordered = [f"{args.output_prefix}.part{index:03d}" for index in range(len(tasks))]
    docs, tokens = merge_shards(ordered, args.output_prefix)
    for name in ordered:
        os.remove(name + ".bin")
        os.remove(name + ".idx")

    with open(args.output_prefix + ".json", "w") as handle:
        json.dump({"docs": docs, "tokens": tokens, "files": files}, handle, indent=2)
    print(f"[convert] done: {docs} docs, {tokens / 1e9:.2f}B tokens, "
          f"seqs@8192={tokens // 8192}, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
