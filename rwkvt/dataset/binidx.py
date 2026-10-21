"""Memory-mapped ``.bin``/``.idx`` dataset reader (Megatron format).

Trimmed down to the read path used by RWKV-PEFT: ``__getitem__``, ``head``,
``pad`` and ``only``. The writer lives in ``json2binidx_tool``.
"""

import os
import struct
from functools import lru_cache

import numpy as np
import torch

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}


def code(dtype):
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path: str) -> str:
    return prefix_path + ".idx"


def data_file_path(prefix_path: str) -> str:
    return prefix_path + ".bin"


def _warmup_mmap_file(path: str) -> None:
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        def __init__(self, path: str, skip_warmup: bool = False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert magic_test == self._HDR_MAGIC, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                (version,) = struct.unpack("<Q", stream.read(8))
                assert version == 1
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize
                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32,
                                        count=self._len, offset=offset)
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len,
                offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup: bool = False):
        super().__init__()
        self._path = path
        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup: bool = False):
        self._path = path
        self._index = self.Index(index_file_path(path), skip_warmup)
        if not skip_warmup:
            _warmup_mmap_file(data_file_path(path))
        self._bin_buffer_mmap = np.memmap(data_file_path(path), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return getattr(self._index, "_truncate_size", None) or len(self._index)

    def head(self, n: int):
        """Zero-copy truncation to the first ``n`` documents (clamped)."""
        self._index._truncate_size = min(n, len(self))
        return self

    def _read(self, idx: int, count: int):
        ptr, _ = self._index[idx]
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                             count=count, offset=ptr)

    def __getitem__(self, idx: int):
        _, size = self._index[idx]
        return self._read(idx, size)

    def get(self, idx: int, offset: int = 0, length: int | None = None):
        """Single document, optionally offset/truncated."""
        ptr, size = self._index[idx]
        length = size - offset if length is None else length
        ptr += offset * np.dtype(self._index.dtype).itemsize
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                             count=length, offset=ptr)

    def pad(self, idx: int, length: int):
        """Return exactly ``length`` tokens, wrapping around the dataset.

        Wrapping continues past the first document, so a dataset that is
        shorter than ``ctx_len`` (small SFT corpora, smoke tests) still works
        instead of reading past the end of the mmap.
        """
        _, size = self._index[idx]
        if size >= length:
            return self._read(idx, length).astype(int), length

        parts = [self._read(idx, size)]
        need = length - size
        doc, n_docs = 0, len(self)
        while need > 0:
            progressed = False
            for _ in range(n_docs):
                take = min(int(self.sizes[doc]), need)
                if take:
                    parts.append(self._read(doc, take))
                    need -= take
                    progressed = True
                doc = (doc + 1) % n_docs
                if need == 0:
                    break
            if not progressed:
                raise ValueError("cannot pad: the dataset contains no tokens")
        return np.concatenate(parts).astype(int), size

    def only(self, idx: int, length: int):
        """Return ``min(size, length)`` tokens without wrapping."""
        _, size = self._index[idx]
        return self._read(idx, min(size, length)).astype(int)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path))
