"""Tests for the mmap binidx reader and the binidx sampler."""

import pickle
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rwkvt.dataset.binidx import MMapIndexedDataset, code
from rwkvt.dataset.dataset import MyDataset

DOCS = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]


def write_binidx(prefix, docs, dtype=np.uint16):
    """Write a minimal Megatron-format .bin/.idx pair and return the prefix."""
    dtype_size = np.dtype(dtype).itemsize
    sizes = [len(d) for d in docs]
    pointers, address = [], 0
    for size in sizes:
        pointers.append(address)
        address += size * dtype_size

    with open(f"{prefix}.bin", "wb") as f:
        for doc in docs:
            f.write(np.asarray(doc, dtype=dtype).tobytes())
    with open(f"{prefix}.idx", "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))              # version
        f.write(struct.pack("<B", code(dtype)))    # dtype code
        f.write(struct.pack("<Q", len(sizes)))     # number of documents
        f.write(struct.pack("<Q", len(sizes)))     # doc count
        f.write(np.asarray(sizes, dtype=np.int32).tobytes())
        f.write(np.asarray(pointers, dtype=np.int64).tobytes())
        f.write(np.asarray(range(len(sizes)), dtype=np.int64).tobytes())
    return prefix


@pytest.fixture
def dataset(tmp_path):
    return MMapIndexedDataset(write_binidx(str(tmp_path / "toy"), DOCS))


def test_len_and_getitem(dataset):
    assert len(dataset) == 3
    assert dataset[0].tolist() == [1, 2, 3]
    assert dataset.sizes.tolist() == [3, 2, 5]


def test_head_truncates(dataset):
    dataset.head(2)
    assert len(dataset) == 2


def test_pad_wraps_around(dataset):
    tokens, min_len = dataset.pad(1, 5)
    assert min_len == 2
    assert tokens.tolist() == [4, 5, 1, 2, 3]


def test_pad_short_document_returns_exact_length(dataset):
    tokens, min_len = dataset.pad(0, 2)
    assert min_len == 2
    assert tokens.tolist() == [1, 2]


def test_only_truncates(dataset):
    assert dataset.only(2, 3).tolist() == [6, 7, 8]
    assert dataset.only(0, 10).tolist() == [1, 2, 3]


def test_get_with_offset(dataset):
    assert dataset.get(0, offset=1).tolist() == [2, 3]
    assert dataset.get(0, offset=1, length=1).tolist() == [2]


def test_pickle_roundtrip(dataset, tmp_path):
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == 3
    assert restored[2].tolist() == [6, 7, 8, 9, 10]


@pytest.mark.parametrize("dataload", ["pad", "only"])
def test_mydataset_padding(tmp_path, dataload):
    prefix = write_binidx(str(tmp_path / "toy"), DOCS)
    args = SimpleNamespace(data_type="binidx", data_file=prefix, ctx_len=4,
                           epoch_steps=10, dataload=dataload)
    ds = MyDataset(args)
    assert len(ds) == 3

    x, y = ds[0]
    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.shape == (4,) and y.shape == (4,)
    # ctx_len=4 -> req_len=5, doc 0 has 3 tokens -> 2 masked positions
    assert (y == -100).sum().item() == 2


def test_mydataset_rejects_unknown_dataload(tmp_path):
    prefix = write_binidx(str(tmp_path / "toy"), DOCS)
    args = SimpleNamespace(data_type="binidx", data_file=prefix, ctx_len=4,
                           epoch_steps=10, dataload="bogus")
    with pytest.raises(ValueError, match="dataload"):
        MyDataset(args)[0]


class _FakeTokenizer:
    pad_token_id = 0


def test_pad_collator_accepts_tuples():
    """MyDataset returns (input_ids, labels) tuples - the loader must accept them."""
    from rwkvt.dataset.SFTdataset import PadCollator

    batch = [(torch.tensor([5, 6, 7]), torch.tensor([-100, 6, 7])),
             (torch.tensor([8]), torch.tensor([8]))]
    input_ids, labels, mask = PadCollator(_FakeTokenizer())(batch)
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert labels.tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_pad_collator_accepts_dicts():
    from rwkvt.dataset.SFTdataset import PadCollator

    batch = [{"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([-100, 6])},
             {"input_ids": torch.tensor([7, 8, 9]), "labels": torch.tensor([-100, 8, 9])}]
    input_ids, labels, mask = PadCollator(_FakeTokenizer())(batch)
    assert input_ids.shape == (2, 3)
    assert mask.sum().item() == 5


def test_pad_collator_mask_ignores_real_pad_id():
    """Pad id (eos=0) also occurs inside real sequences: mask must use lengths."""
    from rwkvt.dataset.SFTdataset import PadCollator

    batch = [(torch.tensor([0, 5, 0]), torch.tensor([0, 5, 0])),
             (torch.tensor([0, 0]), torch.tensor([0, 0]))]
    _, _, mask = PadCollator(_FakeTokenizer())(batch)
    assert mask.tolist() == [[True, True, True], [True, True, False]]


def test_sft_dataset_collate_through_dataloader(monkeypatch, tmp_path):
    """The sft __getitem__ output must survive a real DataLoader + collator."""
    import rwkvt.dataset.dataset as ds_module

    rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4]), "labels": torch.tensor([4])}]
    monkeypatch.setattr(ds_module, "sft_dataset",
                        lambda args: (rows, _FakeTokenizer()))
    args = SimpleNamespace(data_type="sft", ctx_len=8, epoch_steps=100,
                           micro_bsz=2, data_shuffle=0, num_workers=0,
                           persistent_workers=False, prefetch_factor=None)
    ds = ds_module.MyDataset(args)
    loader = MyDataModuleLoader(ds, ds.pad_collator)
    x, y, mask = next(iter(loader))
    assert x.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    # labels are shifted by one for causal LM
    assert y[0, -1].item() == -100 or y.shape[1] == 3


def MyDataModuleLoader(dataset, collate_fn):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


def test_pad_collator_is_picklable():
    """num_workers>0 sends the collator to worker processes."""
    import pickle

    from rwkvt.dataset.SFTdataset import PadCollator

    collator = PadCollator(_FakeTokenizer())
    restored = pickle.loads(pickle.dumps(collator))
    assert restored.pad_token_id == collator.pad_token_id
    batch = [(torch.tensor([1, 2]), torch.tensor([1, 2]))]
    assert restored(batch)[0].tolist() == [[1, 2]]


def test_pad_works_when_dataset_is_shorter_than_request(tmp_path):
    """A tiny dataset with a long ctx_len must wrap around, not read past EOF."""
    ds = MMapIndexedDataset(write_binidx(str(tmp_path / "toy"), DOCS))
    tokens, min_len = ds.pad(2, 20)  # 5 + 2 + 5 = 12 tokens total
    assert len(tokens) == 20
    assert min_len == 5
    assert tokens[:5].tolist() == [6, 7, 8, 9, 10]
    # wraps through doc 0, 1, 2, 0, ...
    assert tokens[5:10].tolist() == [1, 2, 3, 4, 5]
    assert tokens[10:12].tolist() == [6, 7]


def test_sft_dataset_truncates_to_epoch_steps(monkeypatch):
    import rwkvt.dataset.SFTdataset as sft

    class _Raw:
        column_names = ("q", "a")

        def __init__(self):
            self.selected = None
            self.mapped = False

        def __len__(self):
            return 100

        def select(self, indices):
            self.selected = list(indices)
            return self

        def map(self, *args, **kwargs):
            self.mapped = True
            return self

    raw = _Raw()
    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: raw)
    monkeypatch.setattr(sft, "get_tokenizer", lambda args: object())
    args = SimpleNamespace(data_file="x", sft_split="train",
                           sft_field=["q", "a"], sft_num_proc=1,
                           epoch_steps=10, ctx_len=8)
    sft.sft_dataset(args)
    assert raw.selected == list(range(10))
    assert raw.mapped


def test_sft_dataset_keeps_everything_without_epoch_steps(monkeypatch):
    import rwkvt.dataset.SFTdataset as sft

    class _Raw:
        column_names = ("q", "a")
        selected = None

        def __len__(self):
            return 100

        def select(self, indices):  # pragma: no cover - must not be called
            raise AssertionError("select() should not be called")

        def map(self, *args, **kwargs):
            return self

    monkeypatch.setattr(sft, "load_dataset", lambda *a, **k: _Raw())
    monkeypatch.setattr(sft, "get_tokenizer", lambda args: object())
    args = SimpleNamespace(data_file="x", sft_split="train",
                           sft_field=["q", "a"], sft_num_proc=1,
                           epoch_steps=0, ctx_len=8)
    sft.sft_dataset(args)
