"""Datasets and dataloader for RWKV fine-tuning."""

import logging

import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset
from .SFTdataset import PadCollator, sft_dataset

logger = logging.getLogger(__name__)

_PIPELINE = None


def get_pipeline():
    """Lazily build the RWKV tokenizer pipeline (only needed for jsonl data)."""
    global _PIPELINE
    if _PIPELINE is None:
        from rwkv.utils import PIPELINE

        _PIPELINE = PIPELINE("rwkv6", "rwkv_vocab_v20230424")
    return _PIPELINE


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_type = args.data_type
        self.pad_collator = None

        if self.data_type == "sft":
            self.data, tokenizer = sft_dataset(args)
            self.pad_collator = PadCollator(tokenizer)
        elif self.data_type == "jsonl":
            with jsonlines.open(args.data_file) as file:
                self.data = list(file)
            if args.epoch_steps < len(self.data):
                self.data = self.data[:args.epoch_steps]
        elif self.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file).head(args.epoch_steps)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            logger.info("data has %d tokens", self.data_size)
        else:
            raise ValueError(f"unsupported --data_type {self.data_type!r}; "
                             "expected binidx, sft or jsonl")

        logger.info("trimmed to %d samples for epoch_steps %d",
                    len(self), args.epoch_steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "sft":
            row = self.data[idx]
            inputs = torch.as_tensor(row["input_ids"])
            labels = torch.roll(torch.as_tensor(row["labels"]), shifts=-1, dims=-1)
            return inputs, labels
        if self.data_type == "jsonl":
            return self._get_jsonl(idx)
        return self._get_binidx(idx)

    def _get_jsonl(self, idx):
        args = self.args
        req_len = args.ctx_len + 1
        token = torch.tensor(get_pipeline().encode(self.data[idx]["text"]))
        token = token[:req_len]
        pad_len = req_len - len(token)
        dix = F.pad(token, (0, pad_len), value=0)
        label = F.pad(token, (0, pad_len), value=-100)
        return dix[:-1], label[1:]

    def _get_binidx(self, idx):
        args = self.args
        req_len = args.ctx_len + 1
        if args.dataload == "pad":
            dix, min_len = self.data.pad(idx=idx, length=req_len)
        elif args.dataload == "only":
            tokens = self.data.only(idx=idx, length=req_len)
            min_len = len(tokens)
            dix = np.zeros(req_len, dtype=int)  # keep batch shapes static
            dix[:min_len] = tokens
        else:
            raise ValueError(f"unsupported --dataload {args.dataload!r}; "
                             "expected pad or only")

        x = torch.tensor(dix[:-1], dtype=torch.long)
        dix = dix.copy()
        dix[min_len:] = -100
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
