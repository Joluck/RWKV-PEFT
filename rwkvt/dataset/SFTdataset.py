"""Supervised fine-tuning dataset helpers (hf ``datasets`` based)."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import transformers
from datasets import load_dataset

IGNORE_INDEX = -100
EOT_TOKEN = "\x17"
DEFAULT_TOKENIZER = "RWKV/rwkv-5-world-3b"

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def _tokenize_fn(strings: Sequence[str], tokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,
                                truncation=True) for text in strings]
    ids = [np.array(t.input_ids) for t in tokenized_list]
    return {"input_ids": ids, "labels": ids,
            "input_ids_lens": [len(i) for i in ids],
            "labels_lens": [len(i) for i in ids]}


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> dict:
    """Tokenize prompt+answer and mask the prompt tokens out of the loss."""
    examples = [s + t for s, t in zip(sources, targets, strict=True)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = [ids.copy() for ids in input_ids]
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"],
                                strict=True):
        label[:source_len] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map({"instruction": instruction})
               for instruction in examples[query]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples[response]]
    return preprocess(sources, targets, tokenizer)


def get_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.ctx_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sft_dataset(args):
    """Tokenize the configured hf dataset; returns the raw (unpadded) dataset.

    Padding happens per batch in :func:`pad_collate`, so the full dataset never
    has to be materialised as padded tensors.
    """
    tokenizer = get_tokenizer(args)
    raw = load_dataset(args.data_file, split=args.sft_split)
    # ``--epoch_steps`` means "samples per mini-epoch" for every data type, so
    # trim *before* tokenizing: a 200k-row subset tokenizes much faster than the
    # full MetaMathQA and keeps the epoch length predictable.
    epoch_steps = getattr(args, "epoch_steps", 0)
    if epoch_steps and len(raw) > epoch_steps:
        raw = raw.select(range(epoch_steps))
    dataset = raw.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=args.sft_num_proc,
        remove_columns=raw.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer,
                   "query": args.sft_field[0],
                   "response": args.sft_field[1]},
    )
    return dataset, tokenizer


class PadCollator:
    """Pad a batch of variable length samples.

    Items are accepted either as ``(input_ids, labels)`` tuples (what
    :class:`rwkvt.dataset.dataset.MyDataset` returns) or as mappings with
    ``input_ids``/``labels`` keys.

    The attention mask is built from the original lengths instead of
    ``input_ids != pad_token_id``: the RWKV tokenizer uses ``eos`` (=0) as pad,
    and that id also occurs inside real sequences.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    @staticmethod
    def _split(instance):
        if isinstance(instance, Mapping):
            return instance["input_ids"], instance["labels"]
        input_ids, labels = instance
        return input_ids, labels

    def __call__(self, instances: Sequence):
        pairs = [self._split(x) for x in instances]
        lengths = torch.tensor([len(ids) for ids, _ in pairs], dtype=torch.long)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(ids, dtype=torch.long) for ids, _ in pairs],
            batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(labels, dtype=torch.long) for _, labels in pairs],
            batch_first=True, padding_value=IGNORE_INDEX)
        positions = torch.arange(input_ids.size(1)).unsqueeze(0)
        attention_mask = positions < lengths.unsqueeze(1)
        return input_ids, labels, attention_mask
