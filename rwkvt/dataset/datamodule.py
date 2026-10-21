"""Lightning DataModule wiring for :class:`rwkvt.dataset.dataset.MyDataset`."""

import lightning as L
from torch.utils.data import DataLoader

from .dataset import MyDataset


class MyDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.args)

    def train_dataloader(self):
        collate_fn = getattr(self.train_dataset, "pad_collator", None)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=bool(self.args.data_shuffle),
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers and self.args.num_workers > 0,
            prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_fn,
        )
