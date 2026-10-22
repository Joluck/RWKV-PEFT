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
        # idempotent: train.py calls this *before* Trainer.fit so that the
        # dataset is materialised before the distributed strategy takes over
        if self.train_dataset is None:
            self.train_dataset = MyDataset(self.args)

    def train_dataloader(self):
        collate_fn = getattr(self.train_dataset, "pad_collator", None)
        num_workers = self.args.num_workers
        extra = {}
        if num_workers > 0:
            # forking a process that already owns CUDA/NCCL contexts is the
            # classic `Fatal glibc error: pthread_mutex_lock ... __owner == 0`;
            # `--worker_start_method spawn` avoids inheriting those threads
            start_method = getattr(self.args, "worker_start_method", "fork")
            if start_method != "fork":
                extra["multiprocessing_context"] = start_method
            extra["persistent_workers"] = bool(self.args.persistent_workers)
            if self.args.prefetch_factor is not None:
                extra["prefetch_factor"] = self.args.prefetch_factor
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=bool(self.args.data_shuffle),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            **extra,
        )
