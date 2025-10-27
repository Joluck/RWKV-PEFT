########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import torch.nn.functional as F

import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info
from .binidx import MMapIndexedDataset
from rwkvt.args_type import TrainingArgs
from rwkv.utils import PIPELINE
from rwkvt.dataset.SFTdataset import sft_dataset
import time
import jsonlines
from .mask import generate_mask, create_mask,mask_fn_dict
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")


class MyDataset(Dataset):
    def __init__(self, args, processor=None):

        self.args = args
        self.processor = processor
        self.data_type = args.data_type

        if args.data_type == "sft":
            self.data = sft_dataset(args)
        elif args.data_type == "jsonl":
            with jsonlines.open(args.data_file) as file:
                self.data = list(file)

        elif args.data_type == "binidx":
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")
        print(len(self.data))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        args = self.args

        if args.data_type == "sft":

            inputs, labels, attn_mask = self.data[0][idx], self.data[1][idx], self.data[2][idx]
            labels= torch.roll(labels, shifts=-1, dims=-1)

            return inputs, labels, attn_mask
        elif args.data_type == "jsonl":
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            ctx = self.data[idx]['text']
            token = torch.tensor(pipeline.encode(ctx))
            token_len = len(token)
            min_len = min(token_len, req_len)
            if req_len < token_len :
                token = token[:req_len]
                pad_len = 0
            else:
                pad_len = req_len - token_len
        
            # dix = F.pad(token, (pad_len, 0), value=0)
            dix = F.pad(token, (0, pad_len), value=0)
            label = F.pad(token, (0, pad_len), value=-100)
            x = dix[:-1]
            y = label[1:]

        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            data = self.data


            if args.data_type == "binidx":
                if args.dataload == 'pad':
                    dix, min_len = data.pad(idx=idx, length=req_len)
                elif args.dataload == 'only':
                    dix = data.only(idx=idx, length=req_len).astype(int)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            dix[min_len:] = -100
            y = torch.tensor(dix[1:], dtype=torch.long)

        mask_fn = mask_fn_dict.get(args.loss_mask)

        if mask_fn!=None:
            t1 = pipeline.encode('User:')
            t2 = pipeline.encode('Assistant:')
            y = mask_fn(dix, t1, t2, min_len)
        return x, y
    

import lightning as L
from torch.utils.data import DataLoader

class MyDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.args, self.processor)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True,    # Lightning 自动替换成 DistributedSampler
            num_workers=self.args.num_workers,
            pin_memory=True
        )