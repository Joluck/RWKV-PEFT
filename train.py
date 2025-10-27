########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import sys
import logging
# logging.basicConfig(level=logging.INFO)
from typing import Optional, Dict, Sequence, List, Literal

if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from lightning_utilities.core.rank_zero import rank_zero_info
    import lightning as pl
    import json
    from rwkvt.args_type import TrainingArgs
    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str) #binidx / sft
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x052', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser.add_argument("--peft", default="none", type=str)# lora pissa DiSHA
    #parser.add_argument("--train_parts", default=["time", "ln"], type=list)##emb , head
    parser.add_argument("--train_parts", default=["time", "ln"], nargs='*', help="List of parts to train emb head time ln")

    #LORA
    parser.add_argument("--lora_config", default='{"lora_load":"", "lora_r":8, "lora_alpha":32, "lora_dropout":0.01}', type=json.loads)


    # #LISA
    # parser.add_argument("--lisa_config", default='{"lisa_r":2, "lisa_k":100}', type=json.loads)

    #PISSA
    parser.add_argument("--pissa_config", default='{"pissa_load":"", "pissa_init":"", "pissa_r":8, "svd_niter":4}', type=json.loads)

    #Bone
    parser.add_argument("--miss_config", default='{"mode":"mode", "load":"", "r":64}', type=json.loads)
    parser.add_argument("--merge", type=int, default=1, help="1=merge PEFT weights, 0=save PEFT-only")

    #quant
    parser.add_argument("--quant", default="none", type=str)

    #dataset
    parser.add_argument("--dataload", default="pad", type=str)

    parser.add_argument("--chunk_ctx", default=512, type=int)
    #fla
    parser.add_argument("--fla", action="store_true")
    parser.add_argument("--train_type", default="none", type=str)

    #loss_mask
    parser.add_argument("--loss_mask", default="none", type=str)### pad qa se
    parser.add_argument("--mask_id", default='{"mask0":"0", "mask1":"1"}', type=json.loads)
    parser.add_argument("--data_shuffle", default=1, type=int)


    #new optim
    parser.add_argument("--optim", default="none", type=str)

    #acc_grad_batchs
    parser.add_argument("--avg_loss", default=0, type=int)


    parser.add_argument("--sft_field", default=None, type=str, nargs='+', help='List of fields for SFT')
    parser.add_argument("--sft_split", default="train", type=str)


    parser.add_argument("--op", default="cuda", type=str)
    parser.add_argument("--fused_kernel", action='store_true', help="Enable rwkv-fla fused kernel")

    parser.add_argument("--lr_schedule", default="cos", type=str)        #['cos', 'wsd']


    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="auto", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", default=None, type=int)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    # from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    #args.vocab_size = get_vocab_size(args)
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
      # continue forever
    args.max_epochs = args.epoch_count
    if args.dataload =='get':
        args.max_epochs = -1
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    ######state tuning
    if args.peft=='state':
        os.environ["RWKV_TRAIN_TYPE"] = 'state'
    else:
        os.environ["RWKV_TRAIN_TYPE"] = args.train_type


    print(f"########## WKV OP           {args.op}               ##########\n" * 1)
    print(f"########## FUSED OP    {args.fused_kernel}          ##########\n" * 1)
    os.environ["WKV"]= args.op
    os.environ["FUSED_KERNEL"] = '1' if args.fused_kernel else '0'

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size


    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass


    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16", "sft", 'jsonl']

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "0"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from rwkvt.lightning_train.trainer import train_callback
    from rwkvt.rwkvpeft.peft_loading import load_peft_model
    from rwkvt.dataset.dataset import MyDataModule
    args, model = load_peft_model(args)



    trainer = Trainer(accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
    logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
    log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)


 
    train_data = MyDataModule(args)

    trainer.fit(model, train_data)

