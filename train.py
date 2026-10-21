"""RWKV-PEFT training entry point.

Run ``python train.py --help`` for the full option list; ``scripts/*.sh`` contain
ready-to-use examples.
"""

from __future__ import annotations

import datetime
import json
import os
import warnings
from argparse import ArgumentParser, Namespace

SUPPORTED_VERSIONS = ("5", "6", "7")
SUPPORTED_DATA_TYPES = ("binidx", "sft", "jsonl")
SUPPORTED_TRAIN_TYPES = ("none", "state", "infctx", "fullstate")
SUPPORTED_PEFT = ("none", "state", "lora", "miss", "adalora", "prefix")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="RWKV parameter-efficient fine-tuning")

    # ---- model -----------------------------------------------------------
    parser.add_argument("--load_model", default="", type=str,
                        help="path to the base checkpoint (.pth)")
    parser.add_argument("--my_testing", default="x070", type=str,
                        help="model generation: x050 / x060 / x070")
    parser.add_argument("--vocab_size", default=0, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int,
                        help="0 means n_embd")
    parser.add_argument("--dim_ffn", default=0, type=int,
                        help="0 means 3.5 * n_embd")
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--load_partial", default=0, type=int,
                        help="1 = tolerate parameters missing from the checkpoint")

    # ---- peft ------------------------------------------------------------
    parser.add_argument("--peft", default="none", type=str, choices=SUPPORTED_PEFT)
    parser.add_argument(
        "--peft_config", default={}, type=json.loads,
        help='tuner config, e.g. \'{"r":8,"lora_alpha":32,"lora_dropout":0.05}\'')
    parser.add_argument("--merge", type=int, default=1,
                        help="1 = save merged weights, 0 = save the adapter only")

    # ---- data ------------------------------------------------------------
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="binidx", type=str,
                        choices=SUPPORTED_DATA_TYPES)
    parser.add_argument("--dataload", default="pad", type=str,
                        choices=("pad", "only"),
                        help="binidx sampling: pad wraps around, only truncates")
    parser.add_argument("--data_shuffle", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", default=None, type=int)
    parser.add_argument("--sft_field", default=None, type=str, nargs="+",
                        help="dataset columns holding prompt / answer")
    parser.add_argument("--sft_split", default="train", type=str)
    parser.add_argument("--sft_num_proc", default=32, type=int)
    parser.add_argument("--tokenizer_path", default="RWKV/rwkv-5-world-3b", type=str)

    # ---- optimisation ----------------------------------------------------
    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--micro_bsz", default=12, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int,
                        help="samples per mini-epoch")
    parser.add_argument("--epoch_count", default=500, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int,
                        help="epoch offset used when naming checkpoints")
    parser.add_argument("--epoch_save", default=5, type=int,
                        help="save every N epochs, 0 disables intermediate saves")
    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--lr_schedule", default="cos", type=str,
                        choices=("cos", "wsd"))
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--weight_decay_final", default=-1, type=float)
    parser.add_argument("--layerwise_lr", default=1, type=int)
    parser.add_argument("--my_pile_stage", default=0, type=int,
                        help="2 = 5x lr on time-mix parameters")
    parser.add_argument("--grad_cp", default=0, type=int,
                        help="1 = gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--optimizer", default="adamw", type=str,
                        choices=("adamw",))
    parser.add_argument("--train_type", default="none", type=str,
                        choices=SUPPORTED_TRAIN_TYPES)
    parser.add_argument("--chunk_ctx", default=512, type=int,
                        help="chunk length for infctx training")
    parser.add_argument("--op", default="cuda", type=str,
                        choices=("cuda", "fla", "triton"),
                        help="WKV kernel backend")
    parser.add_argument("--fused_kernel", action="store_true",
                        help="use the rwkv-fla fused addcmul/layernorm kernels")

    # ---- runtime ---------------------------------------------------------
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--wandb", default="", type=str,
                        help="wandb project name, empty disables logging")
    parser.add_argument("--random_seed", default="-1", type=int)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="auto", type=str,
                        help="auto | ddp | deepspeed_stage_* | fsdp2")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=("fp32", "tf32", "fp16", "bf16"))
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    return parser


def finalize_args(args: Namespace) -> Namespace:
    """Derive paths/env vars and validate the configuration."""
    version = [v for v in SUPPORTED_VERSIONS if v in args.my_testing]
    if len(version) != 1:
        raise SystemExit(f"--my_testing {args.my_testing!r} is not supported; "
                         f"expected x050 / x060 / x070")
    args.rwkv_version = version[0]

    if not args.load_model:
        raise SystemExit("--load_model is required")
    if not os.path.exists(args.load_model):
        raise SystemExit(f"--load_model {args.load_model} does not exist")
    if args.data_type == "sft" and not args.sft_field:
        raise SystemExit("--sft_field is required for --data_type sft")
    args.fsdp2 = args.strategy == "fsdp2"
    if args.fsdp2 and args.precision == "fp16":
        raise SystemExit("--strategy fsdp2 supports fp32/bf16 only (fp16 master "
                         "weights are unstable under FSDP2)")
    if args.fsdp2 and args.peft == "prefix":
        warnings.warn("prefix tuning under FSDP2 is untested; prefer lora/miss/state",
                      stacklevel=2)
    if args.op == "cuda" and args.precision in ("fp16", "tf32"):
        raise SystemExit(
            f"--precision {args.precision} is not supported by the bundled WKV "
            "CUDA kernel (bf16/fp32 only); use --precision bf16 or --op fla")
    if args.ctx_len % 16 != 0:
        warnings.warn("--ctx_len should be a multiple of 16 for the WKV kernels",
                      stacklevel=2)

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default 3.5x
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    args.max_epochs = args.epoch_count
    os.makedirs(args.proj_dir, exist_ok=True)

    # env vars must be set before the model/operator modules are imported
    train_type = "state" if args.peft == "state" else args.train_type
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_TRAIN_TYPE"] = train_type
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_FLOAT_MODE"] = "fp32" if args.precision in ("fp32", "tf32") else args.precision
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["WKV"] = args.op
    os.environ["FUSED_KERNEL"] = "1" if args.fused_kernel else "0"

    use_tf32 = args.precision != "fp32"
    if args.precision in ("fp32", "tf32"):
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"
    args.use_tf32 = use_tf32
    return args


def main() -> None:
    args = build_parser().parse_args()
    args = finalize_args(args)

    import numpy as np
    import torch
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from lightning_utilities.core.rank_zero import rank_zero_info

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    if args.random_seed >= 0:
        rank_zero_info(f"GLOBAL SEED {args.random_seed} (affects multi-GPU sampling)")
        seed_everything(args.random_seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = args.use_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.use_tf32

    rank_zero_info(f"model={args.my_testing} op={args.op} precision={args.precision}")

    from rwkvt.dataset.datamodule import MyDataModule
    from rwkvt.lightning_train.trainer import train_callback
    from rwkvt.peft_loading import load_peft_model

    args, model = load_peft_model(args)

    strategy = args.strategy
    if args.fsdp2:
        from rwkvt.fsdp import build_fsdp2_strategy

        strategy = build_fsdp2_strategy(args.devices, args.num_nodes)
        rank_zero_info(f"FSDP2 enabled (mesh: dp x 1 = {args.devices} device(s))")

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        logger=False,
        callbacks=[train_callback(args)],
        max_epochs=args.max_epochs,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=int(1e20),
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, MyDataModule(args))


if __name__ == "__main__":
    main()
