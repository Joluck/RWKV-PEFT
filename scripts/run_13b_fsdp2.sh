#!/usr/bin/env bash
# RWKV-7 13.3B (g1k) SFT on MetaMathQA: LoRA (default) or MTP heads.
#
# 默认：8 卡 FSDP2，只保存 adapter（13B 合并保存需要额外一份 26GB CPU 权重）。
#
# 常用组合：
#   bash scripts/run_13b_fsdp2.sh                             # 8 卡 FSDP2 + LoRA
#   DEVICES=4 bash scripts/run_13b_fsdp2.sh                   # 4 卡
#   DEVICES=1 STRATEGY=deepspeed_stage_1 MERGE=1 \
#       bash scripts/run_13b_fsdp2.sh                         # 单卡 80G / DeepSpeed
#   PEFT=mtp bash scripts/run_13b_fsdp2.sh                    # 冻结底座，只训 MTP
#   PEFT=mtp MTP_DEPTH=2 bash scripts/run_13b_fsdp2.sh        # 预测后面 2 个 token
#   EPOCH_STEPS=200 EPOCH_COUNT=1 bash scripts/run_13b_fsdp2.sh   # 200 条冒烟
#   DRY_RUN=1 bash scripts/run_13b_fsdp2.sh                   # 只打印命令
#
# 说明：
#   --epoch_steps 的含义是「每个 mini-epoch 取多少条样本」（现在 SFT 也会真的截断，
#   且在分词之前截断）；0 表示用全量数据集。
# --- 必须是 bash：数组和 pipefail 是 bash 扩展，`sh script.sh`(dash) 会挂 ---
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 路径 / 数据 ----------------------------------------------------------
load_model="${LOAD_MODEL:-/home/rwkv/jl/models/rwkv7-g1k-13.3b-20260930-ctx25600.pth}"
proj_dir="${PROJ_DIR:-/home/rwkv/JL/out_model/rwkv7-13b-lora}"
data_file="${DATA_FILE:-meta-math/MetaMathQA}"
# RWKV-7 用同一个 65536 world 词表；换成你自己的本地 tokenizer 目录也可以
tokenizer_path="${TOKENIZER_PATH:-RWKV/rwkv-5-world-3b}"

# ---- 模型结构 ------------------------------------------------------------
n_layer=61
n_embd=4096
vocab_size=65536

# ---- 训练超参 ------------------------------------------------------------
micro_bsz="${MICRO_BSZ:-8}"
ctx_len="${CTX_LEN:-128}"
epoch_steps="${EPOCH_STEPS:-100000}"   # 0 = 全量（MetaMathQA 约 39.5 万条）
epoch_count="${EPOCH_COUNT:-10}"
epoch_save="${EPOCH_SAVE:-1}"
grad_accum="${GRAD_ACCUM:-1}"
# 默认 0 个 worker：fork 一个已经初始化了 CUDA/NCCL 的进程会触发
# "Fatal glibc error: pthread_mutex_lock ... __owner == 0"。
# 数据加载确实成为瓶颈时再开：
#   NUM_WORKERS=8 WORKER_START_METHOD=spawn PERSISTENT_WORKERS=1 bash scripts/...
num_workers="${NUM_WORKERS:-0}"
# 如果遇到 "Fatal glibc error: pthread_mutex_lock ... __owner == 0"：
#   NUM_WORKERS=0                        # 最稳，13B 训练 GPU 是瓶颈
#   WORKER_START_METHOD=spawn            # 保留 workers 但避免 fork 继承 CUDA/NCCL 线程
worker_start_method="${WORKER_START_METHOD:-fork}"
sft_num_proc="${SFT_NUM_PROC:-8}"
persistent_workers="${PERSISTENT_WORKERS:-0}"
lr="${LR:-1e-5}"

# ---- 并行 / 精度 ---------------------------------------------------------
devices="${DEVICES:-8}"
strategy="${STRATEGY:-fsdp2}"          # fsdp2 | deepspeed_stage_1 | ddp
merge="${MERGE:-0}"                    # 0 = 只存 adapter（推荐）1 = 合并全量权重
precision="${PRECISION:-bf16}"

peft_mode="${PEFT:-lora}"              # lora | miss | mtp
mtp_depth="${MTP_DEPTH:-1}"
mtp_loss_weight="${MTP_LOSS_WEIGHT:-0.3}"
mtp_main_weight="${MTP_MAIN_WEIGHT:-1.0}"

peft_config="${PEFT_CONFIG:-}"
if [ -z "$peft_config" ]; then
  peft_config='{"r":8,"lora_alpha":32,"lora_dropout":0.05}'
fi

if [ ! -f "$load_model" ]; then
  echo "checkpoint not found: $load_model" >&2
  exit 1
fi

cmd=(python train.py
  --load_model "$load_model"
  --proj_dir "$proj_dir"
  --data_file "$data_file"
  --data_type sft --sft_field query response --sft_split train
  --tokenizer_path "$tokenizer_path"
  --vocab_size "$vocab_size"
  --n_layer "$n_layer" --n_embd "$n_embd"
  --ctx_len "$ctx_len" --micro_bsz "$micro_bsz"
  --epoch_steps "$epoch_steps" --epoch_count "$epoch_count"
  --epoch_save "$epoch_save" --accumulate_grad_batches "$grad_accum"
  --lr_init "$lr" --lr_final "$lr" --lr_schedule cos --warmup_steps -1
  --accelerator gpu --devices "$devices" --strategy "$strategy"
  --precision "$precision" --grad_cp 1 --num_workers "$num_workers" \
  --worker_start_method "$worker_start_method" --sft_num_proc "$sft_num_proc" \
  --my_testing "x070"
  --peft "$peft_mode" --peft_config "$peft_config"
  --merge "$merge")

if [ "$peft_mode" = "mtp" ]; then
  cmd+=(--mtp_depth "$mtp_depth" --mtp_loss_weight "$mtp_loss_weight"
        --mtp_main_weight "$mtp_main_weight")
fi

# wandb：默认开启（项目名可用 WANDB= 置空关闭；WANDB_MODE=offline 只写本地）
wandb_project="${WANDB-rwkv7-13b-lora}"
wandb_mode="${WANDB_MODE:-online}"
if [ "$persistent_workers" = "1" ]; then
  cmd+=(--persistent_workers)
fi
if [ -n "$wandb_project" ]; then
  cmd+=(--wandb "$wandb_project" --wandb_mode "$wandb_mode")
fi

echo "=========================================================="
echo " model      : $load_model"
echo " arch       : L$n_layer D$n_embd vocab$vocab_size"
echo " devices    : $devices  strategy: $strategy  precision: $precision"
if [ "$peft_mode" = "mtp" ]; then
  echo " peft       : mtp (depth $mtp_depth, base model frozen)"
else
  echo " peft       : $peft_mode"
fi
echo " batch      : micro $micro_bsz x ctx $ctx_len x grad_accum $grad_accum"
echo " data       : workers=$num_workers ($worker_start_method, persistent=$persistent_workers), sft_num_proc=$sft_num_proc"
echo " epoch      : $epoch_steps samples / epoch x $epoch_count epochs"
echo " save       : every $epoch_save epoch(s), merge=$merge"
echo " proj_dir   : $proj_dir"
echo " wandb      : ${wandb_project:-<disabled>}  mode: ${wandb_mode}"
echo "=========================================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

exec "${cmd[@]}"
