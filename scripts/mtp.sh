#!/usr/bin/env bash
# RWKV-7 13.3B (g1k) + MTP on MetaMathQA：冻结底座，只训练 MTP 头。
#
# 默认：8 卡 FSDP2。MTP 头是随机初始化的，学习率比 LoRA 大 2 个量级（默认 1e-3）。
#
# 常用组合：
#   bash scripts/mtp.sh                                  # 8 卡 FSDP2，depth=1
#   DEVICES=4 bash scripts/mtp.sh                        # 4 卡
#   DEVICES=1 STRATEGY=deepspeed_stage_1 bash scripts/mtp.sh   # 单卡 80G
#   MTP_DEPTH=2 bash scripts/mtp.sh                      # 预测后面 2 个 token
#   MTP_LOAD=rwkv-5-mtp.pth bash scripts/mtp.sh          # 从已有 MTP 头续训
#   EPOCH_STEPS=200 EPOCH_COUNT=1 bash scripts/mtp.sh    # 200 条冒烟
#   DRY_RUN=1 bash scripts/mtp.sh                        # 只打印命令
#
# 说明：
#   --epoch_steps 是「每个 mini-epoch 取多少条样本」（SFT 会在分词前截断），0 = 全量。
#   产物只有 MTP 头：{proj_dir}/rwkv-{epoch}-mtp.pth，底座权重不会变。
# --- 必须是 bash：数组和 pipefail 是 bash 扩展，`sh script.sh`(dash) 会挂 ---
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 路径 / 数据 ----------------------------------------------------------
load_model="${LOAD_MODEL:-/home/rwkv/jl/models/rwkv7-g1k-13.3b-20260930-ctx25600.pth}"
proj_dir="${PROJ_DIR:-/home/rwkv/JL/out_model/rwkv7-13b-mtp}"
data_file="${DATA_FILE:-meta-math/MetaMathQA}"
tokenizer_path="${TOKENIZER_PATH:-RWKV/rwkv-5-world-3b}"

# ---- 模型结构 ------------------------------------------------------------
n_layer=61
n_embd=4096
vocab_size=65536

# ---- MTP ------------------------------------------------------------------
mtp_depth="${MTP_DEPTH:-1}"
# λ: L = mtp_main_weight * L_main + λ * mean_k(L_k)，对齐 DeepSeek-V3（0.3）。
# 本脚本 mtp_main_weight=0，所以 λ 只是全局缩放（Adam 近似尺度无关），不影响优化，
# 只影响 loss_data.jsonl 里 loss 字段的数值；设 1.0 可让 loss == mtp_loss。
mtp_loss_weight="${MTP_LOSS_WEIGHT:-1.0}"
mtp_main_weight="${MTP_MAIN_WEIGHT:-0.0}"
mtp_step_beta="${MTP_STEP_BETA:-0.6}"   # 底座冻结，主损失无梯度，只作监控
mtp_load="${MTP_LOAD:-}"

# ---- 训练超参 ------------------------------------------------------------
micro_bsz="${MICRO_BSZ:-8}"
ctx_len="${CTX_LEN:-128}"
epoch_steps="${EPOCH_STEPS:-100000}"
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
lr="${LR:-3e-4}"
# 余弦衰减到 lr/10：头只有 151M，1e-3 会出现 17~20 nats 的尖峰
lr_final="${LR_FINAL:-3e-5}"
warmup_steps="${WARMUP_STEPS:-100}"

# ---- 并行 / 精度 ---------------------------------------------------------
devices="${DEVICES:-4}"
strategy="${STRATEGY:-deepspeed_stage_1}"          # fsdp2 | deepspeed_stage_1 | ddp
precision="${PRECISION:-bf16}"

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
  --lr_init "$lr" --lr_final "$lr_final" --lr_schedule cos --warmup_steps "$warmup_steps"
  --accelerator gpu --devices "$devices" --strategy "$strategy"
  --precision "$precision" --grad_cp 1 --num_workers "$num_workers" \
  --worker_start_method "$worker_start_method" --sft_num_proc "$sft_num_proc" \
  --my_testing "x070"
  --peft mtp
  --mtp_depth "$mtp_depth" --mtp_loss_weight "$mtp_loss_weight"
  --mtp_main_weight "$mtp_main_weight"
  --mtp_step_beta "$mtp_step_beta")

if [ -n "$mtp_load" ]; then
  cmd+=(--mtp_load "$mtp_load")
fi
# wandb：默认开启（项目名可用 WANDB= 置空关闭；WANDB_MODE=offline 只写本地）
wandb_project="${WANDB-rwkv7-mtp}"
wandb_mode="${WANDB_MODE:-online}"
if [ "$persistent_workers" = "1" ]; then
  cmd+=(--persistent_workers)
fi
if [ -n "$wandb_project" ]; then
  cmd+=(--wandb "$wandb_project" --wandb_mode "$wandb_mode")
fi

echo "=========================================================="
echo " model      : $load_model"
echo " arch       : L$n_layer D$n_embd vocab$vocab_size  (base frozen)"
echo " mtp        : depth=$mtp_depth  loss_weight=$mtp_loss_weight  step_beta=$mtp_step_beta"
echo " devices    : $devices  strategy: $strategy  precision: $precision"
echo " batch      : micro $micro_bsz x ctx $ctx_len x grad_accum $grad_accum"
echo " data       : workers=$num_workers ($worker_start_method, persistent=$persistent_workers), sft_num_proc=$sft_num_proc"
echo " epoch      : $epoch_steps samples / epoch x $epoch_count epochs"
echo " lr         : $lr -> $lr_final (cos, warmup $warmup_steps)  save every $epoch_save epoch(s)"
echo " proj_dir   : $proj_dir"
echo " wandb      : ${wandb_project:-<disabled>}  mode: ${wandb_mode}"
echo "=========================================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

exec "${cmd[@]}"
