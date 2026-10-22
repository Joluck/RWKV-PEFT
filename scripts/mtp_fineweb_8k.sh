#!/usr/bin/env bash
# 8k-context MTP run on fineweb-10BT (~8h budget).
#
# Measured throughput on 4x RTX PRO 6000: ~23k tok/s at ctx 8192,
# micro_bsz 4 x 4 GPUs (131k tokens/step, ~5.7s/step).
#   5000 steps ~ 655M tokens ~ 8 hours
#
# Usage:  bash scripts/mtp_fineweb_8k.sh
#         STEPS=1000 bash scripts/mtp_fineweb_8k.sh   # shorter probe
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

cd "$(dirname "$0")/.."

LOAD_MODEL="${LOAD_MODEL:-/home/rwkv/jl/models/rwkv7-g1k-13.3b-20260930-ctx25600.pth}"
BINIDX="${BINIDX:-/home/rwkv/data/fineweb-10BT/binidx/fineweb}"
PROJ_DIR="${PROJ_DIR:-/home/rwkv/JL/out_model/rwkv7-13b-mtp-fineweb8k}"
DEVICES="${DEVICES:-4}"
MICRO_BSZ="${MICRO_BSZ:-4}"
CTX_LEN="${CTX_LEN:-8192}"
STEPS="${STEPS:-5000}"
LR="${LR:-3e-4}"
LR_FINAL="${LR_FINAL:-3e-5}"
WARMUP="${WARMUP:-100}"
SAVE_PER_STEP="${SAVE_PER_STEP:-1000}"
# 每步的 teacher-forced 接受率直接写进 accept_1..K（零成本，见 mtp.acceptance）
# 每 N 步再跑一次真实 decode 校验（wandb accept/rate + accept_curve.jsonl）
ACCEPT_EVERY="${ACCEPT_EVERY:-1000}"
ACCEPT_TOKENS="${ACCEPT_TOKENS:-32}"
MTP_DEPTH="${MTP_DEPTH:-1}"
WANDB="${WANDB-rwkv7-mtp}"
WANDB_MODE="${WANDB_MODE:-online}"
WAIT="${WAIT:-1}"

seq_per_step=$((MICRO_BSZ * DEVICES))
epoch_steps=$((STEPS * seq_per_step))
tokens=$((epoch_steps * CTX_LEN))

if [ "$WAIT" = "1" ] && [ ! -f "$BINIDX.json" ]; then
  echo "waiting for $BINIDX.json (binidx conversion still running) ..."
  while [ ! -f "$BINIDX.json" ]; do sleep 60; done
  echo "binidx ready:"
  cat "$BINIDX.json"
fi
if [ ! -f "$BINIDX.bin" ]; then
  echo "missing $BINIDX.bin" >&2
  exit 1
fi

echo "=========================================================="
echo " MTP / fineweb-10BT 8k"
echo " model      : $LOAD_MODEL"
echo " data       : $BINIDX"
echo " proj_dir   : $PROJ_DIR"
echo " devices    : $DEVICES  micro_bsz $MICRO_BSZ  ctx $CTX_LEN"
echo " steps      : $STEPS ($epoch_steps seqs, $(printf '%.0f' "$(echo "$tokens/1000000" | bc -l)")M tokens)"
echo " lr         : $LR -> $LR_FINAL (cos, warmup $WARMUP)"
echo " save       : every $SAVE_PER_STEP steps"
echo " accept     : every $ACCEPT_EVERY steps (8 held-out prompts, $ACCEPT_TOKENS tokens)"
echo "=========================================================="

args=(
  --load_model "$LOAD_MODEL" --proj_dir "$PROJ_DIR"
  --data_file "$BINIDX" --data_type binidx
  --vocab_size 65536 --n_layer 61 --n_embd 4096 --head_size_a 64
  --ctx_len "$CTX_LEN" --micro_bsz "$MICRO_BSZ"
  --epoch_steps "$epoch_steps" --epoch_count 1 --epoch_save 0
  --save_per_step "$SAVE_PER_STEP"
  --lr_init "$LR" --lr_final "$LR_FINAL" --lr_schedule cos --warmup_steps "$WARMUP"
  --accelerator gpu --devices "$DEVICES" --strategy deepspeed_stage_1
  --precision bf16 --grad_cp 1 --num_workers 0 --sft_num_proc 0
  --op cuda --my_testing x070
  --peft mtp --mtp_depth "$MTP_DEPTH" --mtp_loss_weight 1.0 --mtp_main_weight 0.0
  --mtp_step_beta 0.6
  --accept_every "$ACCEPT_EVERY" --accept_prompts 8 --accept_tokens "$ACCEPT_TOKENS"
)
if [ -n "$WANDB" ]; then
  args+=(--wandb "$WANDB" --wandb_mode "$WANDB_MODE")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'python train.py'; printf ' %q' "${args[@]}"; echo
  exit 0
fi

exec python train.py "${args[@]}"
