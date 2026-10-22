#!/usr/bin/env bash
# One-shot pipeline for training an MTP head on allenai/tulu-3-sft-mixture:
#   1. download + tokenise the dataset into RWKV binidx
#   2. train the MTP head (frozen 13B base) with per-step acceptance logging
#
# Portable: every path is an env var, nothing points at the original machine.
#
#   MODEL=/data/rwkv7-13b.pth bash scripts/run_mtp_tulu.sh
#   MODEL=... STAGE=prepare bash scripts/run_mtp_tulu.sh    # data only
#   MODEL=... STAGE=train   bash scripts/run_mtp_tulu.sh    # train only
#   DRY_RUN=1 MODEL=... bash scripts/run_mtp_tulu.sh        # print the commands
#
# Requires: torch, lightning, transformers, datasets, and (for the default
# strategy) deepspeed. Use STRATEGY=fsdp2 to drop the deepspeed dependency.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

# ---------------------------------------------------------------- config ----
STAGE="${STAGE:-all}"                 # all | prepare | train
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-}"                    # required for STAGE=train
OUT="${OUT:-$REPO/data/mtp_tulu}"
DATA_PREFIX="${DATA_PREFIX:-$REPO/data/binidx/tulu}"
TOKENIZER="${TOKENIZER:-RWKV/rwkv-5-world-3b}"
# hf   = transformers tokenizer (fast, what all validated runs used)
# rwkv = json2binidx_tool's greedy tokenizer (exact RWKV training pipeline, ~10x slower)
TOKENIZER_IMPL="${TOKENIZER_IMPL:-hf}"
VOCAB_FILE="${VOCAB_FILE:-}"
DATASET="${DATASET:-allenai/tulu-3-sft-mixture}"
MAX_DOCS="${MAX_DOCS:-0}"             # 0 = whole dataset
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 8)}"

DEVICES="${DEVICES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$DEVICES" -ge 1 ] || DEVICES=1
MICRO_BSZ="${MICRO_BSZ:-4}"
CTX_LEN="${CTX_LEN:-8192}"
STEPS="${STEPS:-5000}"
LR="${LR:-3e-4}"
LR_FINAL="${LR_FINAL:-3e-5}"
WARMUP="${WARMUP:-100}"
SAVE_PER_STEP="${SAVE_PER_STEP:-1000}"
ACCEPT_EVERY="${ACCEPT_EVERY:-1000}"
MTP_DEPTH="${MTP_DEPTH:-1}"
STRATEGY="${STRATEGY:-deepspeed_stage_1}"   # or fsdp2 (no deepspeed needed)
PRECISION="${PRECISION:-bf16}"
WANDB="${WANDB-rwkv7-mtp}"
WANDB_MODE="${WANDB_MODE:-online}"
N_LAYER="${N_LAYER:-61}"
N_EMBD="${N_EMBD:-4096}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
HEAD_SIZE_A="${HEAD_SIZE_A:-64}"

seq_per_step=$((MICRO_BSZ * DEVICES))
epoch_steps=$((STEPS * seq_per_step))
tokens=$((epoch_steps * CTX_LEN))

banner() {
  echo "=========================================================="
  echo " MTP / tulu-3-sft-mixture"
  echo " repo       : $REPO"
  echo " python     : $PYTHON"
  echo " stage      : $STAGE"
  echo " model      : ${MODEL:-<unset>}"
  echo " data       : $DATA_PREFIX"
  echo " tokenizer  : $TOKENIZER ($TOKENIZER_IMPL)"
  echo " devices    : $DEVICES  micro_bsz $MICRO_BSZ  ctx $CTX_LEN  ($seq_per_step seq/step)"
  echo " schedule   : $STEPS steps ~ $((tokens / 1000000))M tokens, lr $LR -> $LR_FINAL"
  echo " save       : every $SAVE_PER_STEP steps"
  echo " accept     : every step (teacher-forced) + every $ACCEPT_EVERY (decode)"
  echo " strategy   : $STRATEGY  precision $PRECISION"
  echo " proj_dir   : $OUT"
  echo "=========================================================="
}

# ---------------------------------------------------------------- stages ----
prepare() {
  if [ -f "$DATA_PREFIX.json" ]; then
    echo "[prepare] $DATA_PREFIX.json exists, skipping (delete it to rebuild)"
    cat "$DATA_PREFIX.json"
    return
  fi
  local cmd=("$PYTHON" scripts/prepare_tulu_binidx.py
             --output-prefix "$DATA_PREFIX" --dataset "$DATASET"
             --tokenizer "$TOKENIZER" --tokenizer-impl "$TOKENIZER_IMPL"
             --workers "$WORKERS")
  [ -n "$VOCAB_FILE" ] && cmd+=(--vocab-file "$VOCAB_FILE")
  [ "$MAX_DOCS" -gt 0 ] && cmd+=(--max-docs "$MAX_DOCS")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '  %q' "${cmd[@]}"; echo; return; fi
  "${cmd[@]}"
}

train() {
  if [ -z "$MODEL" ]; then
    echo "MODEL is required for STAGE=train (path to the RWKV-7 13B checkpoint)" >&2
    exit 1
  fi
  if [ ! -f "$MODEL" ]; then
    echo "MODEL=$MODEL does not exist" >&2
    exit 1
  fi
  [ -f "$DATA_PREFIX.bin" ] || { echo "missing $DATA_PREFIX.bin; run STAGE=prepare" >&2; exit 1; }

  if [ "$STRATEGY" = "deepspeed_stage_1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import deepspeed" 2>/dev/null || {
      echo "deepspeed is not installed; rerun with STRATEGY=fsdp2" >&2; exit 1; }
  fi

  local args=(
    --load_model "$MODEL" --proj_dir "$OUT"
    --data_file "$DATA_PREFIX" --data_type binidx
    --vocab_size "$VOCAB_SIZE" --n_layer "$N_LAYER" --n_embd "$N_EMBD"
    --head_size_a "$HEAD_SIZE_A"
    --ctx_len "$CTX_LEN" --micro_bsz "$MICRO_BSZ"
    --epoch_steps "$epoch_steps" --epoch_count 1 --epoch_save 0
    --save_per_step "$SAVE_PER_STEP"
    --lr_init "$LR" --lr_final "$LR_FINAL" --lr_schedule cos --warmup_steps "$WARMUP"
    --accelerator gpu --devices "$DEVICES" --strategy "$STRATEGY"
    --precision "$PRECISION" --grad_cp 1 --num_workers 0 --sft_num_proc 0
    --op cuda --my_testing x070
    --peft mtp --mtp_depth "$MTP_DEPTH" --mtp_loss_weight 1.0 --mtp_main_weight 0.0
    --mtp_step_beta 0.6
    --accept_every "$ACCEPT_EVERY" --accept_prompts 8 --accept_tokens 32
  )
  if [ -n "$WANDB" ]; then
    args+=(--wandb "$WANDB" --wandb_mode "$WANDB_MODE")
  fi

  mkdir -p "$OUT"
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '  %q' "$PYTHON" train.py "${args[@]}"; echo; return; fi
  exec "$PYTHON" train.py "${args[@]}"
}

# ------------------------------------------------------------------ main ----
banner
case "$STAGE" in
  prepare) prepare ;;
  train)   train ;;
  all)     prepare; train ;;
  *) echo "unknown STAGE=$STAGE (all|prepare|train)" >&2; exit 1 ;;
esac
