#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"

TRAIN_SHARDS="${TRAIN_SHARDS:-170}"
MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-17000000000}"
MAX_VAL_CHARS="${MAX_VAL_CHARS:-2000000}"
TOKEN_SHARDS_DIR="${TOKEN_SHARDS_DIR:-/data/nanochat_tokens_32k}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
OPTIMIZER="${OPTIMIZER:-adamw}"

D_MODEL="${D_MODEL:-768}"
N_HEADS="${N_HEADS:-12}"
N_LAYERS="${N_LAYERS:-12}"

RUN_NAME="${RUN_NAME:-text-diffusion-adamw-170shards}"
OUT_DIR="${OUT_DIR:-/runs/${RUN_NAME}}"

COMPILE="${COMPILE:-1}"
WANDB="${WANDB:-1}"
OVERWRITE_TOKENS="${OVERWRITE_TOKENS:-0}"

modal_flags=()
if [[ "${COMPILE}" == "1" ]]; then
  modal_flags+=(--compile)
fi
if [[ "${WANDB}" == "1" ]]; then
  modal_flags+=(--wandb)
fi

pretokenize() {
  local overwrite_flags=()
  if [[ "${OVERWRITE_TOKENS}" == "1" ]]; then
    overwrite_flags+=(--overwrite-tokens)
  fi

  modal run modal_train.py \
    --pretokenize \
    --train-shards "${TRAIN_SHARDS}" \
    --max-train-chars "${MAX_TRAIN_CHARS}" \
    --max-val-chars "${MAX_VAL_CHARS}" \
    --token-shards-dir "${TOKEN_SHARDS_DIR}" \
    "${overwrite_flags[@]}"
}

train() {
  modal run modal_train.py \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --token-shards-dir "${TOKEN_SHARDS_DIR}" \
    --out-dir "${OUT_DIR}" \
    "${modal_flags[@]}"
}

case "${MODE}" in
  pretokenize|prep|data)
    pretokenize
    ;;
  train)
    train
    ;;
  all)
    pretokenize
    train
    ;;
  *)
    echo "usage: $0 [pretokenize|train|all]" >&2
    exit 2
    ;;
esac
