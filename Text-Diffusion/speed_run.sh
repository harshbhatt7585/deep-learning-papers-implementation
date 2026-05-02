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

FP8="${FP8:-1}"
COMPILE="${COMPILE:-$([[ "${FP8}" == "1" ]] && echo 0 || echo 1)}"
WANDB="${WANDB:-1}"
OVERWRITE_TOKENS="${OVERWRITE_TOKENS:-0}"
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:--1}"
CORE_EVAL_CACHE_DIR="${CORE_EVAL_CACHE_DIR:-/data/core_eval}"

modal_flags=()
if [[ "${COMPILE}" == "1" ]]; then
  modal_flags+=(--compile)
fi
if [[ "${FP8}" == "1" ]]; then
  modal_flags+=(--fp8)
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

core_eval() {
  modal run modal_train.py \
    --core-eval \
    --checkpoint-dir "${OUT_DIR}" \
    --eval-cache-dir "${CORE_EVAL_CACHE_DIR}" \
    --max-per-task "${CORE_MAX_PER_TASK}"
}

case "${MODE}" in
  pretokenize|prep|data)
    pretokenize
    ;;
  train)
    train
    ;;
  core|core-eval|eval)
    core_eval
    ;;
  all)
    pretokenize
    train
    core_eval
    ;;
  *)
    echo "usage: $0 [pretokenize|train|core-eval|all]" >&2
    exit 2
    ;;
esac
