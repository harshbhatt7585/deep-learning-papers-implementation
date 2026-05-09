#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
RUN_CONFIG="${RUN_CONFIG:-${2:-8gpu}}"

TRAIN_SHARDS="${TRAIN_SHARDS:-170}"
MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-17000000000}"
MAX_VAL_CHARS="${MAX_VAL_CHARS:-2000000}"
TOKEN_SHARDS_DIR="${TOKEN_SHARDS_DIR:-/data/nanochat_tokens_32k}"
TOKENIZER_THREADS="${TOKENIZER_THREADS:-64}"
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-4096}"
TOKENIZER_TRAIN_SHARDS="${TOKENIZER_TRAIN_SHARDS:-8}"
STREAM_NANOCHAT="${STREAM_NANOCHAT:-1}"
GPU_TYPE="${GPU_TYPE:-H100}"
GPU_TYPE_UPPER="$(printf "%s" "${GPU_TYPE}" | tr '[:lower:]' '[:upper:]')"

MAX_STEPS="${MAX_STEPS:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
TARGET_TOKENS="${TARGET_TOKENS:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-2048}"
OPTIMIZER="${OPTIMIZER:-muon}"

D_MODEL="${D_MODEL:-768}"
N_HEADS="${N_HEADS:-6}"
N_LAYERS="${N_LAYERS:-12}"

case "${RUN_CONFIG}" in
  1gpu|1GPU|1)
    GPU_COUNT=1
    DEFAULT_GRAD_ACCUM_STEPS=8
    RUN_CONFIG_NAME="1gpu"
    ;;
  2gpu|2GPU|2)
    GPU_COUNT=2
    DEFAULT_GRAD_ACCUM_STEPS=4
    RUN_CONFIG_NAME="2gpu"
    ;;
  4gpu|4GPU|4)
    GPU_COUNT=4
    DEFAULT_GRAD_ACCUM_STEPS=2
    RUN_CONFIG_NAME="4gpu"
    ;;
  8gpu|8GPU|8)
    GPU_COUNT=8
    DEFAULT_GRAD_ACCUM_STEPS=1
    RUN_CONFIG_NAME="8gpu"
    ;;
  *)
    echo "unknown run config: ${RUN_CONFIG}" >&2
    echo "usage: $0 [tokenizer|download|pretokenize|train|all] [1gpu|2gpu|4gpu|8gpu]" >&2
    exit 2
    ;;
esac

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${DEFAULT_GRAD_ACCUM_STEPS}}"

DEFAULT_RUN_TIME="$(date +"%Y-%m-%d--%I-%M%p" | tr '[:upper:]' '[:lower:]')"
RUN_NAME="${RUN_NAME:-text-diffusion-${RUN_CONFIG_NAME}--${DEFAULT_RUN_TIME}}"
OUT_DIR="${OUT_DIR:-/runs/${RUN_NAME}}"

if [[ -z "${FP8+x}" ]]; then
  if [[ "${GPU_TYPE_UPPER}" == "H100" ]]; then
    FP8=1
  else
    FP8=0
  fi
fi
if [[ "${GPU_TYPE_UPPER}" != "H100" && "${FP8}" == "1" ]]; then
  echo "warning: disabling FP8 because GPU_TYPE=${GPU_TYPE} is not H100" >&2
  FP8=0
fi
COMPILE="${COMPILE:-1}"
WANDB="${WANDB:-1}"
OVERWRITE_TOKENS="${OVERWRITE_TOKENS:-0}"

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
    --tokenizer-threads "${TOKENIZER_THREADS}" \
    --doc-batch-size "${DOC_BATCH_SIZE}" \
    --tokenizer-train-shards "${TOKENIZER_TRAIN_SHARDS}" \
    "${overwrite_flags[@]}"
}

train_tokenizer() {
  local overwrite_flags=()
  if [[ "${OVERWRITE_TOKENS}" == "1" ]]; then
    overwrite_flags+=(--overwrite-tokens)
  fi

  modal run modal_train.py \
    --pretokenize \
    --tokenizer-only \
    --train-shards "${TOKENIZER_TRAIN_SHARDS}" \
    --max-train-chars "${MAX_TRAIN_CHARS}" \
    --max-val-chars "${MAX_VAL_CHARS}" \
    --token-shards-dir "${TOKEN_SHARDS_DIR}" \
    --tokenizer-threads "${TOKENIZER_THREADS}" \
    --doc-batch-size "${DOC_BATCH_SIZE}" \
    --tokenizer-train-shards "${TOKENIZER_TRAIN_SHARDS}" \
    "${overwrite_flags[@]}"
}

download_data() {
  modal run modal_train.py \
    --pretokenize \
    --download-only \
    --train-shards "${TRAIN_SHARDS}" \
    --max-train-chars "${MAX_TRAIN_CHARS}" \
    --max-val-chars "${MAX_VAL_CHARS}" \
    --token-shards-dir "${TOKEN_SHARDS_DIR}" \
    --tokenizer-threads "${TOKENIZER_THREADS}" \
    --doc-batch-size "${DOC_BATCH_SIZE}"
}

train() {
  local data_flags=()
  if [[ "${STREAM_NANOCHAT}" == "1" ]]; then
    data_flags+=(--stream-nanochat --train-shards "${TRAIN_SHARDS}" --max-val-chars "${MAX_VAL_CHARS}")
  else
    data_flags+=(--token-shards-dir "${TOKEN_SHARDS_DIR}")
  fi

  modal run modal_train.py \
    --gpu-count "${GPU_COUNT}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --gpu-type "${GPU_TYPE}" \
    --d-model "${D_MODEL}" \
    --n-heads "${N_HEADS}" \
    --n-layers "${N_LAYERS}" \
    --target-param-data-ratio "${TARGET_PARAM_DATA_RATIO}" \
    --target-tokens "${TARGET_TOKENS}" \
    --out-dir "${OUT_DIR}" \
    "${data_flags[@]}" \
    "${modal_flags[@]}"
}

case "${MODE}" in
  tokenizer|tok)
    train_tokenizer
    ;;
  download|download-data)
    download_data
    ;;
  pretokenize|prep|data)
    pretokenize
    ;;
  train)
    train
    ;;
  all)
    train_tokenizer
    download_data
    train
    ;;
  *)
    echo "usage: $0 [tokenizer|download|pretokenize|train|all]" >&2
    exit 2
    ;;
esac
