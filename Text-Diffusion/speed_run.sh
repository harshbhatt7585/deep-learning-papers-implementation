#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
RUN_CONFIG="${RUN_CONFIG:-${2:-8gpu}}"

# --- Drafter-mode defaults --------------------------------------------------
# When MODE=draft we're training a DFlash block-diffusion drafter (Chen et al.
# 2026, "Block Diffusion for Flash Speculative Decoding"). The drafter is a
# small cross-attention adapter on top of a frozen target — it shares the
# target's embed_tokens and lm_head and is conditioned on the target's
# intermediate hidden states. The user MUST provide a target checkpoint via
# the TARGET_CHECKPOINT env var.
#
# These ":=" forms only set a value if it isn't already exported in the
# caller's environment, so users can still override any of them on the
# command line. They must appear BEFORE the generic ":-" defaults below,
# otherwise the target-sized defaults win.
#
# Reference DFlash drafters use n_draft_layers=2-3, block_size=16. The drafter
# does NOT own its own d_model — it inherits target_d_model at bind time. The
# args.d_model below is harmlessly ignored by DFlashDraftModel; we keep the
# default small so logging is honest about drafter scale.
if [[ "${MODE}" == "draft" || "${MODE}" == "drafter" ]]; then
  : "${OBJECTIVE:=dflash}"           # block-diffusion drafter objective
  : "${MTP_HEADS:=0}"                # MTP heads irrelevant for dflash
  : "${BLOCK_SIZE:=16}"              # B from the paper (1 anchor + B-1 drafts)
  : "${N_DRAFT_LAYERS:=2}"           # 2 layers per DFlash defaults
  : "${N_HEADS:=4}"                  # drafter attention heads
  : "${FF_MULT:=4}"
  : "${GATED_MLP:=0}"
  : "${BATCH_SIZE:=32}"
  : "${SEQ_LEN:=512}"                # block fits inside seq_len with margin
  : "${MAX_STEPS:=2000}"             # drafter converges fast; raise if accept is low
  : "${EVAL_INTERVAL:=200}"
  : "${SAMPLE_INTERVAL:=0}"          # drafter doesn't generate standalone samples
  # args.d_model isn't used by the drafter (inherits target_d_model), but
  # we set it so target_param_data_ratio computations stay sane.
  : "${D_MODEL:=256}"
  : "${N_LAYERS:=2}"
  : "${TARGET_PARAM_DATA_RATIO:=30}"

  if [[ -z "${TARGET_CHECKPOINT:-}" ]]; then
    echo "[speed_run] ERROR: MODE=${MODE} requires TARGET_CHECKPOINT=<path/to/target/checkpoint.pt>" >&2
    echo "             Example: TARGET_CHECKPOINT=runs/text-diffusion-mtp1-relu2/checkpoint.pt \\" >&2
    echo "                       bash speed_run.sh draft 4gpu" >&2
    exit 64
  fi
fi

TRAIN_SHARDS="${TRAIN_SHARDS:-170}"
MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-17000000000}"
MAX_VAL_CHARS="${MAX_VAL_CHARS:-2000000}"
TOKEN_SHARDS_DIR="${TOKEN_SHARDS_DIR:-/data/nanochat_tokens_32k}"
NANOCHAT_TOKENIZER_CACHE_DIR="${NANOCHAT_TOKENIZER_CACHE_DIR:-/data/nanochat_tokenizer_32k}"
NANOCHAT_TOKENIZER_VOCAB_SIZE="${NANOCHAT_TOKENIZER_VOCAB_SIZE:-32768}"
TOKENIZER_THREADS="${TOKENIZER_THREADS:-64}"
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-4096}"
TOKENIZER_TRAIN_SHARDS="${TOKENIZER_TRAIN_SHARDS:-8}"
STREAM_NANOCHAT="${STREAM_NANOCHAT:-1}"
GPU_TYPE="${GPU_TYPE:-H100}"
GPU_TYPE_UPPER="$(printf "%s" "${GPU_TYPE}" | tr '[:lower:]' '[:upper:]' | tr '_' '-')"
if [[ "${GPU_TYPE_UPPER}" == "A100-80GB" ]]; then
  GPU_TYPE="A100-80GB"
fi

MAX_STEPS="${MAX_STEPS:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
TARGET_TOKENS="${TARGET_TOKENS:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-2048}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-}"
OPTIMIZER="${OPTIMIZER:-muon}"
OBJECTIVE="${OBJECTIVE:-diffusion}"
MTP_HEADS="${MTP_HEADS:-3}"
MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-0.3}"
AURORA_WEIGHT_DECAY="${AURORA_WEIGHT_DECAY:-0.025}"

D_MODEL="${D_MODEL:-768}"
N_HEADS="${N_HEADS:-6}"
N_LAYERS="${N_LAYERS:-12}"
FF_MULT="${FF_MULT:-4}"
GATED_MLP="${GATED_MLP:-0}"

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
if [[ "${MODE}" == "draft" || "${MODE}" == "drafter" ]]; then
  DEFAULT_RUN_NAME="text-diffusion-dflash-drafter-${RUN_CONFIG_NAME}--${DEFAULT_RUN_TIME}"
else
  DEFAULT_RUN_NAME="text-diffusion-${RUN_CONFIG_NAME}--${DEFAULT_RUN_TIME}"
fi
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUT_DIR="${OUT_DIR:-/runs/${RUN_NAME}}"
RESUME="${RESUME:-}"

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
if [[ "${GPU_TYPE_UPPER}" == "A100" || "${GPU_TYPE_UPPER}" == "A100-80GB" ]]; then
  echo "modal gpu request: A100-80GB:${GPU_COUNT}" >&2
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
  local command=(
    modal run modal_train.py::main
    --pretokenize
    --train-shards "${TRAIN_SHARDS}"
    --max-train-chars "${MAX_TRAIN_CHARS}"
    --max-val-chars "${MAX_VAL_CHARS}"
    --token-shards-dir "${TOKEN_SHARDS_DIR}"
    --nanochat-tokenizer-cache-dir "${NANOCHAT_TOKENIZER_CACHE_DIR}"
    --nanochat-tokenizer-vocab-size "${NANOCHAT_TOKENIZER_VOCAB_SIZE}"
    --tokenizer-threads "${TOKENIZER_THREADS}"
    --doc-batch-size "${DOC_BATCH_SIZE}"
    --tokenizer-train-shards "${TOKENIZER_TRAIN_SHARDS}"
  )
  if [[ "${OVERWRITE_TOKENS}" == "1" ]]; then
    command+=(--overwrite-tokens)
  fi
  "${command[@]}"
}

train_tokenizer() {
  local command=(
    modal run modal_train.py::main
    --pretokenize
    --tokenizer-only
    --train-shards "${TOKENIZER_TRAIN_SHARDS}"
    --max-train-chars "${MAX_TRAIN_CHARS}"
    --max-val-chars "${MAX_VAL_CHARS}"
    --token-shards-dir "${TOKEN_SHARDS_DIR}"
    --nanochat-tokenizer-cache-dir "${NANOCHAT_TOKENIZER_CACHE_DIR}"
    --nanochat-tokenizer-vocab-size "${NANOCHAT_TOKENIZER_VOCAB_SIZE}"
    --tokenizer-threads "${TOKENIZER_THREADS}"
    --doc-batch-size "${DOC_BATCH_SIZE}"
    --tokenizer-train-shards "${TOKENIZER_TRAIN_SHARDS}"
  )
  if [[ "${OVERWRITE_TOKENS}" == "1" ]]; then
    command+=(--overwrite-tokens)
  fi
  "${command[@]}"
}

download_data() {
  modal run modal_train.py::main \
    --pretokenize \
    --download-only \
    --train-shards "${TRAIN_SHARDS}" \
    --max-train-chars "${MAX_TRAIN_CHARS}" \
    --max-val-chars "${MAX_VAL_CHARS}" \
    --token-shards-dir "${TOKEN_SHARDS_DIR}" \
    --nanochat-tokenizer-cache-dir "${NANOCHAT_TOKENIZER_CACHE_DIR}" \
    --nanochat-tokenizer-vocab-size "${NANOCHAT_TOKENIZER_VOCAB_SIZE}" \
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

  local command=(
    modal run modal_train.py::main
    --gpu-count "${GPU_COUNT}"
    --max-steps "${MAX_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --seq-len "${SEQ_LEN}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --optimizer "${OPTIMIZER}"
    --objective "${OBJECTIVE}"
    --mtp-heads "${MTP_HEADS}"
    --mtp-loss-weight "${MTP_LOSS_WEIGHT}"
    --aurora-weight-decay "${AURORA_WEIGHT_DECAY}"
    --gpu-type "${GPU_TYPE}"
    --d-model "${D_MODEL}"
    --n-heads "${N_HEADS}"
    --n-layers "${N_LAYERS}"
    --ff-mult "${FF_MULT}"
    --target-param-data-ratio "${TARGET_PARAM_DATA_RATIO}"
    --target-tokens "${TARGET_TOKENS}"
    --out-dir "${OUT_DIR}"
    --nanochat-tokenizer-cache-dir "${NANOCHAT_TOKENIZER_CACHE_DIR}"
    --nanochat-tokenizer-vocab-size "${NANOCHAT_TOKENIZER_VOCAB_SIZE}"
  )
  if [[ "${GATED_MLP}" == "1" ]]; then
    command+=(--gated-mlp)
  fi
  if [[ -n "${RESUME}" ]]; then
    command+=(--resume "${RESUME}")
  fi
  if [[ "${OBJECTIVE}" == "dflash" ]]; then
    command+=(
      --target-checkpoint "${TARGET_CHECKPOINT}"
      --block-size "${BLOCK_SIZE:-16}"
      --n-draft-layers "${N_DRAFT_LAYERS:-2}"
    )
  fi
  if [[ -n "${EVAL_INTERVAL}" ]]; then
    command+=(--eval-interval "${EVAL_INTERVAL}")
  fi
  if [[ -n "${CORE_METRIC_EVERY}" ]]; then
    command+=(--core-metric-every "${CORE_METRIC_EVERY}")
  fi
  if [[ -n "${SAMPLE_INTERVAL}" ]]; then
    command+=(--sample-interval "${SAMPLE_INTERVAL}")
  fi
  command+=("${data_flags[@]}")
  if [[ ${#modal_flags[@]} -gt 0 ]]; then
    command+=("${modal_flags[@]}")
  fi
  "${command[@]}"
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
  draft|drafter)
    # DFlash drafter training. The drafter-specific defaults at the top of
    # this script (OBJECTIVE=dflash, BLOCK_SIZE, N_DRAFT_LAYERS, ...) have
    # already been applied. train() builds a DFlashDraftModel bound to the
    # target loaded from TARGET_CHECKPOINT.
    echo "[speed_run] training dflash drafter:" >&2
    echo "  RUN_NAME=${RUN_NAME}" >&2
    echo "  target:  ${TARGET_CHECKPOINT}" >&2
    echo "  arch:    n_draft_layers=${N_DRAFT_LAYERS} n_heads=${N_HEADS} ff_mult=${FF_MULT} block_size=${BLOCK_SIZE}" >&2
    echo "  obj:     ${OBJECTIVE}" >&2
    echo "  budget:  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN}" >&2
    train
    ;;
  all)
    train_tokenizer
    download_data
    train
    ;;
  *)
    echo "usage: $0 [tokenizer|download|pretokenize|train|draft|all]" >&2
    exit 2
    ;;
esac
