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
    echo "             Example: TARGET_CHECKPOINT=/runs/tinygroot-mtp1-relu2/checkpoint.pt \\" >&2
    echo "                       bash speed_run.sh draft 4gpu" >&2
    exit 64
  fi
fi

if [[ "${MODE}" == "rl" || "${MODE}" == "chat-rl" || "${MODE}" == "chat_rl" ]]; then
  : "${NUM_EPOCHS:=1}"
  : "${EXAMPLES_PER_STEP:=16}"
  : "${NUM_SAMPLES:=16}"
  : "${MAX_NEW_TOKENS:=256}"
  : "${TEMPERATURE:=1.0}"
  : "${TOP_K:=50}"
  : "${EVAL_EXAMPLES:=400}"
  : "${EVAL_INTERVAL:=60}"
  : "${SAVE_EVERY:=60}"
  : "${RL_DEVICE_BATCH_SIZE:=${BATCH_SIZE:-8}}"
  if [[ -z "${RL_CHECKPOINT:-${CHECKPOINT:-}}" ]]; then
    echo "[speed_run] ERROR: MODE=${MODE} requires RL_CHECKPOINT=/runs/<sft-run>/checkpoint.pt" >&2
    exit 64
  fi
  RL_CHECKPOINT="${RL_CHECKPOINT:-${CHECKPOINT}}"
fi

PIPELINE_MODE=0
if [[ "${MODE}" == "pipeline" || "${MODE}" == "full" || "${MODE}" == "pretrain-sft-rl-eval" ]]; then
  PIPELINE_MODE=1
fi

if [[ "${PIPELINE_MODE}" == "1" ]]; then
  : "${NUM_EPOCHS:=1}"
  : "${EXAMPLES_PER_STEP:=16}"
  : "${NUM_SAMPLES:=16}"
  : "${MAX_NEW_TOKENS:=256}"
  : "${TEMPERATURE:=1.0}"
  : "${TOP_K:=50}"
  : "${EVAL_EXAMPLES:=400}"
  : "${EVAL_INTERVAL:=60}"
  : "${SAVE_EVERY:=60}"
  : "${RL_DEVICE_BATCH_SIZE:=${BATCH_SIZE:-8}}"
fi

if [[ "${MODE}" == "sft" || "${MODE}" == "chat-sft" || "${MODE}" == "chat_sft" || "${PIPELINE_MODE}" == "1" ]]; then
  : "${SFT_MAX_STEPS:=-1}"
  : "${SFT_DEVICE_BATCH_SIZE:=16}"
  : "${SFT_TOTAL_BATCH_SIZE:=524288}"
  : "${SFT_EVAL_EVERY:=200}"
  : "${SFT_EVAL_TOKENS:=2097152}"
  if [[ "${PIPELINE_MODE}" == "1" ]]; then
    : "${SFT_CHATCORE_EVERY:=0}"
  else
    : "${SFT_CHATCORE_EVERY:=200}"
  fi
  : "${SFT_CHATCORE_MAX_CAT:=-1}"
  : "${SFT_CHATCORE_MAX_SAMPLE:=24}"
  : "${SFT_SAMPLE_EVERY:=200}"
  : "${SFT_SAMPLE_LENGTH:=128}"
  : "${SFT_OPTIMIZER:=muon}"
  : "${SFT_WANDB_PROJECT:=tinyGroot-sft}"
  : "${SFT_MMLU_EPOCHS:=3}"
  : "${SFT_GSM8K_EPOCHS:=4}"
  : "${SFT_SIMPLE_SPELLING_SIZE:=200000}"
  : "${SFT_SPELLINGBEE_SIZE:=80000}"
  if [[ "${PIPELINE_MODE}" != "1" && -z "${SFT_CHECKPOINT:-${CHECKPOINT:-}}" ]]; then
    echo "[speed_run] ERROR: MODE=${MODE} requires SFT_CHECKPOINT=/runs/<pretrain-run>/checkpoint.pt" >&2
    exit 64
  fi
  SFT_CHECKPOINT="${SFT_CHECKPOINT:-${CHECKPOINT:-}}"
fi

if [[ "${MODE}" == "eval" || "${MODE}" == "chatcore" || "${MODE}" == "chatcore-eval" || "${PIPELINE_MODE}" == "1" ]]; then
  : "${EVAL_SUITE:=chatcore}"
  : "${EVAL_EXAMPLES:=400}"
  : "${EVAL_NUM_SAMPLES:=8}"
  : "${MAX_NEW_TOKENS:=256}"
  : "${TEMPERATURE:=1.0}"
  : "${TOP_K:=50}"
  : "${CHATCORE_MAX_CAT:=-1}"
  : "${CHATCORE_MAX_SAMPLE:=24}"
  : "${CHATCORE_MAX_NEW_TOKENS:=512}"
  : "${CHATCORE_TEMPERATURE:=0.0}"
  : "${CHATCORE_TOP_K:=50}"
  : "${CHATCORE_BATCH_SIZE:=8}"
  if [[ "${PIPELINE_MODE}" != "1" && -z "${EVAL_CHECKPOINT:-${CHECKPOINT:-}}" ]]; then
    echo "[speed_run] ERROR: MODE=${MODE} requires EVAL_CHECKPOINT=/runs/<run>/checkpoint.pt" >&2
    exit 64
  fi
  EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${CHECKPOINT:-}}"
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
STREAM_NANOCHAT="${STREAM_NANOCHAT:-0}"
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
MTP_HEADS="${MTP_HEADS:-3}"
ARCH="${ARCH:-hrm}"
MTP_ARCH="${MTP_ARCH:-linear}"
MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-0.3}"
TST_BAG_SIZE="${TST_BAG_SIZE:-1}"
TST_RATIO="${TST_RATIO:-0}"
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
    echo "usage: $0 [tokenizer|download|pretokenize|train|sft|rl|eval|pipeline|draft|all] [1gpu|2gpu|4gpu|8gpu]" >&2
    exit 2
    ;;
esac

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${DEFAULT_GRAD_ACCUM_STEPS}}"

# Canonical run names come from a single source of truth (tinygroot.exp_naming):
#   groot/{stage}/{YYYYMMDD}-{slug}-{gpu}x{count}-{gitsha}
# SLUG is the one human-facing knob; everything else (date, gpu, git sha) is filled
# in for you, and the full hyperparameter set lives in each run's meta.json + wandb.
GPU_LABEL="$(printf '%s' "${GPU_TYPE_UPPER}" | sed 's/-80GB//')"
SLUG="${SLUG:-}"
gen_name() { python -m tinygroot.exp_naming --stage "$1" --slug "${SLUG}" --gpu-type "${GPU_LABEL}" --gpu-count "${GPU_COUNT}"; }
case "${MODE}" in
  draft|drafter) DEFAULT_STAGE="draft" ;;
  rl|chat-rl|chat_rl) DEFAULT_STAGE="rl" ;;
  sft|chat-sft|chat_sft) DEFAULT_STAGE="sft" ;;
  *) DEFAULT_STAGE="pretrain" ;;
esac
DEFAULT_RUN_NAME="$(gen_name "${DEFAULT_STAGE}")"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUT_DIR="${OUT_DIR:-/runs/${RUN_NAME}}"
RESUME="${RESUME:-}"

if [[ "${PIPELINE_MODE}" == "1" ]]; then
  PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-$(gen_name pretrain)}"
  SFT_RUN_NAME="${SFT_RUN_NAME:-$(gen_name sft)}"
  RL_RUN_NAME="${RL_RUN_NAME:-$(gen_name rl)}"
  PRETRAIN_OUT_DIR="${PRETRAIN_OUT_DIR:-/runs/${PRETRAIN_RUN_NAME}}"
  SFT_OUT_DIR="${SFT_OUT_DIR:-/runs/${SFT_RUN_NAME}}"
  RL_OUT_DIR="${RL_OUT_DIR:-/runs/${RL_RUN_NAME}}"
  SFT_CHECKPOINT="${SFT_CHECKPOINT:-${PRETRAIN_OUT_DIR}/checkpoint.pt}"
  RL_CHECKPOINT="${RL_CHECKPOINT:-${SFT_OUT_DIR}/checkpoint.pt}"
  EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${RL_OUT_DIR}/checkpoint.pt}"
else
  PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-${RUN_NAME}}"
  SFT_RUN_NAME="${SFT_RUN_NAME:-${RUN_NAME}}"
  RL_RUN_NAME="${RL_RUN_NAME:-${RUN_NAME}}"
  PRETRAIN_OUT_DIR="${PRETRAIN_OUT_DIR:-${OUT_DIR}}"
  SFT_OUT_DIR="${SFT_OUT_DIR:-${OUT_DIR}}"
  RL_OUT_DIR="${RL_OUT_DIR:-${OUT_DIR}}"
fi

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
PUSH_TO_HF="${PUSH_TO_HF:-0}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_REVISION="${HF_REVISION:-}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-}"

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
if [[ "${PUSH_TO_HF}" == "1" ]]; then
  if [[ -z "${HF_REPO_ID}" ]]; then
    echo "[speed_run] ERROR: PUSH_TO_HF=1 requires HF_REPO_ID=<namespace/model>" >&2
    exit 64
  fi
  modal_flags+=(--push-to-hf --hf-repo-id "${HF_REPO_ID}")
  if [[ "${HF_PRIVATE}" == "1" ]]; then
    modal_flags+=(--hf-private)
  fi
  if [[ -n "${HF_REVISION}" ]]; then
    modal_flags+=(--hf-revision "${HF_REVISION}")
  fi
  if [[ -n "${HF_COMMIT_MESSAGE}" ]]; then
    modal_flags+=(--hf-commit-message "${HF_COMMIT_MESSAGE}")
  fi
fi

pretokenize() {
  local command=(
    modal run tinygroot/modal/modal_train.py::main
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
    modal run tinygroot/modal/modal_train.py::main
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
  modal run tinygroot/modal/modal_train.py::main \
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
    modal run tinygroot/modal/modal_train.py::main
    --gpu-count "${GPU_COUNT}"
    --max-steps "${MAX_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --seq-len "${SEQ_LEN}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --optimizer "${OPTIMIZER}"
    --mtp-heads "${MTP_HEADS}"
    --arch "${ARCH}"
    --mtp-arch "${MTP_ARCH}"
    --mtp-loss-weight "${MTP_LOSS_WEIGHT}"
    --tst-bag-size "${TST_BAG_SIZE}"
    --tst-ratio "${TST_RATIO}"
    --aurora-weight-decay "${AURORA_WEIGHT_DECAY}"
    --gpu-type "${GPU_TYPE}"
    --d-model "${D_MODEL}"
    --n-heads "${N_HEADS}"
    --n-layers "${N_LAYERS}"
    --ff-mult "${FF_MULT}"
    --target-param-data-ratio "${TARGET_PARAM_DATA_RATIO}"
    --target-tokens "${TARGET_TOKENS}"
    --out-dir "${PRETRAIN_OUT_DIR}"
    --nanochat-tokenizer-cache-dir "${NANOCHAT_TOKENIZER_CACHE_DIR}"
    --nanochat-tokenizer-vocab-size "${NANOCHAT_TOKENIZER_VOCAB_SIZE}"
  )
  if [[ "${GATED_MLP}" == "1" ]]; then
    command+=(--gated-mlp)
  fi
  if [[ -n "${RESUME}" ]]; then
    command+=(--resume "${RESUME}")
  fi
  if [[ "${MODE}" == "draft" || "${MODE}" == "drafter" ]]; then
    command+=(
      --dflash
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

rl() {
  local command=(
    modal run tinygroot/modal/modal_train.py::rl
    --checkpoint "${RL_CHECKPOINT}"
    --out-dir "${RL_OUT_DIR}"
    --gpu-type "${GPU_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --num-epochs "${NUM_EPOCHS}"
    --max-steps "${MAX_STEPS}"
    --device-batch-size "${RL_DEVICE_BATCH_SIZE}"
    --examples-per-step "${EXAMPLES_PER_STEP}"
    --num-samples "${NUM_SAMPLES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-k "${TOP_K}"
    --eval-every "${EVAL_INTERVAL}"
    --eval-examples "${EVAL_EXAMPLES}"
    --save-every "${SAVE_EVERY}"
    --optimizer "${OPTIMIZER}"
  )
  if [[ "${RL_RUN_NAME}" != "${DEFAULT_RUN_NAME}" || "${PIPELINE_MODE}" == "1" ]]; then
    command+=(--run-name "${RL_RUN_NAME}")
  fi
  if [[ ${#modal_flags[@]} -gt 0 ]]; then
    command+=("${modal_flags[@]}")
  fi
  "${command[@]}"
}

sft() {
  local command=(
    modal run tinygroot/modal/modal_chat_sft.py::main
    --gpu-type "${GPU_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --checkpoint "${SFT_CHECKPOINT}"
    --out-dir "${SFT_OUT_DIR}"
    --run-name "${SFT_RUN_NAME}"
    --wandb-project "${SFT_WANDB_PROJECT}"
    --max-steps "${SFT_MAX_STEPS}"
    --device-batch-size "${SFT_DEVICE_BATCH_SIZE}"
    --total-batch-size "${SFT_TOTAL_BATCH_SIZE}"
    --eval-every "${SFT_EVAL_EVERY}"
    --eval-tokens "${SFT_EVAL_TOKENS}"
    --chatcore-every "${SFT_CHATCORE_EVERY}"
    --chatcore-max-cat "${SFT_CHATCORE_MAX_CAT}"
    --chatcore-max-sample "${SFT_CHATCORE_MAX_SAMPLE}"
    --sample-every "${SFT_SAMPLE_EVERY}"
    --sample-length "${SFT_SAMPLE_LENGTH}"
    --optimizer "${SFT_OPTIMIZER}"
    --mmlu-epochs "${SFT_MMLU_EPOCHS}"
    --gsm8k-epochs "${SFT_GSM8K_EPOCHS}"
    --simple-spelling-size "${SFT_SIMPLE_SPELLING_SIZE}"
    --spellingbee-size "${SFT_SPELLINGBEE_SIZE}"
  )
  if [[ -n "${SFT_SEQ_LEN:-}" ]]; then
    command+=(--seq-len "${SFT_SEQ_LEN}")
  fi
  if [[ "${FP8}" == "1" ]]; then
    command+=(--fp8)
  fi
  if [[ "${COMPILE}" == "1" ]]; then
    command+=(--compile)
  fi
  if [[ "${PUSH_TO_HF}" == "1" ]]; then
    if [[ -z "${HF_REPO_ID}" ]]; then
      echo "[speed_run] ERROR: PUSH_TO_HF=1 requires HF_REPO_ID=<namespace/model>" >&2
      exit 64
    fi
    command+=(--push-to-hf --hf-repo-id "${HF_REPO_ID}")
    if [[ "${HF_PRIVATE}" == "1" ]]; then
      command+=(--hf-private)
    fi
    if [[ -n "${HF_REVISION}" ]]; then
      command+=(--hf-revision "${HF_REVISION}")
    fi
    if [[ -n "${HF_COMMIT_MESSAGE}" ]]; then
      command+=(--hf-commit-message "${HF_COMMIT_MESSAGE}")
    fi
  fi
  "${command[@]}"
}

chatcore_eval() {
  local command=(
    modal run tinygroot/modal/modal_train.py::evaluate
    --gpu-type "${GPU_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --checkpoint "${EVAL_CHECKPOINT}"
    --suite "${EVAL_SUITE}"
    --eval-examples "${EVAL_EXAMPLES}"
    --eval-num-samples "${EVAL_NUM_SAMPLES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-k "${TOP_K}"
    --chatcore-max-cat "${CHATCORE_MAX_CAT}"
    --chatcore-max-sample "${CHATCORE_MAX_SAMPLE}"
    --chatcore-max-new-tokens "${CHATCORE_MAX_NEW_TOKENS}"
    --chatcore-temperature "${CHATCORE_TEMPERATURE}"
    --chatcore-top-k "${CHATCORE_TOP_K}"
    --chatcore-batch-size "${CHATCORE_BATCH_SIZE}"
  )
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
  sft|chat-sft|chat_sft)
    echo "[speed_run] training chat SFT:" >&2
    echo "  RUN_NAME=${SFT_RUN_NAME}" >&2
    echo "  checkpoint: ${SFT_CHECKPOINT}" >&2
    echo "  out_dir:    ${SFT_OUT_DIR}" >&2
    sft
    ;;
  draft|drafter)
    # DFlash drafter training. The drafter-specific defaults at the top of
    # this script (BLOCK_SIZE, N_DRAFT_LAYERS, ...) have
    # already been applied. train() builds a DFlashDraftModel bound to the
    # target loaded from TARGET_CHECKPOINT.
    echo "[speed_run] training dflash drafter:" >&2
    echo "  RUN_NAME=${RUN_NAME}" >&2
    echo "  target:  ${TARGET_CHECKPOINT}" >&2
    echo "  arch:    n_draft_layers=${N_DRAFT_LAYERS} n_heads=${N_HEADS} ff_mult=${FF_MULT} block_size=${BLOCK_SIZE}" >&2
    echo "  budget:  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN}" >&2
    train
    ;;
  rl|chat-rl|chat_rl)
    echo "[speed_run] training GSM8K RL:" >&2
    echo "  RUN_NAME=${RL_RUN_NAME}" >&2
    echo "  checkpoint: ${RL_CHECKPOINT}" >&2
    echo "  rollout: examples_per_step=${EXAMPLES_PER_STEP} num_samples=${NUM_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}" >&2
    rl
    ;;
  eval|chatcore|chatcore-eval)
    echo "[speed_run] running standalone eval:" >&2
    echo "  checkpoint: ${EVAL_CHECKPOINT}" >&2
    echo "  suite:      ${EVAL_SUITE}" >&2
    chatcore_eval
    ;;
  pipeline|full|pretrain-sft-rl-eval)
    echo "[speed_run] running full pipeline:" >&2
    echo "  pretrain: ${PRETRAIN_OUT_DIR}" >&2
    echo "  sft:      ${SFT_OUT_DIR}" >&2
    echo "  rl:       ${RL_OUT_DIR}" >&2
    echo "  eval:     ${EVAL_SUITE} on ${EVAL_CHECKPOINT}" >&2
    train
    sft
    rl
    chatcore_eval
    ;;
  all)
    train_tokenizer
    download_data
    train
    ;;
  *)
    echo "usage: $0 [tokenizer|download|pretokenize|train|sft|rl|eval|pipeline|draft|all]" >&2
    exit 2
    ;;
esac
