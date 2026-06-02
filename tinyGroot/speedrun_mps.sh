#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

MODE="${1:-train}"
DEFAULT_RUN_TIME="$(date +"%Y-%m-%d--%I-%M%p" | tr '[:upper:]' '[:lower:]')"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=("uv" "run" "python")
else
  PYTHON_CMD=("python3")
fi

# Laptop-sized defaults. Override any of these from the environment for a
# larger-memory MacBook Pro or a longer run.
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
MAX_TRAIN_CHARS="${MAX_TRAIN_CHARS:-5000000}"
MAX_VAL_CHARS="${MAX_VAL_CHARS:-200000}"
TOKENIZER_TRAIN_SHARDS="${TOKENIZER_TRAIN_SHARDS:-1}"
TOKENIZER_TRAIN_CHARS="${TOKENIZER_TRAIN_CHARS:-5000000}"
TOKENIZER_THREADS="${TOKENIZER_THREADS:-4}"
DOC_BATCH_SIZE="${DOC_BATCH_SIZE:-128}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"

NANOCHAT_CACHE_DIR="${NANOCHAT_CACHE_DIR:-data/nanochat_climbmix}"
TOKENIZER_CACHE_DIR="${TOKENIZER_CACHE_DIR:-data/nanochat_tokenizer_32k}"
TOKEN_SHARDS_DIR="${TOKEN_SHARDS_DIR:-data/nanochat_tokens_mps_32k}"
DATA="${DATA:-}"

RUN_NAME="${RUN_NAME:-tinygroot-mps--${DEFAULT_RUN_TIME}}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
RESUME="${RESUME:-}"

PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-${MAX_STEPS:-100}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
MTP_HEADS="${MTP_HEADS:-1}"
MTP_ARCH="${MTP_ARCH:-linear}"
MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-0.3}"
TST_BAG_SIZE="${TST_BAG_SIZE:-1}"
TST_RATIO="${TST_RATIO:-0}"
AURORA_WEIGHT_DECAY="${AURORA_WEIGHT_DECAY:-0.025}"

D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-4}"
FF_MULT="${FF_MULT:-4}"
GATED_MLP="${GATED_MLP:-0}"

EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-50}"
CORE_EVAL_MAX_PER_TASK="${CORE_EVAL_MAX_PER_TASK:--1}"
STREAM_NANOCHAT="${STREAM_NANOCHAT:-0}"
OVERWRITE_TOKENS="${OVERWRITE_TOKENS:-0}"
COMPILE="${COMPILE:-0}"
WANDB="${WANDB:-0}"
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-0}"
DRY_RUN="${DRY_RUN:-0}"

PRETRAIN_OUT_DIR="${PRETRAIN_OUT_DIR:-${OUT_DIR}}"
SFT_OUT_DIR="${SFT_OUT_DIR:-${OUT_DIR}-sft}"
RL_OUT_DIR="${RL_OUT_DIR:-${OUT_DIR}-rl}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${PRETRAIN_OUT_DIR}}"
RL_CHECKPOINT="${RL_CHECKPOINT:-${SFT_OUT_DIR}}"

SFT_MAX_STEPS="${SFT_MAX_STEPS:-100}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-2}"
SFT_TOTAL_BATCH_SIZE="${SFT_TOTAL_BATCH_SIZE:-2048}"
SFT_SEQ_LEN="${SFT_SEQ_LEN:-${SEQ_LEN}}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-50}"
SFT_EVAL_TOKENS="${SFT_EVAL_TOKENS:-4096}"
SFT_CHATCORE_EVERY="${SFT_CHATCORE_EVERY:-0}"
SFT_SAMPLE_EVERY="${SFT_SAMPLE_EVERY:-50}"
SFT_SAMPLE_LENGTH="${SFT_SAMPLE_LENGTH:-64}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-50}"
SFT_OPTIMIZER="${SFT_OPTIMIZER:-adamw}"
SFT_MMLU_EPOCHS="${SFT_MMLU_EPOCHS:-1}"
SFT_GSM8K_EPOCHS="${SFT_GSM8K_EPOCHS:-1}"
SFT_SIMPLE_SPELLING_SIZE="${SFT_SIMPLE_SPELLING_SIZE:-2000}"
SFT_SPELLINGBEE_SIZE="${SFT_SPELLINGBEE_SIZE:-1000}"
SFT_NO_SMOLTALK="${SFT_NO_SMOLTALK:-0}"
IDENTITY_JSONL="${IDENTITY_JSONL:-data/identity_conversations.jsonl}"
WORDS_PATH="${WORDS_PATH:-data/words_alpha.txt}"

RL_MAX_STEPS="${RL_MAX_STEPS:-100}"
RL_DEVICE_BATCH_SIZE="${RL_DEVICE_BATCH_SIZE:-1}"
RL_EXAMPLES_PER_STEP="${RL_EXAMPLES_PER_STEP:-1}"
RL_NUM_SAMPLES="${RL_NUM_SAMPLES:-2}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-64}"
RL_TEMPERATURE="${RL_TEMPERATURE:-1.0}"
RL_TOP_K="${RL_TOP_K:-50}"
RL_EVAL_EVERY="${RL_EVAL_EVERY:-0}"
RL_EVAL_EXAMPLES="${RL_EVAL_EXAMPLES:-16}"
RL_SAVE_EVERY="${RL_SAVE_EVERY:-50}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adamw}"
RL_LOG_ROLLOUTS_EVERY="${RL_LOG_ROLLOUTS_EVERY:-10}"

# Some PyTorch operations do not have an MPS kernel yet. Let PyTorch use CPU
# for those isolated operations while the model itself remains on MPS.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

usage() {
  cat >&2 <<EOF
usage: $0 [doctor|download|tokenizer|pretokenize|train|pretrain|sft|rl|pipeline|eval|all]

Local Apple Silicon speedrun defaults:
  model:      d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} mtp_heads=${MTP_HEADS}
  pretrain:   steps=${PRETRAIN_MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN} grad_accum=${GRAD_ACCUM_STEPS}
  sft:        steps=${SFT_MAX_STEPS} batch=${SFT_DEVICE_BATCH_SIZE} total_batch=${SFT_TOTAL_BATCH_SIZE}
  rl:         steps=${RL_MAX_STEPS} batch=${RL_DEVICE_BATCH_SIZE} examples=${RL_EXAMPLES_PER_STEP} samples=${RL_NUM_SAMPLES}
  output:     ${PRETRAIN_OUT_DIR}, ${SFT_OUT_DIR}, ${RL_OUT_DIR}

Examples:
  $0 all
  PRETRAIN_MAX_STEPS=10 SFT_MAX_STEPS=10 RL_MAX_STEPS=10 $0 pipeline
  SFT_CHECKPOINT=runs/<pretrain-run> $0 sft
  RL_CHECKPOINT=runs/<sft-run> $0 rl
  CHECKPOINT=runs/<run-name> $0 eval
  DATA=path/to/local.txt $0 pretokenize
EOF
}

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[speedrun_mps] dry run:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

check_mps() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  if "${PYTHON_CMD[@]}" - <<'PY'
import platform
import sys

import torch

available = torch.backends.mps.is_available()
print(
    "[speedrun_mps] "
    f"python={platform.python_version()} "
    f"torch={torch.__version__} "
    f"mps_available={available}",
    flush=True,
)
sys.exit(0 if available else 1)
PY
  then
    return
  fi

  if [[ "${ALLOW_CPU_FALLBACK}" == "1" ]]; then
    echo "[speedrun_mps] WARNING: MPS is unavailable; continuing on CPU because ALLOW_CPU_FALLBACK=1" >&2
    return
  fi

  echo "[speedrun_mps] ERROR: PyTorch MPS is unavailable." >&2
  echo "               Run this on an Apple Silicon Mac with an MPS-enabled PyTorch build." >&2
  echo "               Set ALLOW_CPU_FALLBACK=1 only for a slow CPU smoke test." >&2
  exit 1
}

pretokenize_flags=(
  --nanochat-cache-dir "${NANOCHAT_CACHE_DIR}"
  --nanochat-tokenizer-cache-dir "${TOKENIZER_CACHE_DIR}"
  --token-shards-dir "${TOKEN_SHARDS_DIR}"
  --train-shards "${TRAIN_SHARDS}"
  --max-train-chars "${MAX_TRAIN_CHARS}"
  --max-val-chars "${MAX_VAL_CHARS}"
  --tokenizer-threads "${TOKENIZER_THREADS}"
  --doc-batch-size "${DOC_BATCH_SIZE}"
  --tokenizer-train-shards "${TOKENIZER_TRAIN_SHARDS}"
  --tokenizer-train-chars "${TOKENIZER_TRAIN_CHARS}"
  --tokenizer-vocab-size "${TOKENIZER_VOCAB_SIZE}"
)

if [[ -n "${DATA}" ]]; then
  pretokenize_flags+=(--data "${DATA}")
fi
if [[ "${OVERWRITE_TOKENS}" == "1" ]]; then
  pretokenize_flags+=(--overwrite)
fi

download_data() {
  run "${PYTHON_CMD[@]}" -m tinygroot.training.pretokenize \
    --download-only \
    "${pretokenize_flags[@]}"
}

train_tokenizer() {
  run "${PYTHON_CMD[@]}" -m tinygroot.training.pretokenize \
    --tokenizer-only \
    "${pretokenize_flags[@]}"
}

pretokenize() {
  run "${PYTHON_CMD[@]}" -m tinygroot.training.pretokenize \
    "${pretokenize_flags[@]}"
}

train() {
  local data_flags=()
  if [[ "${STREAM_NANOCHAT}" == "1" ]]; then
    data_flags+=(
      --stream-nanochat
      --nanochat-cache-dir "${NANOCHAT_CACHE_DIR}"
      --nanochat-train-shards "${TRAIN_SHARDS}"
      --max-val-chars "${MAX_VAL_CHARS}"
    )
  else
    data_flags+=(--token-shards-dir "${TOKEN_SHARDS_DIR}")
  fi

  local command=(
    "${PYTHON_CMD[@]}" -m tinygroot.training.train
    --out-dir "${PRETRAIN_OUT_DIR}"
    --nanochat-tokenizer-cache-dir "${TOKENIZER_CACHE_DIR}"
    --max-steps "${PRETRAIN_MAX_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --seq-len "${SEQ_LEN}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --optimizer "${OPTIMIZER}"
    --mtp-heads "${MTP_HEADS}"
    --mtp-arch "${MTP_ARCH}"
    --mtp-loss-weight "${MTP_LOSS_WEIGHT}"
    --tst-bag-size "${TST_BAG_SIZE}"
    --tst-ratio "${TST_RATIO}"
    --aurora-weight-decay "${AURORA_WEIGHT_DECAY}"
    --d-model "${D_MODEL}"
    --n-heads "${N_HEADS}"
    --n-layers "${N_LAYERS}"
    --ff-mult "${FF_MULT}"
    --amp-dtype float32
    --eval-interval "${EVAL_INTERVAL}"
    --core-metric-every "${CORE_METRIC_EVERY}"
    --sample-interval "${SAMPLE_INTERVAL}"
  )

  if [[ "${GATED_MLP}" == "1" ]]; then
    command+=(--gated-mlp)
  fi
  if [[ -n "${RESUME}" ]]; then
    command+=(--resume "${RESUME}")
  fi
  if [[ "${COMPILE}" == "1" ]]; then
    command+=(--compile)
  fi
  if [[ "${WANDB}" == "1" ]]; then
    command+=(--wandb)
  fi

  echo "[speedrun_mps] starting local training:" >&2
  echo "  out_dir: ${PRETRAIN_OUT_DIR}" >&2
  echo "  model:   d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} mtp_heads=${MTP_HEADS}" >&2
  echo "  budget:  max_steps=${PRETRAIN_MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN} grad_accum=${GRAD_ACCUM_STEPS}" >&2
  run "${command[@]}" "${data_flags[@]}"
}

sft() {
  local command=(
    "${PYTHON_CMD[@]}" -m tinygroot.training.chat_sft
    --checkpoint "${SFT_CHECKPOINT}"
    --out-dir "${SFT_OUT_DIR}"
    --run-name "${RUN_NAME}-sft"
    --max-steps "${SFT_MAX_STEPS}"
    --device-batch-size "${SFT_DEVICE_BATCH_SIZE}"
    --total-batch-size "${SFT_TOTAL_BATCH_SIZE}"
    --seq-len "${SFT_SEQ_LEN}"
    --eval-every "${SFT_EVAL_EVERY}"
    --eval-tokens "${SFT_EVAL_TOKENS}"
    --chatcore-every "${SFT_CHATCORE_EVERY}"
    --sample-every "${SFT_SAMPLE_EVERY}"
    --sample-length "${SFT_SAMPLE_LENGTH}"
    --save-every "${SFT_SAVE_EVERY}"
    --optimizer "${SFT_OPTIMIZER}"
    --mmlu-epochs "${SFT_MMLU_EPOCHS}"
    --gsm8k-epochs "${SFT_GSM8K_EPOCHS}"
    --simple-spelling-size "${SFT_SIMPLE_SPELLING_SIZE}"
    --spellingbee-size "${SFT_SPELLINGBEE_SIZE}"
    --identity-jsonl "${IDENTITY_JSONL}"
    --words-path "${WORDS_PATH}"
    --amp-dtype float32
  )

  if [[ "${SFT_NO_SMOLTALK}" == "1" ]]; then
    command+=(--no-smoltalk)
  fi
  if [[ "${COMPILE}" == "1" ]]; then
    command+=(--compile)
  fi
  if [[ "${WANDB}" == "1" ]]; then
    command+=(--wandb)
  fi

  echo "[speedrun_mps] starting local SFT:" >&2
  echo "  checkpoint: ${SFT_CHECKPOINT}" >&2
  echo "  out_dir:    ${SFT_OUT_DIR}" >&2
  echo "  budget:     max_steps=${SFT_MAX_STEPS} batch=${SFT_DEVICE_BATCH_SIZE} total_batch=${SFT_TOTAL_BATCH_SIZE}" >&2
  run "${command[@]}"
}

rl() {
  local command=(
    "${PYTHON_CMD[@]}" -m tinygroot.training.chat_rl
    --checkpoint "${RL_CHECKPOINT}"
    --out-dir "${RL_OUT_DIR}"
    --run-name "${RUN_NAME}-rl"
    --max-steps "${RL_MAX_STEPS}"
    --device-batch-size "${RL_DEVICE_BATCH_SIZE}"
    --examples-per-step "${RL_EXAMPLES_PER_STEP}"
    --num-samples "${RL_NUM_SAMPLES}"
    --max-new-tokens "${RL_MAX_NEW_TOKENS}"
    --temperature "${RL_TEMPERATURE}"
    --top-k "${RL_TOP_K}"
    --eval-every "${RL_EVAL_EVERY}"
    --eval-examples "${RL_EVAL_EXAMPLES}"
    --save-every "${RL_SAVE_EVERY}"
    --optimizer "${RL_OPTIMIZER}"
    --log-rollouts-every "${RL_LOG_ROLLOUTS_EVERY}"
    --words-path "${WORDS_PATH}"
    --amp-dtype float32
  )

  if [[ "${COMPILE}" == "1" ]]; then
    command+=(--compile)
  fi
  if [[ "${WANDB}" == "1" ]]; then
    command+=(--wandb)
  fi

  echo "[speedrun_mps] starting local GSM8K RL:" >&2
  echo "  checkpoint: ${RL_CHECKPOINT}" >&2
  echo "  out_dir:    ${RL_OUT_DIR}" >&2
  echo "  budget:     max_steps=${RL_MAX_STEPS} batch=${RL_DEVICE_BATCH_SIZE} examples=${RL_EXAMPLES_PER_STEP} samples=${RL_NUM_SAMPLES}" >&2
  run "${command[@]}"
}

evaluate() {
  local checkpoint="${CHECKPOINT:-${PRETRAIN_OUT_DIR}}"
  run "${PYTHON_CMD[@]}" -m tinygroot.eval_core \
    --checkpoint-dir "${checkpoint}" \
    --max-per-task "${CORE_EVAL_MAX_PER_TASK}"
}

case "${MODE}" in
  doctor)
    check_mps
    ;;
  download|download-data)
    download_data
    ;;
  tokenizer|tok)
    train_tokenizer
    ;;
  pretokenize|prep|data)
    pretokenize
    ;;
  train)
    check_mps
    train
    ;;
  pretrain|pre-training)
    check_mps
    pretokenize
    train
    ;;
  sft|chat-sft|chat_sft)
    check_mps
    sft
    ;;
  rl|chat-rl|chat_rl)
    check_mps
    rl
    ;;
  eval)
    check_mps
    evaluate
    ;;
  pipeline|full|all)
    check_mps
    pretokenize
    train
    sft
    rl
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
