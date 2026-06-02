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

MAX_STEPS="${MAX_STEPS:-100}"
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

# Some PyTorch operations do not have an MPS kernel yet. Let PyTorch use CPU
# for those isolated operations while the model itself remains on MPS.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

usage() {
  cat >&2 <<EOF
usage: $0 [doctor|download|tokenizer|pretokenize|train|eval|all]

Local Apple Silicon speedrun defaults:
  model:      d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} mtp_heads=${MTP_HEADS}
  training:   max_steps=${MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN} grad_accum=${GRAD_ACCUM_STEPS}
  output:     ${OUT_DIR}

Examples:
  $0 all
  MAX_STEPS=10 $0 train
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
    --out-dir "${OUT_DIR}"
    --nanochat-tokenizer-cache-dir "${TOKENIZER_CACHE_DIR}"
    --max-steps "${MAX_STEPS}"
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
  echo "  out_dir: ${OUT_DIR}" >&2
  echo "  model:   d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} mtp_heads=${MTP_HEADS}" >&2
  echo "  budget:  max_steps=${MAX_STEPS} batch=${BATCH_SIZE} seq_len=${SEQ_LEN} grad_accum=${GRAD_ACCUM_STEPS}" >&2
  run "${command[@]}" "${data_flags[@]}"
}

evaluate() {
  local checkpoint="${CHECKPOINT:-${OUT_DIR}}"
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
  eval)
    check_mps
    evaluate
    ;;
  all)
    check_mps
    pretokenize
    train
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
