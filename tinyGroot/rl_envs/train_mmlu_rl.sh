#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-8}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000}"
OUT_DIR="${OUT_DIR:-/runs/hrm-loop-mmlu-rl}"
MAX_STEPS="${MAX_STEPS:-500}"
EXAMPLES_PER_STEP="${EXAMPLES_PER_STEP:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-tinyGroot-mmlu-rl}"

torchrun --standalone --nproc_per_node="${GPUS}" rl_envs/mmlu_rl.py \
  --hf-checkpoint-repo-id "${CHECKPOINT_REPO}" \
  --out-dir "${OUT_DIR}" \
  --examples-per-step "${EXAMPLES_PER_STEP}" \
  --num-samples "${NUM_SAMPLES}" \
  --device-batch-size "${DEVICE_BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 1.0 \
  --top-k 50 \
  --max-steps "${MAX_STEPS}" \
  --eval-every 50 \
  --save-every 50 \
  --optimizer muon \
  --amp-dtype bfloat16 \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  "$@"
