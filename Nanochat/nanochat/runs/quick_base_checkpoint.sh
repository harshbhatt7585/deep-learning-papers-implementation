#!/usr/bin/env bash
# Train a very short d12 base run so chat_sft (and friends) have a checkpoint under
# $NANOCHAT_BASE_DIR/base_checkpoints/d12/ — useful instead of downloading from the Hub.
#
# Not meant to produce a good model; only enough steps to exercise the pipeline.
#
# Usage (from repo root Nanochat/nanochat, venv active):
#   bash runs/quick_base_checkpoint.sh
#
# Optional env:
#   NANOCHAT_BASE_DIR   default: ~/.cache/nanochat
#   QUICK_BASE_STEPS    optimization steps (default: 50)
#   QUICK_DATA_SHARDS   climbmix train shards to download if data missing (default: 2)
#   QUICK_TOK_CHARS     tokenizer training budget in chars if tokenizer missing (default: 100000000)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

QUICK_BASE_STEPS="${QUICK_BASE_STEPS:-50}"
QUICK_DATA_SHARDS="${QUICK_DATA_SHARDS:-2}"
QUICK_TOK_CHARS="${QUICK_TOK_CHARS:-100000000}"

TOK_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [[ ! -f "$TOK_FILE" ]]; then
  echo "[quick_base] No tokenizer at $TOK_FILE — downloading $QUICK_DATA_SHARDS data shards and training tokenizer..."
  python -m nanochat.dataset -n "$QUICK_DATA_SHARDS"
  python -m scripts.tok_train --max-chars="$QUICK_TOK_CHARS"
else
  echo "[quick_base] Using existing tokenizer at $TOK_FILE"
fi

echo "[quick_base] Training d12 for $QUICK_BASE_STEPS steps -> base_checkpoints/d12/ ..."
python -m scripts.base_train \
  --depth=12 \
  --model-tag=d12 \
  --window-pattern=L \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=4096 \
  --num-iterations="$QUICK_BASE_STEPS" \
  --warmup-steps=5 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --eval-tokens=65536 \
  --save-every=-1 \
  --run=dummy

echo "[quick_base] Done. Next: python -m scripts.chat_sft"
