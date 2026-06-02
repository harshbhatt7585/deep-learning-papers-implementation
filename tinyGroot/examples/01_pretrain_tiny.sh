#!/usr/bin/env bash
# Example 1: tokenize a small shard and pretrain a tiny model locally (MPS/CPU).
# Laptop-sized; finishes in a few minutes. Override any knob via the environment,
# e.g. MAX_STEPS=200 SEQ_LEN=512 bash examples/01_pretrain_tiny.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# speedrun_mps.sh owns the laptop-sized defaults (small data shard, batch 8,
# seq 256, ~100 steps, MTP=1). It tokenizes on first run and caches under data/.
MAX_STEPS="${MAX_STEPS:-100}" bash speedrun_mps.sh train

echo
echo "Done. The run directory (with checkpoint.pt) was printed above as 'OUT_DIR'."
echo "Next:  bash examples/02_sample.sh <that-run-dir>"
