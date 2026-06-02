#!/usr/bin/env bash
# Example 2: sample from a checkpoint produced by example 1.
# Usage: bash examples/02_sample.sh <run-dir> ["your prompt"]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT="${1:?usage: bash examples/02_sample.sh <run-dir> [prompt]}"
PROMPT="${2:-The meaning of life is}"

# Prefer the project venv / uv; fall back to python.
if [[ -x ".venv/bin/python" ]]; then RUN=(".venv/bin/python" -m tinygroot.infer.sample)
elif command -v uv >/dev/null 2>&1; then RUN=(uv run tinygroot-sample)
else RUN=(python3 -m tinygroot.infer.sample); fi

"${RUN[@]}" \
  --checkpoint-dir "${CKPT}" \
  --prompt "${PROMPT}" \
  --gen-length 64 \
  --temperature 0.8 \
  --top-k 50 \
  --num-samples 2
