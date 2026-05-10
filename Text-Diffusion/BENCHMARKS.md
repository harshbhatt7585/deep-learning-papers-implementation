# Experiment Benchmark Framework

Training now writes a self-contained experiment folder under every `OUT_DIR`:

```text
<OUT_DIR>/experiment/
├── experiment.yaml      # description, tags, config, args
├── metrics.jsonl        # train/eval/core/checkpoint metrics
├── samples.jsonl        # sample text over time
├── plots/               # SVG metric plots
└── report.md            # generated summary and scoreboard
```

## 5 Minute Benchmark

At the current 8GPU speed, 5 minutes is roughly:

```text
~1.09M tok/s * 300s = ~327M tokens
~327M / 524,288 tokens_per_step = ~624 steps
```

Use `MAX_STEPS=600` for consistent comparisons. This is enough to compare throughput, early loss slope, early eval loss/BPB, and sample quality. It is not enough to make a final CORE-quality claim.

## Runs To Compare

### A. Pure Causal LM

This tests whether MTP helps at all.

```bash
GPU_TYPE=A100 \
FP8=0 \
MAX_STEPS=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=0 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_LAYERS=12 \
RUN_NAME=bench-causal-lm-600 \
EXPERIMENT_DESCRIPTION="D12 pure causal LM baseline for 600-step benchmark." \
EXPERIMENT_TAGS="benchmark,d12,causal,no-mtp" \
./speed_run.sh train 8gpu
```

### B. MTP Weight 0.1

This tests a weaker auxiliary future-token loss.

```bash
GPU_TYPE=A100 \
FP8=0 \
MAX_STEPS=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_LAYERS=12 \
RUN_NAME=bench-mtp1-w01-600 \
EXPERIMENT_DESCRIPTION="D12 causal-MTP benchmark with one future-token head and weight 0.1." \
EXPERIMENT_TAGS="benchmark,d12,mtp,weight-0.1" \
./speed_run.sh train 8gpu
```

### C. MTP Weight 0.3

This is the current MTP setting.

```bash
GPU_TYPE=A100 \
FP8=0 \
MAX_STEPS=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=1 \
MTP_LOSS_WEIGHT=0.3 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_LAYERS=12 \
RUN_NAME=bench-mtp1-w03-600 \
EXPERIMENT_DESCRIPTION="D12 causal-MTP benchmark with one future-token head and weight 0.3." \
EXPERIMENT_TAGS="benchmark,d12,mtp,weight-0.3" \
./speed_run.sh train 8gpu
```

## Compare Completed Runs

For local runs:

```bash
python experiment.py compare \
  runs/bench-causal-lm-600 \
  runs/bench-mtp1-w01-600 \
  runs/bench-mtp1-w03-600
```

For Modal runs, the reports are inside the Modal run volume:

```text
/runs/<RUN_NAME>/experiment/report.md
```

The comparison utility works once those run directories are available locally or mounted.

## Regenerate A Report

```bash
python experiment.py report runs/bench-mtp1-w01-600
```

## What To Judge

For the 600-step benchmark, rank runs by:

```text
1. best eval/loss at same step
2. best eval/bpb at same step
3. train loss slope
4. samples at step 200/400/600
5. throughput
```

CORE is useful as a weak signal, but it is noisy this early. Use full training to confirm the winner.
