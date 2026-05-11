# Text MTP Progress Log

This is a living blog for the text diffusion / text-MTP experiments. The goal is to keep a clear record of what changed, what ran, what worked, and what still needs to be tested.

## 2026-05-10: From Diffusion To Text-MTP

We started with a masked text diffusion model trained on nanochat/ClimbMix-style data. The diffusion model did learn, but convergence and sample quality were rough: loss improved, but generations were often repetitive, unstable, or failed simple factual prompts. That pushed us toward a hybrid direction inspired by DeepSeek-style multi-token prediction.

The current branch is:

```bash
hybrid-causal-mtp
```

The main new objective is:

```bash
OBJECTIVE=causal_mtp
```

This changes training from masked diffusion to causal next-token prediction with an auxiliary future-token head:

```text
main LM head: hidden[t] -> token[t+1]
MTP head 1:  hidden[t] -> token[t+2]
```

For now, MTP is only used as an auxiliary training loss. Sampling still uses the main autoregressive LM head one token at a time. We are not doing speculative decoding yet.

## Implementation Notes

The model now supports both objectives:

```text
diffusion    - masked text diffusion, old behavior
causal_mtp   - autoregressive next-token loss + MTP auxiliary heads
```

For `causal_mtp`, the model uses causal attention. A hidden state at position `t` cannot see future tokens, so the MTP head predicting `token[t+2]` is not cheating. It predicts future tokens only from prefix context.

The speedrun entrypoint now exposes:

```bash
OBJECTIVE=causal_mtp
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.3
```

We also added:

```bash
GPU_TYPE=A100
GPU_TYPE=H100
```

`GPU_TYPE=A100` now requests A100 80GB instances on Modal. FP8 is disabled for A100 because A100 does not have native FP8 Tensor Cores. H100 can use FP8.

## Current Best MTP Run

Configuration:

```bash
GPU_TYPE=A100
FP8=0
OBJECTIVE=causal_mtp
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.3
OPTIMIZER=muon
D_MODEL=768
N_HEADS=6
N_LAYERS=12
./speed_run.sh train 8gpu
```

Run path:

```text
/runs/text-diffusion-8gpu--2026-05-10--05-42pm/checkpoint.pt
```

Resume path used later:

```text
/runs/text-mtp-8gpu--h100-resume/checkpoint.pt
```

Model scale:

```text
parameters: 160,432,128
tokens_per_step: 524,288
max_steps: 2,448
total_training_tokens: 1,283,457,024
```

Throughput:

```text
~1.09M tok/s on 8x A100 80GB
```

The H100 resume command accidentally ran as A100 in the last completed run, so the logged attention backend was:

```text
SDPA (FA3 unavailable)
fp8: False
```

## Metrics Snapshot

| Step | Val Loss | BPB | CORE | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1800 | 3.0629 | 0.9612 | n/a | Samples became coherent but still weak on facts/reasoning |
| 2000 | 3.0502 | 0.9591 | 0.1001 | Best observed CORE so far |
| 2200 | 3.0258 | 0.9507 | n/a | Val improved, samples improved on gold/Au and simple color |
| 2400 | 3.0089 | 0.9440 | 0.0929 | Val improved, CORE dropped slightly |

Compared to the nanochat D12 reference we discussed:

```text
nanochat D12 CORE: ~0.1059
text-MTP D12 CORE: best observed 0.1001
```

So the text-MTP run is close, but not yet better than nanochat D12.

## Sample Quality

At step 1800, samples were grammatical but weak:

```text
The chemical symbol of gold is silver.
If yesterday was Friday, then tomorrow will be Monday.
The opposite of hot is hot.
If 5*x + 3 = 13, then x is 13.
```

At step 2000, some prompts improved:

```text
The chemical symbol of gold is Au.
```

But factual and reasoning errors remained:

```text
The capital of France is the city of Léon.
If yesterday was Friday, then tomorrow will be Tuesday.
If 5*x + 3 = 13, then x is the base value.
```

At step 2200, generations became more natural, but still unreliable:

```text
The chemical symbol of gold is Au, the symbol for gold.
My favorite color is blue.
If 5*x + 3 = 13, then x is 12/12.
```

At step 2400, language quality was stronger than the original diffusion samples, but the model still hallucinated:

```text
The capital of France is Gorgon...
If yesterday was Friday, then tomorrow will be Friday.
If 5*x + 3 = 13, then x is 14.
```

## What We Learned

Text-MTP is much more language-like than the masked diffusion run. It produces fluent paragraphs and can answer some simple factual prompts.

The current MTP run does not beat nanochat D12 yet. CORE peaked around `0.1001`, below the nanochat D12 reference of about `0.1059`.

The learning rate schedule reached almost zero by step 2448, so the completed run is effectively done. Continuing from that checkpoint without changing schedule will not improve much.

The model has useful language modeling behavior but weak reasoning. This is expected at this scale and token budget, especially with only pretraining and no SFT.

`<|bos|>` appears at both start and end because the nanochat tokenizer uses it as a document boundary. This is okay for training, but samples can look odd.

## Open Issues

The W&B sample table currently warns:

```text
You are mutating a Table with log_mode='IMMUTABLE'
```

This does not affect training, but the logging table should be changed to mutable or recreated per eval.

The H100 resume command was interrupted and the completed resume shown here used A100 settings. We still need a clean H100 run with:

```bash
GPU_TYPE=H100
FP8=1
```

For causal-MTP with FP8/H100, we should verify whether attention stays on FA3. Earlier smoke tests showed q/k could become float32 in some paths, forcing SDPA. That is a speed issue, not a correctness issue.

## 2026-05-11: Fixed 400-Step A100 Repro Check

We reproduced the simple D12 causal-MTP setup and stopped at a fixed 400-step checkpoint so it can be used as a cheap experiment gate before running the full token-proportional ratio-8 schedule.

Configuration:

```bash
GPU_TYPE=A100
FP8=0
MAX_STEPS=400
BATCH_SIZE=32
SEQ_LEN=2048
OBJECTIVE=causal_mtp
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.3
OPTIMIZER=muon
AURORA_WEIGHT_DECAY=0.025
D_MODEL=768
N_HEADS=6
N_LAYERS=12
COMPILE=1
TRAIN_SHARDS=170
MAX_VAL_CHARS=2000000
```

Result at step 400:

```text
train_loss: 5.0072
val_loss: 3.5953
masked_bpb: 1.1289
CORE: 0.0693
lr: 2.85e-04
throughput: ~2.06M tok/s on 8x A100 80GB
```

CORE breakdown highlights:

```text
arc_easy: 0.1840 centered
piqa: 0.1760 centered
lambada_openai: 0.1480
winograd: 0.1282 centered
winogrande: 0.1480 centered
bigbench_cs_algorithms: 0.4180
```

Samples at 400 steps are already much more language-like than the earlier broken sparse/hybrid runs:

```text
The chemical symbol of gold is the symbol for gold.
The opposite of hot is that the cold is about 10 degrees.
```

But the model is still clearly undertrained and repetitive:

```text
The capital of France is the capital of France...
If 5*x + 3 = 13, then x is a number of x...
```

This is useful as a screening run. A configuration that is bad at 400 steps probably should not get a full ratio-8 run. A configuration that reaches around this level or better can be promoted to the full schedule.

## Next Experiments

1. Use the fixed 400-step checkpoint as the first comparison gate:

```bash
MAX_STEPS=400
```

Then promote only promising variants to:

```bash
TARGET_PARAM_DATA_RATIO=8
MAX_STEPS=-1
```

2. Run pure causal LM baseline:

```bash
OBJECTIVE=causal_mtp
MTP_HEADS=0
```

This tells us whether the MTP auxiliary head is helping or hurting.

3. Run MTP with lower auxiliary weight:

```bash
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.1
```

Current `0.3` might be too strong and may hurt next-token quality.

4. Increase token budget:

```bash
TARGET_PARAM_DATA_RATIO=12
```

The current run uses about 8 tokens per parameter, matching the nanochat rule. MTP may need a different schedule.

5. Run clean H100 FP8 test:

```bash
GPU_TYPE=H100
FP8=1
OBJECTIVE=causal_mtp
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.3
OPTIMIZER=muon
D_MODEL=768
N_HEADS=6
N_LAYERS=12
RUN_NAME=text-mtp-8gpu--h100-fp8
./speed_run.sh train 8gpu
```

6. Try Aurora with corrected weight decay:

```bash
OPTIMIZER=aurora
AURORA_WEIGHT_DECAY=0.025
```

Aurora is now wired to use `0.025` matrix weight decay, matching the Aurora release default, instead of inheriting Muon's `0.1`.

## Current Position

The current result is promising but not a win yet:

```text
text-MTP D12 is close to nanochat D12 on CORE, but slightly behind.
text-MTP samples are much better than masked diffusion samples.
The next useful comparison is pure causal LM vs causal-MTP under the same token budget.
```
