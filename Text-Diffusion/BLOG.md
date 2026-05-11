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

## 2026-05-11: GQA MTP2 600-Step Benchmark

The next experiment was to test whether grouped-query attention could make the D12 causal-MTP run cheaper without hurting early validation quality.

The model already had separate query and KV head support internally. We exposed it through:

```bash
N_KV_HEADS=2
```

That changes attention from regular multi-head attention:

```text
q heads: 6
kv heads: 6
```

to grouped-query attention:

```text
q heads: 6
kv heads: 2
```

The run command was:

```bash
GPU_TYPE=A100 \
FP8=0 \
COMPILE=1 \
MAX_STEPS=600 \
CORE_METRIC_EVERY=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=2 \
MTP_LOSS_WEIGHT=0.1 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_KV_HEADS=2 \
N_LAYERS=12 \
RUN_NAME=bench-a100-8gpu-mtp2-gqa2-w01-600-compile \
EXPERIMENT_DESCRIPTION="D12 A100 8GPU causal-MTP quick benchmark: 2 MTP heads, GQA with 2 KV heads, MTP weight 0.1, compile enabled." \
EXPERIMENT_TAGS="benchmark,d12,mtp,heads-2,gqa,kv-2,weight-0.1,a100,8gpu,compile" \
./speed_run.sh train 8gpu
```

The previous comparable 600-step run was the same D12 causal-MTP setup with `MTP_HEADS=2`, `MTP_LOSS_WEIGHT=0.1`, 8x A100, and compile enabled, but without GQA. That run used regular attention with `N_HEADS=6` and implicit `N_KV_HEADS=6`.

Results:

| Run | Attention | MTP Heads | Val Loss | BPB | CORE | Throughput Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MTP2 non-GQA | q=6, kv=6 | 2 | 8.8281 | 2.7712 | 0.0280 | Often around `300K-500K tok/s`, noisy |
| MTP2 GQA | q=6, kv=2 | 2 | 8.7863 | 2.7568 | 0.0366 | Many post-warmup steps around `0.98M-1.01M tok/s`, still noisy |

The GQA run ended with:

```text
step 00600 train_loss 3.8527
step 00600 val_loss 8.7863
step 00600 bpb 2.7568
step 00600 core 0.0366
```

This is an incremental improvement over the non-GQA 600-step MTP2 run:

```text
BPB:  2.7712 -> 2.7568
CORE: 0.0280 -> 0.0366
```

The speed signal is also encouraging. After compile warmup, the GQA run frequently reported close to `1M tok/s`. The per-step logger is still noisy, so the exact throughput should not be over-interpreted, but GQA appears materially faster than the non-GQA run.

Sample quality is still poor at 600 steps. The model remains repetitive and fails simple factual or reasoning prompts like France, gold, and arithmetic. That matches the low CORE score. So this is not a final quality win; it is a useful early signal that `N_KV_HEADS=2` may improve speed and does not obviously hurt early BPB/CORE.

Compared to the longer MTP run from May 10:

```text
longer MTP D12 best CORE: 0.1001
nanochat D12 reference CORE: ~0.1059
new 600-step GQA MTP2 CORE: 0.0366
```

The new GQA run is not close to the longer-run quality yet. Its value is that it gives us a faster configuration to test in longer runs.

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

## Next Experiments

1. Run pure causal LM baseline:

```bash
OBJECTIVE=causal_mtp
MTP_HEADS=0
```

This tells us whether the MTP auxiliary head is helping or hurting.

2. Run MTP with lower auxiliary weight:

```bash
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.1
```

Current `0.3` might be too strong and may hurt next-token quality.

3. Increase token budget:

```bash
TARGET_PARAM_DATA_RATIO=12
```

The current run uses about 8 tokens per parameter, matching the nanochat rule. MTP may need a different schedule.

4. Run clean H100 FP8 test:

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

5. Try Aurora with corrected weight decay:

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
