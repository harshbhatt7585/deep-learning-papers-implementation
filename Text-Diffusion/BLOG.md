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

## 2026-05-11: H100 FP8 MTP2 400-Step Check

Next we tested whether adding a second MTP head helps under the same 400-step gate, this time on H100 with FP8 enabled.

Configuration:

```bash
GPU_TYPE=H100
FP8=1
MAX_STEPS=400
EVAL_INTERVAL=400
CORE_METRIC_EVERY=400
SAMPLE_INTERVAL=400
BATCH_SIZE=32
SEQ_LEN=2048
OBJECTIVE=causal_mtp
MTP_HEADS=2
MTP_LOSS_WEIGHT=0.15
OPTIMIZER=muon
AURORA_WEIGHT_DECAY=0.025
D_MODEL=768
N_HEADS=6
N_LAYERS=12
COMPILE=1
TRAIN_SHARDS=170
MAX_VAL_CHARS=2000000
RUN_NAME=bench-h100-fp8-8gpu-d12-mtp2-w015-relu2-fullattn-400
```

Result at step 400:

```text
train_loss: 4.3863
val_loss: 3.5491
masked_bpb: 1.1157
CORE: 0.0710
lr: 6.04e-09
throughput: ~1.50M tok/s on 8x H100 FP8
checkpoint: /runs/bench-h100-fp8-8gpu-d12-mtp2-w015-relu2-fullattn-400/checkpoint.pt
wandb: https://wandb.ai/harshbhatt7585/text-diffusion/runs/x8z09q7i
```

Compared to the A100 MTP1 400-step gate:

| Run | MTP Heads | MTP Weight | Hardware | FP8 | Val Loss | BPB | CORE | Tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A100 MTP1 | 1 | 0.30 | 8x A100 | 0 | 3.5953 | 1.1289 | 0.0693 | ~2.06M |
| H100 MTP2 | 2 | 0.15 | 8x H100 | 1 | 3.5491 | 1.1157 | 0.0710 | ~1.50M |

MTP2 slightly improved the 400-step gate: lower validation loss, lower BPB, and CORE moved from `0.0693` to `0.0710`. This is not a large enough margin to call it a clear win yet because hardware and FP8 changed at the same time, but it is good enough to justify a clean controlled comparison.

The samples are still fluent but undertrained and repetitive:

```text
The capital of France is the capital of the United States.
The chemical symbol of gold is a symbol of purity...
If 5*x + 3 = 13, then x is 1.5...
```

Takeaway: MTP2 is worth keeping in the experiment queue, but the next clean test should isolate variables:

```text
A100 MTP1 w0.30 vs A100 MTP2 w0.15
or
H100 FP8 MTP1 w0.30 vs H100 FP8 MTP2 w0.15
```

## 2026-05-12: MoE MLP 400-Step Check

We tested whether the observed ReLU2 activation sparsity could be converted into useful expert sparsity by replacing the dense MLP with a top-1 MoE MLP.

Configuration:

```bash
GPU_TYPE=H100
FP8=1
MAX_STEPS=400
EVAL_INTERVAL=400
CORE_METRIC_EVERY=400
SAMPLE_INTERVAL=400
BATCH_SIZE=32
SEQ_LEN=2048
OBJECTIVE=causal_mtp
MTP_HEADS=2
MTP_LOSS_WEIGHT=0.15
OPTIMIZER=muon
D_MODEL=768
N_HEADS=6
N_LAYERS=12
MLP_TYPE=moe
FF_MULT=1
MOE_NUM_EXPERTS=4
MOE_TOP_K=1
MOE_AUX_LOSS_WEIGHT=0.01
COMPILE=1
```

The MoE path required several fixes before it would run on H100 FP8:

```text
dynamic expert routing caused torch.compile cache churn
FP8 expert matmuls required routed token counts divisible by 16
top-1 routing left some expert weights unused, producing None gradients for Muon
```

A 1-GPU H100 FP8 compile smoke eventually completed after guarding the dynamic routing path, padding expert batches, and ensuring all expert parameters participated in the graph. The full 8-GPU 400-step run then completed.

Result at step 400:

```text
train_loss: 4.6509
val_loss: 3.6385
masked_bpb: 1.1439
CORE: 0.0688
throughput: ~1.14M tok/s on 8x H100 FP8
checkpoint: /runs/bench-h100-fp8-8gpu-d12-mtp2-w015-moe4top1-ff1-400/checkpoint.pt
wandb: https://wandb.ai/harshbhatt7585/text-diffusion/runs/cmau2fvv
```

Compared to the dense ReLU2 H100 FP8 MTP2 run:

| Run | MLP | Val Loss | BPB | CORE | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense ReLU2 | dense ff=4 | 3.5491 | 1.1157 | 0.0710 | ~1.50M |
| MoE top-1 | 4 experts, ff=1 | 3.6385 | 1.1439 | 0.0688 | ~1.14M |

This MoE variant did not improve the model. It was worse on validation loss, BPB, CORE, throughput, and sample repetition. Keeping total parameters similar forced each active expert to be much narrower than the dense `ff=4` MLP, so the run mostly added routing complexity without adding useful active compute.

Decision: remove the MoE implementation and keep dense ReLU2 as the current baseline. If MoE is revisited, it should be done with a proper static-capacity/fused expert kernel and a more meaningful capacity tradeoff.

## 2026-05-12: GQA3 400-Step Check

We also tested grouped-query attention to see whether reducing KV heads could keep quality while improving throughput. The setup was the same H100 FP8 MTP2 400-step gate, except attention used 6 query heads and 3 KV heads.

Configuration difference:

```bash
N_HEADS=6
N_KV_HEADS=3
RUN_NAME=bench-h100-fp8-8gpu-d12-mtp2-w015-gqa3-relu2-400
```

Result at step 400:

```text
train_loss: 4.4088
val_loss: 3.5655
masked_bpb: 1.1188
CORE: 0.0681
throughput: ~1.50M tok/s on 8x H100 FP8
checkpoint: /runs/bench-h100-fp8-8gpu-d12-mtp2-w015-gqa3-relu2-400/checkpoint.pt
wandb: https://wandb.ai/harshbhatt7585/text-diffusion/runs/ul7o5nlv
```

Compared to the full multi-head attention baseline:

| Run | KV Heads | Val Loss | BPB | CORE | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full MHA | 6 | 3.5491 | 1.1157 | 0.0710 | ~1.50M |
| GQA3 | 3 | 3.5655 | 1.1188 | 0.0681 | ~1.50M |

GQA3 did not help this small model. It slightly worsened validation loss, BPB, and CORE, while throughput stayed basically unchanged. The samples also remained repetitive and factually weak:

```text
The capital of France is composed of six major cities...
The opposite of hot is a hot one.
If 5*x + 3 = 13, then x is the other x - x = 13...
```

Decision: keep full attention for the current D12 baseline. GQA is useful for KV-cache savings during long-context inference, but this experiment is training a small model at short context, so the quality tradeoff is not worth it here.

## 2026-05-12: Tied Embeddings 400-Step Check

We tested tying the token embedding matrix and LM head to reduce the D12 parameter count. With a 32k tokenizer and `d_model=768`, this saves one full vocabulary matrix:

```text
saved parameters: 32,768 * 768 = 25,165,824
```

Configuration difference:

```bash
tie_word_embeddings: True
RUN_NAME=bench-h100-fp8-d12-tied-mtp2-w015-fullattn-400
```

Result at step 400:

```text
train_loss: 4.3332
val_loss: 3.5672
masked_bpb: 1.1210
CORE: 0.0615
throughput: ~1.52M tok/s on 8x H100 FP8
checkpoint: /runs/bench-h100-fp8-d12-tied-mtp2-w015-fullattn-400/checkpoint.pt
wandb: https://wandb.ai/harshbhatt7585/text-diffusion/runs/v2lguits
```

Compared to the untied H100 FP8 MTP2 baseline:

| Run | Params | Val Loss | BPB | CORE | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untied MTP2 | ~185.6M | 3.5491 | 1.1157 | 0.0710 | ~1.50M |
| Tied MTP2 | ~160.4M | 3.5672 | 1.1210 | 0.0615 | ~1.52M |

Tying embeddings made the model meaningfully smaller and slightly faster, but the 400-step quality gate got worse. The validation loss and BPB regressed slightly, while CORE dropped more noticeably from `0.0710` to `0.0615`. Samples also stayed repetitive:

```text
The opposite of hot is the water vapor...
My favorite color is blue... I want to be a little girl...
If 5*x + 3 = 13, then x is 0.001*x...
```

Decision: tied embeddings are useful for size reduction, but not for the current score baseline. Keep untied embeddings when optimizing for CORE:

```bash
NO_TIE_WORD_EMBEDDINGS=1
```

If we revisit tying, the next fair test should tune the shared embedding initialization and learning rate instead of assuming the untied embedding/head hyperparameters transfer cleanly.

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
