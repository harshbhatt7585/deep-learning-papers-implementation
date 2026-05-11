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

## 2026-05-11: H100 Diffusion Remasking Run

The next experiment was to revisit the diffusion objective using the newer low-confidence remasking sampler and GQA. The goal was to see whether a more LLaDA-style sampling loop plus H100 throughput could make the diffusion path more promising.

The run was:

```bash
GPU_TYPE=H100 \
FP8=1 \
COMPILE=1 \
MAX_STEPS=600 \
CORE_METRIC_EVERY=600 \
OBJECTIVE=diffusion \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_KV_HEADS=2 \
N_LAYERS=12 \
RUN_NAME=bench-h100-8gpu-diffusion-gqa2-remask-600-regional-compile \
EXPERIMENT_DESCRIPTION="D12 H100 8GPU masked diffusion benchmark: GQA kv=2, low-confidence remasking sampler, regional torch.compile, FP8." \
EXPERIMENT_TAGS="benchmark,d12,diffusion,gqa,kv-2,low-confidence-remask,h100,8gpu,fp8,regional-compile" \
./speed_run.sh train 8gpu
```

The run was very fast after warmup:

```text
typical throughput: ~2.64M-2.72M tok/s
```

But the quality was poor. Intermediate evals:

```text
step 0200 val_loss 10.1857 masked_bpb 0.9607
step 0400 val_loss 10.0995 masked_bpb 0.9528
step 0600 val_loss 9.8851  masked_bpb 0.9333
step 0600 core -0.0068
```

The samples were not usable. They collapsed into repetitive local patterns:

```text
The capital of France is since the since of the since...
The chemical symbol of gold is ... chemical chemical chemical...
If yesterday was Friday, then tomorrow will be ,,,,,,,,,,
The opposite of hot is ... -- -- -- -- ...
My favorite color is up. up. up. up...
```

This is worse than the causal-MTP path. Even though masked BPB improved during the 600 steps and the H100 throughput was excellent, the generated text was highly repetitive and CORE was negative.

Comparison against the nearby 600-step causal-MTP GQA run:

| Run | Objective | Hardware | BPB Metric | CORE | Sample Quality |
| --- | --- | --- | ---: | ---: | --- |
| MTP2 GQA | causal-MTP | 8x A100 | `bpb 2.7568` | `0.0366` | Poor but language-like |
| Diffusion GQA remask | masked diffusion | 8x H100 | `masked_bpb 0.9333` | `-0.0068` | Repetitive collapse |

The BPB numbers are not directly comparable because the objectives are different: causal-MTP logs autoregressive BPB, while diffusion logs masked-token BPB. CORE and samples are the more useful comparison here, and both favor causal-MTP.

Conclusion: this diffusion remasking experiment was not good. The sampler is closer to LLaDA-style low-confidence remasking, but the training objective/model scale still does not produce useful continuation behavior. For now, the better path remains causal-MTP with GQA.

## 2026-05-11: H100 Hybrid Attention Causal-MTP Run

The next experiment tested a practical DeepSeek-style idea: hybrid full/local attention instead of full attention in every layer. The goal was not to reproduce DeepSeek CSA/HCA exactly, but to test a simpler local/full pattern that could improve training speed while keeping enough global layers.

Configuration:

```text
objective: causal_mtp
MTP heads: 1
MTP loss weight: 0.1
GPU: 8x H100
FP8: on
compile: on
GQA: q=6, kv=2
attention pattern: hybrid
attention window: 512
full attention every: 3 layers
parameters: 150,994,944
tokens_per_step: 524,288
total_training_tokens: 314,572,800
```

The command was:

```bash
GPU_TYPE=H100 \
FP8=1 \
COMPILE=1 \
MAX_STEPS=600 \
EVAL_INTERVAL=600 \
SAMPLE_INTERVAL=600 \
CORE_METRIC_EVERY=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_KV_HEADS=2 \
N_LAYERS=12 \
ATTENTION_WINDOW=512 \
FULL_ATTENTION_EVERY=3 \
RUN_NAME=bench-h100-8gpu-mtp1-gqa2-hybrid512-w01-600-compile \
EXPERIMENT_DESCRIPTION="D12 H100 8GPU causal-MTP benchmark: 1 MTP head, GQA kv=2, hybrid full/local attention window 512, MTP weight 0.1, FP8." \
EXPERIMENT_TAGS="benchmark,d12,mtp,heads-1,weight-0.1,gqa,kv-2,hybrid-attn,window-512,full-every-3,h100,8gpu,fp8,compile" \
./speed_run.sh train 8gpu
```

Startup confirmed the intended path:

```text
attention_heads: q=6 kv=2 head_dim=128
attention_pattern: hybrid window=512 full_every=3
attention_backend: FA3 (Hopper GPU with bf16 tensors)
fp8: True
```

Throughput was strong and stable after compile warmup:

```text
typical throughput: ~2.10M-2.14M tok/s
```

Final metrics:

```text
step 0600 train_loss 3.9426
step 0600 val_loss 8.9344
step 0600 bpb 2.8061
step 0600 core 0.0317
```

Comparison against the earlier 600-step GQA MTP2 run:

| Run | Hardware | MTP Heads | Attention | BPB | CORE | Throughput Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| MTP2 GQA | 8x A100 | 2 | full, q=6 kv=2 | 2.7568 | 0.0366 | many steps near `0.98M-1.01M tok/s` |
| MTP1 GQA hybrid512 | 8x H100 | 1 | hybrid, q=6 kv=2, window=512 | 2.8061 | 0.0317 | many steps near `2.10M-2.14M tok/s` |

This run is much faster, but quality did not improve at 600 steps. BPB and CORE are both slightly worse than the earlier MTP2 GQA A100 run, though this comparison changes multiple variables at once: hardware, MTP head count, attention pattern, and FP8.

Samples were still weak and repetitive. They were somewhat language-like, but failed simple facts and reasoning:

```text
The capital of France is ... 3D connectinguries...
The chemical symbol of gold is ... symbol ... symbol ...
If yesterday was Friday, then tomorrow will be ... childrening...
The planets of the solar system are: " " " ...
If 5*x + 3 = 13, then x is 5-6.
```

Conclusion: hybrid local/full attention is promising for speed on H100, but this specific 600-step run is not a quality win. To isolate the attention change properly, we need a matched baseline with the same `MTP_HEADS=1`, `MTP_LOSS_WEIGHT=0.1`, H100, FP8, and GQA but full attention. Until then, this result only says hybrid attention is fast and not obviously catastrophic, not that it improves learning.

## 2026-05-11: H100 Hybrid Attention + Gated MLP Run

The next experiment kept the H100 hybrid-attention setup and changed only the MLP from the original `relu2` feed-forward block to a parameter-matched gated MLP. The goal was to test whether a SwiGLU-style gated feed-forward block improves quality without increasing the model size.

Command:

```bash
GPU_TYPE=H100 \
FP8=1 \
COMPILE=1 \
COMPILE_MODE=default \
MAX_STEPS=600 \
EVAL_INTERVAL=600 \
SAMPLE_INTERVAL=600 \
CORE_METRIC_EVERY=600 \
OBJECTIVE=causal_mtp \
MTP_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
OPTIMIZER=muon \
D_MODEL=768 \
N_HEADS=6 \
N_KV_HEADS=2 \
N_LAYERS=12 \
MLP_TYPE=gated \
ATTENTION_WINDOW=512 \
FULL_ATTENTION_EVERY=3 \
RUN_NAME=bench-h100-8gpu-mtp1-gqa2-hybrid512-gated-w01-600-compile \
EXPERIMENT_DESCRIPTION="D12 H100 8GPU causal-MTP benchmark: 1 MTP head, GQA kv=2, gated MLP, hybrid attention window 512, MTP weight 0.1, FP8." \
EXPERIMENT_TAGS="benchmark,d12,mtp,heads-1,weight-0.1,gqa,kv-2,gated-mlp,hybrid512,h100,8gpu,fp8,compile" \
./speed_run.sh train 8gpu
```

Startup confirmed the intended path:

```text
parameters: 150,994,944
attention_heads: q=6 kv=2 head_dim=128
mlp_type: gated
attention_pattern: hybrid window=512 full_every=3
attention_backend: FA3 (Hopper GPU with bf16 tensors)
fp8: True
compile_mode: default
```

Throughput improved versus the previous H100 hybrid `relu2` run:

```text
previous hybrid relu2: ~2.10M-2.14M tok/s
new hybrid gated:      ~2.35M-2.40M tok/s
```

Final metrics:

```text
step 0600 train_loss 3.7734
step 0600 val_loss 8.8747
step 0600 bpb 2.7901
step 0600 core 0.0334
```

Comparison:

| Run | Hardware | MTP Heads | MLP | Attention | BPB | CORE | Throughput Notes |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| MTP1 GQA hybrid512 | 8x H100 | 1 | relu2 | hybrid, window=512 | 2.8061 | 0.0317 | `~2.10M-2.14M tok/s` |
| MTP1 GQA hybrid512 gated | 8x H100 | 1 | gated | hybrid, window=512 | 2.7901 | 0.0334 | `~2.35M-2.40M tok/s` |
| MTP2 GQA | 8x A100 | 2 | relu2 | full | 2.7568 | 0.0366 | `~0.98M-1.01M tok/s` |

The gated MLP run is an incremental improvement over the previous H100 hybrid `relu2` run: lower BPB, higher CORE, lower train loss, and better throughput. It still does not beat the earlier A100 MTP2 GQA run on 600-step CORE, but that run used two MTP heads and full attention, so it is not an isolated comparison.

Conclusion: gated MLP is worth keeping for the H100 hybrid path unless a clean full-attention H100 baseline shows otherwise.

## 2026-05-11: Sparse ReLU2 MLP Training Run

The next run switched back to `MLP_TYPE=relu2` and enabled an activation sparsity penalty. The goal was to test whether sparse ReLU2 activations improve short-run quality without damaging H100 throughput.

The important difference is:

```text
MLP_TYPE=relu2
SPARSE_L1_WEIGHT=1e-4
```

Training showed the sparse objective working. MLP sparsity rose quickly, then plateaued around `0.795-0.800`:

```text
step 00105 train_loss 5.4567 mlp_sparsity 0.690 tok/s 2,323,854
step 00150 train_loss 4.8391 mlp_sparsity 0.732 tok/s 2,326,795
step 00200 train_loss 4.6076 mlp_sparsity 0.771 tok/s 2,321,675
step 00250 train_loss 4.2817 mlp_sparsity 0.788 tok/s 2,316,178
step 00300 train_loss 4.2205 mlp_sparsity 0.797 tok/s 2,311,438
step 00343 train_loss 4.0047 mlp_sparsity 0.801 tok/s 2,308,831
step 00500 train_loss 3.6400 mlp_sparsity 0.797 tok/s 2,323,451
step 00552 train_loss 3.8979 mlp_sparsity 0.796 tok/s 1,105,460
step 00600 train_loss 3.9502 mlp_sparsity 0.795 tok/s 2,320,898
```

Final metrics:

```text
step 0600 train_loss 3.9502
step 0600 val_loss 8.9313
step 0600 bpb 2.8051
step 0600 core 0.0474
final mlp_sparsity 0.795
```

CORE highlights:

```text
hellaswag_zeroshot centered 0.0480
arc_challenge centered -0.0267
copa centered 0.2000
commonsense_qa centered 0.2275
bigbench_cs_algorithms centered 0.4380
bigbench_dyck_languages centered 0.0720
agi_eval_lsat_ar centered 0.1033
bigbench_language_identification centered 0.1881
```

The samples were still poor and repetitive. The arithmetic prompt failed, and factual prompts still produced broken repeated fragments. So this is not a usable language model yet, but the benchmark signal improved.

Comparison:

| Run | MLP | Sparse L1 | BPB | CORE | MLP Sparsity | Throughput Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A100 MTP2 GQA | relu2 | 0 | 2.7568 | 0.0366 | n/a | `~0.98M-1.01M tok/s` |
| H100 hybrid relu2 MTP1 | relu2 | 0 | 2.8061 | 0.0317 | n/a | `~2.10M-2.14M tok/s` |
| H100 hybrid gated MTP1 | gated | 0 | 2.7901 | 0.0334 | n/a | `~2.35M-2.40M tok/s` |
| H100 hybrid sparse relu2 MTP1 | relu2 | `1e-4` | 2.8051 | 0.0474 | 0.795 | `~2.30M-2.32M tok/s` |

Conclusion: sparse-L1 gave the best 600-step CORE score so far in this short-run set, even though BPB did not improve. The likely value is regularization, not speed: training still uses dense MLP matmuls, so sparsity does not reduce training FLOPs. For now, the sparse kernel path is not worth keeping in the training pipeline; continue with `relu2` and `gated` experiments, and treat sparse-L1 as an optional quality regularizer rather than a speed feature.

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
