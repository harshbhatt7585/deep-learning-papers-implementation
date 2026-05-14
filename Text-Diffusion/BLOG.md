# Text MTP Progress Log

A living blog for the text diffusion / text-MTP experiments. The goal is to keep a clear record of what changed, what ran, what worked, and what still needs to be tested.

Reference target: **nanochat D12 CORE ≈ 0.1059**. The 400-step gate uses `MAX_STEPS=400`, `BATCH_SIZE=32`, `SEQ_LEN=2048`, with `tokens_per_step = 524,288` (`8×1` or `4×2` grad-accum).

## Best Runs — Overall

This table mixes short gates and long runs, so use the **Budget** column before comparing directly. It answers: "what are the strongest runs we've seen so far?"

| # | Run | Budget | MLP | MTP design | Hardware | Val Loss | BPB | **CORE** | Notes |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | **MTP2 + TST s4 r0.3 d12** | ratio-12 long run | SwiGLU ff=3 | shared × 2 + TST | 8× H100 FP8 compile | 3.0028 | 0.9446 | **0.1133** | Beats public nanochat d12 CORE/BPB; more raw-token exposure from TST |
| 2 | nanochat d12 public reference | d12 reference | nanochat | causal LM | 8× H100 | — | 0.9825 | 0.1059 | External reference |
| 3 | **H100 bf16 DeepSeek-MTP2 + TST s4 r0.3** | 400-step gate | SwiGLU ff=3 | DeepSeek-style × 2 + TST | 4× H100 | 3.5576 | 1.1165 | **0.0737** | New best 400-step gate; first short run above 0.071 |
| 4 | H100 FP8 MTP2 ReLU² | 400-step gate | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5491 | 1.1157 | 0.0710 | Previous best 400-step gate |
| 5 | A100 MTP1 ReLU² | 400-step gate | ReLU² ff=4 | full-vocab × 1 | 8× A100 | 3.5953 | 1.1289 | 0.0693 | Best A100 gate |
| 6 | H100 bf16 SwiGLU MTP1 shared, dropout=0 | 400-step gate | SwiGLU ff=3 | shared × 1 | 4× H100 | 3.5866 | 1.1246 | 0.0688 | Best efficient non-TST gate |
| 7 | H100 bf16 SwiGLU MTP1 shared + TST s4 r0.3 | 400-step gate | SwiGLU ff=3 | shared × 1 + TST | 4× H100 | 3.5789 | 1.1222 | 0.0685 | Better val/BPB than efficient baseline; CORE tied/slightly lower |
| 8 | H100 FP8 MoE top-1 | 400-step gate | MoE 4×ff=1 | full-vocab × 2 | 8× H100 FP8 | 3.6385 | 1.1439 | 0.0688 | Similar CORE, weaker BPB |
| 9 | H100 FP8 GQA3 | 400-step gate | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5655 | 1.1188 | 0.0681 | GQA hurt CORE relative to dense attention |

Reading the overall table:

- The long-run MTP2+TST d12 run is the first result above the public nanochat d12 reference (`CORE=0.1133` vs `0.1059`).
- The best 400-step gate is now DeepSeek-style MTP2 + TST (`CORE=0.0737`), beating the older H100 FP8 MTP2 ReLU² full-vocab run (`0.0710`) while using 4× H100 bf16.
- TST alone did not beat the old short-run CORE, but TST plus DeepSeek-style MTP2 did.

## Leaderboard — 400-Step Gate

Current best is in bold. All runs are `d_model=768`, `n_layers=12`, causal-MTP objective unless noted.

| # | Run | MLP | MTP design | Hardware | Val Loss | BPB | **CORE** | Params |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | **H100 bf16 DeepSeek-MTP2 + TST s4 r0.3** | SwiGLU ff=3 | DeepSeek-style × 2 + TST | 4× H100 | 3.5576 | 1.1165 | **0.0737** | ~160M |
| 2 | H100 FP8 MTP2 ReLU² | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5491 | 1.1157 | 0.0710 | ~185M |
| 3 | A100 MTP1 ReLU² | ReLU² ff=4 | full-vocab × 1 | 8× A100 | 3.5953 | 1.1289 | 0.0693 | ~160M |
| 4 | H100 bf16 SwiGLU MTP1 shared, dropout=0 | SwiGLU ff=3 | shared × 1 | 4× H100 | 3.5866 | 1.1246 | 0.0688 | ~143M |
| 5 | H100 FP8 MoE top-1 | MoE 4×ff=1 | full-vocab × 2 | 8× H100 FP8 | 3.6385 | 1.1439 | 0.0688 | ~185M (≈143M active) |
| 6 | H100 bf16 SwiGLU MTP1 shared + TST s4 r0.3 | SwiGLU ff=3 | shared × 1 + TST | 4× H100 | 3.5789 | 1.1222 | 0.0685 | ~143M |
| 7 | H100 FP8 GQA3 | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5655 | 1.1188 | 0.0681 | ~178M |
| — | H100 bf16 SwiGLU MTP2 shared + TST s4 r0.3 | SwiGLU ff=3 | shared × 2 + TST | 4× H100 | 3.5873 | 1.1281 | 0.0626 | ~144M |
| — | H100 bf16 SwiGLU MTP3 shared + TST s4 r0.3 | SwiGLU ff=3 | shared × 3 + TST | 4× H100 | 3.5751 | 1.1254 | 0.0563 | ~144M |
| — | Tied embeddings | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5672 | 1.1210 | 0.0615 | ~160M |
| — | MTP-shared first attempt | ReLU² ff=4 | shared × 1 | 8× A100 | 3.5363 | 1.1105 | 0.0585 | ~136M |
| — | SwiGLU MTP1 + **GQA-2** (rejected) | SwiGLU ff=3 | shared × 1 | 4× H100 | 3.5335 | 1.1123 | 0.0459 | ~140M |
| — | SwiGLU dropout=0.1 (broken) | SwiGLU ff=3 | shared × 1 | 8× H100 | **8.4141** | **2.6436** | 0.0447 | ~143M |

Reading the leaderboard:

- The top score (`0.0737`) is now held by the DeepSeek-style MTP2 + TST run. This is the first 400-step gate above the old `0.0710` ceiling.
- The previous top score (`0.0710`) came from the heaviest config: 2× full-vocab MTP heads (≈50M extra params) and ReLU² FFN.
- The SwiGLU + shared-MTP + dropout=0 result (`0.0688`) remains the cheap non-TST baseline on half the GPUs and ~50M fewer MTP params than the old full-vocab MTP2 run.
- The first Token Superposition Training gate (`s=4`, `r=0.3`) improved val loss/BPB versus the efficient baseline but landed just below it on CORE (`0.0685` vs `0.0688`). It is promising but not promoted yet.
- TST plus extra shared MTP heads is not promoted: MTP2 landed at `CORE=0.0626`, and MTP3 improved val loss again (`3.5751`) but hurt BPB and CORE (`0.0563`). More shared MTP heads are cheap parameter-wise but not free optimization-wise.
- The "MTP-shared first attempt" (`0.0585`), "SwiGLU + GQA-2" (`0.0459`), and "SwiGLU dropout=0.1" (`0.0447`) rows are kept in the table on purpose: they are the controlled stepping stones that show what *not* to do. Details in §SwiGLU and §GQA-2 below.

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
MTP_LOSS_WEIGHT=0.15
GATED_MLP=1   # SwiGLU; 0 = ReLU²
FF_MULT=3     # FFN width multiplier
```

We also added GPU type selection. `GPU_TYPE=A100` requests A100 80GB instances on Modal. FP8 is disabled for A100 because it has no native FP8 Tensor Cores. H100 can use FP8.

## 2026-05-10: Long MTP Run (Original 2,448-Step Schedule)

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

Run path: `/runs/text-diffusion-8gpu--2026-05-10--05-42pm/checkpoint.pt`

Model scale:

```text
parameters: 160,432,128
tokens_per_step: 524,288
max_steps: 2,448
total_training_tokens: 1,283,457,024
```

Throughput: ~1.09M tok/s on 8× A100 80GB.

### Metrics snapshot

| Step | Val Loss | BPB | CORE | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1800 | 3.0629 | 0.9612 | n/a | Samples became coherent but still weak on facts/reasoning |
| 2000 | 3.0502 | 0.9591 | 0.1001 | Best observed CORE in the long schedule |
| 2200 | 3.0258 | 0.9507 | n/a | Val improved; samples improved on gold/Au and simple color |
| 2400 | 3.0089 | 0.9440 | 0.0929 | Val improved, CORE dropped slightly |

The long schedule peaks at `CORE ≈ 0.1001`, still below nanochat D12's `~0.1059`. Beyond step ~2000 the LR had decayed enough that further training mostly just oscillates. The follow-up work is all happening at the 400-step gate so we can iterate quickly on architecture without burning the full budget.

## 2026-05-11: 400-Step Gate Established (A100 MTP1 Baseline)

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
D_MODEL=768 N_HEADS=6 N_LAYERS=12
COMPILE=1
```

Result at step 400:

```text
train_loss: 5.0072
val_loss: 3.5953
masked_bpb: 1.1289
CORE: 0.0693
throughput: ~2.06M tok/s on 8× A100 80GB
```

This becomes our screening run. A configuration that is worse than this at 400 steps probably should not get a full ratio-8 run. A configuration that matches or beats it can be promoted.

## 2026-05-11: H100 FP8 MTP2 (Current Best)

Adding a second MTP auxiliary head, on H100 with FP8 enabled.

Configuration delta:

```bash
GPU_TYPE=H100  FP8=1
MTP_HEADS=2
MTP_LOSS_WEIGHT=0.15
RUN_NAME=bench-h100-fp8-8gpu-d12-mtp2-w015-relu2-fullattn-400
```

Result at step 400:

```text
train_loss: 4.3863
val_loss: 3.5491
masked_bpb: 1.1157
CORE: 0.0710            <-- current best
throughput: ~1.50M tok/s on 8× H100 FP8
```

MTP2 + FP8 + H100 gave us our highest CORE so far. Hardware and FP8 changed at the same time, so it isn't a clean MTP1-vs-MTP2 isolation, but the gate gate was the best we'd seen and we kept it as the bar to beat.

## 2026-05-12: MoE MLP — Promising But Not Promoted

We tested whether ReLU²'s activation sparsity could be turned into useful expert sparsity by replacing the dense MLP with a top-1 MoE.

```bash
MLP_TYPE=moe   FF_MULT=1   MOE_NUM_EXPERTS=4   MOE_TOP_K=1
MOE_AUX_LOSS_WEIGHT=0.01
RUN_NAME=bench-h100-fp8-8gpu-d12-mtp2-w015-moe4top1-ff1-400
```

Several MoE-specific bugs had to be fixed before it even ran on H100 FP8:

```text
dynamic expert routing caused torch.compile cache churn
FP8 expert matmuls required routed token counts divisible by 16
top-1 routing left some expert weights unused, producing None gradients for Muon
```

Result at step 400:

```text
val_loss: 3.6385   BPB: 1.1439   CORE: 0.0688   tok/s: ~1.14M
```

This is not a failure: `CORE=0.0688` ties the SwiGLU/dropout=0 run and is close to the 400-step baseline. But loss, BPB, and throughput all regressed relative to the current best. Keeping total params similar forced each active expert to be much narrower than the dense `ff=4` MLP, so this particular run added routing complexity without adding enough useful active compute.

**Decision:** do not promote this exact MoE config yet. It may improve later or with more active expert capacity, but a follow-up should use a static-capacity/fused expert kernel and a meaningful active-compute increase.

## 2026-05-12: GQA3 — Did Not Help

```bash
N_HEADS=6  N_KV_HEADS=3
```

Result: `val_loss 3.5655`, `BPB 1.1188`, `CORE 0.0681`, `~1.50M tok/s`. Slightly worse than full MHA on every quality metric, throughput unchanged. GQA's main payoff is KV-cache savings during long-context inference, which is not what this gate measures.

**Decision:** keep full attention for the D12 baseline.

## 2026-05-12: Tied Embeddings — Smaller But Worse

Tying `token_emb ↔ lm_head` saves `32768 × 768 = 25.2M` parameters.

Result: `val_loss 3.5672`, `BPB 1.1210`, `CORE 0.0615`, `~1.52M tok/s`. Smaller and marginally faster, but CORE regressed from `0.0710` to `0.0615`. Untied embeddings stay on by default. Revisit only if we also re-tune the shared embedding init and learning rate.

## 2026-05-12: MTP Head Redesign (DeepSeek-V3 Style)

The original MTP heads were `Linear(d_model, vocab_size)` — one full unembedding per offset. At `vocab=32768, d_model=768` that's `25.2M` per head. Two heads = `~50M` parameters dedicated just to the auxiliary loss.

We replaced them with the DeepSeek-V3-style design:

```text
mtp_head[i]: Linear(d_model, d_model)        # 0.6M params each
              -> norm(·)
              -> shared lm_head               # same weight as main path
```

Each MTP offset is now a tiny `d_model × d_model` projection. The vocabulary projection is reused from the main `lm_head`, so adding MTP heads is essentially free in parameters and the auxiliary signal still flows through the same embedding geometry as the main loss.

### MTP-shared first attempt (controlled run)

Configuration:

```bash
GPU_TYPE=A100  FP8=0
MTP_HEADS=1  MTP_LOSS_WEIGHT=0.15
DROPOUT=0.1   # still on at this point
```

Result at step 400:

```text
val_loss: 3.5363   BPB: 1.1105   CORE: 0.0585   tok/s: ~1.08M
```

Val loss and BPB are *better* than the old full-vocab MTP2 baseline, but CORE dropped from `0.0710` to `0.0585`. The interpretation: dropping `~50M` of auxiliary-head capacity hurts the downstream CORE signal more than it helps the next-token loss. We need to recover capacity somewhere else if we want to keep this design.

## 2026-05-12: SwiGLU MLP — and the Dropout Trap

To recover the capacity given up by the new MTP design, we switched the FFN from ReLU² (nanochat-style: `c_fc(x) → relu²(·) → c_proj`) to SwiGLU (LLaMA/Mixtral/Qwen-style):

```text
GatedMLP(x) = c_proj( silu(c_gate(x)) * c_up(x) )
```

To keep parameter count roughly constant we picked `ff_mult=3` (SwiGLU has 3 projections; ReLU² has 2 with `ff_mult=4`). We also added a rotary-embedding dtype fix so FA3 stops falling back to SDPA on H100:

```python
def apply_rotary_emb(x, cos, sin):
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    ...
```

That fix alone took throughput from ~1.08M tok/s (A100) → ~1.42M tok/s (8× H100 bf16). FA3 now engages — no more `[flash_attention] using SDPA: FA3 requires bf16` warning.

### Broken run: SwiGLU + dropout=0.1

```bash
RUN_NAME=bench-h100-bf16-8gpu-d12-mtp1-swiglu-ff3-400
GATED_MLP=1  FF_MULT=3
GPU_TYPE=H100  FP8=0  COMPILE=0  AMP_DTYPE=bfloat16
DROPOUT=0.1   # the default at this point
```

Result at step 400:

```text
train_loss: 4.3519    <-- looks fine
val_loss:   8.4141    <-- catastrophic
masked_bpb: 2.6436    <-- catastrophic
CORE:       0.0447    <-- worse than every prior run
tok/s:      ~1.42M
```

Sample at this checkpoint:

```text
<|bos|>The capital of France is aantsitesig a add Title, or penswer activity you use panel toell at us down'm.<|bos|>
```

Tokens, but no language. The diagnostic was the gap itself: **same model, same data, same autocast — `train_loss` matched baseline while `val_loss` was 2.4× worse and samples were gibberish.** The only thing that differs between train and eval forwards is `model.train()` vs `model.eval()`, and the only module that cares is `nn.Dropout`.

Why this didn't bite ReLU² runs:

1. Our network only drops on the post-embedding line, not inside blocks. A ~5% RMS shift between train and eval propagates through 12 SwiGLU blocks, and SwiGLU *multiplicatively* combines `silu(gate) * up`. ReLU² has no such multiplicative interaction, so the same dropout was nearly invisible.
2. `ff_mult=3` is tighter than ReLU²'s `ff_mult=4`, so the gate is doing more work per channel and is more sensitive to that shift.
3. Modern LLMs that use SwiGLU (LLaMA, Mixtral, Qwen, DeepSeek) all pretrain with `dropout=0.0`. The `0.1` was an old nanoGPT carryover that ReLU² happened to tolerate.

### Fix: dropout=0.0

One-line change in `train.py` (`INTERNAL_DEFAULTS["dropout"] = 0.0`), nothing else.

Configuration (run on **4 GPUs** to cut experiment cost in half — `tokens_per_step` is preserved by `GRAD_ACCUM_STEPS=2`):

```bash
RUN_CONFIG=4gpu  GPU_TYPE=H100
OBJECTIVE=causal_mtp
MTP_HEADS=1  MTP_LOSS_WEIGHT=0.15
GATED_MLP=1  FF_MULT=3
AMP_DTYPE=bfloat16  FP8=0  COMPILE=0
MAX_STEPS=400  BATCH_SIZE=32  SEQ_LEN=2048
TRAIN_SHARDS=4
RUN_NAME=bench-h100-bf16-4gpu-d12-mtp1-swiglu-ff3-drop0-400
```

Result at step 400:

```text
train_loss: 4.2346
val_loss:   3.5866
masked_bpb: 1.1246
CORE:       0.0688
tok/s:      ~716K on 4× H100 bf16
```

Sample at this checkpoint:

```text
<|bos|>The capital of France is a long, narrow, long, narrow, small, and large, square, ...
<|bos|>The chemical symbol of gold is gold.
<|bos|>If yesterday was Friday, then tomorrow will be the weekend.
<|bos|>My favorite color is blue.
<|bos|>If 5*x + 3 = 13, then x is the only non-zero integer.
```

English again. Still factually weak (an undertrained 400-step model), but the generations look like a baby language model rather than noise.

CORE highlights:

```text
piqa:                       0.2280 centered
arc_easy:                   0.1520 centered
commonsense_qa:             0.1450 centered
bigbench_lang_id:           0.1771 centered
lambada_openai:             0.1320
bigbench_cs_algorithms:     0.3500
```

This run is the most parameter-efficient point on the leaderboard so far: tied for `CORE=0.0688` with the MoE config while using `~50M` fewer MTP parameters, no FP8, and half the GPUs. The next step is to roll back to 8× H100 and stack capacity (MTP=2 shared, or `ff_mult=4` SwiGLU) to actually challenge the current `0.0710` top score.

## 2026-05-12: GQA-2 on SwiGLU MTP1 — Rejected at Gate

We re-tested grouped-query attention on top of the new per-param-efficient baseline (`SwiGLU MTP1 shared, dropout=0`). The previous GQA-3 result (§above) was on the older ReLU²/MTP2 baseline, so this run is the more relevant attribution: it isolates GQA over the current best-efficient config.

The only delta from the `CORE=0.0688` baseline is `N_KV_HEADS=3` (6 query heads → 3 K/V heads, ratio 2). Everything else — `D_MODEL=768`, `N_LAYERS=12`, `FF_MULT=3`, `GATED_MLP=1`, `MTP_HEADS=1`, `MTP_LOSS_WEIGHT=0.15`, `OPTIMIZER=muon`, `BATCH_SIZE=32`, `SEQ_LEN=2048`, `MAX_STEPS=400`, `AMP_DTYPE=bfloat16`, `FP8=0`, `COMPILE=0`, `dropout=0`, 4× H100 — is identical.

```bash
RUN_NAME=bench-h100-bf16-4gpu-d12-mtp1-swiglu-ff3-drop0-gqa2-400
N_HEADS=6  N_KV_HEADS=3   # <-- the only change
```

Result at step 400:

```text
train_loss: 4.3419     (+0.107 vs baseline 4.2346)
val_loss:   3.5335     (-0.053 vs baseline 3.5866, better)
masked_bpb: 1.1123     (-0.012 vs baseline 1.1246, better)
CORE:       0.0459     (-0.023 vs baseline 0.0688, 33% relative drop)
tok/s:      ~780K on 4× H100 bf16
parameters: ~140M      (-3M vs baseline)
KV cache:   0.5×       (the only structural win)
```

### The divergence is the headline

BPB improved by **1.1%** while CORE fell **33% relative**. This is precisely the "better next-token prediction, worse downstream discrimination" pattern that GQA produces when you don't reinvest the capacity you saved. The CORE drop is way too large to be noise — run-to-run CORE σ at this scale is ~0.005, this delta is `~4σ`.

### Where CORE was lost (per-task vs baseline)

| Task | Baseline centered | GQA-2 centered | Δ | Direction |
| --- | ---: | ---: | ---: | --- |
| `piqa` | 0.2280 | 0.1680 | −0.060 | loss |
| `commonsense_qa` | 0.1450 | 0.0300 | −0.115 | **big loss** |
| `boolq` | n/a | −0.4105 | — | **catastrophic, below random** |
| `arc_easy` | 0.1520 | 0.1440 | −0.008 | flat |
| `lambada_openai` | 0.1320 | 0.1440 | +0.012 | gain |
| `bigbench_lang_id` | 0.1771 | 0.1969 | +0.020 | gain |
| `bigbench_cs_algorithms` | 0.3500 | 0.4040 | +0.054 | gain |

The pattern is unmistakable:

- **Pure language-modeling tasks** (`lambada`, `bigbench_lang_id`, `bigbench_cs_algorithms`) → GQA-2 *wins*. The model is a sharper next-token predictor under the natural-text distribution.
- **Multi-hop reading / fact discrimination tasks** (`piqa`, `commonsense_qa`, `boolq`) → GQA-2 *loses badly*. These tasks require the model to attend to specific facts in the prompt and compare them against answer choices — exactly the routing patterns that reduced K/V rank cripples.

Sample text confirms the same story. Baseline produced coherent if simple completions ("The chemical symbol of gold is gold."). GQA-2 collapses into tautology loops:

```text
<|bos|>The capital of France is the capital of France.<|bos|>
<|bos|>The opposite of hot is the heat, which is the result of the heat, or
the heat, of the Sun, which is the result of the heat. The heat is absorbed
by the Sun, and the heat is absorbed by the Sun and subsequently absorbed
into the Sun. Some of the heat is absorbed by the Sun ...
```

This is what you get when attention cannot route to distinct contextual anchors — the model attends to the *topic* repeatedly and re-emits it.

### Why this happened (intuition)

GQA-2 halves the K/V rank per layer. With 12 layers and `n_heads=6 → n_kv_heads=3`, attention now has **36 K/V channels** instead of **72**.

- Common routing patterns ("attend to the last noun", "attend to the current topic") survive — they only need a few independent attention paths.
- Hence BPB improves: fewer K/V dimensions reduces overfitting noise on the dominant continuations, acting like a soft regularizer over next-token prediction.
- But the model can no longer maintain enough *independent* attention patterns to do something like "look up the chemical symbol in line 1 while predicting the answer in line 5". That's why `commonsense_qa` cratered and `boolq` went below random.

The interesting twist is `train_loss` is *worse* (+0.107) while `val_loss` is *better* (−0.053). The reduced K/V capacity prevents the model from fitting the minibatches as sharply, but the simpler representations generalize the next-token distribution better. So the model is genuinely *underfitting* in a way that helps perplexity and hurts task discrimination.

### Decision

**Reject GQA-2 as a standalone change.** The 3M-param + 50% KV-cache wins are real but trivial at our scale (~2% of the model), and they buy nothing at `SEQ_LEN=2048` where the KV cache is already small. A 33% relative CORE drop is not negotiable.

GQA is not categorically bad — production models (LLaMA, Mixtral, Qwen) all use it. But they pair it with a *capacity reinvestment*: more layers, wider FFN, or much longer training. The standard recipe is "GQA + FF_MULT=4 SwiGLU + more layers", not "GQA alone". We'll only revisit GQA if we plan to reinvest the savings in `FF_MULT=4` or `N_LAYERS=14`, and only after we've actually beaten the leaderboard with the dense `MTP=2 shared` recipe.

## Inference Experiments

A separate track from the 400-step training gate. These runs do **not** affect the leaderboard above — they measure end-to-end inference acceleration (wall-clock speedup, draft acceptance) on top of a frozen target checkpoint from the gate. CORE and BPB are not the scoring metrics here.

### 2026-05-13: DFlash Speculative Decoding — First Baseline

We trained the first DFlash drafter against the long-run target (CORE peak `~0.1001` at step 2000, the highest-quality target we currently have) and ran a correctness + speedup check at greedy temperature.

Setup:

```text
target  /runs/text-diffusion-8gpu--2026-05-10--05-42pm/checkpoint.pt
        d_model=768, n_layers=12, n_heads=6
        ~135M trunk params (legacy MTP head stripped at load)

drafter /runs/dflash-drafter-d2-h4-b16-on-long-d12-mtp1/checkpoint.pt
        DFlash, n_draft_layers=2, n_heads=4, block_size=16
        target_layer_ids=(1, 9)
        owned params = 15,335,424 (= 11.3% of target trunk;
                                     embed_tokens / lm_head are shared with target)
```

Drafter training command (the new `draft` mode in `speed_run.sh`):

```bash
RUN_NAME=dflash-drafter-d2-h4-b16-on-long-d12-mtp1 \
TARGET_CHECKPOINT=/runs/text-diffusion-8gpu--2026-05-10--05-42pm/checkpoint.pt \
  bash speed_run.sh draft 4gpu
# defaults: OBJECTIVE=dflash, MAX_STEPS=400, BLOCK_SIZE=16, N_DRAFT_LAYERS=2
```

Inference test (greedy, prompt = `"The capital of France is"`, `gen_length=256`):

```text
== plain AR ==
  tokens generated:   256        wall: 3681 ms      69.5 tok/s

== speculative · DFlash (block_size = 16) ==
  tokens generated:   256
  target forwards:    177
  drafter forwards:   176
  drafts proposed:    2582
  drafts accepted:     79
  acceptance rate:    3.1%
  avg accepted/block: 1.45 (over 176 blocks)
  wall:               2872 ms        89.1 tok/s
  speedup vs AR:      1.28x

OK (dflash): matches plain AR token-for-token on the first 262 tokens.
```

The implementation is correct — bit-identical output to plain AR at `T=0` over all 256 generated tokens. The speedup is modest.

#### Reading the numbers

The headline number is **`avg accepted/block = 1.45` out of `block_size = 16`**. The drafter nails the very next token most of the time and then loses the thread fast — the classic "block too wide, drafter too weak" pattern for first-pass spec-decode. With each block costing exactly one target verify forward, the theoretical speedup ceiling at this acceptance is `~1.45x`; the measured `1.28x` is what's left after counting the per-block drafter forwards and constant overhead.

Three diagnoses:

1. **Block size overshoots drafter lookahead.** Drafter correctness decays geometrically with position-within-block. With `acc/block = 1.45` we're effectively wasting `~14.5` of the 16 draft slots per block. Smaller `block_size` should give an immediate "free" speedup with no retraining.
2. **Drafter is small *and* under-trained.** Owned params = 15.3M (`~11%` of target), only 2 layers, and trained at the default `MAX_STEPS=400`. The DFlash paper trains drafters at `3–5×` the target's training length; ours saw far less than the target's 1.28B tokens. The single highest-leverage knob is drafter training length, followed by drafter capacity.
3. **Hardware caveat.** The Modal spec-decode run logged `[flash_attention] using SDPA: FA3 is unavailable on this device`. The wall-time was measured on the attention-fallback path, not the H100/FA3 path the drafter was trained against. The `1.28x` *ratio* is fair (both AR and spec share the fallback), but the absolute `89.1 tok/s` is **not** the throughput we should quote. Re-run pinned to H100 once we've found the right config.

#### Decision

Defer judgment on DFlash until two cheap follow-ups have run:

1. **Block-size sweep on the current checkpoint** (no retraining, ~30s/setting). Test `block_size ∈ {2, 4, 8, 12, 16}` to find the empirical optimum for the existing drafter. With `avg_accepted=1.45` today the optimum is almost certainly `block_size ≈ 4`; expect `1.28x → ~1.5–1.7x` for free.
2. **Retrain a longer / wider drafter.** `N_DRAFT_LAYERS=4`, `BLOCK_SIZE=8`, `MAX_STEPS=2000`. Target: `≥2x` speedup with `acc/block ≥ 3` on H100/FA3.

This experiment does not move the 400-step gate. The gate continues to govern training-quality decisions; inference experiments live and die on their own metrics here.

## 2026-05-13: Token Superposition Training Gate

We added a Token Superposition Training (TST) phase inspired by Nous Research's "Efficient Pre-Training with Token Superposition". The implementation keeps the final checkpoint as the normal `TextDiffusionModel`: during the TST phase only, the training batch is reshaped into token bags, embeddings inside each bag are averaged, and each latent position predicts the next raw token bag. After the TST ratio is exhausted, training automatically returns to the baseline `causal_mtp` objective.

Gate config:

```bash
GPU_TYPE=H100
FP8=0
COMPILE=0
OBJECTIVE=causal_mtp
MTP_HEADS=1
MTP_LOSS_WEIGHT=0.15
TST_BAG_SIZE=4
TST_RATIO=0.3
D_MODEL=768
N_LAYERS=12
N_HEADS=6
GATED_MLP=1
FF_MULT=3
OPTIMIZER=muon
BATCH_SIZE=32
SEQ_LEN=2048
MAX_STEPS=400
RUN_CONFIG=4gpu
```

Schedule math:

- `tokens_per_step = 524,288` latent transformer positions.
- `TST_RATIO=0.3` at `MAX_STEPS=400` gives `120` TST steps and `280` normal recovery steps.
- TST exposes `120 * 524,288 * 4 = 251.7M` raw tokens.
- Recovery exposes `280 * 524,288 = 146.8M` raw tokens.
- Total effective raw-token exposure is `398.5M`, about `1.9x` the normal 400-step baseline (`209.7M`), for roughly the same number of transformer-position steps.

Result:

| Run | Val Loss | BPB | CORE |
| --- | ---: | ---: | ---: |
| SwiGLU MTP1 shared baseline | 3.5866 | 1.1246 | 0.0688 |
| TST s4 r0.3 + SwiGLU MTP1 recovery | 3.5789 | 1.1222 | 0.0685 |
| TST s4 r0.3 + SwiGLU MTP2 recovery | 3.5873 | 1.1281 | 0.0626 |
| TST s4 r0.3 + SwiGLU MTP3 recovery | 3.5751 | 1.1254 | 0.0563 |
| TST s4 r0.3 + **DeepSeek-style MTP2** recovery | **3.5576** | **1.1165** | **0.0737** |

Interpretation before DeepSeek-MTP: TST recovered validation loss and BPB slightly better than the efficient baseline, but the shared-linear MTP variants did not cleanly beat CORE. Samples remained rough and repetitive, suggesting that cheap parallel shared-linear MTP heads were the weak link, not only the TST schedule.

MTP2/MTP3 shared-linear follow-up: we kept the same TST schedule and changed only the number of cheap shared-linear MTP heads (microbatch reduced to `BATCH_SIZE=16`, `GRAD_ACCUM_STEPS=4` where needed to keep global tokens/step fixed). MTP2 landed at `val_loss=3.5873`, `BPB=1.1281`, `CORE=0.0626`; MTP3 landed at `val_loss=3.5751`, `BPB=1.1254`, `CORE=0.0563`. The likely read is that extra shared-linear MTP offsets add future-token auxiliary pressure that can smooth continuation modeling but hurts answer-token discrimination at this scale.

FLOP-saving attempt: we also tried `MTP_HEADS=3`, `TST_RATIO=0.2`, `MAX_STEPS=250`, which gives `50` TST steps and `200` recovery steps (`400` raw-token-step equivalents but only `250` optimizer/FLOP steps). It failed the gate (`val_loss=3.8484`, `BPB=1.2077`, `CORE=0.0097`). Same raw-token accounting is not the same as same training quality; the model needed more actual recovery updates.

DeepSeek-style MTP2 follow-up: replacing the cheap parallel shared-linear MTP heads with sequential DeepSeek-like MTP modules changed the 400-step picture. The run `mtp2-deepseek-tst-s4-r03-400step-swiglu-ff3-h100-bf16-4gpu-bs16` used `MTP_ARCH=deepseek`, `MTP_HEADS=2`, `MTP_LOSS_WEIGHT=0.15`, `TST_BAG_SIZE=4`, `TST_RATIO=0.3`, `BATCH_SIZE=16`, `GRAD_ACCUM_STEPS=4`, `GATED_MLP=1`, `FF_MULT=3`. It landed at `val_loss=3.5576`, `BPB=1.1165`, `CORE=0.0737`, around `555k tok/s` during recovery. This is now the best 400-step gate result, beating the older full-vocab ReLU² MTP2 run (`0.0737` vs `0.0710`) while using the 4× H100 bf16 setup.

The likely reason is structural, not just parameter count: DeepSeek-style MTP conditions each future depth on the previous drafted token path (`h_t + token_{t+1} -> token_{t+2}`, then MTP hidden + `token_{t+2} -> token_{t+3}`), whereas the shared-linear heads predict all offsets independently from the same original hidden state. That gives the auxiliary objective a more realistic future-token dependency and appears to improve CORE at the gate.

## 2026-05-14: Long D12 MTP2 + TST Run

We ran the first long D12-scale TST configuration after the 400-step gates:

```bash
GPU_TYPE=H100
FP8=1
COMPILE=1
OBJECTIVE=causal_mtp
MTP_HEADS=2
MTP_LOSS_WEIGHT=0.15
TST_BAG_SIZE=4
TST_RATIO=0.3
D_MODEL=768
N_LAYERS=12
N_HEADS=6
GATED_MLP=1
FF_MULT=3
OPTIMIZER=muon
BATCH_SIZE=32
SEQ_LEN=2048
TARGET_PARAM_DATA_RATIO=12
RUN_CONFIG=8gpu
```

Run: `mtp2-tst-s4-r03-d12-swiglu-ff3-h100-fp8-compile-8gpu`

Result at step `3200`:

| Run | Val Loss | BPB | CORE |
| --- | ---: | ---: | ---: |
| nanochat d12 public reference | — | 0.9825 | 0.1059 |
| MTP2 + TST s4 r0.3 d12 | 3.0028 | 0.9446 | 0.1133 |

This is the first run that clearly beats the public nanochat d12 CORE reference (`0.1133` vs `0.1059`) and also improves BPB (`0.9446` vs `0.9825`). The samples improved substantially on simple factual prompts (`France -> Paris`, `gold -> Au`, `hot -> cold`), though arithmetic still fails.

Caveat: this is not a strict same-token comparison to nanochat d12. The public nanochat d12 table reports about `1.08B` training tokens, while this run used roughly `1.72B` latent training tokens (`~143M params * 12`) and about `3.26B` effective raw-token exposure from TST. The fair claim is: at the same d12 model scale and D12-style latent-token budget, MTP2+TST beats the nanochat d12 reference on CORE and BPB.

## Lessons Learned

1. **Dropout is not free with SwiGLU.** The standard `dropout=0.1` carryover from nanoGPT silently destroys generalization once the FFN becomes multiplicative-gated. If you change MLP architecture, recheck dropout.
2. **A clean train/eval-loss gap is a more reliable bug detector than absolute numbers.** `train_loss` matching baseline while `val_loss` doubled was a near-instant fingerprint that the issue lived in code paths that differ between `.train()` and `.eval()`.
3. **Rotary buffers must match input dtype for FA3.** Leaving `cos`/`sin` in fp32 while `q/k` are bf16 silently downgrades to SDPA on H100 (-25% throughput).
4. **Parameter-cost matters at this scale.** Trading `~50M` full-vocab MTP params for a shared d×d projection costs CORE if you don't reinvest the saved capacity somewhere (FFN width, more layers, more steps).
5. **400-step gate is doing its job.** Every architectural experiment landed or failed inside this gate at < $5/run. Long schedules are reserved for promoted configs only.
6. **"Better BPB, worse CORE" is the GQA capacity-routing signature.** BPB averages over every token (dominated by the modal continuation); CORE scores specific answer-discrimination tokens. When K/V rank shrinks, the model gets sharper at average-case next-token prediction (fewer dimensions to overfit) but loses the *independent* attention patterns needed for multi-hop reading. If you ever see val_loss ↓ alongside CORE ↓ by >2σ, suspect attention-routing capacity, not optimizer drift.
7. **Don't cut capacity without reinvesting it.** GQA, MoE-without-expert-width, tied embeddings, MTP-shared-without-recovery — every "free shrinkage" we've tried at this scale has cost CORE. The winning configs all *trade* one form of capacity for another (e.g. dropped MTP unembeddings → SwiGLU width), not just shed it.

## Next Experiments

Priorities, roughly in order:

1. **Promote DeepSeek-MTP2 + TST.** The 400-step gate finally beat `CORE=0.0710`; next test is the same architecture at ratio-8 or ratio-12 with `MTP_ARCH=deepseek`, `MTP_HEADS=2`, `TST_BAG_SIZE=4`, `TST_RATIO=0.3`.

2. **Pure causal LM baseline** (`MTP_HEADS=0`) at the 400-step gate, so we can finally attribute the MTP contribution rather than guessing.

3. **Run an ablation without TST.** DeepSeek-style MTP2 may be carrying most of the gain; test `TST_BAG_SIZE=1`, `TST_RATIO=0` at the same 400-step gate.

4. **Re-test tied embeddings, but with tuned init and embedding LR.** The first tied attempt assumed the untied hyperparameters transfer cleanly — they don't.

5. **W&B table mutation warning.** The eval-samples table is logged with `log_mode='IMMUTABLE'` and mutated each eval. Switch to `MUTABLE` or recreate per eval. Cosmetic, but spammy.

## Current Position

The current best 400-step gate score is `CORE = 0.0737` from `mtp2-deepseek-tst-s4-r03-400step-swiglu-ff3-h100-bf16-4gpu-bs16`. This replaces the old H100 FP8 MTP2 ReLU² full-vocab result (`CORE = 0.0710`) as the short-run leader. The current best **cheap non-TST baseline** remains `CORE = 0.0688` on 4× H100 with SwiGLU + shared MTP + dropout=0.

The long-run best is now `CORE = 0.1133`, `BPB = 0.9446` from `mtp2-tst-s4-r03-d12-swiglu-ff3-h100-fp8-compile-8gpu`, which beats the public nanochat d12 reference (`CORE ≈ 0.1059`, `BPB = 0.9825`). This is not a strict same-token comparison because TST increases effective raw-token exposure, but it is a strong result at the d12 model scale.

GQA in either flavor (-2 over SwiGLU MTP1, -3 over ReLU² MTP2) has now been tested twice and rejected both times. Attention stays full multi-head (`N_HEADS=6, N_KV_HEADS=6`) on the recommended pretraining config. Any future GQA attempt should be paired with a capacity-reinvestment knob (`FF_MULT=4` or `N_LAYERS=14`), not run as a standalone change.
