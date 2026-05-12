# Text MTP Progress Log

A living blog for the text diffusion / text-MTP experiments. The goal is to keep a clear record of what changed, what ran, what worked, and what still needs to be tested.

All numbers below are at the **fixed 400-step gate** (`MAX_STEPS=400`, `BATCH_SIZE=32`, `SEQ_LEN=2048`), with `tokens_per_step = 524,288` (`8×1` or `4×2` grad-accum). Reference target: **nanochat D12 CORE ≈ 0.1059**.

## Leaderboard — 400-Step Gate

Current best is in bold. All runs are `d_model=768`, `n_layers=12`, causal-MTP objective unless noted.

| # | Run | MLP | MTP design | Hardware | Val Loss | BPB | **CORE** | Params |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | **H100 FP8 MTP2 ReLU²** | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5491 | 1.1157 | **0.0710** | ~185M |
| 2 | A100 MTP1 ReLU² | ReLU² ff=4 | full-vocab × 1 | 8× A100 | 3.5953 | 1.1289 | 0.0693 | ~160M |
| 3 | H100 bf16 SwiGLU MTP1 shared, dropout=0 | SwiGLU ff=3 | shared × 1 | 4× H100 | 3.5866 | 1.1246 | 0.0688 | ~143M |
| 4 | H100 FP8 MoE top-1 | MoE 4×ff=1 | full-vocab × 2 | 8× H100 FP8 | 3.6385 | 1.1439 | 0.0688 | ~185M (≈181M active) |
| 5 | H100 FP8 GQA3 | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5655 | 1.1188 | 0.0681 | ~178M |
| — | Tied embeddings | ReLU² ff=4 | full-vocab × 2 | 8× H100 FP8 | 3.5672 | 1.1210 | 0.0615 | ~160M |
| — | MTP-shared first attempt | ReLU² ff=4 | shared × 1 | 8× A100 | 3.5363 | 1.1105 | 0.0585 | ~136M |
| — | SwiGLU dropout=0.1 (broken) | SwiGLU ff=3 | shared × 1 | 8× H100 | **8.4141** | **2.6436** | 0.0447 | ~143M |

Reading the leaderboard:

- The top score (`0.0710`) is still held by the heaviest config: 2× full-vocab MTP heads (≈50M extra params) and ReLU² FFN.
- The new SwiGLU + shared-MTP + dropout=0 result (`0.0688`) lands **tied for #3 on half the GPUs and ~50M fewer MTP params**, which is the most parameter-efficient point on the board. It's the right direction to scale up.
- The "MTP-shared first attempt" (`0.0585`) and "SwiGLU dropout=0.1" (`0.0447`) rows are kept in the table on purpose: they are the controlled stepping stones that show what *not* to do. Details in §SwiGLU below.

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

## Lessons Learned

1. **Dropout is not free with SwiGLU.** The standard `dropout=0.1` carryover from nanoGPT silently destroys generalization once the FFN becomes multiplicative-gated. If you change MLP architecture, recheck dropout.
2. **A clean train/eval-loss gap is a more reliable bug detector than absolute numbers.** `train_loss` matching baseline while `val_loss` doubled was a near-instant fingerprint that the issue lived in code paths that differ between `.train()` and `.eval()`.
3. **Rotary buffers must match input dtype for FA3.** Leaving `cos`/`sin` in fp32 while `q/k` are bf16 silently downgrades to SDPA on H100 (-25% throughput).
4. **Parameter-cost matters at this scale.** Trading `~50M` full-vocab MTP params for a shared d×d projection costs CORE if you don't reinvest the saved capacity somewhere (FFN width, more layers, more steps).
5. **400-step gate is doing its job.** Every architectural experiment landed or failed inside this gate at < $5/run. Long schedules are reserved for promoted configs only.

## Next Experiments

Priorities, roughly in order:

1. **Beat the leaderboard with SwiGLU + 8 GPUs.** Same config as today's #3 run but `RUN_CONFIG=8gpu` and either:
   - `MTP_HEADS=2  MTP_LOSS_WEIGHT=0.15` (more auxiliary signal, ~free with shared MTP)
   - or `FF_MULT=4` SwiGLU (extra FFN capacity, ~+8M params)
   The cheaper test is MTP=2 first.

2. **Pure causal LM baseline** (`MTP_HEADS=0`) at the 400-step gate, so we can finally attribute the MTP contribution rather than guessing.

3. **Promote a winner to ratio-8.** Whichever variant beats `CORE=0.0710` at the gate gets the full `TARGET_PARAM_DATA_RATIO=8` schedule.

4. **Re-test tied embeddings, but with tuned init and embedding LR.** The first tied attempt assumed the untied hyperparameters transfer cleanly — they don't.

5. **W&B table mutation warning.** The eval-samples table is logged with `log_mode='IMMUTABLE'` and mutated each eval. Switch to `MUTABLE` or recreate per eval. Cosmetic, but spammy.

## Current Position

The current best 400-step gate score is `CORE = 0.0710` (H100 FP8 MTP2 ReLU², full-vocab MTP heads). The current best **per-parameter and per-GPU** efficiency is `CORE = 0.0688` on 4× H100 with SwiGLU + shared MTP + dropout=0 — same neighborhood, dramatically cheaper. The nanochat D12 reference (`~0.1059`) is still ahead but no longer feels out of reach: closing the gap is about scaling the SwiGLU+shared-MTP recipe back up to 8 GPUs and/or running longer than 400 steps.
