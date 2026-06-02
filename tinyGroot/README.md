# 🌱 tinyGroot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**Small-scale LLM training you can actually read end to end** — nanochat-style
pretraining, multi-token prediction (MTP), Token Superposition Training (TST),
self-speculative decoding, chat SFT, and GSM8K RL — in one tidy package that runs
on a laptop or scales to 8× H100 on [Modal](https://modal.com).

> **Headline result:** the DeepSeek-MTP2 + TST recipe reaches **CORE ≈ 0.1162**,
> beating the public **nanochat d12** reference (**0.1059**) at the same budget.
> Full leaderboard in [BLOG.md](BLOG.md).

---

## ✨ Features

- **Decoder-only Transformer** with RoPE, RMSNorm, GQA, SwiGLU/ReLU² MLPs, and an
  FP8 training path (Hopper) that falls back cleanly to bf16/SDPA elsewhere.
- **Multi-token prediction** (Medusa-style shared heads *and* DeepSeek-style depth
  heads) — better pretraining sample-efficiency *and* free speculative-decoding drafters.
- **Speculative decoding** (`speculate_mtp`, `speculate_dflash`) with a correctness
  smoke test against plain autoregressive decoding.
- **Static KV-cache rollout engine** for fast, allocation-free RL generation.
- **Full post-training pipeline:** chat SFT → GSM8K REINFORCE/DAPO-style RL → eval.
- **One launcher, anywhere:** `speed_run.sh` (Modal, 1–8 GPUs) and `speedrun_mps.sh`
  (Apple-Silicon / CPU laptop).

## 🚀 Quickstart (local, no GPU required)

```bash
git clone https://github.com/harshbhatt7585/deep-learning-papers-implementation.git
cd deep-learning-papers-implementation/tinyGroot

# Recommended: uv (https://docs.astral.sh/uv/)
uv sync                      # creates .venv and installs the core deps
# ...or with pip:
# python -m venv .venv && source .venv/bin/activate && pip install -e .

# Tiny end-to-end pretrain on your Mac/laptop (downloads a small data shard):
make quickstart              # ≈ a few hundred steps on MPS/CPU
```

`make quickstart` wraps `speedrun_mps.sh` with laptop-sized defaults. See
[`examples/`](examples/) for individual steps (tokenize → train → sample).

## 📦 Install

| What you want | Command |
| --- | --- |
| Core (train/SFT/sample locally) | `uv sync` or `pip install -e .` |
| + Modal remote launchers | `pip install -e ".[modal]"` |
| + Weights & Biases logging | `pip install -e ".[wandb]"` |
| + FlashAttention-3 kernels (Hopper) | `pip install -e ".[gpu]"` |
| Everything + dev tools | `pip install -e ".[all,dev]"` |

The core install has **no Modal/wandb/CUDA-kernel dependency** — those are lazily
imported only when you use them, so a laptop run stays lightweight. On Linux+CUDA,
`uv` pulls the cu124 PyTorch wheels automatically (see `[tool.uv.sources]`).

## 🔁 The pipeline

Each stage produces a checkpoint the next stage consumes. Runs are named by a single
source of truth (`tinygroot.exp_naming`): `groot/{stage}/{date}-{slug}-{gpu}x{n}-{gitsha}`.

```
pretrain ──▶ chat SFT ──▶ GSM8K RL ──▶ eval
 train.py     chat_sft.py   chat_rl.py    eval.py
```

### Run it on Modal (1–8 GPUs)

```bash
# 1) Pretrain
SLUG=d768 D_MODEL=768 N_HEADS=6 N_LAYERS=12 MTP_HEADS=3 bash speed_run.sh train 8gpu

# 2) Chat SFT  (point at the pretrain checkpoint)
SLUG=mtp3 SFT_CHECKPOINT=/runs/groot/pretrain/<...>/checkpoint.pt bash speed_run.sh sft 8gpu

# 3) GSM8K RL  (point at the SFT checkpoint; fp8 auto-on for H100)
SLUG=gsm8k RL_CHECKPOINT=/runs/groot/sft/<...>/checkpoint.pt bash speed_run.sh rl 8gpu

# 4) Eval
EVAL_CHECKPOINT=/runs/groot/rl/<...>/checkpoint.pt bash speed_run.sh eval 8gpu

# ...or the whole thing, auto-chained:
SLUG=run1 bash speed_run.sh pipeline 8gpu
```

`SLUG` is the only name you type; date, GPU, and git SHA are filled in for you, and
the full hyperparameter set is recorded in each run's `meta.json` + wandb.

## 🖥️ Console commands

After install, these are on your `PATH` (each also runs as `python -m tinygroot.<...>`):

| Command | Does |
| --- | --- |
| `tinygroot-train` | Pretraining (causal LM + MTP/TST) |
| `tinygroot-chat-sft` | Chat supervised fine-tuning |
| `tinygroot-chat-rl` | GSM8K reinforcement learning |
| `tinygroot-pretokenize` | Build token shards + tokenizer |
| `tinygroot-sample` | Sample from a checkpoint |
| `tinygroot-chat-infer` | Interactive chat with an SFT checkpoint |
| `tinygroot-spec-decode` | Speculative-decoding benchmark/smoke test |
| `tinygroot-eval-core` | CORE / chat evals |
| `tinygroot-hf-upload` | Push a checkpoint dir to the Hugging Face Hub |

## 🗂️ Layout

```text
tinygroot/
  model.py        decoder-only Transformer (RoPE, GQA, MTP heads, static KV cache)
  engine.py       KV-cached generation engine used by RL rollouts
  flash_attention.py  FA3-with-SDPA-fallback shim
  exp_naming.py   canonical run-name generator (single source of truth)
  training/       train.py · chat_sft.py · chat_rl.py · pretokenize.py
  infer/          sample.py · chat_infer.py · spec_decode.py
  modal/          Modal launchers for remote training/inference
speed_run.sh      Modal launcher (pretrain/sft/rl/eval/pipeline, 1–8 GPUs)
speedrun_mps.sh   laptop launcher (Apple-Silicon / CPU)
examples/         minimal copy-paste runnable steps
```

## ☁️ Hugging Face uploads

```bash
HF_TOKEN=... tinygroot-hf-upload --checkpoint-dir runs/my-run --repo-id username/my-run
```
Training can also push automatically with `--push-to-hf --hf-repo-id username/my-run`.

## 📚 More

- [BLOG.md](BLOG.md) — experiment log, leaderboards, and the full results table.
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, linting, tests.
- [CHANGELOG.md](CHANGELOG.md) — notable changes.

## 📄 License

MIT © Harsh Bhatt — see [LICENSE](LICENSE).
