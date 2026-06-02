# Changelog

All notable changes to tinyGroot are documented here. This project loosely follows
[Keep a Changelog](https://keepachangelog.com/) and [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Packaging polish: MIT `LICENSE`, `CONTRIBUTING.md`, `CITATION.cff`, this changelog,
  `examples/`, and a `Makefile` with a one-command local `quickstart`.
- Optional dependency extras (`modal`, `wandb`, `gpu`, `dev`, `all`) so the core
  install is laptop-light; Modal/wandb/FA3 kernels are pulled in only on demand.
- `tinygroot.exp_naming`: a single source of truth for run names
  (`groot/{stage}/{date}-{slug}-{gpu}x{n}-{gitsha}`), shared by `speed_run.sh` and the
  Modal entrypoints.
- `StaticKVCache`: pre-allocated KV buffers for the RL rollout engine — removes the
  per-step `torch.cat` reallocation and routes decode through `flash_attn_with_kvcache`.

### Changed
- Modal launcher (`modal/modal_train.py`) deduplicated: the 16 near-identical
  per-`(gpu, count)` functions are now registered from a single loop, and CLI argv is
  built via an `args_to_argv` passthrough instead of hand-maintained flag lists.
- `README` rewritten as the package front page; `BLOG.md` remains the experiment log.

### Fixed
- RL rollout produced NaN logits under FA3/bf16 because the static KV buffer was read
  before initialization during prefill. Buffers are now zero-initialized and prefill
  uses the proven `flash_attn_func` path; decode uses `flash_attn_with_kvcache`.

## [0.1.0]
- Initial tinyGroot: decoder-only Transformer (RoPE, GQA, SwiGLU/ReLU²), MTP heads
  (Medusa- and DeepSeek-style), TST, FP8 training, speculative decoding, chat SFT,
  GSM8K RL, and Modal launchers.
