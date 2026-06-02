# Contributing to tinyGroot

Thanks for your interest! tinyGroot is a research codebase, so the bar is
"clear, correct, and reproducible" rather than "production-hardened."

## Dev setup

```bash
cd tinyGroot
uv sync --extra dev          # core + ruff/pytest/matplotlib
# or: pip install -e ".[dev]"
```

This is a [uv](https://docs.astral.sh/uv/) project (`[tool.uv] package = true`).
On Linux+CUDA, uv resolves the cu124 PyTorch wheels via `[tool.uv.sources]`; on
macOS you get the default CPU/MPS build.

## Before you open a PR

```bash
ruff check .          # lint
ruff format .         # format (line length 110)
pytest                # if you added tests
```

- Keep changes scoped; match the surrounding style (naming, comment density, idioms).
- New runs must use the canonical naming from `tinygroot.exp_naming` — don't hand-roll
  run names. Hyperparameters belong in `meta.json`/wandb, not in the name.
- Heavy deps (`modal`, `wandb`, FA3 `kernels`) must stay **lazily imported** so the
  core install runs on a laptop. If you add a top-level `import` of one of these to a
  core module, move it behind a function-local import or an extra instead.
- Attention changes: validate against a full (non-cached) forward — the static KV
  cache and speculative decoders are expected to be numerically equivalent to plain
  autoregressive decoding (see `infer/spec_decode.py`'s correctness check).

## Running things

- Local laptop: `make quickstart` or `bash speedrun_mps.sh <mode>`.
- Remote (Modal): `bash speed_run.sh <mode> <1gpu|2gpu|4gpu|8gpu>`; needs
  `pip install -e ".[modal]"` and `modal token new`.

## Reporting issues

Open a GitHub issue with the command you ran, the stage (pretrain/sft/rl/eval),
hardware, and the full traceback. For training-quality regressions, include the
run name (which encodes date + git SHA) so it's reproducible.
