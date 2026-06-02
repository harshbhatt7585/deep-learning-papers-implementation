"""Canonical experiment naming — the single source of truth for run names.

A run name is an *identifier*, not a config dump: full hyperparameters live in
each run's ``meta.json`` and in wandb. Names stay short, sortable, and parseable:

    groot/{stage}/{YYYYMMDD}-{slug}-{gpu}x{count}-{gitsha}

e.g. ``groot/rl/20260602-gsm8k-h100x8-a1b2c3d``.

Both the Modal entrypoints (``modal_train.py``) and ``speed_run.sh`` resolve names
through here so bash-side pipeline chaining and direct ``modal run`` invocations
agree on the path. Kept dependency-free (stdlib only) so ``speed_run.sh`` can call
``python -m tinygroot.exp_naming`` without importing torch/modal.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
from pathlib import Path

STAGES = ("pretrain", "sft", "rl", "eval", "draft")


def git_short_sha(repo_dir: str | None = None) -> str:
    cwd = repo_dir or str(Path(__file__).resolve().parents[1])
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def experiment_name(
    stage: str,
    slug: str = "",
    *,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    date: str | None = None,
    sha: str | None = None,
) -> str:
    """Build ``groot/{stage}/{date}-{slug}-{gpu}x{count}-{sha}``.

    ``date``/``sha`` can be pinned by the caller so a single resolution is reused
    across a chained pipeline; otherwise they are read from the clock and git HEAD.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    parts = [date or datetime.datetime.now().strftime("%Y%m%d")]
    if slug:
        parts.append(slug)
    if gpu_type and gpu_count:
        parts.append(f"{gpu_type.lower()}x{gpu_count}")
    parts.append(sha or git_short_sha())
    return f"groot/{stage}/" + "-".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a canonical experiment name.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--slug", default="")
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    args = parser.parse_args()
    print(experiment_name(args.stage, args.slug, gpu_type=args.gpu_type, gpu_count=args.gpu_count))


if __name__ == "__main__":
    main()
