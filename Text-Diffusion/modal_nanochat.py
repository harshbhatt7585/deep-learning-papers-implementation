from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "nanochat-speedrun"
WORKDIR = Path("/workspace/nanochat")

nanochat_cache = modal.Volume.from_name("nanochat-cache", create_if_missing=True)


def ignore_local_source(path: Path) -> bool:
    ignored_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    if any(part in ignored_dirs for part in path.parts):
        return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .pip_install("uv")
    .add_local_dir("nanochat", remote_path=str(WORKDIR), ignore=ignore_local_source)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H100:8",
    cpu=64,
    memory=262144,
    timeout=8 * 60 * 60,
    volumes={
        "/root/.cache/nanochat": nanochat_cache,
    },
)
def speedrun(wandb_run: str = "dummy") -> None:
    env = os.environ.copy()
    if wandb_run != "dummy" and not env.get("WANDB_API_KEY"):
        print(
            "WANDB_API_KEY is not configured in the Modal container; "
            "falling back to WANDB_RUN=dummy.",
            flush=True,
        )
        wandb_run = "dummy"
    env["WANDB_RUN"] = wandb_run
    if wandb_run == "dummy":
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env.pop("WANDB_API_KEY", None)
    print(f"Launching nanochat speedrun with WANDB_RUN={wandb_run!r}", flush=True)
    try:
        subprocess.run(
            [
                "env",
                f"WANDB_RUN={wandb_run}",
                f"WANDB_MODE={env.get('WANDB_MODE', 'online')}",
                f"WANDB_DISABLED={env.get('WANDB_DISABLED', 'false')}",
                "bash",
                "-e",
                "-u",
                "-o",
                "pipefail",
                "runs/speedrun.sh",
            ],
            cwd=WORKDIR,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    finally:
        nanochat_cache.commit()


@app.local_entrypoint()
def main(wandb_run: str = "dummy") -> None:
    speedrun.remote(wandb_run=wandb_run)
