from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "text-diffusion-train"
WORKDIR = Path("/workspace")

data_volume = modal.Volume.from_name("text-diffusion-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("text-diffusion-runs", create_if_missing=True)


def ignore_local_files(path: Path) -> bool:
    ignored = {".git", ".venv", "__pycache__", "data", "runs"}
    return any(part in ignored for part in path.parts)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(frozen=False)
    .add_local_dir(".", remote_path=str(WORKDIR), ignore=ignore_local_files)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100:4",
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_4gpu(
    *,
    max_steps: int = 10_000,
    train_shards: int = 8,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    out_dir: str = "/runs/text-diffusion-4gpu",
    wandb: bool = False,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        str(WORKDIR / "train.py"),
        "--nanochat",
        "--nanochat-cache-dir",
        "/data/nanochat_climbmix",
        "--nanochat-train-shards",
        str(train_shards),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--max-steps",
        str(max_steps),
        "--eval-interval",
        "500",
        "--eval-batches",
        "20",
        "--save-interval",
        "1000",
        "--sample-interval",
        "1000",
        "--d-model",
        str(d_model),
        "--n-heads",
        str(n_heads),
        "--n-layers",
        str(n_layers),
        "--amp-dtype",
        "bfloat16",
        "--out-dir",
        out_dir,
    ]
    if wandb:
        command.append("--wandb")

    try:
        subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()
        runs_volume.commit()


@app.local_entrypoint()
def main(
    max_steps: int = 10_000,
    train_shards: int = 8,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    out_dir: str = "/runs/text-diffusion-4gpu",
    wandb: bool = False,
) -> None:
    train_4gpu.remote(
        max_steps=max_steps,
        train_shards=train_shards,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum_steps=grad_accum_steps,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        out_dir=out_dir,
        wandb=wandb,
    )
