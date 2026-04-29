from __future__ import annotations

import os
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
GPU_COUNT = 8


@app.function(
    image=image,
    gpu="H100:8",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_h100_8gpu(
    *,
    max_steps: int = 10_000,
    train_shards: int = 8,
    max_train_chars: int = 100_000_000,
    max_val_chars: int = 2_000_000,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    optimizer: str = "adamw",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    out_dir: str = "/runs/text-diffusion-4gpu",
    compile: bool = False,
    wandb: bool = False,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={GPU_COUNT}",
        str(WORKDIR / "train.py"),
        "--nanochat",
        "--nanochat-cache-dir",
        "/data/nanochat_climbmix",
        "--nanochat-train-shards",
        str(train_shards),
        "--max-train-chars",
        str(max_train_chars),
        "--max-val-chars",
        str(max_val_chars),
        "--nanochat-tokenizer-cache-dir",
        "/data/nanochat_tokenizer_32k",
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--optimizer",
        optimizer,
        "--max-steps",
        str(max_steps),
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
    if compile:
        command.append("--compile")
    if wandb:
        command.append("--wandb")

    try:
        env = os.environ.copy()
        if compile:
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()
        runs_volume.commit()


@app.local_entrypoint()
def main(
    max_steps: int = 10_000,
    train_shards: int = 8,
    max_train_chars: int = 100_000_000,
    max_val_chars: int = 2_000_000,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    optimizer: str = "adamw",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    out_dir: str = "/runs/text-diffusion-4gpu",
    compile: bool = False,
    wandb: bool = False,
) -> None:
    train_h100_8gpu.remote(
        max_steps=max_steps,
        train_shards=train_shards,
        max_train_chars=max_train_chars,
        max_val_chars=max_val_chars,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum_steps=grad_accum_steps,
        optimizer=optimizer,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        out_dir=out_dir,
        compile=compile,
        wandb=wandb,
    )
