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
    cpu=32,
    memory=131072,
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_volume,
    },
)
def pretokenize_nanochat(
    *,
    train_shards: int = 170,
    max_train_chars: int = 17_000_000_000,
    max_val_chars: int = 2_000_000,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    tokenizer_threads: int = 32,
    doc_batch_size: int = 2048,
    overwrite_tokens: bool = False,
) -> None:
    command = [
        "python",
        str(WORKDIR / "pretokenize.py"),
        "--nanochat-cache-dir",
        "/data/nanochat_climbmix",
        "--nanochat-tokenizer-cache-dir",
        "/data/nanochat_tokenizer_32k",
        "--token-shards-dir",
        token_shards_dir,
        "--train-shards",
        str(train_shards),
        "--max-train-chars",
        str(max_train_chars),
        "--max-val-chars",
        str(max_val_chars),
        "--tokenizer-threads",
        str(tokenizer_threads),
        "--doc-batch-size",
        str(doc_batch_size),
    ]
    if overwrite_tokens:
        command.append("--overwrite")

    try:
        subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()


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
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    compile: bool = False,
    fp8: bool = False,
    wandb: bool = False,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={GPU_COUNT}",
        str(WORKDIR / "train.py"),
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
    if token_shards_dir:
        command.extend(["--token-shards-dir", token_shards_dir])
    else:
        command.extend(
            [
                "--nanochat",
                "--nanochat-cache-dir",
                "/data/nanochat_climbmix",
                "--nanochat-train-shards",
                str(train_shards),
                "--max-train-chars",
                str(max_train_chars),
                "--max-val-chars",
                str(max_val_chars),
            ]
        )
    if compile:
        command.append("--compile")
    if fp8:
        command.append("--fp8")
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


@app.function(
    image=image,
    gpu="H100:8",
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def core_eval_h100_8gpu(
    *,
    checkpoint_dir: str,
    eval_cache_dir: str = "/data/core_eval",
    max_per_task: int = -1,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={GPU_COUNT}",
        str(WORKDIR / "eval_core.py"),
        "--checkpoint-dir",
        checkpoint_dir,
        "--eval-cache-dir",
        eval_cache_dir,
        "--max-per-task",
        str(max_per_task),
    ]
    try:
        subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)
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
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    tokenizer_threads: int = 32,
    doc_batch_size: int = 2048,
    compile: bool = False,
    fp8: bool = False,
    wandb: bool = False,
    pretokenize: bool = False,
    overwrite_tokens: bool = False,
    core_eval: bool = False,
    checkpoint_dir: str | None = None,
    eval_cache_dir: str = "/data/core_eval",
    max_per_task: int = -1,
) -> None:
    if pretokenize:
        pretokenize_nanochat.remote(
            train_shards=train_shards,
            max_train_chars=max_train_chars,
            max_val_chars=max_val_chars,
            token_shards_dir=token_shards_dir,
            tokenizer_threads=tokenizer_threads,
            doc_batch_size=doc_batch_size,
            overwrite_tokens=overwrite_tokens,
        )
        return

    if core_eval:
        core_eval_h100_8gpu.remote(
            checkpoint_dir=checkpoint_dir or out_dir,
            eval_cache_dir=eval_cache_dir,
            max_per_task=max_per_task,
        )
        return

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
        token_shards_dir=token_shards_dir,
        compile=compile,
        fp8=fp8,
        wandb=wandb,
    )
