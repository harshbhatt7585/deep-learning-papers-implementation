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


PROJECT_FILES = [
    "core_eval.py",
    "eval_core.py",
    "experiment.py",
    "experiment_tracker.py",
    "flash_attention.py",
    "model.py",
    "fp8.py",
    "nanochat_optim.py",
    "pretokenize.py",
    "sample.py",
    "tokenizer.py",
    "train.py",
    "utils.py",
]


def ignore_local_source(path: Path) -> bool:
    ignored_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "data",
        "runs",
        "exercise",
        ".codex",
        "nanochat",
    }
    if any(part in ignored_dirs for part in path.parts):
        return True
    if path.is_dir():
        return False
    return path.name not in PROJECT_FILES


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(frozen=False)
    .add_local_dir(".", remote_path=str(WORKDIR), ignore=ignore_local_source)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=64,
    memory=262144,
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
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    tokenizer_threads: int = 64,
    doc_batch_size: int = 4096,
    tokenizer_train_shards: int = 8,
    tokenizer_only: bool = False,
    download_only: bool = False,
    overwrite_tokens: bool = False,
) -> None:
    command = [
        "python",
        str(WORKDIR / "pretokenize.py"),
        "--nanochat-cache-dir",
        "/data/nanochat_climbmix",
        "--nanochat-tokenizer-cache-dir",
        nanochat_tokenizer_cache_dir,
        "--token-shards-dir",
        token_shards_dir,
        "--tokenizer-vocab-size",
        str(nanochat_tokenizer_vocab_size),
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
        "--tokenizer-train-shards",
        str(tokenizer_train_shards),
    ]
    if tokenizer_only:
        command.append("--tokenizer-only")
    if download_only:
        command.append("--download-only")
    if overwrite_tokens:
        command.append("--overwrite")

    try:
        subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()


def run_train(
    *,
    gpu_count: int,
    max_steps: int = -1,
    target_param_data_ratio: float = 8.0,
    target_tokens: int = -1,
    train_shards: int = 8,
    max_train_chars: int = 100_000_000,
    max_val_chars: int = 2_000_000,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    optimizer: str = "adamw",
    objective: str = "diffusion",
    mtp_heads: int = 3,
    mtp_loss_weight: float = 0.3,
    aurora_weight_decay: float = 0.025,
    d_model: int = 256,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    n_layers: int = 4,
    attention_window: int = 0,
    full_attention_every: int = 0,
    out_dir: str = "/runs/text-diffusion-4gpu",
    resume: str | None = None,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    stream_nanochat: bool = False,
    compile: bool = False,
    compile_mode: str = "default",
    fp8: bool = False,
    wandb: bool = False,
    experiment_description: str | None = None,
    experiment_tags: str | None = None,
    experiment_notes: str | None = None,
    eval_interval: int | None = None,
    core_metric_every: int | None = None,
    sample_interval: int | None = None,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        str(WORKDIR / "train.py"),
        "--nanochat-tokenizer-cache-dir",
        nanochat_tokenizer_cache_dir,
        "--nanochat-tokenizer-vocab-size",
        str(nanochat_tokenizer_vocab_size),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--optimizer",
        optimizer,
        "--objective",
        objective,
        "--mtp-heads",
        str(mtp_heads),
        "--mtp-loss-weight",
        str(mtp_loss_weight),
        "--aurora-weight-decay",
        str(aurora_weight_decay),
        "--max-steps",
        str(max_steps),
        "--target-param-data-ratio",
        str(target_param_data_ratio),
        "--target-tokens",
        str(target_tokens),
        "--d-model",
        str(d_model),
        "--n-heads",
        str(n_heads),
        "--n-layers",
        str(n_layers),
        "--attention-window",
        str(attention_window),
        "--full-attention-every",
        str(full_attention_every),
        "--amp-dtype",
        "bfloat16",
        "--out-dir",
        out_dir,
    ]
    if n_kv_heads is not None:
        command.extend(["--n-kv-heads", str(n_kv_heads)])
    if resume:
        command.extend(["--resume", resume])
    if stream_nanochat:
        command.extend(
            [
                "--stream-nanochat",
                "--nanochat-cache-dir",
                "/data/nanochat_climbmix",
                "--nanochat-train-shards",
                str(train_shards),
                "--max-val-chars",
                str(max_val_chars),
                "--token-shards-dir",
                token_shards_dir,
            ]
        )
    elif token_shards_dir:
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
        command.extend(["--compile-mode", compile_mode])
    if fp8:
        command.append("--fp8")
    if wandb:
        command.append("--wandb")
    if experiment_description:
        command.extend(["--experiment-description", experiment_description])
    if experiment_tags:
        command.extend(["--experiment-tags", experiment_tags])
    if experiment_notes:
        command.extend(["--experiment-notes", experiment_notes])
    if eval_interval is not None:
        command.extend(["--eval-interval", str(eval_interval)])
    if core_metric_every is not None:
        command.extend(["--core-metric-every", str(core_metric_every)])
    if sample_interval is not None:
        command.extend(["--sample-interval", str(sample_interval)])

    try:
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if compile:
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/torchinductor_cache")
            env.setdefault("TRITON_CACHE_DIR", "/data/triton_cache")
            env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
            Path(env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()
        runs_volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_a100_1gpu(**kwargs) -> None:
    run_train(gpu_count=1, **kwargs)


@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_a100_2gpu(**kwargs) -> None:
    run_train(gpu_count=2, **kwargs)


@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_a100_4gpu(**kwargs) -> None:
    run_train(gpu_count=4, **kwargs)


@app.function(
    image=image,
    gpu="A100-80GB:8",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_a100_8gpu(**kwargs) -> None:
    run_train(gpu_count=8, **kwargs)


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_h100_1gpu(**kwargs) -> None:
    run_train(gpu_count=1, **kwargs)


@app.function(
    image=image,
    gpu="H100:2",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_h100_2gpu(**kwargs) -> None:
    run_train(gpu_count=2, **kwargs)


@app.function(
    image=image,
    gpu="H100:4",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
    ],
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def train_h100_4gpu(**kwargs) -> None:
    run_train(gpu_count=4, **kwargs)


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
def train_h100_8gpu(**kwargs) -> None:
    run_train(gpu_count=8, **kwargs)


@app.local_entrypoint()
def main(
    gpu_count: int = 8,
    max_steps: int = -1,
    target_param_data_ratio: float = 8.0,
    target_tokens: int = -1,
    train_shards: int = 8,
    max_train_chars: int = 100_000_000,
    max_val_chars: int = 2_000_000,
    batch_size: int = 32,
    seq_len: int = 128,
    grad_accum_steps: int = 1,
    optimizer: str = "adamw",
    objective: str = "diffusion",
    mtp_heads: int = 3,
    mtp_loss_weight: float = 0.3,
    aurora_weight_decay: float = 0.025,
    gpu_type: str = "H100",
    d_model: int = 256,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    n_layers: int = 4,
    attention_window: int = 0,
    full_attention_every: int = 0,
    out_dir: str = "/runs/text-diffusion-4gpu",
    resume: str | None = None,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    tokenizer_threads: int = 64,
    doc_batch_size: int = 4096,
    tokenizer_train_shards: int = 8,
    compile: bool = False,
    compile_mode: str = "default",
    fp8: bool = False,
    wandb: bool = False,
    experiment_description: str | None = None,
    experiment_tags: str | None = None,
    experiment_notes: str | None = None,
    eval_interval: int | None = None,
    core_metric_every: int | None = None,
    sample_interval: int | None = None,
    pretokenize: bool = False,
    tokenizer_only: bool = False,
    download_only: bool = False,
    overwrite_tokens: bool = False,
    stream_nanochat: bool = False,
) -> None:
    gpu_type = gpu_type.upper()
    if gpu_type != "H100" and fp8:
        raise ValueError("--fp8 is only supported for H100/Hopper in this training path; disable FP8 for A100.")

    if pretokenize:
        pretokenize_nanochat.remote(
            train_shards=train_shards,
            max_train_chars=max_train_chars,
            max_val_chars=max_val_chars,
            token_shards_dir=token_shards_dir,
            nanochat_tokenizer_cache_dir=nanochat_tokenizer_cache_dir,
            nanochat_tokenizer_vocab_size=nanochat_tokenizer_vocab_size,
            tokenizer_threads=tokenizer_threads,
            doc_batch_size=doc_batch_size,
            tokenizer_train_shards=tokenizer_train_shards,
            tokenizer_only=tokenizer_only,
            download_only=download_only,
            overwrite_tokens=overwrite_tokens,
        )
        return

    gpu_type = gpu_type.upper()
    train_functions = {
        ("A100", 1): train_a100_1gpu,
        ("A100", 2): train_a100_2gpu,
        ("A100", 4): train_a100_4gpu,
        ("A100", 8): train_a100_8gpu,
        ("H100", 1): train_h100_1gpu,
        ("H100", 2): train_h100_2gpu,
        ("H100", 4): train_h100_4gpu,
        ("H100", 8): train_h100_8gpu,
    }
    train_function = train_functions.get((gpu_type, gpu_count))
    if train_function is None:
        raise ValueError("--gpu-type must be A100 or H100, and --gpu-count must be one of: 1, 2, 4, 8")

    train_function.remote(
        max_steps=max_steps,
        train_shards=train_shards,
        max_train_chars=max_train_chars,
        max_val_chars=max_val_chars,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum_steps=grad_accum_steps,
        optimizer=optimizer,
        objective=objective,
        mtp_heads=mtp_heads,
        mtp_loss_weight=mtp_loss_weight,
        aurora_weight_decay=aurora_weight_decay,
        target_param_data_ratio=target_param_data_ratio,
        target_tokens=target_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        attention_window=attention_window,
        full_attention_every=full_attention_every,
        out_dir=out_dir,
        resume=resume,
        token_shards_dir=token_shards_dir,
        nanochat_tokenizer_cache_dir=nanochat_tokenizer_cache_dir,
        nanochat_tokenizer_vocab_size=nanochat_tokenizer_vocab_size,
        stream_nanochat=stream_nanochat,
        compile=compile,
        compile_mode=compile_mode,
        fp8=fp8,
        wandb=wandb,
        experiment_description=experiment_description,
        experiment_tags=experiment_tags,
        experiment_notes=experiment_notes,
        eval_interval=eval_interval,
        core_metric_every=core_metric_every,
        sample_interval=sample_interval,
    )
