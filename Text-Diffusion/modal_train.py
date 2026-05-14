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
    "dflash_model.py",
    "eval_core.py",
    "flash_attention.py",
    "model.py",
    "fp8.py",
    "nanochat_optim.py",
    "pretokenize.py",
    "sample.py",
    "spec_decode.py",
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
    eval_interval: int | None = None,
    core_metric_every: int | None = None,
    sample_interval: int | None = None,
    optimizer: str = "adamw",
    objective: str = "diffusion",
    mtp_heads: int = 3,
    mtp_loss_weight: float = 0.3,
    tst_bag_size: int = 1,
    tst_ratio: float = 0.0,
    aurora_weight_decay: float = 0.025,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_mult: int = 4,
    gated_mlp: bool = False,
    out_dir: str = "/runs/text-diffusion-4gpu",
    resume: str | None = None,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    stream_nanochat: bool = False,
    compile: bool = False,
    fp8: bool = False,
    wandb: bool = False,
    # --- DFlash drafter flags (used only when objective == "dflash") -------
    target_checkpoint: str | None = None,
    block_size: int = 16,
    n_draft_layers: int = 2,
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
        "--tst-bag-size",
        str(tst_bag_size),
        "--tst-ratio",
        str(tst_ratio),
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
        "--ff-mult",
        str(ff_mult),
        "--amp-dtype",
        "bfloat16",
        "--out-dir",
        out_dir,
    ]
    if gated_mlp:
        command.append("--gated-mlp")
    if resume:
        command.extend(["--resume", resume])
    if objective == "dflash":
        if not target_checkpoint:
            raise ValueError("--objective dflash requires --target-checkpoint")
        if not os.path.exists(target_checkpoint):
            raise SystemExit(
                f"[modal_train] target checkpoint not found inside the container: {target_checkpoint}\n"
                f"  -> Run 'modal run modal_train.py::list_runs' from your Mac to discover the\n"
                f"     real path on the text-diffusion-runs volume, then pass it via\n"
                f"     'TARGET_CHECKPOINT=<actual-path> bash speed_run.sh draft 4gpu'."
            )
        command.extend([
            "--target-checkpoint", target_checkpoint,
            "--block-size", str(block_size),
            "--n-draft-layers", str(n_draft_layers),
        ])
    if eval_interval is not None:
        command.extend(["--eval-interval", str(eval_interval)])
    if core_metric_every is not None:
        command.extend(["--core-metric-every", str(core_metric_every)])
    if sample_interval is not None:
        command.extend(["--sample-interval", str(sample_interval)])
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
    if fp8:
        command.append("--fp8")
    if wandb:
        command.append("--wandb")

    try:
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if compile:
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
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
    eval_interval: int | None = None,
    core_metric_every: int | None = None,
    sample_interval: int | None = None,
    optimizer: str = "adamw",
    objective: str = "diffusion",
    mtp_heads: int = 3,
    mtp_loss_weight: float = 0.3,
    tst_bag_size: int = 1,
    tst_ratio: float = 0.0,
    aurora_weight_decay: float = 0.025,
    gpu_type: str = "H100",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_mult: int = 4,
    gated_mlp: bool = False,
    out_dir: str = "/runs/text-diffusion-4gpu",
    resume: str | None = None,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    tokenizer_threads: int = 64,
    doc_batch_size: int = 4096,
    tokenizer_train_shards: int = 8,
    compile: bool = False,
    fp8: bool = False,
    wandb: bool = False,
    pretokenize: bool = False,
    tokenizer_only: bool = False,
    download_only: bool = False,
    overwrite_tokens: bool = False,
    stream_nanochat: bool = False,
    # --- DFlash drafter flags (only used with --objective dflash) ---------
    target_checkpoint: str | None = None,
    block_size: int = 16,
    n_draft_layers: int = 2,
) -> None:
    gpu_type = gpu_type.upper().replace("_", "-")
    if gpu_type == "A100-80GB":
        gpu_type = "A100"
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
        raise ValueError("--gpu-type must be A100, A100-80GB, or H100, and --gpu-count must be one of: 1, 2, 4, 8")

    modal_gpu_request = f"A100-80GB:{gpu_count}" if gpu_type == "A100" and gpu_count > 1 else gpu_type
    if gpu_type == "A100" and gpu_count == 1:
        modal_gpu_request = "A100-80GB"
    print(f"modal_gpu_request: {modal_gpu_request}", flush=True)

    train_function.remote(
        max_steps=max_steps,
        train_shards=train_shards,
        max_train_chars=max_train_chars,
        max_val_chars=max_val_chars,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum_steps=grad_accum_steps,
        eval_interval=eval_interval,
        core_metric_every=core_metric_every,
        sample_interval=sample_interval,
        optimizer=optimizer,
        objective=objective,
        mtp_heads=mtp_heads,
        mtp_loss_weight=mtp_loss_weight,
        tst_bag_size=tst_bag_size,
        tst_ratio=tst_ratio,
        aurora_weight_decay=aurora_weight_decay,
        target_param_data_ratio=target_param_data_ratio,
        target_tokens=target_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        gated_mlp=gated_mlp,
        out_dir=out_dir,
        resume=resume,
        token_shards_dir=token_shards_dir,
        nanochat_tokenizer_cache_dir=nanochat_tokenizer_cache_dir,
        nanochat_tokenizer_vocab_size=nanochat_tokenizer_vocab_size,
        stream_nanochat=stream_nanochat,
        compile=compile,
        fp8=fp8,
        wandb=wandb,
        target_checkpoint=target_checkpoint,
        block_size=block_size,
        n_draft_layers=n_draft_layers,
    )


@app.function(
    image=image,
    timeout=5 * 60,
    volumes={"/runs": runs_volume},
)
def _list_runs_remote() -> list[dict]:
    """List every run directory on the runs volume so users can discover what
    actually got checkpointed before pointing spec_decode at it."""
    import os
    root = "/runs"
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        entry = {"run": name, "path": run_dir, "has_checkpoint": False, "size_mb": 0.0, "files": []}
        ckpt = os.path.join(run_dir, "checkpoint.pt")
        if os.path.exists(ckpt):
            entry["has_checkpoint"] = True
            entry["size_mb"] = round(os.path.getsize(ckpt) / 1e6, 2)
        try:
            entry["files"] = sorted(os.listdir(run_dir))[:10]
        except Exception:
            pass
        out.append(entry)
    return out


@app.local_entrypoint()
def list_runs() -> None:
    """List every run directory on the Modal runs volume.

    Use this to find a real target checkpoint path before invoking ``spec`` or
    ``speed_run.sh draft``.

    Example::

        modal run modal_train.py::list_runs
    """
    entries = _list_runs_remote.remote()
    if not entries:
        print("[list_runs] no runs found on volume text-diffusion-runs.")
        return
    print(f"[list_runs] found {len(entries)} run directories on /runs:")
    for e in entries:
        marker = "ckpt" if e["has_checkpoint"] else "----"
        print(f"  [{marker}] {e['path']:<70s}  size={e['size_mb']:>8.2f} MB")
        if e["files"]:
            print(f"         files: {', '.join(e['files'])}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
)
def run_spec_decode(
    *,
    checkpoint: str,
    prompt: str = "The capital of France is",
    gen_length: int = 64,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    tokenizer_dir: str | None = None,
    seed: int = 0,
    warmup: int = 1,
    # --- DFlash spec-decode args (optional) -------------------------------
    drafter_checkpoint: str | None = None,
    mode: str = "auto",
    block_size: int = 16,
) -> None:
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"[modal_train] target checkpoint not found inside the container: {checkpoint}\n"
            f"  -> Run 'modal run modal_train.py::list_runs' to find the exact /runs path."
        )
    if drafter_checkpoint is not None and not os.path.exists(drafter_checkpoint):
        raise SystemExit(
            f"[modal_train] drafter checkpoint not found inside the container: {drafter_checkpoint}\n"
            f"  -> Run 'modal run modal_train.py::list_runs' to find the exact /runs path."
        )
    command = [
        "python",
        "-m",
        "spec_decode",
        "--checkpoint",
        checkpoint,
        "--prompt",
        prompt,
        "--gen-length",
        str(gen_length),
        "--temperature",
        str(temperature),
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--mode",
        mode,
        "--block-size",
        str(block_size),
    ]
    if top_k is not None:
        command += ["--top-k", str(top_k)]
    if top_p is not None:
        command += ["--top-p", str(top_p)]
    if tokenizer_dir is not None:
        command += ["--tokenizer-dir", tokenizer_dir]
    if drafter_checkpoint is not None:
        command += ["--drafter-checkpoint", drafter_checkpoint]
    subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)


@app.local_entrypoint()
def spec(
    checkpoint: str = "/runs/bench-h100-bf16-4gpu-d12-mtp1-swiglu-ff3-drop0-400/checkpoint.pt",
    prompt: str = "The capital of France is",
    gen_length: int = 64,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    tokenizer_dir: str | None = None,
    seed: int = 0,
    warmup: int = 1,
    drafter_checkpoint: str | None = None,
    mode: str = "auto",
    block_size: int = 16,
) -> None:
    """Run the speculative-decoding smoke test on a Modal A10G.

    MTP-only (target has MTP heads, no drafter)::

        modal run modal_train.py::spec \\
            --checkpoint /runs/text-diffusion-mtp1-relu2/checkpoint.pt \\
            --prompt "The capital of France is" --gen-length 64

    DFlash (target + DFlash drafter checkpoint)::

        modal run modal_train.py::spec \\
            --checkpoint /runs/text-diffusion-mtp1-relu2/checkpoint.pt \\
            --drafter-checkpoint /runs/text-diffusion-dflash-drafter-.../checkpoint.pt \\
            --mode dflash --block-size 16 \\
            --prompt "The capital of France is" --gen-length 64
    """
    run_spec_decode.remote(
        checkpoint=checkpoint,
        prompt=prompt,
        gen_length=gen_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        tokenizer_dir=tokenizer_dir,
        seed=seed,
        warmup=warmup,
        drafter_checkpoint=drafter_checkpoint,
        mode=mode,
        block_size=block_size,
    )
