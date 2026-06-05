from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


def args_to_argv(mapping: dict) -> list[str]:
    """Flatten ``{dest: value}`` into CLI argv, replacing the hand-maintained
    per-flag ``command.extend([...])`` walls. ``None``/``False`` are omitted,
    ``True`` becomes a bare flag, and underscores become hyphens
    (``num_samples=16`` -> ``["--num-samples", "16"]``)."""
    argv: list[str] = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def resolve_run_paths(
    stage: str,
    slug: str,
    out_dir: str,
    run_name: str | None,
    *,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
) -> tuple[str, str]:
    """Resolve the canonical out_dir (``/runs/groot/...``) and wandb run name from
    a slug, unless the caller passed an explicit ``out_dir`` (e.g. speed_run.sh's
    pipeline chaining, which pre-resolves paths via ``tinygroot.exp_naming``). When
    an explicit out_dir is given, the run name is derived from it so the two stay
    consistent.

    ``experiment_name`` is imported lazily (not at module top level): this module is
    re-imported inside the Modal container as a top-level ``modal_train`` module with
    no ``tinygroot`` package on its path, and naming is only ever resolved locally in
    the entrypoints before ``.remote()``."""
    from tinygroot.exp_naming import experiment_name

    if out_dir:
        name = out_dir[len("/runs/"):] if out_dir.startswith("/runs/") else out_dir
    else:
        name = experiment_name(stage, slug, gpu_type=gpu_type, gpu_count=gpu_count)
        out_dir = f"/runs/{name}"
    if run_name is None:
        run_name = name.replace("/", "-")
    return out_dir, run_name


APP_NAME = "tinygroot-train"
WORKDIR = Path("/workspace")

# Keep the legacy Modal volume names so existing checkpoints and datasets remain mounted.
data_volume = modal.Volume.from_name("text-diffusion-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("text-diffusion-runs", create_if_missing=True)


PROJECT_FILES = [
    "__init__.py",
    "cache_management.py",
    "chat_core_eval.py",
    "chat_sft.py",
    "core_eval.py",
    "dflash_model.py",
    "eval.py",
    "eval_core.py",
    "engine.py",
    "flash_attention.py",
    "hf_upload.py",
    "model.py",
    "fp8.py",
    "nanochat_optim.py",
    "chat_rl.py",
    "pretokenize.py",
    "sample.py",
    "sft_chat.py",
    "sft_data.py",
    "spec_decode.py",
    "tokenizer.py",
    "train.py",
    "modal_train.py",
    "modal_chat_sft.py",
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
    return path.name not in PROJECT_FILES and path.name not in {"pyproject.toml", "uv.lock"}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(frozen=False)
    .add_local_dir(".", remote_path=str(WORKDIR), ignore=ignore_local_source)
)

app = modal.App(APP_NAME)


def add_workspace_pythonpath(env: dict[str, str]) -> None:
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{WORKDIR}:{pythonpath}" if pythonpath else str(WORKDIR)


def hf_upload_secrets() -> list[modal.Secret]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return [modal.Secret.from_dict({"HF_TOKEN": token})]
    secret_name = os.environ.get("MODAL_HF_SECRET_NAME")
    if secret_name:
        return [modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])]
    return []


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    secrets=hf_upload_secrets(),
    volumes={"/runs": runs_volume},
)
def upload_checkpoint_to_hf(
    *,
    checkpoint_dir: str,
    repo_id: str,
    private: bool = False,
    revision: str | None = None,
    commit_message: str | None = None,
) -> None:
    command = [
        "python",
        "-m",
        "tinygroot.hf_upload",
        "--checkpoint-dir",
        checkpoint_dir,
        "--repo-id",
        repo_id,
    ]
    if private:
        command.append("--private")
    if revision is not None:
        command.extend(["--revision", revision])
    if commit_message is not None:
        command.extend(["--commit-message", commit_message])
    try:
        subprocess.run(command, cwd=WORKDIR, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        runs_volume.commit()


def run_chat_eval(
    *,
    gpu_count: int,
    checkpoint: str,
    suite: str,
    eval_examples: int,
    eval_num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    chatcore_max_cat: int,
    chatcore_max_sample: int,
    chatcore_max_new_tokens: int,
    chatcore_temperature: float,
    chatcore_top_k: int,
    chatcore_batch_size: int,
) -> None:
    checkpoint_path = Path(checkpoint)
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    has_legacy_checkpoint = checkpoint_path.exists()
    has_split_checkpoint = (checkpoint_dir / "model.pt").exists() and (checkpoint_dir / "meta.json").exists()
    if not has_legacy_checkpoint and not has_split_checkpoint:
        raise SystemExit(
            f"[modal_train] eval checkpoint not found inside the container: {checkpoint}\n"
            f"  -> New-format checkpoints are directories containing model.pt + meta.json.\n"
            f"  -> Run 'modal run tinygroot/modal/modal_train.py::list_runs' to find the exact /runs path."
        )
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "-m",
        "tinygroot.eval",
        *args_to_argv(
            {
                "checkpoint": checkpoint,
                "suite": suite,
                "eval_examples": eval_examples,
                "eval_num_samples": eval_num_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "chatcore_max_cat": chatcore_max_cat,
                "chatcore_max_sample": chatcore_max_sample,
                "chatcore_max_new_tokens": chatcore_max_new_tokens,
                "chatcore_temperature": chatcore_temperature,
                "chatcore_top_k": chatcore_top_k,
                "chatcore_batch_size": chatcore_batch_size,
            }
        ),
    ]
    env = os.environ.copy()
    add_workspace_pythonpath(env)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)


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
        "-m",
        "tinygroot.training.pretokenize",
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
    dflash: bool = False,
    mtp_heads: int = 3,
    arch: str = "hrm",
    mtp_arch: str = "linear",
    mtp_loss_weight: float = 0.3,
    tst_bag_size: int = 1,
    tst_ratio: float = 0.0,
    aurora_weight_decay: float = 0.025,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_mult: int = 4,
    gated_mlp: bool = False,
    out_dir: str = "/runs/tinygroot-4gpu",
    resume: str | None = None,
    token_shards_dir: str = "/data/nanochat_tokens_32k",
    nanochat_tokenizer_cache_dir: str = "/data/nanochat_tokenizer_32k",
    nanochat_tokenizer_vocab_size: int = 32_768,
    stream_nanochat: bool = False,
    compile: bool = False,
    fp8: bool = False,
    wandb: bool = False,
    # --- DFlash drafter flags (used only with --dflash) --------------------
    target_checkpoint: str | None = None,
    block_size: int = 16,
    n_draft_layers: int = 2,
) -> None:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "tinygroot/training/train.py",
        *args_to_argv(
            {
                "nanochat_tokenizer_cache_dir": nanochat_tokenizer_cache_dir,
                "nanochat_tokenizer_vocab_size": nanochat_tokenizer_vocab_size,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "optimizer": optimizer,
                "mtp_heads": mtp_heads,
                "arch": arch,
                "mtp_arch": mtp_arch,
                "mtp_loss_weight": mtp_loss_weight,
                "tst_bag_size": tst_bag_size,
                "tst_ratio": tst_ratio,
                "aurora_weight_decay": aurora_weight_decay,
                "max_steps": max_steps,
                "target_param_data_ratio": target_param_data_ratio,
                "target_tokens": target_tokens,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "ff_mult": ff_mult,
                "amp_dtype": "bfloat16",
                "out_dir": out_dir,
                # bool flags + optional values; args_to_argv drops False/None.
                "gated_mlp": gated_mlp,
                "resume": resume,
                "eval_interval": eval_interval,
                "core_metric_every": core_metric_every,
                "sample_interval": sample_interval,
                "compile": compile,
                "fp8": fp8,
                "wandb": wandb,
            }
        ),
    ]
    if dflash:
        if not target_checkpoint:
            raise ValueError("--dflash requires --target-checkpoint")
        if not os.path.exists(target_checkpoint):
            raise SystemExit(
                f"[modal_train] target checkpoint not found inside the container: {target_checkpoint}\n"
                f"  -> Run 'modal run tinygroot/modal/modal_train.py::list_runs' from your Mac to discover the\n"
                f"     real path on the runs volume, then pass it via\n"
                f"     'TARGET_CHECKPOINT=<actual-path> bash speed_run.sh draft 4gpu'."
            )
        command.extend([
            "--dflash",
            "--target-checkpoint", target_checkpoint,
            "--block-size", str(block_size),
            "--n-draft-layers", str(n_draft_layers),
        ])
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

    try:
        env = os.environ.copy()
        add_workspace_pythonpath(env)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if compile:
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()
        runs_volume.commit()


def run_chat_rl(
    *,
    gpu_count: int,
    checkpoint: str,
    out_dir: str,
    run_name: str | None,
    wandb_project: str,
    num_epochs: int,
    max_steps: int,
    device_batch_size: int,
    examples_per_step: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eval_suite: str,
    eval_every: int,
    eval_examples: int,
    chatcore_max_cat: int,
    chatcore_max_sample: int,
    chatcore_max_new_tokens: int,
    chatcore_temperature: float,
    chatcore_top_k: int,
    chatcore_batch_size: int,
    save_every: int,
    optimizer: str,
    wandb: bool,
    compile: bool,
    fp8: bool,
) -> None:
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"[modal_train] RL checkpoint not found inside the container: {checkpoint}\n"
            f"  -> Run 'modal run tinygroot/modal/modal_train.py::list_runs' to find the exact /runs path."
        )
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "tinygroot/training/chat_rl.py",
        *args_to_argv(
            {
                "checkpoint": checkpoint,
                "out_dir": out_dir,
                "run_name": run_name,
                "wandb_project": wandb_project,
                "num_epochs": num_epochs,
                "max_steps": max_steps,
                "device_batch_size": device_batch_size,
                "examples_per_step": examples_per_step,
                "num_samples": num_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "eval_suite": eval_suite,
                "eval_every": eval_every,
                "eval_examples": eval_examples,
                "chatcore_max_cat": chatcore_max_cat,
                "chatcore_max_sample": chatcore_max_sample,
                "chatcore_max_new_tokens": chatcore_max_new_tokens,
                "chatcore_temperature": chatcore_temperature,
                "chatcore_top_k": chatcore_top_k,
                "chatcore_batch_size": chatcore_batch_size,
                "save_every": save_every,
                "optimizer": optimizer,
                "wandb": wandb,
                "compile": compile,
                "fp8": fp8,
            }
        ),
    ]

    try:
        env = os.environ.copy()
        add_workspace_pythonpath(env)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if compile:
            env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        runs_volume.commit()


def _gpu_spec(gpu_type: str, gpu_count: int) -> str:
    base = "A100-80GB" if gpu_type == "A100" else gpu_type
    return f"{base}:{gpu_count}" if gpu_count > 1 else base


# The train / RL / eval Modal functions differ only by GPU type x count (and whether
# they need the wandb secret), so register them in a loop instead of hand-writing 16
# near-identical defs. Each is given a unique name so Modal can import it by name in
# the container, and is exposed both as a module global and via the registry that
# select_*_function() looks up.
_DAY = 24 * 60 * 60
_WANDB_SECRET = [modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])]
_STAGE_RUNNERS = {
    "train": (run_train, _WANDB_SECRET),
    "rl": (run_chat_rl, _WANDB_SECRET),
    "eval": (run_chat_eval, []),
}
_GPU_MATRIX = [("A100", n) for n in (1, 2, 4, 8)] + [("H100", n) for n in (1, 2, 4, 8)]

TRAIN_FUNCTIONS: dict = {}
RL_FUNCTIONS: dict = {}
EVAL_FUNCTIONS: dict = {}
_STAGE_REGISTRIES = {"train": TRAIN_FUNCTIONS, "rl": RL_FUNCTIONS, "eval": EVAL_FUNCTIONS}


def _make_stage_fn(runner, gpu_count: int, name: str):
    def fn(**kwargs):
        runner(gpu_count=gpu_count, **kwargs)

    # Modal's @app.function rejects non-global functions (qualname containing
    # "<locals>") unless serialized=True — but serialized functions require the
    # local and image Python versions to match (we have 3.13 local / 3.11 image).
    # Giving the closure a bare module-level qualname + name satisfies the global
    # check and lets Modal re-import it by name in the container, where re-running
    # this loop re-registers the same name. No serialization needed.
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__module__ = __name__
    return fn


for _stage, (_runner, _secrets) in _STAGE_RUNNERS.items():
    for _gpu_type, _gpu_count in _GPU_MATRIX:
        _fn_name = f"{_stage}_{_gpu_type.lower()}_{_gpu_count}gpu"
        _fn = app.function(
            image=image,
            gpu=_gpu_spec(_gpu_type, _gpu_count),
            name=_fn_name,
            timeout=_DAY,
            secrets=_secrets,
            volumes={"/data": data_volume, "/runs": runs_volume},
        )(_make_stage_fn(_runner, _gpu_count, _fn_name))
        globals()[_fn_name] = _fn
        _STAGE_REGISTRIES[_stage][(_gpu_type, _gpu_count)] = _fn


def select_train_function(gpu_type: str, gpu_count: int):
    return TRAIN_FUNCTIONS.get((gpu_type, gpu_count))


def select_rl_function(gpu_type: str, gpu_count: int):
    return RL_FUNCTIONS.get((gpu_type, gpu_count))


def select_eval_function(gpu_type: str, gpu_count: int):
    return EVAL_FUNCTIONS.get((gpu_type, gpu_count))


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
    dflash: bool = False,
    mtp_heads: int = 3,
    arch: str = "hrm",
    mtp_arch: str = "linear",
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
    slug: str = "",
    out_dir: str = "",
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
    push_to_hf: bool = False,
    hf_repo_id: str | None = None,
    hf_private: bool = False,
    hf_revision: str | None = None,
    hf_commit_message: str | None = None,
    pretokenize: bool = False,
    tokenizer_only: bool = False,
    download_only: bool = False,
    overwrite_tokens: bool = False,
    stream_nanochat: bool = False,
    # --- DFlash drafter flags (only used with --dflash) --------------------
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

    train_function = select_train_function(gpu_type, gpu_count)
    if train_function is None:
        raise ValueError("--gpu-type must be A100, A100-80GB, or H100, and --gpu-count must be one of: 1, 2, 4, 8")

    stage = "draft" if dflash else "pretrain"
    out_dir, _ = resolve_run_paths(stage, slug, out_dir, None, gpu_type=gpu_type, gpu_count=gpu_count)
    print(f"modal_gpu_request: {_gpu_spec(gpu_type, gpu_count)}  out_dir: {out_dir}", flush=True)

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
        dflash=dflash,
        mtp_heads=mtp_heads,
        arch=arch,
        mtp_arch=mtp_arch,
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
    if push_to_hf:
        if not hf_repo_id:
            raise ValueError("--push-to-hf requires --hf-repo-id")
        upload_checkpoint_to_hf.remote(
            checkpoint_dir=out_dir,
            repo_id=hf_repo_id,
            private=hf_private,
            revision=hf_revision,
            commit_message=hf_commit_message,
        )


@app.local_entrypoint()
def evaluate(
    checkpoint: str,
    gpu_type: str = "H100",
    gpu_count: int = 8,
    suite: str = "both",
    eval_examples: int = 400,
    eval_num_samples: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    chatcore_max_cat: int = -1,
    chatcore_max_sample: int = 24,
    chatcore_max_new_tokens: int = 512,
    chatcore_temperature: float = 0.0,
    chatcore_top_k: int = 50,
    chatcore_batch_size: int = 8,
) -> None:
    """Run standalone chat evals for a checkpoint on Modal."""
    gpu_type = gpu_type.upper().replace("_", "-")
    if gpu_type == "A100-80GB":
        gpu_type = "A100"
    eval_function = select_eval_function(gpu_type, gpu_count)
    if eval_function is None:
        raise ValueError("--gpu-type must be A100, A100-80GB, or H100, and --gpu-count must be one of: 1, 2, 4, 8")
    eval_function.remote(
        checkpoint=checkpoint,
        suite=suite,
        eval_examples=eval_examples,
        eval_num_samples=eval_num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        chatcore_max_cat=chatcore_max_cat,
        chatcore_max_sample=chatcore_max_sample,
        chatcore_max_new_tokens=chatcore_max_new_tokens,
        chatcore_temperature=chatcore_temperature,
        chatcore_top_k=chatcore_top_k,
        chatcore_batch_size=chatcore_batch_size,
    )


@app.local_entrypoint()
def rl(
    checkpoint: str,
    slug: str = "gsm8k",
    out_dir: str = "",
    run_name: str | None = None,
    wandb_project: str = "tinyGroot-rl",
    gpu_type: str = "H100",
    gpu_count: int = 8,
    num_epochs: int = 1,
    max_steps: int = -1,
    device_batch_size: int = 8,
    examples_per_step: int = 16,
    num_samples: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    eval_suite: str = "gsm8k-passk",
    eval_every: int = 60,
    eval_examples: int = 400,
    chatcore_max_cat: int = -1,
    chatcore_max_sample: int = 24,
    chatcore_max_new_tokens: int = 512,
    chatcore_temperature: float = 0.0,
    chatcore_top_k: int = 50,
    chatcore_batch_size: int = 8,
    save_every: int = 60,
    optimizer: str = "muon",
    wandb: bool = False,
    compile: bool = False,
    fp8: bool = False,
    push_to_hf: bool = False,
    hf_repo_id: str | None = None,
    hf_private: bool = False,
    hf_revision: str | None = None,
    hf_commit_message: str | None = None,
) -> None:
    """Run nanochat-style GSM8K RL from an SFT checkpoint on Modal."""
    gpu_type = gpu_type.upper().replace("_", "-")
    if gpu_type == "A100-80GB":
        gpu_type = "A100"
    if gpu_type != "H100" and fp8:
        raise ValueError("--fp8 is only supported for H100/Hopper in this training path; disable FP8 for A100.")
    rl_function = select_rl_function(gpu_type, gpu_count)
    if rl_function is None:
        raise ValueError("--gpu-type must be A100, A100-80GB, or H100, and --gpu-count must be one of: 1, 2, 4, 8")
    out_dir, run_name = resolve_run_paths("rl", slug, out_dir, run_name, gpu_type=gpu_type, gpu_count=gpu_count)
    print(f"rl run: out_dir={out_dir} run_name={run_name}", flush=True)
    rl_function.remote(
        checkpoint=checkpoint,
        out_dir=out_dir,
        run_name=run_name,
        wandb_project=wandb_project,
        num_epochs=num_epochs,
        max_steps=max_steps,
        device_batch_size=device_batch_size,
        examples_per_step=examples_per_step,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eval_suite=eval_suite,
        eval_every=eval_every,
        eval_examples=eval_examples,
        chatcore_max_cat=chatcore_max_cat,
        chatcore_max_sample=chatcore_max_sample,
        chatcore_max_new_tokens=chatcore_max_new_tokens,
        chatcore_temperature=chatcore_temperature,
        chatcore_top_k=chatcore_top_k,
        chatcore_batch_size=chatcore_batch_size,
        save_every=save_every,
        optimizer=optimizer,
        wandb=wandb,
        compile=compile,
        fp8=fp8,
    )
    if push_to_hf:
        if not hf_repo_id:
            raise ValueError("--push-to-hf requires --hf-repo-id")
        upload_checkpoint_to_hf.remote(
            checkpoint_dir=out_dir,
            repo_id=hf_repo_id,
            private=hf_private,
            revision=hf_revision,
            commit_message=hf_commit_message,
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
        model = os.path.join(run_dir, "model.pt")
        meta = os.path.join(run_dir, "meta.json")
        if os.path.exists(ckpt):
            entry["has_checkpoint"] = True
            entry["size_mb"] = round(os.path.getsize(ckpt) / 1e6, 2)
        elif os.path.exists(model) and os.path.exists(meta):
            entry["has_checkpoint"] = True
            entry["size_mb"] = round(os.path.getsize(model) / 1e6, 2)
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

        modal run tinygroot/modal/modal_train.py::list_runs
    """
    entries = _list_runs_remote.remote()
    if not entries:
        print("[list_runs] no runs found on the runs volume.")
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
            f"  -> Run 'modal run tinygroot/modal/modal_train.py::list_runs' to find the exact /runs path."
        )
    if drafter_checkpoint is not None and not os.path.exists(drafter_checkpoint):
        raise SystemExit(
            f"[modal_train] drafter checkpoint not found inside the container: {drafter_checkpoint}\n"
            f"  -> Run 'modal run tinygroot/modal/modal_train.py::list_runs' to find the exact /runs path."
        )
    command = [
        "python",
        "-m",
        "tinygroot.infer.spec_decode",
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

        modal run tinygroot/modal/modal_train.py::spec \\
            --checkpoint /runs/tinygroot-mtp1-relu2/checkpoint.pt \\
            --prompt "The capital of France is" --gen-length 64

    DFlash (target + DFlash drafter checkpoint)::

        modal run tinygroot/modal/modal_train.py::spec \\
            --checkpoint /runs/tinygroot-mtp1-relu2/checkpoint.pt \\
            --drafter-checkpoint /runs/tinygroot-dflash-drafter-.../checkpoint.pt \\
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
