from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "tinygroot-mmlu-rl"
WORKDIR = Path("/workspace")

data_volume = modal.Volume.from_name("text-diffusion-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("text-diffusion-runs", create_if_missing=True)


def args_to_argv(mapping: dict) -> list[str]:
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


def ignore_local_source(path: Path) -> bool:
    ignored_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "assets",
        "data",
        "runs",
        "exercise",
        ".codex",
        "nanochat",
        "local_chat_deploy",
    }
    if any(part in ignored_dirs for part in path.parts):
        return True
    if path.is_dir():
        return False
    if path.name in {"pyproject.toml", "uv.lock"}:
        return False
    return path.suffix not in {".py", ".sh"}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(frozen=False)
    .add_local_dir(".", remote_path=str(WORKDIR), ignore=ignore_local_source)
)

app = modal.App(APP_NAME)


def hf_secrets() -> list[modal.Secret]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return [modal.Secret.from_dict({"HF_TOKEN": token})]
    secret_name = os.environ.get("MODAL_HF_SECRET_NAME")
    if secret_name:
        return [modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])]
    return []


def wandb_secrets() -> list[modal.Secret]:
    if os.environ.get("WANDB_API_KEY"):
        return [modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})]
    secret_name = os.environ.get("MODAL_WANDB_SECRET_NAME", "wandb")
    return [modal.Secret.from_name(secret_name, required_keys=["WANDB_API_KEY"])]


def add_workspace_pythonpath(env: dict[str, str]) -> None:
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{WORKDIR}:{pythonpath}" if pythonpath else str(WORKDIR)


def _gpu_spec(gpu_type: str, gpu_count: int) -> str:
    base = "A100-80GB" if gpu_type == "A100" else gpu_type
    return f"{base}:{gpu_count}" if gpu_count > 1 else base


def run_mmlu_rl(
    *,
    gpu_count: int,
    checkpoint: str | None = None,
    hf_checkpoint_repo_id: str | None = "harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000",
    hf_checkpoint_revision: str | None = None,
    hf_checkpoint_cache_dir: str = "/runs/hf_checkpoints",
    out_dir: str = "/runs/hrm-loop-mmlu-rl",
    run_name: str | None = None,
    train_split: str = "auxiliary_train",
    eval_split: str = "test",
    train_stop: int = -1,
    eval_examples: int = 1000,
    num_epochs: int = 1,
    max_steps: int = 500,
    device_batch_size: int = 64,
    examples_per_step: int = 64,
    num_samples: int = 16,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
    eval_every: int = 50,
    save_every: int = 50,
    log_rollouts_every: int = 5,
    log_rollout_samples: int = 4,
    log_rollout_chars: int = 240,
    stream_rollouts: bool = False,
    optimizer: str = "muon",
    lr: float = 3e-4,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    matrix_lr: float = 0.02,
    scalar_lr: float = 0.5,
    weight_decay: float = 0.0,
    init_lr_frac: float = 0.05,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    amp_dtype: str = "bfloat16",
    compile: bool = False,
    compile_mode: str = "default",
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    seed: int = 0,
    pin_memory: bool = True,
    wandb: bool = True,
    wandb_project: str = "tinyGroot-mmlu-rl",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    push_to_hf: bool = False,
    hf_repo_id: str | None = None,
    hf_private: bool = False,
    hf_revision: str | None = None,
    hf_commit_message: str | None = None,
) -> None:
    if checkpoint is None and hf_checkpoint_repo_id is None:
        raise ValueError("Pass either checkpoint or hf_checkpoint_repo_id.")

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "rl_envs/mmlu_rl.py",
        *args_to_argv(
            {
                "checkpoint": checkpoint,
                "hf_checkpoint_repo_id": hf_checkpoint_repo_id,
                "hf_checkpoint_revision": hf_checkpoint_revision,
                "hf_checkpoint_cache_dir": hf_checkpoint_cache_dir,
                "out_dir": out_dir,
                "run_name": run_name,
                "train_split": train_split,
                "eval_split": eval_split,
                "train_stop": train_stop,
                "eval_examples": eval_examples,
                "num_epochs": num_epochs,
                "max_steps": max_steps,
                "device_batch_size": device_batch_size,
                "examples_per_step": examples_per_step,
                "num_samples": num_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "eval_every": eval_every,
                "save_every": save_every,
                "log_rollouts_every": log_rollouts_every,
                "log_rollout_samples": log_rollout_samples,
                "log_rollout_chars": log_rollout_chars,
                "stream_rollouts": stream_rollouts,
                "optimizer": optimizer,
                "lr": lr,
                "embedding_lr": embedding_lr,
                "unembedding_lr": unembedding_lr,
                "matrix_lr": matrix_lr,
                "scalar_lr": scalar_lr,
                "weight_decay": weight_decay,
                "init_lr_frac": init_lr_frac,
                "muon_momentum": muon_momentum,
                "muon_ns_steps": muon_ns_steps,
                "amp_dtype": amp_dtype,
                "compile": compile,
                "compile_mode": compile_mode,
                "fp8": fp8,
                "fp8_recipe": fp8_recipe,
                "seed": seed,
                "pin_memory": pin_memory,
                "wandb": wandb,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "wandb_group": wandb_group,
                "push_to_hf": push_to_hf,
                "hf_repo_id": hf_repo_id,
                "hf_private": hf_private,
                "hf_revision": hf_revision,
                "hf_commit_message": hf_commit_message,
            }
        ),
    ]
    env = dict(os.environ)
    add_workspace_pythonpath(env)
    try:
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        runs_volume.commit()


_DAY = 24 * 60 * 60
_GPU_MATRIX = [("A100", n) for n in (1, 2, 4, 8)] + [("H100", n) for n in (1, 2, 4, 8)]
MMLU_RL_FUNCTIONS: dict[tuple[str, int], modal.Function] = {}


def _make_gpu_fn(gpu_count: int, name: str):
    def fn(**kwargs):
        run_mmlu_rl(gpu_count=gpu_count, **kwargs)

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__module__ = __name__
    return fn


for _gpu_type, _gpu_count in _GPU_MATRIX:
    _fn_name = f"mmlu_rl_{_gpu_type.lower()}_{_gpu_count}gpu"
    _fn = app.function(
        image=image,
        gpu=_gpu_spec(_gpu_type, _gpu_count),
        name=_fn_name,
        timeout=_DAY,
        secrets=[*wandb_secrets(), *hf_secrets()],
        volumes={"/data": data_volume, "/runs": runs_volume},
    )(_make_gpu_fn(_gpu_count, _fn_name))
    globals()[_fn_name] = _fn
    MMLU_RL_FUNCTIONS[(_gpu_type, _gpu_count)] = _fn


def select_mmlu_rl_function(gpu_type: str, gpu_count: int):
    return MMLU_RL_FUNCTIONS.get((gpu_type, gpu_count))


@app.local_entrypoint()
def main(
    gpu_type: str = "H100",
    gpu_count: int = 8,
    checkpoint: str | None = None,
    hf_checkpoint_repo_id: str | None = "harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000",
    hf_checkpoint_revision: str | None = None,
    hf_checkpoint_cache_dir: str = "/runs/hf_checkpoints",
    out_dir: str = "/runs/hrm-loop-mmlu-rl",
    run_name: str | None = None,
    train_split: str = "auxiliary_train",
    eval_split: str = "test",
    train_stop: int = -1,
    eval_examples: int = 1000,
    num_epochs: int = 1,
    max_steps: int = 500,
    device_batch_size: int = 64,
    examples_per_step: int = 64,
    num_samples: int = 16,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
    eval_every: int = 50,
    save_every: int = 50,
    log_rollouts_every: int = 5,
    log_rollout_samples: int = 4,
    log_rollout_chars: int = 240,
    stream_rollouts: bool = False,
    optimizer: str = "muon",
    lr: float = 3e-4,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    matrix_lr: float = 0.02,
    scalar_lr: float = 0.5,
    weight_decay: float = 0.0,
    init_lr_frac: float = 0.05,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    amp_dtype: str = "bfloat16",
    compile: bool = False,
    compile_mode: str = "default",
    fp8: bool = False,
    fp8_recipe: str = "tensorwise",
    seed: int = 0,
    pin_memory: bool = True,
    wandb: bool = True,
    wandb_project: str = "tinyGroot-mmlu-rl",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    push_to_hf: bool = False,
    hf_repo_id: str | None = None,
    hf_private: bool = False,
    hf_revision: str | None = None,
    hf_commit_message: str | None = None,
) -> None:
    gpu_type = gpu_type.upper().replace("_", "-")
    if gpu_type == "A100-80GB":
        gpu_type = "A100"
    if gpu_type != "H100" and fp8:
        raise ValueError("--fp8 is only supported for H100/Hopper; disable FP8 for A100.")

    train_fn = select_mmlu_rl_function(gpu_type, gpu_count)
    if train_fn is None:
        raise ValueError("--gpu-type must be A100, A100-80GB, or H100, and --gpu-count must be one of: 1, 2, 4, 8")
    print(f"modal_gpu_request: {_gpu_spec(gpu_type, gpu_count)}  out_dir: {out_dir}", flush=True)
    train_fn.remote(
        checkpoint=checkpoint,
        hf_checkpoint_repo_id=hf_checkpoint_repo_id,
        hf_checkpoint_revision=hf_checkpoint_revision,
        hf_checkpoint_cache_dir=hf_checkpoint_cache_dir,
        out_dir=out_dir,
        run_name=run_name,
        train_split=train_split,
        eval_split=eval_split,
        train_stop=train_stop,
        eval_examples=eval_examples,
        num_epochs=num_epochs,
        max_steps=max_steps,
        device_batch_size=device_batch_size,
        examples_per_step=examples_per_step,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eval_every=eval_every,
        save_every=save_every,
        log_rollouts_every=log_rollouts_every,
        log_rollout_samples=log_rollout_samples,
        log_rollout_chars=log_rollout_chars,
        stream_rollouts=stream_rollouts,
        optimizer=optimizer,
        lr=lr,
        embedding_lr=embedding_lr,
        unembedding_lr=unembedding_lr,
        matrix_lr=matrix_lr,
        scalar_lr=scalar_lr,
        weight_decay=weight_decay,
        init_lr_frac=init_lr_frac,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        amp_dtype=amp_dtype,
        compile=compile,
        compile_mode=compile_mode,
        fp8=fp8,
        fp8_recipe=fp8_recipe,
        seed=seed,
        pin_memory=pin_memory,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        push_to_hf=push_to_hf,
        hf_repo_id=hf_repo_id,
        hf_private=hf_private,
        hf_revision=hf_revision,
        hf_commit_message=hf_commit_message,
    )
