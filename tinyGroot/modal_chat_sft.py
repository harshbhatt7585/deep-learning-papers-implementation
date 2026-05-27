from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "text-diffusion-chat-sft"
WORKDIR = Path("/workspace")

data_volume = modal.Volume.from_name("text-diffusion-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("text-diffusion-runs", create_if_missing=True)

PROJECT_FILES = [
    "chat_core_eval.py",
    "chat_sft.py",
    "core_eval.py",
    "eval_core.py",
    "flash_attention.py",
    "fp8.py",
    "model.py",
    "nanochat_optim.py",
    "sft_chat.py",
    "sft_data.py",
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
        "assets",
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


def run_sft(
    *,
    gpu_count: int,
    checkpoint: str,
    out_dir: str,
    run_name: str,
    wandb_project: str,
    max_steps: int,
    device_batch_size: int,
    total_batch_size: int,
    seq_len: int | None,
    eval_every: int,
    eval_tokens: int,
    chatcore_every: int,
    chatcore_max_cat: int,
    chatcore_max_sample: int,
    sample_every: int,
    sample_length: int,
    optimizer: str,
    fp8: bool,
    compile: bool,
    train_mtp_heads: bool,
    identity_jsonl: str,
    mmlu_epochs: int,
    gsm8k_epochs: int,
    simple_spelling_size: int,
    spellingbee_size: int,
) -> None:
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        "--",
        str(WORKDIR / "chat_sft.py"),
        "--checkpoint",
        checkpoint,
        "--out-dir",
        out_dir,
        "--run",
        run_name,
        "--wandb",
        "--wandb-project",
        wandb_project,
        "--device-batch-size",
        str(device_batch_size),
        "--total-batch-size",
        str(total_batch_size),
        "--max-steps",
        str(max_steps),
        "--eval-every",
        str(eval_every),
        "--eval-tokens",
        str(eval_tokens),
        "--chatcore-every",
        str(chatcore_every),
        "--chatcore-max-cat",
        str(chatcore_max_cat),
        "--chatcore-max-sample",
        str(chatcore_max_sample),
        "--sample-every",
        str(sample_every),
        "--sample-length",
        str(sample_length),
        "--optimizer",
        optimizer,
        "--identity-jsonl",
        identity_jsonl,
        "--mmlu-epochs",
        str(mmlu_epochs),
        "--gsm8k-epochs",
        str(gsm8k_epochs),
        "--simple-spelling-size",
        str(simple_spelling_size),
        "--spellingbee-size",
        str(spellingbee_size),
        "--words-path",
        "/data/words_alpha.txt",
    ]
    if seq_len is not None:
        command.extend(["--seq-len", str(seq_len)])
    if fp8:
        command.append("--fp8")
    if compile:
        command.append("--compile")
    command.append("--train-mtp-heads" if train_mtp_heads else "--no-train-mtp-heads")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if compile:
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    try:
        subprocess.run(command, cwd=WORKDIR, env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        data_volume.commit()
        runs_volume.commit()


@app.function(
    image=image,
    gpu="H100:8",
    cpu=64,
    memory=262144,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    volumes={"/data": data_volume, "/runs": runs_volume},
)
def sft_h100_8gpu(**kwargs) -> None:
    run_sft(gpu_count=8, **kwargs)


@app.function(
    image=image,
    gpu="H100:4",
    cpu=64,
    memory=262144,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    volumes={"/data": data_volume, "/runs": runs_volume},
)
def sft_h100_4gpu(**kwargs) -> None:
    run_sft(gpu_count=4, **kwargs)


@app.function(
    image=image,
    gpu="A100-80GB:8",
    cpu=64,
    memory=262144,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    volumes={"/data": data_volume, "/runs": runs_volume},
)
def sft_a100_8gpu(**kwargs) -> None:
    run_sft(gpu_count=8, **kwargs)


@app.local_entrypoint()
def main(
    gpu_type: str = "H100",
    gpu_count: int = 8,
    checkpoint: str = "/runs/mtp2-tst-s4-r03-d12-swiglu-ff3-h100-fp8-compile-8gpu/checkpoint.pt",
    out_dir: str = "/runs/sft-rank2-pretrain-shared-mtp2-tst-20260527",
    run_name: str = "sft-rank2-pretrain-shared-mtp2-tst-20260527",
    wandb_project: str = "tinyGroot-sft",
    max_steps: int = -1,
    device_batch_size: int = 16,
    total_batch_size: int = 524_288,
    seq_len: int | None = None,
    eval_every: int = 200,
    eval_tokens: int = 2_097_152,
    chatcore_every: int = 200,
    chatcore_max_cat: int = -1,
    chatcore_max_sample: int = 24,
    sample_every: int = 200,
    sample_length: int = 128,
    optimizer: str = "muon",
    fp8: bool = False,
    compile: bool = False,
    train_mtp_heads: bool = True,
    identity_jsonl: str = "/data/identity_conversations.jsonl",
    mmlu_epochs: int = 3,
    gsm8k_epochs: int = 4,
    simple_spelling_size: int = 200_000,
    spellingbee_size: int = 80_000,
) -> None:
    gpu_type = gpu_type.upper().replace("_", "-")
    if (gpu_type, gpu_count) == ("H100", 8):
        fn = sft_h100_8gpu
    elif (gpu_type, gpu_count) == ("H100", 4):
        fn = sft_h100_4gpu
    elif gpu_type in {"A100", "A100-80GB"} and gpu_count == 8:
        fn = sft_a100_8gpu
    else:
        raise ValueError("Supported SFT Modal shapes: H100x8, H100x4, A100-80GBx8")

    fn.remote(
        checkpoint=checkpoint,
        out_dir=out_dir,
        run_name=run_name,
        wandb_project=wandb_project,
        max_steps=max_steps,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        seq_len=seq_len,
        eval_every=eval_every,
        eval_tokens=eval_tokens,
        chatcore_every=chatcore_every,
        chatcore_max_cat=chatcore_max_cat,
        chatcore_max_sample=chatcore_max_sample,
        sample_every=sample_every,
        sample_length=sample_length,
        optimizer=optimizer,
        fp8=fp8,
        compile=compile,
        train_mtp_heads=train_mtp_heads,
        identity_jsonl=identity_jsonl,
        mmlu_epochs=mmlu_epochs,
        gsm8k_epochs=gsm8k_epochs,
        simple_spelling_size=simple_spelling_size,
        spellingbee_size=spellingbee_size,
    )
