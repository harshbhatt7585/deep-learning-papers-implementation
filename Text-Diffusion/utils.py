from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import TextDiffusionModel
from tokenizer import LLaDA21Tokenizer, SimpleCharTokenizer


NANOCHAT_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
NANOCHAT_MAX_SHARD = 6542
Tokenizer = LLaDA21Tokenizer | SimpleCharTokenizer


@dataclass
class Runtime:
    device: torch.device
    local_rank: int
    non_blocking: bool


@dataclass
class TokenData:
    tokenizer: Tokenizer
    train_text: str
    val_text: str | None
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return rank() == 0


def log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def setup_distributed() -> int:
    if "RANK" not in os.environ:
        return 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_runtime(args: argparse.Namespace) -> Runtime:
    local_rank = setup_distributed()
    configure_cuda()
    device = pick_device(local_rank)
    torch.manual_seed(args.seed + rank())
    return Runtime(
        device=device,
        local_rank=local_rank,
        non_blocking=args.pin_memory and device.type == "cuda",
    )


def autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def unwrap_model(model: torch.nn.Module) -> TextDiffusionModel:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def args_as_plain_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def init_wandb(args: argparse.Namespace, config, runtime: Runtime):
    if not args.wandb or not is_main_process():
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or remove --wandb.") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.out_dir.name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        dir=args.wandb_dir,
        config={
            **args_as_plain_dict(args),
            "device": str(runtime.device),
            "world_size": world_size(),
            "model": asdict(config),
        },
    )


def learning_rate(step: int, *, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    runtime: Runtime,
) -> torch.Tensor:
    starts = torch.randint(0, data.numel() - seq_len, (batch_size,))
    batch = torch.stack([data[start:start + seq_len] for start in starts])
    return batch.to(runtime.device, non_blocking=runtime.non_blocking)


def nanochat_shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_nanochat_shard(index: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / nanochat_shard_name(index)
    if path.exists():
        return path

    url = f"{NANOCHAT_BASE_URL}/{path.name}"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    log(f"downloading {url}")

    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with tmp_path.open("wb") as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
            tmp_path.replace(path)
            return path
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 5:
                raise RuntimeError(f"failed to download {path.name}") from exc
            wait_s = 2 ** attempt
            log(f"download failed for {path.name}, retrying in {wait_s}s: {exc}")
            time.sleep(wait_s)

    return path


def read_parquet_text(path: Path, *, max_chars: int | None) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading nanochat parquet shards requires: pip install pyarrow") from exc

    pieces: list[str] = []
    total_chars = 0
    parquet_file = pq.ParquetFile(path)
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=["text"])
        for text in table.column("text").to_pylist():
            if max_chars is not None and total_chars >= max_chars:
                return "\n".join(pieces)
            if max_chars is not None:
                text = text[: max_chars - total_chars]
            pieces.append(text)
            total_chars += len(text)
    return "\n".join(pieces)


def load_nanochat_text(args: argparse.Namespace) -> tuple[str, str]:
    train_pieces: list[str] = []
    remaining_chars = args.max_train_chars

    for shard_index in range(args.nanochat_train_shards):
        path = download_nanochat_shard(shard_index, args.nanochat_cache_dir)
        text = read_parquet_text(path, max_chars=remaining_chars)
        train_pieces.append(text)
        if remaining_chars is not None:
            remaining_chars -= len(text)
            if remaining_chars <= 0:
                break

    val_path = download_nanochat_shard(NANOCHAT_MAX_SHARD, args.nanochat_cache_dir)
    val_text = read_parquet_text(val_path, max_chars=args.max_val_chars)
    return "\n".join(train_pieces), val_text


def load_raw_text(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.nanochat:
        return load_nanochat_text(args)
    if args.data is None:
        raise ValueError("Pass either --data path.txt or --nanochat.")
    return args.data.read_text(), None


def build_tokenizer(args: argparse.Namespace, train_text: str, val_text: str | None) -> Tokenizer:
    if args.tokenizer == "llada21":
        return LLaDA21Tokenizer.from_pretrained(
            local_files_only=args.tokenizer_local_files_only,
        )
    return SimpleCharTokenizer.from_texts([train_text, val_text or "", args.sample_prompt])


def tokenize_data(args: argparse.Namespace, runtime: Runtime) -> TokenData:
    train_text, val_text = load_raw_text(args)
    tokenizer = build_tokenizer(args, train_text, val_text)
    train_ids = torch.tensor(tokenizer.encode(train_text, add_eos=True), dtype=torch.long)

    if val_text is None:
        split = int(0.95 * train_ids.numel())
        train_tokens = train_ids[:split]
        val_tokens = train_ids[split:]
    else:
        train_tokens = train_ids
        val_tokens = torch.tensor(tokenizer.encode(val_text, add_eos=True), dtype=torch.long)

    if train_tokens.numel() <= args.seq_len or val_tokens.numel() <= args.seq_len:
        raise ValueError("Dataset is too small for --seq-len. Use more text or lower --seq-len.")

    if runtime.non_blocking:
        train_tokens = train_tokens.pin_memory()
        val_tokens = val_tokens.pin_memory()

    return TokenData(
        tokenizer=tokenizer,
        train_text=train_text,
        val_text=val_text,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )


def save_checkpoint(
    *,
    out_dir: Path,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_model = unwrap_model(model)

    tokenizer_path = out_dir / ("tokenizer_hf" if isinstance(tokenizer, LLaDA21Tokenizer) else "tokenizer.json")
    tokenizer.save(tokenizer_path)
    torch.save(
        {
            "step": step,
            "config": asdict(source_model.config),
            "model_state": source_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_type": "llada21" if isinstance(tokenizer, LLaDA21Tokenizer) else "char",
            "args": args_as_plain_dict(args),
        },
        out_dir / "checkpoint.pt",
    )
