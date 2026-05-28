from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tinygroot.model import TinyGrootModel
from tinygroot.tokenizer import NanochatTokenizer


NANOCHAT_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
NANOCHAT_MAX_SHARD = 6542
Tokenizer = NanochatTokenizer


def iter_text_documents(text: str):
    start = 0
    while start < len(text):
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        line = text[start:end]
        if line.endswith("\r"):
            line = line[:-1]
        if line:
            yield line
        start = end + 1


@dataclass
class TokenStore:
    shards: list[torch.Tensor]
    names: list[str]
    sample_weights: torch.Tensor
    total_tokens: int

    def numel(self) -> int:
        return self.total_tokens


class StreamingNanochatTokenStore:
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        cache_dir: Path,
        train_shards: int,
        seq_len: int,
        batch_size: int,
        runtime: Runtime,
        tokenizer_batch_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.train_shards = train_shards
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.runtime = runtime
        self.tokenizer_batch_size = tokenizer_batch_size
        self.shard_indices = list(range(rank(), train_shards, world_size()))
        if not self.shard_indices:
            raise ValueError(f"rank {rank()} has no nanochat shards assigned")
        self.position = 0
        self.current_tokens = torch.empty(0, dtype=torch.long)
        self.current_name = ""
        self.batch_calls = 0
        self.batches_per_shard = 1
        self._load_current_shard()

    def numel(self) -> int:
        return int(self.current_tokens.numel())

    def _encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode_documents(
            iter_text_documents(text),
            add_bos=True,
            add_eos=False,
            batch_size=self.tokenizer_batch_size,
        )
        if not ids:
            ids = self.tokenizer.encode(text, add_eos=True)
        return torch.tensor(ids, dtype=torch.long)

    def _load_current_shard(self) -> None:
        attempts = 0
        while attempts < len(self.shard_indices):
            shard_index = self.shard_indices[self.position % len(self.shard_indices)]
            path = download_nanochat_shard(shard_index, self.cache_dir)
            log(f"streaming train shard rank={rank()} shard={path.name}")
            text = read_parquet_text(path, max_chars=None)
            tokens = self._encode_text(text)
            self.position += 1
            attempts += 1
            if tokens.numel() > self.seq_len:
                self.current_tokens = tokens
                self.current_name = path.name
                self.batch_calls = 0
                self.batches_per_shard = max(1, tokens.numel() // max(1, self.batch_size * self.seq_len))
                log(
                    f"streamed {path.name}: {tokens.numel():,} tokens "
                    f"batches_per_shard={self.batches_per_shard}"
                )
                return
            log(f"skipping streamed shard shorter than seq_len: {path.name}")
        raise ValueError("No streamed nanochat shard is long enough for --seq-len.")

    def get_batch(self, *, batch_size: int, seq_len: int, runtime: Runtime) -> torch.Tensor:
        if self.batch_calls >= self.batches_per_shard:
            self._load_current_shard()
        self.batch_calls += 1
        starts = torch.randint(0, self.current_tokens.numel() - seq_len + 1, (batch_size,))
        batch = torch.stack([self.current_tokens[start:start + seq_len] for start in starts])
        return batch.to(runtime.device, non_blocking=runtime.non_blocking)


TokenSource = torch.Tensor | TokenStore | StreamingNanochatTokenStore


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
    train_tokens: TokenSource
    val_tokens: TokenSource
    train_chars: int | None = None
    val_chars: int | None = None


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


def barrier(local_rank: int | None = None) -> None:
    if not is_dist():
        return
    if torch.cuda.is_available():
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def setup_distributed() -> int:
    if "RANK" not in os.environ:
        return 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, device_id=torch.device("cuda", local_rank))
    else:
        dist.init_process_group(backend=backend)
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


def unwrap_model(model: torch.nn.Module) -> TinyGrootModel:
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

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
        group["lr"] = lr * group.get("lr_multiplier", 1.0)


def get_batch(
    data: TokenSource,
    *,
    batch_size: int,
    seq_len: int,
    runtime: Runtime,
) -> torch.Tensor:
    if isinstance(data, StreamingNanochatTokenStore):
        return data.get_batch(batch_size=batch_size, seq_len=seq_len, runtime=runtime)

    if isinstance(data, TokenStore):
        shard_ids = torch.multinomial(data.sample_weights, batch_size, replacement=True)
        rows = []
        for shard_id in shard_ids.tolist():
            shard = data.shards[shard_id]
            start = torch.randint(0, shard.numel() - seq_len + 1, (1,)).item()
            rows.append(shard[start:start + seq_len])
        batch = torch.stack(rows).to(dtype=torch.long)
        return batch.to(runtime.device, non_blocking=False)

    starts = torch.randint(0, data.numel() - seq_len + 1, (batch_size,))
    batch = torch.stack([data[start:start + seq_len] for start in starts]).to(dtype=torch.long)
    return batch.to(runtime.device, non_blocking=runtime.non_blocking)


def nanochat_shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_nanochat_shard(index: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / nanochat_shard_name(index)
    if path.exists():
        log(f"using cached {path.name}")
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
        shard_max_chars = remaining_chars
        log(f"reading train shard {shard_index + 1}/{args.nanochat_train_shards}: {path.name}")
        text = read_parquet_text(path, max_chars=remaining_chars)
        train_pieces.append(text)
        log(f"loaded train shard {path.name}: {len(text):,} chars")
        if remaining_chars is not None:
            remaining_chars -= len(text)
            if remaining_chars <= 0:
                log(f"reached max_train_chars={args.max_train_chars:,}")
                break
        if shard_max_chars is not None and len(text) < shard_max_chars:
            log(f"shard {path.name} ended before max char cap")

    val_path = download_nanochat_shard(NANOCHAT_MAX_SHARD, args.nanochat_cache_dir)
    log(f"reading val shard: {val_path.name}")
    val_text = read_parquet_text(val_path, max_chars=args.max_val_chars)
    train_text = "\n".join(train_pieces)
    log(f"loaded nanochat train text: {len(train_text):,} chars")
    log(f"loaded nanochat val text: {len(val_text):,} chars")
    return train_text, val_text


def load_raw_text(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.nanochat:
        return load_nanochat_text(args)
    if args.data is None:
        raise ValueError("Pass either --data path.txt or --nanochat.")
    return args.data.read_text(), None


def build_tokenizer(args: argparse.Namespace, train_text: str, val_text: str | None) -> Tokenizer:
    if args.tokenizer == "nanochat":
        if is_dist():
            tokenizer = None
            if rank() == 0:
                tokenizer = NanochatTokenizer.from_pretrained(
                    args.nanochat_tokenizer_cache_dir,
                    train_text=train_text,
                    vocab_size=args.nanochat_tokenizer_vocab_size,
                    train_chars=args.nanochat_tokenizer_train_chars,
                    doc_cap=args.nanochat_tokenizer_doc_cap,
                    local_files_only=args.tokenizer_local_files_only,
                )
            barrier()
            if tokenizer is None:
                tokenizer = NanochatTokenizer.from_pretrained(
                    args.nanochat_tokenizer_cache_dir,
                    local_files_only=True,
                )
            return tokenizer
        return NanochatTokenizer.from_pretrained(
            args.nanochat_tokenizer_cache_dir,
            train_text=train_text,
            vocab_size=args.nanochat_tokenizer_vocab_size,
            train_chars=args.nanochat_tokenizer_train_chars,
            doc_cap=args.nanochat_tokenizer_doc_cap,
            local_files_only=args.tokenizer_local_files_only,
        )
    raise ValueError(f"unsupported tokenizer {args.tokenizer!r}; only nanochat is supported")


def configure_tokenizer_parallelism(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.nanochat_tokenizer_threads))
    log(
        "tokenizer parallelism: "
        f"TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']} "
        f"RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']} "
        f"batch_size={args.nanochat_tokenizer_batch_size}"
    )


def load_token_array(path: Path) -> torch.Tensor:
    array = np.load(path, mmap_mode="r+")
    return torch.from_numpy(array)


def load_token_store(paths: list[Path], *, seq_len: int) -> TokenStore:
    shards = []
    names = []
    weights = []
    total_tokens = 0

    for path in paths:
        shard = load_token_array(path)
        if shard.numel() <= seq_len:
            log(f"skipping token shard shorter than seq_len: {path.name}")
            continue
        shards.append(shard)
        names.append(path.name)
        weights.append(float(shard.numel() - seq_len + 1))
        total_tokens += int(shard.numel())

    if not shards:
        raise ValueError(f"No token shards in {paths[0].parent if paths else '<empty>'} are long enough for --seq-len.")

    sample_weights = torch.tensor(weights, dtype=torch.float64)
    sample_weights /= sample_weights.sum()
    return TokenStore(
        shards=shards,
        names=names,
        sample_weights=sample_weights,
        total_tokens=total_tokens,
    )


def load_pretokenized_data(args: argparse.Namespace) -> TokenData:
    token_dir = args.token_shards_dir
    metadata_path = token_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing pretokenized metadata: {metadata_path}. Run pretokenization first.")

    metadata = json.loads(metadata_path.read_text())
    train_paths = sorted(token_dir.glob("train_*.npy"))
    val_path = token_dir / "val.npy"
    if not train_paths:
        raise FileNotFoundError(f"No train_*.npy token shards found in {token_dir}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation token shard: {val_path}")

    tokenizer = NanochatTokenizer.load(args.nanochat_tokenizer_cache_dir)
    train_tokens = load_token_store(train_paths, seq_len=args.seq_len)
    val_tokens = load_token_array(val_path)
    if val_tokens.numel() <= args.seq_len:
        raise ValueError("Pretokenized validation data is too small for --seq-len.")

    log(f"loaded pretokenized train shards: {len(train_tokens.shards)}")
    log(f"loaded pretokenized val shard: {val_path.name}")
    return TokenData(
        tokenizer=tokenizer,
        train_text="",
        val_text="",
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_chars=metadata.get("train_chars"),
        val_chars=metadata.get("val_chars"),
    )


def encode_text_documents(tokenizer: Tokenizer, text: str, *, batch_size: int) -> torch.Tensor:
    ids = tokenizer.encode_documents(
        iter_text_documents(text),
        add_bos=True,
        add_eos=False,
        batch_size=batch_size,
    )
    if not ids:
        ids = tokenizer.encode(text, add_eos=True)
    return torch.tensor(ids, dtype=torch.long)


def load_streaming_nanochat_data(args: argparse.Namespace, runtime: Runtime) -> TokenData:
    configure_tokenizer_parallelism(args)
    tokenizer = NanochatTokenizer.load(args.nanochat_tokenizer_cache_dir)
    train_tokens = StreamingNanochatTokenStore(
        tokenizer=tokenizer,
        cache_dir=args.nanochat_cache_dir,
        train_shards=args.nanochat_train_shards,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        runtime=runtime,
        tokenizer_batch_size=args.nanochat_tokenizer_batch_size,
    )

    val_path = download_nanochat_shard(NANOCHAT_MAX_SHARD, args.nanochat_cache_dir)
    log(f"reading streaming val shard: {val_path.name}")
    val_text = read_parquet_text(val_path, max_chars=args.max_val_chars)
    val_tokens = encode_text_documents(
        tokenizer,
        val_text,
        batch_size=args.nanochat_tokenizer_batch_size,
    )
    if val_tokens.numel() <= args.seq_len:
        raise ValueError("Streaming validation data is too small for --seq-len.")
    if runtime.non_blocking:
        val_tokens = val_tokens.pin_memory()

    return TokenData(
        tokenizer=tokenizer,
        train_text="",
        val_text=val_text,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_chars=None,
        val_chars=len(val_text),
    )


def tokenize_data(args: argparse.Namespace, runtime: Runtime) -> TokenData:
    if getattr(args, "stream_nanochat", False):
        return load_streaming_nanochat_data(args, runtime)
    if args.token_shards_dir is not None:
        return load_pretokenized_data(args)

    train_text, val_text = load_raw_text(args)
    log(f"loading tokenizer: {args.tokenizer}")
    tokenizer = build_tokenizer(args, train_text, val_text)
    log("tokenizing train text")
    train_ids = torch.tensor(tokenizer.encode(train_text, add_eos=True), dtype=torch.long)
    log(f"tokenized train text: {train_ids.numel():,} tokens")

    if val_text is None:
        split = int(0.95 * train_ids.numel())
        train_tokens = train_ids[:split]
        val_tokens = train_ids[split:]
    else:
        train_tokens = train_ids
        log("tokenizing val text")
        val_tokens = torch.tensor(tokenizer.encode(val_text, add_eos=True), dtype=torch.long)
        log(f"tokenized val text: {val_tokens.numel():,} tokens")

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
        train_chars=len(train_text),
        val_chars=len(val_text) if val_text is not None else None,
    )


def save_checkpoint(
    *,
    out_dir: Path,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    step: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_model = unwrap_model(model)

    tokenizer.save(out_dir / "tokenizer_hf")
    torch.save(
        {
            "step": step,
            "config": asdict(source_model.config),
            "model_state": source_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "tokenizer_type": tokenizer.tokenizer_type,
            "args": args_as_plain_dict(args),
        },
        out_dir / "checkpoint.pt",
    )
