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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import (
    TextDiffusionConfig,
    TextDiffusionModel,
    diffusion_loss,
    generate,
    make_masked_inputs,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny text diffusion model.")

    parser.add_argument("--data", type=Path, help="Plain text dataset path.")
    parser.add_argument("--nanochat", action="store_true", help="Use nanochat's ClimbMix parquet dataset.")
    parser.add_argument("--nanochat-cache-dir", type=Path, default=Path("data/nanochat_climbmix"))
    parser.add_argument("--nanochat-train-shards", type=int, default=1)
    parser.add_argument("--max-train-chars", type=int, default=5_000_000)
    parser.add_argument("--max-val-chars", type=int, default=1_000_000)

    parser.add_argument("--tokenizer", choices=["llada21", "char"], default="llada21")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")

    parser.add_argument("--out-dir", type=Path, default=Path("runs/text-diffusion-char"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--mask-prob", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.add_argument("--sample-prompt", type=str, default="The ")
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--sample-block-length", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-threshold", type=float, default=0.5)

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="text-diffusion")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/wandb"))

    return parser.parse_args()


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

    non_blocking = args.pin_memory and device.type == "cuda"
    return Runtime(device=device, local_rank=local_rank, non_blocking=non_blocking)


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


def init_wandb(args: argparse.Namespace, config: TextDiffusionConfig, runtime: Runtime):
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


def build_config(args: argparse.Namespace, tokenizer: Tokenizer) -> TextDiffusionConfig:
    sample_len = len(tokenizer.encode(args.sample_prompt)) + args.sample_length
    max_seq_len = max(args.seq_len, sample_len + args.sample_block_length)
    return TextDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )


def build_model(args: argparse.Namespace, config: TextDiffusionConfig, runtime: Runtime) -> torch.nn.Module:
    model: torch.nn.Module = TextDiffusionModel(config).to(runtime.device)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    return model


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, runtime: Runtime) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
    }
    if args.fused_adamw and runtime.device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def learning_rate(step: int, *, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_eval_metrics(
    model: torch.nn.Module,
    data: torch.Tensor,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, float]:
    model.eval()
    source_model = unwrap_model(model)
    total_loss = 0.0
    total_masked_tokens = 0
    total_bytes = 0

    for _ in range(args.eval_batches):
        batch = get_batch(
            data,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runtime=runtime,
        )
        noised, labels = make_masked_inputs(
            batch,
            mask_token_id=source_model.config.mask_token_id,
            pad_token_id=source_model.config.pad_token_id,
            mask_prob=args.mask_prob,
        )
        attention_mask = noised != source_model.config.pad_token_id
        with autocast_context(runtime.device, args.amp_dtype):
            logits = model(noised, attention_mask=attention_mask)
            loss_sum = F.cross_entropy(
                logits.view(-1, source_model.config.vocab_size),
                labels.view(-1),
                reduction="sum",
            )

        total_loss += float(loss_sum.item())
        total_masked_tokens += int((labels != -100).sum().item())
        total_bytes += sum(
            len(tokenizer.decode(row.detach().cpu().tolist()).encode("utf-8"))
            for row in batch
        )

    model.train()
    totals = torch.tensor(
        [total_loss, total_masked_tokens, total_bytes],
        dtype=torch.float64,
        device=runtime.device,
    )
    if is_dist():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss = float(totals[0].item())
    total_masked_tokens = max(1.0, float(totals[1].item()))
    total_bytes = max(1.0, float(totals[2].item()))
    return {
        "loss": total_loss / total_masked_tokens,
        "masked_bpb": total_loss / (math.log(2) * total_bytes),
        "masked_tokens": total_masked_tokens,
    }


@torch.no_grad()
def sample_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> str:
    source_model = unwrap_model(model)
    prompt_ids = torch.tensor(
        tokenizer.encode(args.sample_prompt),
        dtype=torch.long,
        device=runtime.device,
    )
    output = generate(
        source_model,
        prompt_ids,
        gen_length=args.sample_length,
        block_length=args.sample_block_length,
        steps=args.sample_steps,
        threshold=args.sample_threshold,
        editing_threshold=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    sample = tokenizer.decode(output.detach().cpu())
    log(f"sample: {sample!r}")
    model.train()
    return sample


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


def log_startup(args: argparse.Namespace, data: TokenData, config: TextDiffusionConfig, model: torch.nn.Module, runtime: Runtime) -> None:
    val_chars = len(data.val_text) if data.val_text is not None else len(data.train_text) - int(0.95 * len(data.train_text))
    log(f"device: {runtime.device}")
    log(f"world_size: {world_size()}")
    log(f"data_source: {'nanochat/climbmix-400b-shuffle' if args.nanochat else args.data}")
    log(f"tokenizer: {args.tokenizer}")
    log(f"train chars: {len(data.train_text):,}")
    log(f"val chars: {val_chars:,}")
    log(f"train tokens: {data.train_tokens.numel():,}")
    log(f"val tokens: {data.val_tokens.numel():,}")
    log(f"vocab_size: {config.vocab_size:,}")
    log(f"parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}")
    log(f"tokens_per_step: {args.batch_size * args.seq_len * args.grad_accum_steps * world_size():,}")
    log(f"amp_dtype: {args.amp_dtype}")
    log(f"compile: {args.compile}")


def log_train_metrics(wandb_run, step: int, metrics: dict[str, float]) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def train_one_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_tokens: torch.Tensor,
    args: argparse.Namespace,
    runtime: Runtime,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0

    for micro_step in range(args.grad_accum_steps):
        batch = get_batch(
            train_tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runtime=runtime,
        )
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1
            else nullcontext()
        )
        with sync_context:
            with autocast_context(runtime.device, args.amp_dtype):
                loss = diffusion_loss(model, batch, mask_prob=args.mask_prob)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
        step_loss += loss.detach().item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return step_loss / args.grad_accum_steps


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    data = tokenize_data(args, runtime)
    config = build_config(args, data.tokenizer)
    model = build_model(args, config, runtime)
    optimizer = build_optimizer(args, model, runtime)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(runtime.device.type == "cuda" and args.amp_dtype == "float16"),
    )
    wandb_run = init_wandb(args, config, runtime)

    log_startup(args, data, config, model, runtime)
    if wandb_run is not None:
        wandb_run.summary["parameters"] = sum(p.numel() for p in unwrap_model(model).parameters())
        wandb_run.summary["train_tokens"] = data.train_tokens.numel()
        wandb_run.summary["val_tokens"] = data.val_tokens.numel()

    model.train()
    running_loss = 0.0
    last_log_time = time.time()

    for step in range(args.max_steps):
        step_id = step + 1
        lr = learning_rate(
            step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
        )
        set_lr(optimizer, lr)

        step_loss = train_one_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_tokens=data.train_tokens,
            args=args,
            runtime=runtime,
        )
        running_loss += step_loss

        if args.log_interval > 0 and step_id % args.log_interval == 0:
            elapsed = time.time() - last_log_time
            tokens = args.batch_size * args.seq_len * args.grad_accum_steps * args.log_interval * world_size()
            tokens_per_second = tokens / max(elapsed, 1e-9)
            train_loss = running_loss / args.log_interval
            log(f"step {step_id:05d} train_loss {train_loss:.4f} lr {lr:.2e} tok/s {tokens_per_second:,.0f}")
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens": step_id * args.batch_size * args.seq_len * args.grad_accum_steps * world_size(),
                },
            )
            running_loss = 0.0
            last_log_time = time.time()

        if step == 0 or (args.eval_interval > 0 and step_id % args.eval_interval == 0):
            eval_metrics = estimate_eval_metrics(
                model,
                data.val_tokens,
                data.tokenizer,
                args,
                runtime,
            )
            log(
                f"step {step_id:05d} "
                f"train_loss {step_loss:.4f} "
                f"val_loss {eval_metrics['loss']:.4f} "
                f"masked_bpb {eval_metrics['masked_bpb']:.4f} "
                f"lr {lr:.2e}"
            )
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "eval/loss": eval_metrics["loss"],
                    "eval/masked_bpb": eval_metrics["masked_bpb"],
                    "eval/masked_tokens": eval_metrics["masked_tokens"],
                    "eval/train_loss_at_eval": step_loss,
                    "train/lr": lr,
                },
            )

        if is_main_process() and args.sample_interval > 0 and step_id % args.sample_interval == 0:
            sample = sample_text(model, data.tokenizer, args, runtime)
            if wandb_run is not None:
                import wandb

                wandb_run.log({"sample/text": wandb.Html(f"<pre>{sample}</pre>")}, step=step_id)

        if is_main_process() and args.save_interval > 0 and step_id % args.save_interval == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                model=model,
                tokenizer=data.tokenizer,
                optimizer=optimizer,
                step=step_id,
                args=args,
            )
            log(f"saved checkpoint: {args.out_dir / 'checkpoint.pt'}")
            log_train_metrics(wandb_run, step_id, {"checkpoint/step": step_id})

    if is_main_process():
        save_checkpoint(
            out_dir=args.out_dir,
            model=model,
            tokenizer=data.tokenizer,
            optimizer=optimizer,
            step=args.max_steps,
            args=args,
        )
        log(f"saved final checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = args.max_steps
            wandb_run.finish()


def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        train(args, runtime)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
