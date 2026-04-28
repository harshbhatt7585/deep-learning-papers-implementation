from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import (
    TextDiffusionConfig,
    TextDiffusionModel,
    generate,
    make_masked_inputs,
)
from utils import (
    Runtime,
    TokenData,
    Tokenizer,
    autocast_context,
    cleanup_distributed,
    create_runtime,
    get_batch,
    init_wandb,
    is_dist,
    is_main_process,
    learning_rate,
    log,
    save_checkpoint,
    set_lr,
    tokenize_data,
    unwrap_model,
    world_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny text diffusion model.")

    parser.add_argument("--data", type=Path, help="Plain text dataset path.")
    parser.add_argument("--nanochat", action="store_true", help="Use nanochat's ClimbMix parquet dataset.")
    parser.add_argument("--nanochat-cache-dir", type=Path, default=Path("data/nanochat_climbmix"))
    parser.add_argument("--nanochat-train-shards", type=int, default=1)
    parser.add_argument("--max-train-chars", type=int, default=5_000_000)
    parser.add_argument("--max-val-chars", type=int, default=1_000_000)

    parser.add_argument("--tokenizer", choices=["llada21", "nanochat"], default="llada21")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--nanochat-tokenizer-cache-dir", type=Path, default=Path("data/nanochat_tokenizer_32k"))
    parser.add_argument("--nanochat-tokenizer-vocab-size", type=int, default=32_768)
    parser.add_argument("--nanochat-tokenizer-train-chars", type=int, default=2_000_000_000)
    parser.add_argument("--nanochat-tokenizer-doc-cap", type=int, default=10_000)

    parser.add_argument("--out-dir", type=Path, default=Path("runs/text-diffusion-llada21"))
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
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument("--sample-top-k", type=int, default=None)
    parser.add_argument("--sample-top-p", type=float, default=None)

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


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    masked = labels != -100
    return F.cross_entropy(logits[masked], labels[masked], reduction=reduction)


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
            loss_sum = masked_cross_entropy(logits, labels, reduction="sum")

        total_loss += float(loss_sum.item())
        total_masked_tokens += int((labels != -100).sum().item())
        total_bytes += sum(
            len(tokenizer.decode(row.detach().cpu().tolist()).encode("utf-8"))
            for row in batch
        )

    model.train()
    totals_device = torch.device("cpu") if runtime.device.type == "mps" else runtime.device
    totals = torch.tensor(
        [total_loss, total_masked_tokens, total_bytes],
        dtype=torch.float64,
        device=totals_device,
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
        temperature=args.sample_temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    sample = tokenizer.decode(output.detach().cpu())
    log(f"sample: {sample!r}")
    model.train()
    return sample


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


def create_wandb_sample_table(wandb_run):
    if wandb_run is None:
        return None

    import wandb

    return wandb.Table(
        columns=[
            "step",
            "prompt",
            "generated_text",
            "eval_loss",
            "eval_masked_bpb",
            "eval_masked_tokens",
            "sample_length",
            "block_length",
            "steps",
            "threshold",
            "temperature",
            "top_k",
            "top_p",
        ]
    )


def log_wandb_sample_table(
    wandb_run,
    table,
    *,
    step: int,
    sample: str,
    eval_metrics: dict[str, float] | None,
    args: argparse.Namespace,
) -> None:
    if wandb_run is None or table is None:
        return

    metrics = eval_metrics or {}
    table.add_data(
        step,
        args.sample_prompt,
        sample,
        metrics.get("loss"),
        metrics.get("masked_bpb"),
        metrics.get("masked_tokens"),
        args.sample_length,
        args.sample_block_length,
        args.sample_steps,
        args.sample_threshold,
        args.sample_temperature,
        args.sample_top_k,
        args.sample_top_p,
    )
    wandb_run.log({"eval/generated_samples": table}, step=step)


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
    source_model = unwrap_model(model)
    config = source_model.config
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
        noised, labels = make_masked_inputs(
            batch,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
            mask_prob=args.mask_prob,
        )
        attention_mask = noised != config.pad_token_id

        with sync_context:
            with autocast_context(runtime.device, args.amp_dtype):
                logits = model(noised, attention_mask=attention_mask)
                loss = masked_cross_entropy(logits, labels)
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
    latest_eval_metrics: dict[str, float] | None = None
    wandb_sample_table = create_wandb_sample_table(wandb_run)

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
            latest_eval_metrics = eval_metrics
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
                log_wandb_sample_table(
                    wandb_run,
                    wandb_sample_table,
                    step=step_id,
                    sample=sample,
                    eval_metrics=latest_eval_metrics,
                    args=args,
                )

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
