from __future__ import annotations

import argparse
import math
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch

from model import TextDiffusionConfig, TextDiffusionModel, diffusion_loss, generate
from tokenizer import LLaDA21Tokenizer, SimpleCharTokenizer


NANOCHAT_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
NANOCHAT_MAX_SHARD = 6542


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    starts = torch.randint(0, data.numel() - seq_len, (batch_size,))
    batch = torch.stack([data[start:start + seq_len] for start in starts])
    return batch.to(device)


@torch.no_grad()
def estimate_loss(
    model: TextDiffusionModel,
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    mask_prob: float,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        batch = get_batch(data, batch_size=batch_size, seq_len=seq_len, device=device)
        loss = diffusion_loss(model, batch, mask_prob=mask_prob)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, *, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    *,
    out_dir: Path,
    model: TextDiffusionModel,
    tokenizer: LLaDA21Tokenizer | SimpleCharTokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / ("tokenizer_hf" if isinstance(tokenizer, LLaDA21Tokenizer) else "tokenizer.json")
    tokenizer.save(tokenizer_path)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "step": step,
            "config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_type": "llada21" if isinstance(tokenizer, LLaDA21Tokenizer) else "char",
            "args": args_dict,
        },
        out_dir / "checkpoint.pt",
    )


@torch.no_grad()
def print_sample(
    *,
    model: TextDiffusionModel,
    tokenizer: LLaDA21Tokenizer | SimpleCharTokenizer,
    prompt: str,
    gen_length: int,
    block_length: int,
    steps: int,
    threshold: float,
    device: torch.device,
) -> None:
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    output = generate(
        model,
        prompt_ids,
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        threshold=threshold,
        editing_threshold=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("sample:", repr(tokenizer.decode(output.detach().cpu())))
    model.train()


def nanochat_shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_nanochat_shard(index: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = nanochat_shard_name(index)
    path = cache_dir / filename
    if path.exists():
        return path

    url = f"{NANOCHAT_BASE_URL}/{filename}"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {url}")
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with tmp_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp_path.replace(path)
            return path
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 5:
                raise RuntimeError(f"failed to download {filename}") from exc
            wait_s = 2 ** attempt
            print(f"download failed for {filename}, retrying in {wait_s}s: {exc}")
            time.sleep(wait_s)

    return path


def read_parquet_text(path: Path, *, max_chars: int | None = None) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading nanochat parquet shards requires pyarrow. Install it with: pip install pyarrow"
        ) from exc

    pieces = []
    total_chars = 0
    parquet_file = pq.ParquetFile(path)
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=["text"])
        for text in table.column("text").to_pylist():
            if max_chars is not None and total_chars >= max_chars:
                return "\n".join(pieces)
            if max_chars is not None:
                remaining = max_chars - total_chars
                text = text[:remaining]
            pieces.append(text)
            total_chars += len(text)
    return "\n".join(pieces)


def load_nanochat_text(
    *,
    cache_dir: Path,
    num_train_shards: int,
    max_train_chars: int | None,
    max_val_chars: int | None,
) -> tuple[str, str]:
    train_pieces = []
    remaining_train_chars = max_train_chars
    for shard_index in range(num_train_shards):
        path = download_nanochat_shard(shard_index, cache_dir)
        text = read_parquet_text(path, max_chars=remaining_train_chars)
        train_pieces.append(text)
        if remaining_train_chars is not None:
            remaining_train_chars -= len(text)
            if remaining_train_chars <= 0:
                break

    val_path = download_nanochat_shard(NANOCHAT_MAX_SHARD, cache_dir)
    val_text = read_parquet_text(val_path, max_chars=max_val_chars)
    return "\n".join(train_pieces), val_text


def load_training_text(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.nanochat:
        return load_nanochat_text(
            cache_dir=args.nanochat_cache_dir,
            num_train_shards=args.nanochat_train_shards,
            max_train_chars=args.max_train_chars,
            max_val_chars=args.max_val_chars,
        )

    if args.data is None:
        raise ValueError("Pass either --data path.txt or --nanochat.")
    return args.data.read_text(), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny text diffusion model.")
    parser.add_argument("--data", type=Path, help="Plain text dataset path.")
    parser.add_argument("--nanochat", action="store_true", help="Use nanochat's current ClimbMix parquet dataset.")
    parser.add_argument("--tokenizer", choices=["llada21", "char"], default="llada21")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--nanochat-cache-dir", type=Path, default=Path("data/nanochat_climbmix"))
    parser.add_argument("--nanochat-train-shards", type=int, default=1)
    parser.add_argument("--max-train-chars", type=int, default=5_000_000)
    parser.add_argument("--max-val-chars", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/text-diffusion-char"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--mask-prob", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-prompt", type=str, default="The ")
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--sample-block-length", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    train_text, val_text = load_training_text(args)
    if args.tokenizer == "llada21":
        tokenizer = LLaDA21Tokenizer.from_pretrained(
            local_files_only=args.tokenizer_local_files_only,
        )
    else:
        tokenizer = SimpleCharTokenizer.from_texts([train_text, val_text or "", args.sample_prompt])
    train_ids = torch.tensor(tokenizer.encode(train_text, add_eos=True), dtype=torch.long)

    if val_text is None:
        split = int(0.95 * train_ids.numel())
        train_data = train_ids[:split]
        val_data = train_ids[split:]
    else:
        train_data = train_ids
        val_data = torch.tensor(tokenizer.encode(val_text, add_eos=True), dtype=torch.long)
    if train_data.numel() <= args.seq_len or val_data.numel() <= args.seq_len:
        raise ValueError("Dataset is too small for --seq-len. Use more text or lower --seq-len.")

    max_sample_len = len(tokenizer.encode(args.sample_prompt)) + args.sample_length
    config = TextDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max(args.seq_len, max_sample_len + args.sample_block_length),
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = TextDiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    print(f"device: {device}")
    print(f"data_source: {'nanochat/climbmix-400b-shuffle' if args.nanochat else args.data}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"train chars: {len(train_text):,}")
    print(f"val chars: {len(val_text) if val_text is not None else len(train_text) - int(0.95 * len(train_text)):,}")
    print(f"train tokens: {train_data.numel():,}")
    print(f"val tokens: {val_data.numel():,}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(args.max_steps):
        lr = learning_rate(
            step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = get_batch(
            train_data,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, batch, mask_prob=args.mask_prob)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                mask_prob=args.mask_prob,
                eval_batches=args.eval_batches,
                device=device,
            )
            print(
                f"step {step + 1:05d} "
                f"train_loss {loss.item():.4f} "
                f"val_loss {val_loss:.4f} "
                f"lr {lr:.2e}"
            )

        if (step + 1) % args.sample_interval == 0:
            print_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=args.sample_prompt,
                gen_length=args.sample_length,
                block_length=args.sample_block_length,
                steps=args.sample_steps,
                threshold=args.sample_threshold,
                device=device,
            )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                step=step + 1,
                args=args,
            )
            print(f"saved checkpoint: {args.out_dir / 'checkpoint.pt'}")

    save_checkpoint(
        out_dir=args.out_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        step=args.max_steps,
        args=args,
    )
    print(f"saved final checkpoint: {args.out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
