from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from model import TinyGrootConfig, TinyGrootModel
from sft_chat import generate_with_tools, render_prompt_for_completion
from tokenizer import NanochatTokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    checkpoint_path = path / "checkpoint.pt" if path.is_dir() else path
    checkpoint_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return checkpoint_path, checkpoint_dir


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    tokenizer_dir: Path | None = None,
) -> tuple[TinyGrootModel, NanochatTokenizer, dict[str, Any]]:
    checkpoint_path, checkpoint_dir = resolve_checkpoint(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint:
        raise KeyError(f"checkpoint at {checkpoint_path} has no 'config' key")

    tokenizer_dir = tokenizer_dir or checkpoint_dir / "tokenizer_hf"
    tokenizer = NanochatTokenizer.load(tokenizer_dir)
    config_blob = checkpoint["config"]
    config = config_blob if isinstance(config_blob, TinyGrootConfig) else TinyGrootConfig(**config_blob)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config).to(device)
    model.load_state_dict(clean_state_dict(checkpoint["model_state"]), strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat inference for a tinyGroot SFT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint.pt or its run directory.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Deprecated alias for --checkpoint.")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Defaults to <checkpoint-dir>/tokenizer_hf.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or args.checkpoint_dir
    if checkpoint_path is None:
        raise SystemExit("pass --checkpoint /path/to/checkpoint.pt or --checkpoint-dir /path/to/run")

    device = pick_device()
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device, tokenizer_dir=args.tokenizer_dir)
    prompt_ids = render_prompt_for_completion(
        tokenizer,
        {"messages": [{"role": "user", "content": args.prompt}]},
        max_tokens=max(1, model.config.max_seq_len - args.max_new_tokens),
    )
    output = generate_with_tools(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    print(f"loaded checkpoint step: {checkpoint['step']}")
    print(tokenizer.decode(output, skip_special=True).strip())


if __name__ == "__main__":
    main()
