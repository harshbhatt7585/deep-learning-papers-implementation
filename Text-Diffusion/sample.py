from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import TextDiffusionConfig, TextDiffusionModel, generate
from tokenizer import LLaDA21Tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_dir: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    tokenizer_type = checkpoint.get("tokenizer_type")
    if tokenizer_type != "llada21":
        raise ValueError(
            f"unsupported tokenizer_type {tokenizer_type!r}; only llada21 checkpoints are supported"
        )
    tokenizer = LLaDA21Tokenizer.load(checkpoint_dir / "tokenizer_hf")
    config = TextDiffusionConfig(**checkpoint["config"])

    model = TextDiffusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained text diffusion checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Hello ")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint_dir, device)
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device)

    output = generate(
        model,
        prompt_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        editing_threshold=None,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"loaded checkpoint step: {checkpoint['step']}")
    print(tokenizer.decode(output.detach().cpu()))


if __name__ == "__main__":
    main()
