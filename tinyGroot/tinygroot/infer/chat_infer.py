from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinygroot.model import TinyGrootConfig, TinyGrootModel
from tinygroot.sft_chat import generate_with_tools, render_prompt_for_completion
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.utils import load_meta, load_model_state, resolve_checkpoint_dir


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


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    tokenizer_dir: Path | None = None,
) -> tuple[TinyGrootModel, NanochatTokenizer, dict[str, Any]]:
    meta = load_meta(path)
    if "config" not in meta:
        raise KeyError(f"checkpoint at {path} has no 'config'")

    tokenizer_dir = tokenizer_dir or resolve_checkpoint_dir(path) / "tokenizer_hf"
    tokenizer = NanochatTokenizer.load(tokenizer_dir)
    config_blob = meta["config"]
    config = config_blob if isinstance(config_blob, TinyGrootConfig) else TinyGrootConfig(**config_blob)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config).to(device)
    model.load_state_dict(clean_state_dict(load_model_state(path, map_location=device)), strict=True)
    model.eval()
    return model, tokenizer, meta


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
    model, tokenizer, meta = load_checkpoint(checkpoint_path, device, tokenizer_dir=args.tokenizer_dir)
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
    print(f"loaded checkpoint step: {meta.get('step')}")
    print(tokenizer.decode(output, skip_special=True).strip())


if __name__ == "__main__":
    main()
