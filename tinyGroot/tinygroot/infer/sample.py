from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinygroot.engine import Engine
from tinygroot.model import TinyGrootConfig, TinyGrootModel, infer_arch_from_state_dict
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.utils import load_meta, load_model_state, resolve_checkpoint_dir


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_dir: Path, device: torch.device):
    meta = load_meta(checkpoint_dir)
    tokenizer_type = meta.get("tokenizer_type")
    if tokenizer_type != "nanochat":
        raise ValueError(
            f"unsupported tokenizer_type {tokenizer_type!r}; only nanochat checkpoints are supported"
        )
    tokenizer = NanochatTokenizer.load(resolve_checkpoint_dir(checkpoint_dir) / "tokenizer_hf")
    state = load_model_state(checkpoint_dir, map_location=device)
    cfg = dict(meta["config"])
    cfg["arch"] = infer_arch_from_state_dict(state)
    config = TinyGrootConfig(**cfg)

    model = TinyGrootModel(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained causal/MTP checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Hello ")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model forward for faster decode steps.")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, meta = load_checkpoint(args.checkpoint_dir, device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    prompt_ids = tokenizer.encode(args.prompt)
    engine = Engine(model, tokenizer)
    sequences, _masks = engine.generate_batch(
        prompt_ids,
        num_samples=args.num_samples,
        max_tokens=args.gen_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        seed=args.seed,
    )

    print(f"loaded checkpoint step: {meta.get('step')}")
    for idx, seq in enumerate(sequences):
        if args.num_samples > 1:
            print(f"--- sample {idx} ---")
        print(tokenizer.decode(seq))


if __name__ == "__main__":
    main()
