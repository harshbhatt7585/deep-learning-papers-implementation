"""
Download nanochat checkpoints from Hugging Face into NANOCHAT_BASE_DIR
(default ~/.cache/nanochat), in the layout expected by checkpoint_manager / tokenizer.

Defaults target depth-12 (d12): there is no karpathy/nanochat-d12 on the Hub, so we use the
nested bundle pankajmathur/nanochat-d12-newarch (base_checkpoints/d12/ + tokenizer/).

For the Transformers-docs Karpathy release (flat repo root), use explicit flags:

  python -m scripts.pull_hf_checkpoint --repo karpathy/nanochat-d32 --layout flat --model-tag d32

Layouts:

  nested (default for d12): e.g. base_checkpoints/d12/model_*.pt inside the repo.

  flat: model_*.pt and meta_*.json at repo root — e.g. karpathy/nanochat-d32.

Examples:

  # d12 from Hub (default)
  python -m scripts.pull_hf_checkpoint
  python -m scripts.chat_sft

  # or train a tiny base checkpoint locally (no Hub)
  bash runs/quick_base_checkpoint.sh
  python -m scripts.chat_sft

  # Karpathy d32 (Transformers NanoChat docs)
  python -m scripts.pull_hf_checkpoint --repo karpathy/nanochat-d32 --layout flat --model-tag d32
  python -m scripts.chat_sft --model-tag d32

Gated models: set HF_TOKEN or run `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from nanochat.common import get_base_dir

PHASE_TO_DIR = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


def _download_nested(repo_id: str, base: Path, model_tag: str, phase: str, tokenizer: bool) -> None:
    prefix = PHASE_TO_DIR[phase]
    patterns = [f"{prefix}/{model_tag}/*"]
    if tokenizer:
        patterns.append("tokenizer/*")
    print(f"Downloading from {repo_id} into {base} (patterns: {patterns})")
    snapshot_download(repo_id=repo_id, local_dir=str(base), allow_patterns=patterns)


def _download_flat(repo_id: str, base: Path, model_tag: str, tokenizer: bool) -> None:
    print(f"Downloading flat layout from {repo_id} into {base} (model tag {model_tag})")
    with tempfile.TemporaryDirectory() as td:
        snapshot_download(
            repo_id=repo_id,
            local_dir=td,
            allow_patterns=["model_*.pt", "meta_*.json", "tokenizer.pkl", "token_bytes.pt"],
        )
        td_path = Path(td)
        dest_ckpt = base / "base_checkpoints" / model_tag
        dest_ckpt.mkdir(parents=True, exist_ok=True)
        for pattern in ("model_*.pt", "meta_*.json"):
            for p in sorted(td_path.glob(pattern)):
                shutil.copy2(p, dest_ckpt / p.name)
        if not any(dest_ckpt.glob("model_*.pt")):
            raise FileNotFoundError(f"No model_*.pt found under {td_path} after download — check repo id / files.")
        if tokenizer:
            tok_dest = base / "tokenizer"
            tok_dest.mkdir(parents=True, exist_ok=True)
            for name in ("tokenizer.pkl", "token_bytes.pt"):
                src = td_path / name
                if src.exists():
                    shutil.copy2(src, tok_dest / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=str,
        default="pankajmathur/nanochat-d12-newarch",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="d12",
        help="Checkpoint subdirectory name under base_checkpoints/ (e.g. d12, d32)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="base",
        choices=list(PHASE_TO_DIR.keys()),
        help="Training stage folder on Hub (nested layout only)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="nested",
        choices=("nested", "flat"),
        help="Repo file layout (Karpathy checkpoints use flat; many d12 bundles use nested)",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer files (only if ~/.cache/nanochat/tokenizer is already valid)",
    )
    args = parser.parse_args()

    base = Path(get_base_dir())
    tokenizer = not args.no_tokenizer
    if args.layout == "nested":
        _download_nested(args.repo, base, args.model_tag, args.phase, tokenizer)
    else:
        _download_flat(args.repo, base, args.model_tag, tokenizer)

    print("Done.")
    if args.layout == "nested" or args.phase == "base":
        print(f"Try: python -m scripts.chat_sft --model-tag {args.model_tag}")


if __name__ == "__main__":
    main()
