from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def _checkpoint_summary(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    except Exception:
        return {"step": None, "tokenizer_type": None, "config": {}, "args": {}}
    config = checkpoint.get("config", {})
    args = checkpoint.get("args", {})
    return {
        "step": checkpoint.get("step"),
        "tokenizer_type": checkpoint.get("tokenizer_type"),
        "config": config if isinstance(config, dict) else {},
        "args": args if isinstance(args, dict) else {},
    }


def write_model_card(checkpoint_dir: Path, repo_id: str, *, overwrite: bool = False) -> Path:
    readme = checkpoint_dir / "README.md"
    if readme.exists() and not overwrite:
        return readme

    summary = _checkpoint_summary(checkpoint_dir / "checkpoint.pt")
    config = summary["config"]
    args = summary["args"]
    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        "- tinygroot",
        "- causal-lm",
        "---",
        "",
        f"# {repo_id}",
        "",
        "tinyGroot checkpoint uploaded from the training pipeline.",
        "",
        "## Checkpoint",
        "",
        f"- Step: `{summary['step']}`",
        f"- Tokenizer: `{summary['tokenizer_type']}`",
        f"- Layers: `{config.get('n_layers')}`",
        f"- Hidden size: `{config.get('d_model')}`",
        f"- Heads: `{config.get('n_heads')}`",
        f"- MTP heads: `{config.get('n_mtp_heads')}`",
        f"- Max sequence length: `{config.get('max_seq_len')}`",
        f"- Source run: `{args.get('wandb_name') or args.get('out_dir') or 'unknown'}`",
        "",
        "## Files",
        "",
        "- `checkpoint.pt`: PyTorch training checkpoint containing model weights, optimizer state, config, tokenizer metadata, and training args.",
        "- `tokenizer_hf/tokenizer.json`: tokenizer used by this checkpoint.",
        "",
        "Load this checkpoint with the tinyGroot codebase, not the Transformers `AutoModel` API.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def push_checkpoint_to_hub(
    *,
    checkpoint_dir: Path,
    repo_id: str,
    private: bool = False,
    revision: str | None = None,
    commit_message: str | None = None,
    token: str | None = None,
    overwrite_model_card: bool = False,
) -> str:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    tokenizer_path = checkpoint_dir / "tokenizer_hf" / "tokenizer.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to push checkpoints to Hugging Face.") from exc

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    write_model_card(checkpoint_dir, repo_id, overwrite=overwrite_model_card)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo=".",
        revision=revision,
        commit_message=commit_message or f"Upload tinyGroot checkpoint from {checkpoint_dir.name}",
    )
    return str(commit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a tinyGroot checkpoint directory to Hugging Face Hub.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing checkpoint.pt and tokenizer_hf/tokenizer.json.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/model-name.")
    parser.add_argument("--private", action="store_true", help="Create the HF model repo as private.")
    parser.add_argument("--revision", default=None, help="Optional branch or revision to upload to.")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--overwrite-model-card", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit_url = push_checkpoint_to_hub(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        private=args.private,
        revision=args.revision,
        commit_message=args.commit_message,
        token=args.token,
        overwrite_model_card=args.overwrite_model_card,
    )
    print(f"uploaded checkpoint to Hugging Face: {commit_url}", flush=True)


if __name__ == "__main__":
    main()
