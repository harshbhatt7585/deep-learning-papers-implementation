from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


HF_TEXT_PREFIXES = (
    "model.language_model.",
    "language_model.",
)

IGNORED_PREFIXES = (
    "model.visual.",
    "visual.",
    "model.vision",
    "vision.",
    "mtp.",
    "model.mtp.",
)


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu")


def resolve_model_path(model_name_or_path: str, allow_patterns: list[str] | None = None) -> Path:
    path = Path(model_name_or_path)
    if path.exists():
        return path

    snapshot_dir = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=allow_patterns,
    )
    return Path(snapshot_dir)


def load_config_dict(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    return json.loads((model_dir / "config.json").read_text())


def remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(IGNORED_PREFIXES):
            continue

        for prefix in HF_TEXT_PREFIXES:
            if key.startswith(prefix):
                key = "model." + key[len(prefix) :]
                break

        remapped[key] = value

    return remapped


def _load_sharded_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors_files = sorted(model_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

        state_dict: dict[str, torch.Tensor] = {}
        for file_path in safetensors_files:
            state_dict.update(load_file(file_path))
        return state_dict

    index = json.loads(index_path.read_text())
    state_dict = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(model_dir / shard_name))
    return state_dict


def load_weights(model, model_dir: str | Path):
    model_dir = Path(model_dir)
    state_dict = remap_state_dict_keys(_load_sharded_state_dict(model_dir))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            missing = [key for key in missing if key != "lm_head.weight"]
    return missing, unexpected
