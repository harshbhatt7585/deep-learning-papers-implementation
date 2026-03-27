from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class Qwen35Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    hidden_act: str
    attention_dropout: float
    attention_bias: bool
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    rope_parameters: dict[str, Any]
    layer_types: list[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen35Config":
        data = json.loads(Path(path).read_text())
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            rms_norm_eps=data["rms_norm_eps"],
            hidden_act=data["hidden_act"],
            attention_dropout=data.get("attention_dropout", 0.0),
            attention_bias=data.get("attention_bias", False),
            pad_token_id=data.get("pad_token_id"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            rope_parameters=data["rope_parameters"],
            layer_types=data["layer_types"],
            linear_num_key_heads=data["linear_num_key_heads"],
            linear_num_value_heads=data["linear_num_value_heads"],
            linear_key_head_dim=data["linear_key_head_dim"],
            linear_value_head_dim=data["linear_value_head_dim"],
            linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
        )