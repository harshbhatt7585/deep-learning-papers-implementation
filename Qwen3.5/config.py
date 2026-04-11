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
    tie_word_embeddings: bool = False

    @staticmethod
    def _normalize_dict(data: dict[str, Any]) -> dict[str, Any]:
        if "text_config" in data:
            data = data["text_config"]

        rope_parameters = data.get("rope_parameters")
        if rope_parameters is None:
            rope_parameters = {
                "rope_theta": data.get("theta", 10000.0),
                "partial_rotary_factor": data.get("rotaty_factor", data.get("rotary_factor", 1.0)),
                "mrope_section": data.get("mrope_section", [11, 11, 10]),
            }

        return {
            "vocab_size": data["vocab_size"],
            "hidden_size": data["hidden_size"],
            "intermediate_size": data["intermediate_size"],
            "num_hidden_layers": data["num_hidden_layers"],
            "num_attention_heads": data["num_attention_heads"],
            "num_key_value_heads": data["num_key_value_heads"],
            "head_dim": data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            "rms_norm_eps": data["rms_norm_eps"],
            "hidden_act": data["hidden_act"],
            "attention_dropout": data.get("attention_dropout", 0.0),
            "attention_bias": data.get("attention_bias", False),
            "pad_token_id": data.get("pad_token_id"),
            "bos_token_id": data.get("bos_token_id"),
            "eos_token_id": data.get("eos_token_id"),
            "rope_parameters": rope_parameters,
            "layer_types": data["layer_types"],
            "linear_num_key_heads": data.get("linear_num_key_heads", data.get("num_k_heads")),
            "linear_num_value_heads": data.get("linear_num_value_heads", data.get("num_v_heads")),
            "linear_key_head_dim": data.get("linear_key_head_dim", data.get("head_k_dim")),
            "linear_value_head_dim": data.get("linear_value_head_dim", data.get("head_v_dim")),
            "linear_conv_kernel_dim": data.get("linear_conv_kernel_dim", data.get("linear_conv_kernel_size")),
            "tie_word_embeddings": data.get("tie_word_embeddings", False),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen35Config":
        return cls(**cls._normalize_dict(data))

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen35Config":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_namespace(cls, namespace: Any) -> "Qwen35Config":
        return cls.from_dict(vars(namespace))

    @classmethod
    def simple(cls, **overrides: Any) -> "Qwen35Config":
        data = {
            "vocab_size": 256,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attention_dropout": 0.0,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "intermediate_size": 256,
            "num_v_heads": 4,
            "num_k_heads": 2,
            "head_k_dim": 32,
            "head_v_dim": 32,
            "linear_conv_kernel_size": 4,
            "rotaty_factor": 1.0,
            "theta": 10000.0,
            "dim": 32,
            "mrope_section": [11, 11, 10],
        }
        data.update(overrides)
        return cls.from_dict(data)

    @property
    def num_v_heads(self) -> int:
        return self.linear_num_value_heads

    @property
    def num_k_heads(self) -> int:
        return self.linear_num_key_heads

    @property
    def head_k_dim(self) -> int:
        return self.linear_key_head_dim

    @property
    def head_v_dim(self) -> int:
        return self.linear_value_head_dim

    @property
    def linear_conv_kernel_size(self) -> int:
        return self.linear_conv_kernel_dim

    @property
    def theta(self) -> float:
        return self.rope_parameters["rope_theta"]

    @property
    def rotaty_factor(self) -> float:
        return self.rope_parameters.get("partial_rotary_factor", 1.0)

    @property
    def dim(self) -> int:
        return int(self.head_dim * self.rotaty_factor)

    @property
    def mrope_section(self) -> list[int]:
        return self.rope_parameters.get("mrope_section", [11, 11, 10])
