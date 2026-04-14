from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

config = SimpleNamespace(
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        hidden_size=128,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        intermediate_size=256,
        num_v_heads=4,
        num_k_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        linear_conv_kernel_size=4,
        rotaty_factor=1.0,
        theta=10000.0,
        dim=32,
        mrope_section=[11, 11, 10],
    )


@dataclass
class Qwen35Config:
    vocab_size: int
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    tie_word_embeddings: bool

    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_dropout: float
    attention_bias: bool

    rms_norm_eps: float
    hidden_act: str
    intermediate_size: int

    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    linear_conv_kernel_size: int

    theta: float = 10000.0
    rotaty_factor: float = 1.0
    dim: int | None = None
    mrope_section: list[int] | None = None

    def __post_init__(self) -> None:
        if self.dim is None:
            self.dim = int(self.head_dim * self.rotaty_factor)
        if self.mrope_section is None:
            self.mrope_section = [11, 11, 10]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen35Config":
        if "text_config" in data:
            data = data["text_config"]

        rope_parameters = data.get("rope_parameters", {})
        rotaty_factor = data.get(
            "rotaty_factor",
            data.get("rotary_factor", rope_parameters.get("partial_rotary_factor", 1.0)),
        )
        head_dim = data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])

        return cls(
            vocab_size=data["vocab_size"],
            pad_token_id=data.get("pad_token_id"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            layer_types=data["layer_types"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=head_dim,
            attention_dropout=data.get("attention_dropout", 0.0),
            attention_bias=data.get("attention_bias", False),
            rms_norm_eps=data["rms_norm_eps"],
            hidden_act=data["hidden_act"],
            intermediate_size=data["intermediate_size"],
            num_v_heads=data.get("num_v_heads", data.get("linear_num_value_heads")),
            num_k_heads=data.get("num_k_heads", data.get("linear_num_key_heads")),
            head_k_dim=data.get("head_k_dim", data.get("linear_key_head_dim")),
            head_v_dim=data.get("head_v_dim", data.get("linear_value_head_dim")),
            linear_conv_kernel_size=data.get(
                "linear_conv_kernel_size",
                data.get("linear_conv_kernel_dim"),
            ),
            theta=data.get("theta", rope_parameters.get("rope_theta", 10000.0)),
            rotaty_factor=rotaty_factor,
            dim=data.get("dim", int(head_dim * rotaty_factor)),
            mrope_section=data.get("mrope_section", rope_parameters.get("mrope_section", [11, 11, 10])),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen35Config":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_namespace(cls, namespace: Any) -> "Qwen35Config":
        return cls.from_dict(vars(namespace))
