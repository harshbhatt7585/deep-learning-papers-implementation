from dataclasses import dataclass
from tkinter import NO
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
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rope_parameters: dict[str, Any] | None = None
    layer_types: list[str] | None = None

    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    