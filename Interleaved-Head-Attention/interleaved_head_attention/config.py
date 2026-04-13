from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IHAConfig:
    hidden_size: int
    num_attention_heads: int
    num_pseudo_heads: int | None = None
    attention_dropout: float = 0.0
    bias: bool = True
    causal: bool = True
    window_size: int | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
