from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CollapseMode = Literal["per_head", "global"]


@dataclass
class IHAConfig:
    hidden_size: int
    num_attention_heads: int
    num_pseudo_heads: int | None = None
    attention_dropout: float = 0.0
    bias: bool = True
    causal: bool = True
    window_size: int | None = None
    collapse_mode: CollapseMode = "per_head"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
