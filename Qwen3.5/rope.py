from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., : x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), dim=-1)
