from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



class RoPE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.theta = config.theta
        self.dim = min(config.dim, config.head_dim)
        self.mrope_section = getattr(config, "mrope_section", [11, 11, 10])
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)



    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim != 3:
            raise ValueError(f"Expected 2D or 3D position ids, got {tuple(position_ids.shape)}")

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = apply_rotary_pos_emb(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        theta=10000.0,
        rotaty_factor=1.0,
        dim=20,
        head_dim=20,
        mrope_section=[3, 2, 2],
    )
    rope = Qwen35RotaryEmbedding(config, "cpu")

    q = torch.randn(4, 8, 20, 20)
    k = torch.randn(4, 8, 20, 20)
    position_ids = torch.arange(20).unsqueeze(0).expand(4, -1)
    cos, sin = rope(q, position_ids)

    print(cos.shape)
    print(q.shape)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    print(q_embed.shape, k_embed.shape)
