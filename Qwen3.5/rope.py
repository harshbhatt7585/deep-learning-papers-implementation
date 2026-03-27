from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., : x.shape[-1] // 2: ]
    return torch.cat((-x2, x1), dim=-1)


class QwenRotaryEmbedding(torch.nn.Module):
    def __init__(self, config: device=None):
        super().__init__()
        base = config.base_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arrange(0, dim, 2, dtype=torch.float(), device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, positon_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positon_ids.dim == 2:
            positon_ids = positon_ids[None, ...].expand(3, positon_ids.shape[0], -1)
        
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            positon_ids.shape[0],
            positon_ids.shape[1],
            -1,
            1
        )
        positon_ids_exapanded = positon_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ positon_ids_exapanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


