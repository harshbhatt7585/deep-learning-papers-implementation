import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import attention, functional as F


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LinearNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_mult * config.d_model),
            nn.GELU(),
            nn.Linear(config.ff_mult * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        key_padding_mask = None
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                key_padding_mask = ~attention.bool()
            elif attention_mask.ndim in 3:
                keep_mask = attention_mask
                kee_mask = keep_mask.to(dtype=torch.bool, device=x.device)
                attn_mask = ~kee_mask
                attn_mask = attn_mask[:, None].expand(
                    keep_mask.shape[0],
                    self.attn.num_heads,
                    keep_mask.shape[-2],
                    keep_mask.shape[-1]
                )
                attn_mask = attn_mask.reshape(
                    keep_mask.shape[0] * self.attn.num_heads,
                    keep_mask.shape[-2],
                    kee_mask.shape[-1]
                )
            else:
                pass
                
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x