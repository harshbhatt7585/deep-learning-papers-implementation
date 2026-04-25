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
                batch_size, query_len, key_len = attention_mask
                attn_mask = ~attention_mask.bool().to(device=x.device)
                attn_mask = attn_mask[:, None].expand(
                    batch_size,
                    self.attn.num_heads,
                    query_len,
                    key_len
                )
                attn_mask = attn_mask.reshape(
                    batch_size * self.attn.num_heads,
                    query_len,
                    key_len
                )
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


class TextDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len. config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ):
        batch_size, seq_len = input_ids.shape
        
        positons = torch.arange(seq_len, device=input_ids.device)[None, :]
        positons = positons.expand(batch_size seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positons)
        x = self.drop(x)
    
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.norm(x)
        return self.lm_head(x)

        
