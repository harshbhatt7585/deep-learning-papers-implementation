"""
GPT model (rewrite, a lot simpler)
Notable featues:
 -- rotary embeddings (and no positonal embedings)
 -- QK norma
 -- united weights for token embedding and lm_head
 -- relu^2 activation in MLP
 -- norm after token embedding
 -- no learnable params in rmsnorm 
 -- no bias in linear layers
 -- Group-Query Attention (GQA) support for more efficient inference
 -- Flash Attention 3 itegration
"""

from functools import patial
from dataclasses import dataclass
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MounAdamW, DistMounAdamW

from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layers: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    # Slinding window attention pattern string titles across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"= all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1), )) # this will run on bf16

class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    "Returns True of GPT layer should have Value Embedding (alternating, last layer always included)"
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_em(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head  config.n_kv_head
        self.n_emebd = config.n_emebd
        self.head_dim = self.n_emebd // self.n_head
        assert self.n_emebd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_emebd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_emebd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_emebd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_emebd, self.n_emebd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys and values
        # Shape: (B, T, H, D) -- FA3's native layout, no transpose needed
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input_dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])) # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_em(q, cos, sin), apply_rotary_em(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, Pytorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, 
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seq_lens,
                causal=True,
                window_size=window_size
            )
            
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
            
        
        y = y.contigious().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init()

        self.c_fc = Linear(config.n_embd, 4 * config.n_emdb, bias=False)
        self.c_proj = Linear(4 * config.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
        



