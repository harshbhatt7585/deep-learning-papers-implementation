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
        

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        self.window_size = self._compute_window_size(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} tp {padded_vocab_size} for efficiency")
        
        self.transformer = nn.ModuleList({
            "wte": nn.Embedding(padded_vocab_size, config.n_emebd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        })
        self.lm_head = Linear(config)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends inital embedding back in at each layer (init 0.0 = disabled)
        # Seperate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer)) # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer)) # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: substract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_emebds = nn.ModuleList({str(i): nn.Embedding(padded_vocab_size, kv_dim)} for i in range(config.n_layer) if has_vae(i, config.n_layer))

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding): normal, std=1.0
        lm_head: norm, std=0.001
        for each block:
            attn.c_q: unifrom, std=1/sqrt(n_emebd)
            attn.c_k: uniform, std=1/sqrt(n_emebd)
            attn.c_v: uniform, std=1/sqrt(n_emebd)
            attn.c_proj: zeros
            mlp.c_fc: uniform, std=1/sqrt(n_embd)
            mlp.c_proj: zeros

        """
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std
        n_emebd = self.config.n_emdb
        s = 3**0.5 * n_emebd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zeros
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        
        # per-layer scalars
        # per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (1.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        
        # Smean/backout scalars and smear gate must be explicitly initalized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_emebds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        
        # Gate weights init with samll positive values so gates start strictly above neurtal
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        
        # Rotary Embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScalar cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_emebds.values():
                ve.to(dtype=COMPUTE_DTYPE)
        



