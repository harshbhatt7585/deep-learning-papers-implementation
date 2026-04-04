from __future__ import annotations

import sys
from pathlib import Path
from turtle import forward
from types import SimpleNamespace

import torch
from torch import nn

# Allow `python exercise/exercise10.py` to import modules from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm import Qwen35RMSNorm
from rope import apply_rotary_pos_emb
from utils import repeat_kv


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.scaling = self.head_dim ** 0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim * 2, # twice because it packs two things: actual query vector and a gate vector
            bias=config.attention_bias
        )
        
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )

        self.out_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positonal_embedding: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
    ) -> torch.Tensor:
        
        # hidden_states: [batch, seq_len, hidden_dim]
        # positonal_embeddings: [batch, seq_len]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        input_shape = (batch_size, seq_len, hidden_dim)

        
        query_shape = (batch_size, self.num_attention_heads, seq_len,  self.head_dim * 2)
        key_value_shape = (batch_size, self.num_key_value_heads, seq_len, self.head_dim)

        # [batch, seq_len, heads, head_dim]
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim * 2),
            2,
            dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.reshape(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).reshape(key_value_shape)).transpose(1, 2)
        value_states = self.v_proj(self.v_proj(hidden_states).reshape(value_states)).transpose(1, 2)

        cos, sin = positonal_embedding
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contigious().reshape(*input_shape, self.num_attention_heads)
        attn_output = attn_output * torch.sigmpid(gate)
        return self.out_proj(attn_output)


        
