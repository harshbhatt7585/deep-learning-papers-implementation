from timeit import repeat
import torch
from torch import nn
from norm import Qwen35RMSNorm
from rope import apply_rotary_pos_emd
from utils import repeat_kv

from types import SimpleNamespace
import torch



class Qwen35Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.q_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps
        )
        self.k_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps
        )

    
    def forward(
        self,
        hidden_states,
        positon_embeddings,
        attention_mask,
        past_key_values=None
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1
        )
        gate= gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = positon_embeddings
        query_states, key_states = apply_rotary_pos_emd(query_states,key_states, cos, sin)

        if past_key_values is not None:
            attn_weights = attn_weights + attention_mask
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contagious()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return attn_output

if __name__ == "__main__":
    a = torch.randn(4,5,20)
    config = SimpleNamespace(
          hidden_size=20,
          num_attention_heads=4,
          num_key_value_heads=2,
          head_dim=20,
          attention_dropout=0.0,
          attention_bias=False,
          rms_norm_eps=1e-6,
      )
    cos = torch.ones(4, 5, 20)
    sin = torch.zeros(4, 5, 20)
    mask = None
    model = Qwen35Attention(config, 1)
    o = model(a, (cos, sin), mask)
    print(o)
