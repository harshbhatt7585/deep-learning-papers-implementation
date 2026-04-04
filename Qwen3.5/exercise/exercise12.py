
import sys
from pathlib import Path
from turtle import forward
from types import SimpleNamespace

# Allow `python exercise/exercise10.py` to import modules from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rope import apply_rotary_pos_emb, Qwen35RotaryEmbedding
import torch
from torch import nn
from utils import repeat_kv




class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int
    ): 
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attention_bias = config.attention_bias
    
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=self.attention_bias
        )
        
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias
        )
        
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias
        )

        self.out_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positonal_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value=None
    ):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # positonal_embeddings: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]

        batch_size, seq_len, hidden_size = hidden_states.shape
        input_shape = (batch_size, seq_len, hidden_size)

        query_proj = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(query_proj, 2, dim=-1)

        query_states = query_states.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        

        cos, sin = positonal_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)        

        
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, key_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_size)       
        attn_output = attn_output * torch.sigmoid(gate)
        return self.out_proj(attn_output)


if __name__ == "__main__":
    config = SimpleNamespace(
        hidden_size=128,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000, "mrope_section": [11, 11, 10]},
    )

    batch_size = 4
    seq_len = 20
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rotary = Qwen35RotaryEmbedding(config)
    position_embeddings = rotary(hidden_states, position_ids)

    attention_mask = None
    attn = Attention(config=config, layer_idx=1)
    output = attn(hidden_states, position_embeddings, attention_mask)
    print(output.shape)
