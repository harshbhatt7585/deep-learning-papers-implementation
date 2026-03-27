from torch import nn
from .norm import Qwen35RMSNorm
from .rope import apply_rotary_pos_emd
from .utils import repeat_kv


class Qwen35Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = self.attention_dropout

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