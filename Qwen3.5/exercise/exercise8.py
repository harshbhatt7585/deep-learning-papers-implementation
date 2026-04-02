from __future__ import annotations

from torch import nn

from attention import Qwen35Attention
from delta import Qwen35GatedDeltaNet
from mlp import Qwen3MLP
from norm import Qwen35RMSNorm

class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        
        if self.layer_type == "linear_attention":
            self.token_mixer = Qwen35GatedDeltaNet(config, layer_idx)
        
        else:
            self.token_mixer = Qwen35Attention(config, layer_idx)
    
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    
    def forward(self, hidden_states, positon_embeddings, attention_mask=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.token_mixer(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask
            )
        else:
            hidden_states = self.token_mixer(
                hidden_states=hidden_states,
                positon_embeddings=positon_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



