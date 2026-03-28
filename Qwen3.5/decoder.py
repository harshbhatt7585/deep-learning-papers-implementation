from __future__ import annotations

from torch import nn

from .attention import Qwen35Attention
from .delta import Qwen35GatedDeltaNet
from .mlp import Qwen3MLP, forward
from .norm import Qwen35RMSNorm

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

