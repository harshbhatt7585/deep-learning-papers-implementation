from __future__ import annotations
from multiprocessing import Value

import torch
from torch import nn


from .cache import Qwen35DynamicCache
from .decoder import Qwen35DecoderLayer
from .mask import build_casual_mask
from .norm import Qwen35RMSNorm
from .rope import Qwen35RotaryEmbedding


class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)
    

    def forward(self, input_ids=None, attention_mask=None, positon_ids=None, past_key_values=None, input_emebds=None, use_cache=True):
        if (input_ids is None) == (input_emebds is None):
            raise ValueError("You must specify exactly one of input_ids or input_emebds")
        
        if input_emebds is None:
            input_emebds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(self.config)

        batch_size, seq_len, _ = input_emebds.shape

        