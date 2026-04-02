from __future__ import annotations
from math import e
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
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    

    def forward(self, input_ids=None, attention_mask=None, positon_ids=None, past_key_values=None, input_emebds=None, use_cache=True):
        if (input_ids is None) == (input_emebds is None):
            raise ValueError("You must specify exactly one of input_ids or input_emebds")
        
        if input_emebds is None:
            input_emebds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(self.config)

        batch_size, seq_len, _ = input_emebds.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if positon_ids is None:
            positon_ids = torch.arange(seq_len, device=input_emebds.shape) + past_seen_tokens
            positon_ids = positon_ids.view(1, -1).expand(batch_size, -1)
        
        rope_positon_ids = positon_ids[None, ...].expand(3, batch_size, -1)
        positonal_embeddings = self.rotary_emb(input_emebds, rope_positon_ids)

        kv_length = seq_len + past_seen_tokens
        casual_mask = build_casual_mask(
            attenton_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=input_emebds.device,
            dtype=input_emebds.dtype
        )

        hidden_states = input_emebds

        for layer in self.layers:
            layer_mask = attention_mask if getattr(layer, "layer_type", None) == "linear_attention" else casual_mask
            hidden_states = layer(
                hidden_states,
                positonal_embeddings=positonal_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
        