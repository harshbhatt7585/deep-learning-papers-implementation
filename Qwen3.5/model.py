from __future__ import annotations

import torch
from torch import nn

from cache import Qwen35DynamicCache
from decoder import Qwen35DecoderLayer
from mask import build_causal_mask
from norm import Qwen35RMSNorm
from rope import Qwen35RotaryEmbedding


class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Qwen35DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Qwen35DynamicCache | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(self.config)

        batch_size, seq_len, _ = inputs_embeds.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)

        rope_position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, rope_position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Qwen35DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Qwen35DynamicCache | None]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
