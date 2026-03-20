from typing import Any


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states, gate=None):
        input_dtypes = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtypes)


class Qwen3NextDynamicCache:
    is_compileable = False


    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(self.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    
    def __len__(self):
        return len(self.layer_types)
    

    def update(
        self,
        key_states: torch.Tesnor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            beam_idx = beam_idx.to(device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx]
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)
        
        if self.conv_states[layer_idx] is not None:
            device = self.conv_states[layer_idx].device
            beam_idx = beam_idx.to(device)
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
            self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)
    

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    
    def get_mask_size(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        kv_offset = 0
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset
    
    @property
    def has_previous_state(self):
        return self.conv_states[self.last_linear_layer] is not None
    
    
    

