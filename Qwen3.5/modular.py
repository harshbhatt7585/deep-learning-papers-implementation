from typing import Any, Optional
import torch


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

    

class Qwen3NextRotataryEmbedding():
    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3NextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len = int | None = None
    ) -> tuple["torch.Tensor", float]:

        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** torch (
                torch.arange(0, dim, 2, dtype=int64).to(device=device, dtype=torch.float / dim) 
            )
        ) 
        return inv_freq, attention_factor


class Qwen3NextRMSNorm(Gemma3RMSNorm):
    pass


class Qwen3NextAttention(Qwen3MoeAttention):
    def __init__(self, config: QwenNextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attetion_heads * self.head_dim * 2, bias=config.attention_bias
        )
        del self.sliding_window

    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        positional_encodings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[: -1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(
                hidden_states
            ).view(*input_shape, -1, self.head_dim * 2),
            2, 
            dim=-1
        )

        gate= gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(
            hidden_shape
        )).transpose(1,2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        value_states = self.v_proj(
            hidden_states
        ).view(hidden_shape).transpose(1,2)

        cos, sin = positional_encodings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin
        )
        
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



def torch_casual_conv1d_update(
    hidden_states,
    conv_states,
    weight,
    bias=None,
    activation=None
):
    _, hidden_size, seq_len = hidden_size.shape
    state_len = conv_states.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy(hidden_states_new[:, :, -state_len: ])
    out = F.conv1d(hidden_states_new, weight.unsnsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len: ])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int= -1, eps: float - 1e-6):
    inv_norm = torch.rsqrt(
        (x * x).sum(dim=dim, keepdim=True) + eps
    )
    return x + inv_norm


        




    

    

