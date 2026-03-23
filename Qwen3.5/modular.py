from re import M
from typing import Any, Callable, Optional
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


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g, 
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False
):
    inital_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contigious().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim= key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(
        query,
        (0, 0, 0, pad_size)
    ),
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(key, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = value * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.resahpe(x.shape[0], x.shape[1], -1, chunk_size, x.shpae[-1] for x in (query, key, value, k_beta, v_beta))
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, dtype=torch.bool, device=query.device, diagonal=0)
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsueeze(-2).tril().exp().float().tril()))
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqeeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device))

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i]) = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contigious().to(inital_dtype)
    return core_attn_out, last_recurrent_state



class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_new_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads)
        )

        A = torch.empty(
            self.num_v_heads
        ).uniform(0, 16)

        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            Qwen3_5GatedDeltaNet(
                self.head_v_dim,
                eps=self.layer_norm_epsilon
            )
            if FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_normalization,
                activaition=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=True)

        self.casual_conv1d_fn = casual_conv1d_fn
        self.casual_conv1d_update = casual_conv1d_update or torch.casual_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule pr torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=True
        )            

        self.in_proj_z = nn.Linear(
            self.hidden_size, 
            self.num_v_heads,
            bias=True
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=True
        )

        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=True
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3_5DynamicsCache | None = None,
        attention_mask: torch.Tensor | None = None
    ):
        hidden_states = apply_mask_to_padding_states(
            hidden_states,
            attention_mask
        )
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.casual_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv,
                    (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
            
            if self.casual_conv1d_fn is not None:
                mixed_qkv = self.casual_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None
                )
            else:
                mixed_qkv = F.silu(
                    self.conv1d(
                        mixed_qkv[:, :, :seq_len]
                    )
                )

        mixed_qkv = mixed_qkv.trasnpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim
            ],
            dim=-1
        )


        query = query.reshape(
            batch_size,
            seq_len,
            -1,
            self.head_k_dim
        )
        key = key.reshape(
            batch_size,
            seq_len,
            -1,
            self.head_k_dim
        )
        value = value.reshape(
            batch_size,
            seq_len,
            -1,
            self.head_v_dim
        )

        beta = b.sigmoid()

        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2,
            )
            key = key.repeat_interleave(
                self.num_v_heads // self.num_k_heads,
                dim=2
            )

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                inital_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True
            )
        
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                inital_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(
            -1,
            self.head_v_dim
        )
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(
            batch_size,
            seq_len,
            -1
        )

        output = self.out_proj(core_attn_out)
        return output



def rotate_half(x):
    x1 = x[..., x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1), dim=-1
    )


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotate_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotate_dim], q[..., rotate_dim:]
    k_rot, k_pass = k[..., :rotate_dim], k[..., rotate_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    K_emebd = (k_rot * cos) + (rotate_dim(k_rot) * sin)

    q_emebd = torch.cat([q_emebd, q_pass], dim=-1)
    k_emebd = torch.cat([K_emebd, k_pass], dim=-1)

    return q_embed, k_emebd



def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int
):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim
    )

def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout,
): 
    key_states = repeat_kv(
        key,
        module.num_key_value_groups
    )
    value_states = repeat_kv(
        value,
        module.num_key_value_groups
    )

    attn_weights = torch.matmul(
        query,
        key_states.transpose(2,3) * scaling
    )
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.funcational,softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32
    ).to(query.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training
    )
    attn_output = torch.matmul(
        attn_weights,
        value_states
    )

    attn_output = attn_output.transpose(1,2).contigious()
    
    return attn_output, attn_weights


@use_kernelised_func(apply_rotary_pos_emb)
class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_casual = True
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
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps
        )


    
    def forward(
        self,
        hidden_states,
        postional_embeddings,
        attention_mask,
        past_key_values
    ):
        input_shape = hidden_states.shape[:-1]

        query_states, gate = torch.chunk(
            self.q_proj(
                hidden_states
            ).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1
        )
        
        key_states = self.q_norm(
            query_states.view(hidden_states)
        ).transpose(1, 2)

        value_states = self.v_proj(
            hidden_states
        ).view(hidden_states).transpose(1, 2)

        cos, sin = postional_embeddings
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
            self.config_attn_implenentation,
            eager_attention_forward
        )

        attn_output, attn_weight = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1).contigious()
        attn_output = attn_output * torch.sigmoid()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weight





