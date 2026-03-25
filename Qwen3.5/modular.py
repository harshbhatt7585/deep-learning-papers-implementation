from dataclasses import dataclass
import itertools
from nt import device_encoding
from re import M, S
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




class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermidiate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermidiate_size,
            bias=True
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermidiate_size,
            bias=True
        )
        self.down_proj = nn.Linear(
            self.intermidiate_size,
            self.hidden_size,
            bias=True
        )
        self.act_fn = ACT2FN[config.hidden_act]

    
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x) * self.up_proj(x)
        ))
        return down_proj


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())

        return output.type_as(x)


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        
        self.mlp = Qwen3_5MLP(config, config.intermidate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        postional_encodings: tuple[torch.tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        postion_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None
    ):
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask
            )
        
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                postion_ids=postion_ids,
                past_key_values=past_key_values,
                postional_encodings=postional_encodings,
                **kwargs
            )
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_5VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.attn = Qwen3_5VisionBlock(config=config)
        self.mlp = Qwen3_5VisionBlock(config=config)

    
    @auto_docstring
    def forward(
        self,
        hidden_states:  torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        postional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            postional_embeddings=postional_embeddings,
            **kwargs
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(Qwen3_5PretrainedModel):
    config: Qwen3_5VisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen3_5VisionBlock"]
    _can_rocord_outputs = {
        "hidden_states": Qwen3_5VisionBlock,
        "attentions": Qwen3_5VisionAttention
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_szie = config.spatial_size
        self.spaital_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_emebd = Qwen3_5VisionPatchEmbed(
            config=config
        )

        self.pos_embed = nn.Embedding(config.num_positional_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_positional_embeddings ** 0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3_5VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False
        )
        self.gradient_checkpointing = False
        self.post_init()
    

    def rot_pos_emb(
        self,
        grid_thw: torch.Tensor
    ):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.decice

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None:]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size)


            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings
    

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t,h,w in grid_thw_list:
            h_idxs = torch.linsapce(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idx_floor = w_idxs.int()
            h_idx_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idx_ciel = (w_idxs.int() + 1).clip(max=self.num_grid_per_side -1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idx_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ciel = h_idx_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idx_ciel[None]).flatten(),
                (base_h_ciel[None].T + w_idx_ciel[None]).flatten(),
                (base_h_ciel[None].T + w_idx_ciel[None]).flatten()
            ] 

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None.T] * dw[None]).flatten(),
                ((dh[None].T * (1 - dw)[None])).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weight_list[i].tolist())
        
        idx_tensor = torch.tensor(
            idx_list,
            dtype=torch.long,
            device=device
        )
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=device
        )
        pos_emebd = self.pos_embed(
            idx_tensor
        ).to(device) * weight_tensor[:, :, None]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_emebd = pos_emebd.repeat(t, 1)
            pos_emebd = (
                pos_emebd.view(t, h // merge_size, merge_size,  w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
            )
            patch_pos_embeds_permute.append(pos_embed)
        
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds


    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs
    ): 
        hidden_states = self.patch_emebd(hidden_states)

        pos_emebd = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_emebd

        roatary_pos_emb = self.rotary_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        roatary_pos_emb = roatary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((roatary_pos_emb, roatary_pos_emb), dim=-1)
        positional_embeddings = (emb.cos(), emb.sin())

        cu_seqlen = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlen, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlen=cu_seqlens,
                positional_embeddings=positional_embeddings,
                **kwargs
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states
        )

@dataclass
@auto_docstring(
    custom_intro="""
    Base Class for LLava outputs, with hidden states and attention
    """
)

class Qwen3_5ModelOutputWithPast(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


class Qwen3_5TextModel(Qwen3_5PretrainedModel):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(
            config=config
        )
        self.gradient_checkpointing = True
        self.post_init()

        @merge_with_congif_defaults
        @capture_outputs
        @auto_docstring
        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            positon_ids: torch.LongTensor,
            past_key_values: Cache | None = None,
            inputs_emebds: torch.FloatTensor = None,
            use_cache: bool | None = None
        ):
            if (input_ids is None) ^ (inputs_emebds is not None):
                raise ValueError(
                    "You Must specify exactly one of input_ids or input_emebds"
                )
            
            if inputs_emebds is None:
                inputs_emebds = self.embed_tokens(input_ids)
            
            if use_cache and past_key_values is None:
                past_key_values = Qwen3_5DynamicCache(config=self.config)
            
            if positon_ids is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arrange(inputs_emebds.shape[1], device=inputs_emebds.device) + past_seen_tokens
                position_ids = positon_ids.view(1, 1, -1).expand(4, inputs_emebds.shape[0], -1)
            
            elif positon_ids.ndim == 2:
                positon_ids = positon_ids[None, ...].expand(4, position_ids.shape[0], 1)
            
            if positon_ids.ndim == 3 and positon_ids.shape[0] == 4:
                text_position_ids = positon_ids[0]
                positon_ids = positon_ids[1: ]
            
            else:
                text_position_ids = None
            
            casual_mask = create_casual_mask(
                config=self.config,
                input_emebds=inputs_emebds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids
            )

            linear_attn_mask = self._update_linear_attn_mask(
                attention_mask, past_key_values
            )
            hidden_states = inputs_emebds
            positonal_embeddings = self.roatary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(self.layers[:, self.config.num_hidden_layers]):
                layer_mask = linear_attn_mask if self.config.layer_types[layer_idx] == "linear_attention" else casual_mask

                hidden_states = decoder_layer(
                    hidden_states,
                    positonal_embeddings=positonal_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs
                )
                hidden_states = self.norm(hidden_states)

                return Qwen3_5ModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values
                )
        

        def _update_linear_attn_mask(
            self,
            attention_mask,
            past_key_values
        ):
            linear_attn_mask = attention_mask
            if (past_key_values is not None) and past_key_values.has_previous_state or (
                attention_mask is not None and torch.all(attention_mask == 1)
            ):
                linear_attn_mask = None

            return linear_attn_mask

@auto_docstring
class Qwen3_5Model(
    Qwen3_5PreTrainedModel
):
    base_model_prefix = "model"
    accept_loss_kwargs = False
    config: Qwen3_5Config
    _no_split_modules: ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3_5VisonModel._from_config(config.vision_config)
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embedding(self):
        return self.language_model.set_input_embeddings(value)
    
    def get_vision_positon_ids(self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None
    ):
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size
        )

        image_seq_len = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(
            start_position,
            start_positon + llm_grid_w,
            device=device
        ).repeat(
            llm_grid_h * llm_grid_t
        )
        positon_height = torch.arange(
            start_positon,
            start_positon +  llm_grid_h,
            device=device_encoding
        ).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        positon_temporal = torch.full((image_seq_len,), start_positon, device=device, dtype=torch.long)
        position_temporal = positon_temporal * time_interval
        vision_positon_ids = torch.stack(
            [position_temporal,
            positon_height,
            position_width]
        )
        return vision_positon_ids
    

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw,
                video_grid_thw[:, 0],
                dim=0
            )
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_positon_deltas = []
        positon_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_seq_len is not None else None,
            2: iter(image_grid_thw) if video_grid_thw is not None else None
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]
            
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))
            
            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arrange(text_len, device=input_ids.device).view(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters(modality_type))
                    vision_positon_ids = self.get_vision_positon_ids(
                        current_pos, grid_thw, 1, spatial_merge_size,
                        device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_positon_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
                
            
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                positon_ids[:, batch_size, attention_mask[batch_size].bool()] = llm_positons.to(positon_ids.device)
            else:
                position_ids[:, batch_size] = llm_positions.to(positon_height.device)
            
            mrope_position_deltas.append(llm_position.max() + 1 - len(current_input_ids))
        
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    





    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TrasnformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.get_image_features(
            pixel_values_videos,
            video_grid_thw,
            **kwargs
        )


