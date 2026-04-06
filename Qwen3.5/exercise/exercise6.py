"""
In GatedDeltaNet

we have multiple layers, like
mixed_qkv
z
b
a
core_attn

qkv goes through torch_causal_conv1d_update

if it's per step decoding (inference), then we can use cache

beta = sigmoid(b)
g = -exp(A_log) * softplus(A_raw + dt_bias)


input: [batch, seq_len, hidden_size]
output: [batch, seq_len, hidden_size]

"""


from delta import apply_mask_to_padding_states
from norm import Qwen35RMSNorm
from torch import nn
import torch
import torch.nn.functional as F



def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=True):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states[:, :, -state_len])
    out = F.conv1d(hidden_states_new, weight, bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len])
    return out.to(hidden_states.dtype)


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state

    

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int 
    ):
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
    
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Paramerter(torch.log(torch.empty(self.num_v_heads)._unfiform(0, 16)))
        self.norm = Qwen35RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.head_v_dim, self.hidden_size)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=True)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=True)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_headsm bias=True)

    
    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1
        conv_state = cache_params.conv_state[self.layer_idx] if cache_params is not None else None
        recurrent_state = cache_params.recurrent_state[self.layer_idx] if cache_params is not None else None

        # [batch, seq_len, dim]
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2) # -> [batch, dim, seq_len]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_v_dim)

            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

            if self.num_v_heads // self.num_k_heads > 1:
                rep = self.num_v_heads // self.num_k_heads
                query = query.repeat_interleave(rep, dim=2)
                key = key.repeat_interleave(rep, dim=2)
            
            if use_precomputed_states:
                core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                    query, key, value, g, beta, recurrent_state, cache_params is not None
                )
            
            else:
                core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                    query, key, value, g, beta, recurrent_state, cache_params is not None
                )

            if cache_params is not None:
                cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
            
            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
            return self.out_proj(core_attn_out)
        


