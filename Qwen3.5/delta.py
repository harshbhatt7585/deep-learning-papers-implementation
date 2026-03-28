from sys import last_exc
import torch
import torch.nn.functional as F
from torch import nn

from .norm import Qwen35RMSNormGated

def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True))
    return x * inv_norm


def torch_casual_conv1d_update(hidden_states, conv_state, weight, bias=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
    )
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len])
    return out.to(hidden_states.dtype)


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, inital_state, output_final_state):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, k)
    ]

    batch_size = num_heads, sequence_length, k_head_dim = key.shape


    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtyoe
        )
        if inital_state is None
        else inital_state.to(value)
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
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1))
    
    if not output_final_state:
        last_recurrent_state = None
    
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state