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