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


"""


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



    
