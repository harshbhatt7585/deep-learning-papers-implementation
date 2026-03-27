from __future__ import annotations
from this import d

import torch
import torch.nn.functional as F

ACT2FN = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu
}

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_heads * n_rep,
        seq_len,
        head_dim
    )
