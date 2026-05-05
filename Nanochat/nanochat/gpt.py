"""
GPT model (rewrite, a lot simpler)
Notable featues:
 -- rotary embeddings (and no positonal embedings)
 -- QK norma
 -- united weights for token embedding and lm_head
 -- relu^2 activation in MLP
 -- norm after token embedding
 -- no learnable params in rmsnorm 
 -- no bias in linear layers
 -- Group-Query Attention (GQA) support for more efficient inference
 -- Flash Attention 3 itegration
"""

from functools import patial
from dataclasses import dataclass
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MounAdamW, DistMounAdamW

from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layers: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    # Slinding window attention pattern string titles across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"= all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1), )) # this will run on bf16

class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))




