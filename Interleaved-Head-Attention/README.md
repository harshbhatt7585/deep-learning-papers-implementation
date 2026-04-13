# Interleaved Head Attention

This directory contains a self-contained PyTorch implementation of [Interleaved Head Attention (IHA)](https://arxiv.org/pdf/2602.21371).

## What is implemented

- Standard `q_proj`, `k_proj`, `v_proj`, and `o_proj` layers.
- Learned pseudo-head mixing tensors `alpha_q`, `alpha_k`, and `alpha_v` with shape `(H, H, P)`.
- Token-major interleaving of pseudo heads into an expanded sequence of length `N * P`.
- Global `H * P -> H` collapse after attention.
- Flat causal masking over the fully interleaved virtual sequence.
- Optional local sliding-window restriction using `window_size` in original-token units.


## Quick smoke check

Run the module directly to execute a small random-input check:

```bash
python interleaved_head_attention/attention.py
```

It will construct a layer, run a forward pass on random inputs, and print the tensor shapes.

## Minimal usage

```python
import torch

from interleaved_head_attention import (
    InterleavedHeadAttention,
    IHAConfig,
)

config = IHAConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_pseudo_heads=8,
    attention_dropout=0.0,
    causal=True,
)

layer = InterleavedHeadAttention(config)
hidden_states = torch.randn(2, 128, 512)
output = layer(hidden_states)
print(output.shape)  # torch.Size([2, 128, 512])
```

## RoPE note

The paper assigns every pseudo token a distinct virtual position. If you apply RoPE externally, use `expand_position_ids(...)` to build those virtual positions before applying the rotary embedding to the interleaved `Q` and `K` tensors.
