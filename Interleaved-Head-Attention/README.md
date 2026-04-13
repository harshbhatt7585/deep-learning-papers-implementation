# Interleaved Head Attention

This directory now contains a self-contained PyTorch implementation of Interleaved Head Attention (IHA) from the paper by Duvvuri et al.

## What is implemented

- Standard `q_proj`, `k_proj`, `v_proj`, and `o_proj` layers.
- Learned pseudo-head mixing tensors `alpha_q`, `alpha_k`, and `alpha_v` with shape `(H, H, P)`.
- Token-major interleaving of pseudo heads into an expanded sequence of length `N * P`.
- Two collapse variants:
  - `per_head`: matches Algorithm 1 in the paper.
  - `global`: matches the broader `H * P -> H` collapse used in Definition 3.
- Three masking modes:
  - `token_causal`: causal at the original token level.
  - `flat_causal`: strict causal masking over the fully interleaved virtual sequence.
  - `none`: no causal masking.
- Optional local sliding-window restriction using `window_size` in original-token units.

## Files

- `interleaved_head_attention/attention.py`: core implementation and config dataclass.

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
    InterleavedHeadAttentionConfig,
)

config = InterleavedHeadAttentionConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_pseudo_heads=8,
    attention_dropout=0.0,
    causal=True,
    mask_mode="token_causal",
    collapse_mode="per_head",
)

layer = InterleavedHeadAttention(config)
hidden_states = torch.randn(2, 128, 512)
output, _ = layer(hidden_states)
print(output.shape)  # torch.Size([2, 128, 512])
```

## RoPE note

The paper assigns every pseudo token a distinct virtual position. If you apply RoPE externally, use `expand_position_ids(...)` to build those virtual positions before applying the rotary embedding to the interleaved `Q` and `K` tensors.
