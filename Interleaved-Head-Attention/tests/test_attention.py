from __future__ import annotations

import torch

from interleaved_head_attention import (
    InterleavedHeadAttention,
    InterleavedHeadAttentionConfig,
)


def _reference_mha_output(layer: InterleavedHeadAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    q = layer._reshape_projection(layer.q_proj(hidden_states))
    k = layer._reshape_projection(layer.k_proj(hidden_states))
    v = layer._reshape_projection(layer.v_proj(hidden_states))

    attn_scores = torch.matmul(q, k.transpose(-1, -2)) * layer.scaling
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, layer.hidden_size)
    )
    return layer.o_proj(attn_output)


def test_forward_preserves_shape_and_returns_weights() -> None:
    torch.manual_seed(0)
    config = InterleavedHeadAttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_pseudo_heads=3,
    )
    layer = InterleavedHeadAttention(config)
    hidden_states = torch.randn(2, 5, 32)

    output, weights = layer(hidden_states, need_weights=True)

    assert output.shape == hidden_states.shape
    assert weights is not None
    assert weights.shape == (2, 4, 15, 15)


def test_position_ids_expand_into_virtual_positions() -> None:
    config = InterleavedHeadAttentionConfig(
        hidden_size=24,
        num_attention_heads=3,
        num_pseudo_heads=3,
    )
    layer = InterleavedHeadAttention(config)
    position_ids = torch.tensor([[5, 6]])

    expanded = layer.expand_position_ids(position_ids)

    expected = torch.tensor([[15, 16, 17, 18, 19, 20]])
    assert torch.equal(expanded, expected)


def test_matches_standard_attention_when_pseudos_are_duplicates_in_non_causal_mode() -> None:
    torch.manual_seed(1)
    config = InterleavedHeadAttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_pseudo_heads=2,
        causal=False,
        mask_mode="none",
        attention_dropout=0.0,
    )
    layer = InterleavedHeadAttention(config).eval()
    hidden_states = torch.randn(2, 6, 32)

    output, _ = layer(hidden_states)
    reference = _reference_mha_output(layer, hidden_states)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)


def test_padding_mask_zeroes_padded_tokens() -> None:
    torch.manual_seed(2)
    config = InterleavedHeadAttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_pseudo_heads=2,
    )
    layer = InterleavedHeadAttention(config).eval()
    hidden_states = torch.randn(2, 4, 32)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    output, _ = layer(hidden_states, attention_mask=attention_mask)

    assert torch.count_nonzero(output[0, 3]).item() == 0
    assert torch.count_nonzero(output[1, 1:]).item() == 0


def test_global_collapse_mode_runs() -> None:
    torch.manual_seed(3)
    config = InterleavedHeadAttentionConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_pseudo_heads=2,
        collapse_mode="global",
    )
    layer = InterleavedHeadAttention(config).eval()
    hidden_states = torch.randn(1, 7, 32)

    output, weights = layer(hidden_states, need_weights=True)

    assert output.shape == hidden_states.shape
    assert weights is not None
