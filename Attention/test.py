import torch
from attention import MultiHeadAttention
import math

def test_attention_shape():
    """Test if the output shape matches the input shape."""
    batch_size, seq_len, d_embed, n_heads = 2, 5, 64, 8  # Sample dimensions
    mha = MultiHeadAttention(n_heads, d_embed)
    
    x = torch.randn(batch_size, seq_len, d_embed)
    output = mha(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, but got {output.shape}"
    print("test_attention_shape passed!")

def test_qkv_splitting():
    """Test if Q, K, V tensors are split and reshaped correctly."""
    batch_size, seq_len, d_embed, n_heads = 2, 5, 64, 8
    mha = MultiHeadAttention(n_heads, d_embed)
    
    x = torch.randn(batch_size, seq_len, d_embed)
    x_proj = mha.in_proj(x)  # (batch, seq, 3*d_embed)
    
    q, k, v = x_proj.chunk(3, dim=-1)
    
    assert q.shape == (batch_size, seq_len, d_embed), f"Q shape mismatch: {q.shape}"
    assert k.shape == (batch_size, seq_len, d_embed), f"K shape mismatch: {k.shape}"
    assert v.shape == (batch_size, seq_len, d_embed), f"V shape mismatch: {v.shape}"
    
    print("test_qkv_splitting passed!")

def test_masking():
    """Test if causal masking is applied correctly."""
    batch_size, n_heads, seq_len, d_heads = 2, 8, 5, 8
    weight = torch.randn(batch_size, n_heads, seq_len, seq_len)

    mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
    weight.masked_fill_(mask, -torch.inf)

    assert (weight[:, :, torch.triu_indices(seq_len, seq_len, 1)[0], torch.triu_indices(seq_len, seq_len, 1)[1]] == -torch.inf).all(), \
        "Masking failed: Upper triangle should be -inf"

    print("test_masking passed!")

def test_softmax_stability():
    """Test if softmax produces valid probabilities (no NaNs or Infs)."""
    batch_size, n_heads, seq_len, d_heads = 2, 8, 5, 8
    weight = torch.randn(batch_size, n_heads, seq_len, seq_len)
    weight /= math.sqrt(d_heads)  # Apply scaling

    mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
    weight.masked_fill_(mask, -torch.inf)

    softmax_weight = torch.nn.functional.softmax(weight, dim=-1)

    assert torch.isnan(softmax_weight).sum() == 0, "Softmax output contains NaNs"
    assert torch.isinf(softmax_weight).sum() == 0, "Softmax output contains Infs"

    print("test_softmax_stability passed!")

# Run all tests
test_attention_shape()
test_qkv_splitting()
test_masking()
test_softmax_stability()