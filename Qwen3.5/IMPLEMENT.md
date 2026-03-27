# Qwen3.5 Implementation Guide

This document is a build companion for implementing Qwen3.5 from scratch in plain PyTorch, without depending on `transformers` at runtime.

The goal is:

- load the real Qwen3.5 weights
- run inference with your own model code
- understand every block well enough to debug shape and cache issues

This guide follows the architecture used by the official Hugging Face implementation, but explains it in implementation order rather than file order.

## Scope

There are two parts to Qwen3.5:

1. Text model
2. Multimodal model

You should implement the text model first. It is the critical path for inference parity.

The multimodal model adds:

- vision patch embedding
- vision attention blocks
- multimodal rotary position ids
- image/video placeholder replacement

If your goal is text inference only, stop after the text model and LM head.

---

## Recommended Build Order

Implement in this order:

1. Configuration
2. Tokenizer
3. RMSNorm
4. RoPE
5. Attention
6. MLP
7. Dynamic cache
8. Gated delta net
9. Decoder layer
10. Text model
11. LM head and generation
12. Multimodal extension

Do not start with the vision stack. It adds complexity without helping you validate the text path.

---

## 1. Configuration

You need a config object that contains every hyperparameter used by the model.

For a checkpoint, the values come from `config.json`.

The key fields you will see in Qwen3.5 text are:

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `rms_norm_eps`
- `attention_dropout`
- `hidden_act`
- `rope_parameters`
- `layer_types`
- `linear_num_key_heads`
- `linear_num_value_heads`
- `linear_key_head_dim`
- `linear_value_head_dim`
- `linear_conv_kernel_dim`

### Why config matters

Every module uses config values to determine:

- tensor shapes
- number of heads
- whether a layer is full attention or linear attention
- rotary embedding size
- cache size
- MLP hidden size

### Minimal config skeleton

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class Qwen35Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rope_parameters: dict[str, Any] | None = None
    layer_types: list[str] | None = None

    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
```

### What `layer_types` means

Qwen3.5 is hybrid:

- some layers are `full_attention`
- some layers are `linear_attention`

The model uses this list to decide which token mixer to instantiate in each decoder block.

Example:

```python
layer_types = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]
```

The exact pattern is checkpoint-specific, so read it from the config.

---

## 2. Tokenizer

Qwen3.5 uses byte-level BPE.

That means:

- text is normalized
- text is split by a regex pretokenizer
- bytes are encoded into token ids
- decoding reverses the byte-level mapping

### What you need

You need:

- `vocab.json` or equivalent vocab dictionary
- `merges.txt` or equivalent merge list

### Why this is separate from the model

The model only consumes token ids.
The tokenizer converts strings to ids and back.

### Minimal interface

```python
class Qwen35Tokenizer:
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...
```

### Implementation notes

- Use byte-level BPE, not whitespace tokenization.
- Preserve special tokens like `<|endoftext|>`.
- Keep the regex pretokenizer behavior close to the official tokenizer if you want parity.

---

## 3. RMSNorm

Qwen3.5 uses RMSNorm, but the exact formula matters.

The version in the official implementation behaves like:

```python
output = normalize(x) * (1 + weight)
```

where:

- `normalize(x)` is the root-mean-square normalization
- `weight` is a learnable vector initialized to zeros

### Why this matters

This is slightly different from some models that use:

```python
output = normalize(x) * weight
```

Here the layer starts as identity because `1 + 0 = 1`.

### Skeleton

```python
import torch
from torch import nn


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._norm(x.float())
        y = y * (1.0 + self.weight.float())
        return y.to(dtype=x.dtype)
```

### Shape

Input:

```python
[batch, seq_len, hidden_size]
```

Output:

```python
[batch, seq_len, hidden_size]
```

---

## 4. Rotary Position Embeddings

RoPE injects position information into attention by rotating query and key vectors.

Qwen3.5 text uses standard text RoPE.
The multimodal model uses 3-axis position ids, but the math is still RoPE-based.

### Core helper

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)
```

### Apply RoPE

```python
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

### Why split into `q_rot` and `q_pass`

Not all dimensions need to be rotated.
Some models only rotate a fraction of the head dimension.

The remaining dimensions are passed through unchanged.

---

## 5. KV Repetition

Qwen3.5 uses grouped key/value heads.

That means:

- query heads can be more numerous than key/value heads
- keys and values are repeated to match query head count

### Helper

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)
```

### Why this exists

If you have:

- 16 query heads
- 4 key/value heads

then each K/V head is shared across 4 Q heads.

This saves memory and compute.

---

## 6. Standard Attention

The full attention block is the familiar transformer attention, but with two Q branches:

- one branch becomes the query
- the other branch becomes a gate

### Important projection shapes

- `q_proj`: `hidden_size -> num_attention_heads * head_dim * 2`
- `k_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `v_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `o_proj`: `num_attention_heads * head_dim -> hidden_size`

### Why `q_proj` outputs 2x

The output is split into:

- actual query vectors
- a gating vector

The gate is applied after attention output.

### Skeleton

```python
class Qwen35Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_and_gate = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

### What to watch for

- `query_states` shape is `[batch, heads, seq, head_dim]`
- `key_states` and `value_states` are grouped before repetition
- gating happens after attention output, not before

---

## 7. MLP

The MLP is SwiGLU-style:

- `gate_proj`
- `up_proj`
- elementwise multiply after activation
- `down_proj`

### Skeleton

```python
class Qwen35MLP(nn.Module):
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### Intuition

- `gate_proj(x)` controls which channels are open
- `up_proj(x)` provides the value stream
- the product creates the gated MLP effect

---

## 8. Dynamic Cache

Qwen3.5 needs a cache because generation is autoregressive.

There are two cache families:

1. Full attention cache
2. Linear attention cache

### Full attention cache

Stores:

- keys
- values

Shape:

```python
[batch, heads, seq_len, head_dim]
```

### Linear attention cache

Stores:

- convolution state
- recurrent state

These are constant-shape states that do not grow with sequence length.

### Cache skeleton

```python
class Qwen35DynamicCache:
    def __init__(self, config):
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    @property
    def has_previous_state(self):
        return self.conv_states[self.last_linear_layer] is not None
```

### Why `reorder_cache` exists

Beam search keeps several candidate sequences.
When beams are pruned or reordered, the cache must be reordered too.

---

## 9. Linear Attention / Gated Delta Net

This is the most nonstandard part of Qwen3.5.

The linear-attention layers are not normal self-attention.
They use a gated delta rule with:

- query
- key
- value
- `beta`
- `g`

### Projections

The layer creates:

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_b`
- `in_proj_a`

Meaning:

- `qkv` provides query/key/value content
- `z` is an output gate
- `b` becomes `beta = sigmoid(b)`
- `a` becomes the decay signal `g`

### Why this layer exists

It gives the model a fast linear recurrent path alongside normal attention layers.

### Core helpers

```python
def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
```

This zeros out padded tokens before the mixing layers see them.

### Causal conv1d update

```python
def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = torch.nn.functional.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = torch.nn.functional.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out
```

### L2 normalization helper

```python
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
```

### Chunk gated delta rule

The official code contains a chunked version and a recurrent version.

You can implement the torch fallback first, then optimize later.

The important sequence of operations is:

1. transpose to `[batch, heads, seq, dim]`
2. optionally L2-normalize query and key
3. pad to chunk size
4. compute decay terms from `g`
5. build chunk attention
6. update recurrent state
7. return output and final state

### Why this is hard

This code mixes:

- attention-like matrix products
- recurrent state updates
- exponential decay
- chunk-level computation

Do not treat it like ordinary attention.

---

## 10. Gated Delta Layer

This is the layer that wraps the linear-attention math.

### Forward flow

1. Mask padding tokens.
2. Read previous conv and recurrent cache if available.
3. Project hidden states into `qkv`, `z`, `b`, `a`.
4. Apply causal convolution.
5. Split `qkv` into query/key/value.
6. Compute:
   - `beta = sigmoid(b)`
   - `g = -exp(A_log) * softplus(a + dt_bias)`
7. Repeat query/key heads if needed.
8. Run chunked or recurrent gated delta rule.
9. Save recurrent state in cache.
10. Apply gated RMSNorm with `z`.
11. Project back to hidden size.

### Skeleton

```python
class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
            )
        else:
            if cache_params is not None:
                conv_state = torch.nn.functional.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            mixed_qkv = torch.nn.functional.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=None, output_final_state=cache_params is not None
            )
        else:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state, output_final_state=cache_params is not None
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)
```

### What `z` means

`z` is an output gate.
It modulates the final state after the delta-rule output is normalized.

### What `b` means

`b` becomes `beta`.
It controls the strength of the update.

### What `a` means

`a` becomes a decay signal.
It controls how much old state survives.

---

## 11. Gated RMSNorm

This norm takes:

- hidden states
- a gate tensor

and applies normalization followed by SiLU gating.

### Skeleton

```python
class Qwen35RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
```

### Why this is used

It lets the model suppress or enhance channels based on the gate.

---

## 12. Decoder Layer

The decoder layer is the core residual block.

It always has:

- input norm
- token mixer
- residual add
- post-attention norm
- MLP
- residual add

### Token mixer choice

If the layer is:

- `full_attention`, use attention
- `linear_attention`, use gated delta net

### Skeleton

```python
class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config, layer_idx)

        self.mlp = Qwen35MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

### Why residuals are everywhere

Residual connections let the model:

- train deeper stacks
- preserve information
- make optimization stable

---

## 13. Text Model

The text model is:

- token embeddings
- decoder layer stack
- final RMSNorm
- optional cache
- optional position ids

### Forward flow

1. Ensure exactly one of `input_ids` or `inputs_embeds`.
2. Convert ids to embeddings if needed.
3. Create cache if generation uses it.
4. Build text position ids.
5. Build causal mask.
6. Build rotary embeddings.
7. Loop through all decoder layers.
8. Apply final norm.
9. Return final hidden states and cache.

### Skeleton

```python
class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(config=self.config)

        hidden_states = inputs_embeds

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        text_position_ids = position_ids[0] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else None
        causal_mask = build_causal_mask(self.config, inputs_embeds, attention_mask, past_key_values, text_position_ids)
        position_embeddings = build_rope_embeddings(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

### Why position ids can be 4D in the model

Qwen3.5 reserves one position axis for text and three for multimodal RoPE.
For text-only use, you can still keep the interface consistent.

---

## 14. LM Head

The LM head maps hidden states to vocabulary logits.

### Skeleton

```python
class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
```

### Weight tying

In the official model, the LM head is tied to the embedding table in some variants.
If you want exact compatibility, mirror the checkpoint convention.

---

## 15. Generation

Generation is not part of the model, but you need it for inference.

### Minimal greedy decode

1. Encode prompt to ids.
2. Run the full prompt once.
3. Keep the cache.
4. Feed the last generated token back in.
5. Repeat.

### Why caching matters

Without cache, every new token recomputes the whole prompt.
With cache, each step only processes the new token.

### What to implement

- greedy decode
- temperature sampling
- top-k
- top-p
- beam search

Beam search requires `reorder_cache`.

---

## 16. Multimodal Extension

Only implement this after text parity is stable.

The multimodal stack adds:

- image patch embedding
- vision attention blocks
- patch merger
- multimodal position ids
- image and video placeholder replacement

### Vision patch embedding

It converts a 3D input grid into patch tokens with a 3D convolution.

### Vision attention

The vision branch uses standard attention, not the gated delta net.

### `cu_seqlens`

This is a cumulative sequence-length array used by packed attention kernels.

For sequences with lengths `[4, 6, 3]`, `cu_seqlens = [0, 4, 10, 13]`.

It tells the kernel where each sample starts and ends in a packed tensor.

### Multimodal RoPE

The model computes separate position ids for:

- time
- height
- width

Then it builds RoPE embeddings from those ids.

---

## 17. Weight Loading

To load real Qwen3.5 weights, your module names should be stable.

### Strategy

1. Mirror official module names as closely as possible.
2. Load the checkpoint `state_dict`.
3. Rename keys only if your local names differ.
4. Verify tensor shapes before assignment.

### Typical mapping examples

- `model.embed_tokens.weight`
- `layers.0.self_attn.q_proj.weight`
- `layers.0.self_attn.k_proj.weight`
- `layers.0.mlp.gate_proj.weight`
- `lm_head.weight`

### Best practice

Write a converter script that prints:

- missing keys
- unexpected keys
- mismatched shapes

Do not silently drop keys.

---

## 18. Validation Plan

You should verify in this order:

1. Embedding output shape
2. One decoder layer output shape
3. Full prompt forward pass
4. LM head logits shape
5. Cache update
6. One-token incremental decode
7. Tokenizer roundtrip
8. Logit parity against the reference model

### Debugging rule

If outputs diverge:

- compare after embeddings
- compare after first attention layer
- compare after final norm
- compare logits

The first mismatch tells you where the bug is.

---

## 19. Common Failure Points

- Wrong tensor transpose before attention
- Wrong RoPE dimension
- Wrong grouped KV repetition
- Missing gate application in attention
- Wrong RMSNorm formula
- Broken cache update dimension
- Using full attention for layers that should be linear attention
- Forgetting to mask padding tokens in linear attention
- Loading weights into modules with mismatched names

---

## 20. What To Implement First In Code

If you want the shortest path to a working model, implement these files first:

```text
config.py
tokenizer.py
norm.py
rope.py
cache.py
attention.py
mlp.py
delta_net.py
decoder.py
model.py
generate.py
```

Then add:

```text
vision.py
multimodal.py
weight_loader.py
```

---

## 21. Practical Advice

- Keep every tensor shape explicit in comments.
- Print shapes at the boundaries of each module.
- Use a tiny test prompt before touching the full checkpoint.
- Prefer exactness over speed first.
- Only optimize after the outputs match.

---

## 22. Summary

The core idea of Qwen3.5 is:

- a hybrid text stack
- standard attention layers interleaved with linear recurrent layers
- RoPE for position encoding
- gated output mixing
- cache-aware generation

If you implement the text model carefully, the rest becomes a matter of extending the same patterns into the vision branch.

If you want, the next step is to split this guide into actual starter files:

- `config.py`
- `cache.py`
- `attention.py`
- `delta_net.py`
- `model.py`
- `generate.py`

and I can write those files one by one in the same style.
