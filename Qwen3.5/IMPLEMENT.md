# Build Qwen3.5 From Scratch

This guide is not a model summary. It is a build plan for taking a blank directory and ending with a runnable, weight-loaded, text-only Qwen3.5 inference implementation in plain PyTorch.

The target is:

- no `transformers` in your runtime model code
- your own modules and generation loop
- ability to load real Qwen3.5 text weights
- enough structure to extend to multimodal later

This guide intentionally focuses on the text model first. Qwen3.5 multimodal is a second phase. If you try to implement vision, MRoPE, placeholder replacement, and the hybrid text stack at the same time, you will make debugging much harder than it needs to be.

## What you are building

For the first working version, build this:

```text
qwen35/
  __init__.py
  config.py
  utils.py
  norm.py
  rope.py
  mask.py
  cache.py
  attention.py
  mlp.py
  delta.py
  decoder.py
  model.py
  loader.py
  generate.py
run.py
```

The end state of phase 1 is:

1. `python run.py --prompt "Hello"` works.
2. The model loads a real checkpoint.
3. Greedy decoding works with cache.
4. The outputs are at least structurally correct.
5. Then you compare logits against the reference implementation and tighten parity.

## Ground rules

Before you write any model code, keep these constraints in mind:

- Qwen3.5 text is not a plain transformer.
- It is a hybrid stack of `full_attention` and `linear_attention`.
- The linear path is implemented by `Qwen3_5GatedDeltaNet`.
- If you skip the gated delta path and replace it with standard attention, you are no longer implementing Qwen3.5.
- The shortest correct path is text-only first.

## Phase 0: Understand what is essential

You do not need all of Hugging Face.

You do need these ideas:

1. Config loading
2. Token embedding
3. RMSNorm
4. RoPE
5. Full attention block
6. Gated delta block
7. Hybrid decoder stack
8. KV/recurrent cache
9. LM head
10. Generation loop
11. Weight loading

You can defer these:

- multimodal inputs
- vision tower
- packed vision attention
- MRoPE for images/videos
- beam search
- optimized kernels

## Phase 1: Create the project skeleton

Create the package structure above first. Do not start by writing one giant `model.py`. You need separable modules so you can debug each piece.

The ownership of each file should be:

- `config.py`: read and validate checkpoint config
- `utils.py`: helpers like `repeat_kv`, activation lookup
- `norm.py`: RMSNorm and gated RMSNorm
- `rope.py`: RoPE helpers
- `mask.py`: causal mask construction
- `cache.py`: full-attention and recurrent cache
- `attention.py`: Qwen3.5 full-attention block
- `mlp.py`: SwiGLU MLP
- `delta.py`: linear-attention path
- `decoder.py`: decoder layer
- `model.py`: text model and LM model
- `loader.py`: checkpoint loading and key mapping
- `generate.py`: autoregressive decode
- `run.py`: command-line entrypoint

## Phase 2: Implement config loading first

The config is the contract between the checkpoint and your code. If the config is wrong, every shape after that is wrong.

Write `qwen35/config.py` first.

Use a dataclass and a `from_json` loader:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    rms_norm_eps: float
    hidden_act: str
    attention_dropout: float
    attention_bias: bool
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    rope_parameters: dict[str, Any]
    layer_types: list[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen35Config":
        data = json.loads(Path(path).read_text())
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            rms_norm_eps=data["rms_norm_eps"],
            hidden_act=data["hidden_act"],
            attention_dropout=data.get("attention_dropout", 0.0),
            attention_bias=data.get("attention_bias", False),
            pad_token_id=data.get("pad_token_id"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            rope_parameters=data["rope_parameters"],
            layer_types=data["layer_types"],
            linear_num_key_heads=data["linear_num_key_heads"],
            linear_num_value_heads=data["linear_num_value_heads"],
            linear_key_head_dim=data["linear_key_head_dim"],
            linear_value_head_dim=data["linear_value_head_dim"],
            linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
        )
```

### Why this first

Every other file depends on these fields. Do not hardcode the architecture if you want to load real weights.

### Stop and test

Before moving on:

1. Point this loader at a real Qwen3.5 `config.json`.
2. Print `hidden_size`, `num_hidden_layers`, and `layer_types[:8]`.
3. Confirm the values look sensible.

If this step is wrong, fix it now.

## Phase 3: Implement utilities and activation lookup

Write `qwen35/utils.py`.

You need:

- `repeat_kv`
- activation lookup

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


ACT2FN = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
```

### Why this matters

Qwen3.5 uses grouped KV attention. Query heads and KV heads are not always equal.

## Phase 4: Implement normalization

Write `qwen35/norm.py`.

You need two layers:

1. `Qwen35RMSNorm`
2. `Qwen35RMSNormGated`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float())
        out = out * (1.0 + self.weight.float())
        return out.to(dtype=x.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)
```

### Important detail

The regular RMSNorm here is not `weight * normalized_x`. It is `(1 + weight) * normalized_x`.

That detail matters for weight parity.

## Phase 5: Implement RoPE

Write `qwen35/rope.py`.

For text-only inference, start with standard RoPE. Ignore multimodal position ids for now.

```python
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen35RotaryEmbedding(torch.nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            position_ids.shape[0], position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


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

### Why this is enough for phase 1

Text-only Qwen3.5 still needs RoPE. You can keep the same interface shape as the reference model and ignore multimodal semantics at first.

## Phase 6: Implement causal masking

Write `qwen35/mask.py`.

Start with a standard causal mask builder.

```python
from __future__ import annotations

import torch


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + kv_length - query_length)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, kv_length)

    if attention_mask is None:
        return causal

    # attention_mask: [batch, kv_length]
    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value
    return causal + padding_mask
```

### Why start simple

You do not need to reimplement every HF masking branch to get a working model. You need a correct causal mask for decode and prompt processing.

## Phase 7: Implement cache

Write `qwen35/cache.py`.

The cache must support both:

- full attention K/V cache
- linear-attention recurrent state

```python
from __future__ import annotations

import torch


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

    def update(self, key_states, value_states, layer_idx):
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
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(self.key_cache[layer_idx].device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(self.value_cache[layer_idx].device))
            if self.conv_states[layer_idx] is not None:
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(self.conv_states[layer_idx].device))
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx.to(self.recurrent_states[layer_idx].device))

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    @property
    def has_previous_state(self) -> bool:
        return self.conv_states[self.last_linear_layer] is not None
```

### Stop and test

Make a fake tensor:

```python
torch.randn(2, 4, 5, 32)
```

Append another one of length 1 and check the seq length becomes 6.

## Phase 8: Implement the full-attention block

Write `qwen35/attention.py`.

This is not standard attention copied from a tutorial. Qwen3.5 adds:

- doubled Q projection
- query gate
- q/k RMSNorm on the head dimension
- grouped KV heads

```python
from __future__ import annotations

import torch
from torch import nn

from .norm import Qwen35RMSNorm
from .rope import apply_rotary_pos_emb
from .utils import repeat_kv


class Qwen35Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
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
        return attn_output
```

### Stop and test

Construct one attention block with a tiny fake config and verify:

- input `[B, T, H]`
- output `[B, T, H]`

Do that before touching the linear-attention path.

## Phase 9: Implement the MLP

Write `qwen35/mlp.py`.

```python
from __future__ import annotations

from torch import nn

from .utils import ACT2FN


class Qwen35MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

This part is straightforward. If this breaks, the issue is almost always a shape mismatch or wrong activation.

## Phase 10: Implement the linear-attention path

Write `qwen35/delta.py`.

This is the hardest file in the project.

Do not try to be clever here. Port the reference math as literally as possible.

### Step 10.1: helper functions

Start with the helpers:

```python
from __future__ import annotations

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
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)
```

### Step 10.2: recurrent kernel

Implement the recurrent version first because it is conceptually easier:

```python
def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
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
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state
```

### Step 10.3: chunk path

For prompt prefill, you need the chunked path too. Port it directly from the reference implementation. Do not simplify the algorithm until you have parity.

The reference math is long. Keep the exact shape transitions:

1. transpose to `[B, H, T, D]`
2. pad to `chunk_size`
3. compute `v_beta` and `k_beta`
4. reshape into chunks
5. `g = g.cumsum(dim=-1)`
6. compute decay masks
7. do within-chunk recurrence
8. update last recurrent state per chunk
9. reshape back to `[B, T, H, Dv]`

If you do not port this carefully, prompt processing will differ from one-token decoding.

### Step 10.4: full layer wrapper

Now wrap these helpers into the actual module:

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
        conv_state = cache_params.conv_states[self.layer_idx] if cache_params is not None else None
        recurrent_state = cache_params.recurrent_states[self.layer_idx] if cache_params is not None else None

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
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if use_precomputed_states:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g, beta, recurrent_state, cache_params is not None
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g, beta, initial_state=None, output_final_state=cache_params is not None
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)
```

### What this layer is doing

This layer replaces normal self-attention in the linear-attention blocks. It learns:

- content vectors through `qkv`
- an update strength through `b`
- a decay term through `a`
- an output gate through `z`

That is the core of Qwen3.5’s nonstandard architecture.

## Phase 11: Implement the decoder layer

Write `qwen35/decoder.py`.

```python
from __future__ import annotations

from torch import nn

from .attention import Qwen35Attention
from .delta import Qwen35GatedDeltaNet
from .mlp import Qwen35MLP
from .norm import Qwen35RMSNorm


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.token_mixer = Qwen35GatedDeltaNet(config, layer_idx)
        else:
            self.token_mixer = Qwen35Attention(config, layer_idx)

        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.token_mixer(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = self.token_mixer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

At this point, you now have the actual building blocks of the text stack.

## Phase 12: Implement the text model

Write `qwen35/model.py`.

Start with the base text model:

```python
from __future__ import annotations

import torch
from torch import nn

from .cache import Qwen35DynamicCache
from .decoder import Qwen35DecoderLayer
from .mask import build_causal_mask
from .norm import Qwen35RMSNorm
from .rope import Qwen35RotaryEmbedding


class Qwen35TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=True):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen35DynamicCache(self.config)

        batch_size, seq_len, _ = inputs_embeds.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)

        # Keep the same interface style as the reference model.
        rope_position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, rope_position_ids)

        kv_length = seq_len + past_seen_tokens
        causal_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_mask = attention_mask if getattr(layer, "layer_type", None) == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

### Important note on masks

For the linear-attention path, the reference model uses a different mask behavior than full attention. For your first pass, preserving the padding mask behavior for linear layers is enough to get the architecture wired correctly.

## Phase 13: Add the LM head

Continue in `qwen35/model.py`:

```python
class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=True):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
```

### Stop and test

With a toy config, run:

```python
model = Qwen35ForCausalLM(config)
ids = torch.randint(0, config.vocab_size, (2, 8))
logits, cache = model(input_ids=ids)
print(logits.shape)
```

Expected shape:

```python
[2, 8, vocab_size]
```

If this fails, do not touch generation yet. Fix the forward path first.

## Phase 14: Implement checkpoint loading

Write `qwen35/loader.py`.

The pragmatic version is:

1. load `safetensors` or PyTorch checkpoint
2. inspect key names
3. map keys if necessary
4. call `load_state_dict(strict=False)`
5. print missing and unexpected keys

Example skeleton:

```python
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu")


def load_weights(model, checkpoint_path: str | Path):
    state_dict = load_state_dict_file(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected
```

### Reality check

You will probably need a mapping function because real checkpoints are usually sharded and sometimes prefixed.

What you want here is not elegance. You want visibility:

- which keys matched
- which keys did not
- which tensors had mismatched names

## Phase 15: Implement greedy generation

Write `qwen35/generate.py`.

Start with greedy decode only.

```python
from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate(model, input_ids, attention_mask=None, max_new_tokens: int = 32):
    model.eval()
    logits, cache = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    generated = input_ids

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

        logits, cache = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    return generated
```

### Why only greedy first

Sampling and beam search are easy to add later. If greedy does not work, nothing else matters.

## Phase 16: Add a runnable entrypoint

Write `run.py`.

The first version does not need a full tokenizer implementation. You can temporarily test using fake ids or a minimal external tokenizer wrapper if your immediate goal is model validation.

Minimal structure:

```python
from pathlib import Path

import torch

from qwen35.config import Qwen35Config
from qwen35.generate import greedy_generate
from qwen35.loader import load_weights
from qwen35.model import Qwen35ForCausalLM


def main():
    config = Qwen35Config.from_json("config.json")
    model = Qwen35ForCausalLM(config)
    missing, unexpected = load_weights(model, "model.safetensors")
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))

    # Temporary fake prompt for plumbing test.
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    out = greedy_generate(model, input_ids, max_new_tokens=4)
    print(out.shape)


if __name__ == "__main__":
    main()
```

### Why this is acceptable

At this stage, your objective is not user-facing inference yet. Your objective is to prove:

- the model instantiates
- the weights load
- the forward pass runs
- the cache updates
- generation loops without shape errors

## Phase 17: Tokenizer

You do need the real tokenizer before this becomes useful. But tokenizer work should not block model debugging.

There are two sensible paths:

1. Implement Qwen’s tokenizer yourself from vocab and merges.
2. Temporarily use a reference tokenizer only for preprocessing while your runtime model remains your own.

If your stated goal is to avoid Hugging Face entirely, do option 1. But do it after the model path is stable.

Tokenizer implementation tasks:

1. Read vocab
2. Read merges
3. Implement byte-level preprocessing
4. Implement BPE merge loop
5. Implement special-token handling
6. Implement decode

If you do tokenizer first, you will spend time debugging text normalization while your model still does not run. That is the wrong order.

## Phase 18: Validation strategy

Once the model runs, validate in layers.

### Validation ladder

1. Check module output shapes.
2. Check one block at a time on random inputs.
3. Check full model forward on random token ids.
4. Check cached decoding with a one-token step.
5. Compare logits against the reference implementation.

### Best debugging workflow

If logits do not match:

1. compare embeddings
2. compare first decoder output
3. compare first full-attention layer
4. compare first linear-attention layer
5. compare final norm output
6. compare LM logits

The first mismatch is the bug location.

## Phase 19: What will likely break

These are the highest-risk parts:

1. Wrong RoPE dimensions
2. Wrong grouped-KV repetition
3. Wrong RMSNorm formula
4. Wrong cache concat dimension
5. Wrong linear-attention recurrent update
6. Wrong conv state padding
7. Wrong mask handed to linear-attention layers
8. Wrong parameter names during weight load

If your first attempt fails, assume the issue is one of those.

## Phase 20: What is enough for the first milestone

You have reached milestone 1 when all of this is true:

1. `Qwen35ForCausalLM(config)` instantiates.
2. A random-token forward pass returns logits.
3. A real checkpoint loads with only understood key mismatches.
4. One-token cached generation works.
5. No shape errors occur in either full-attention or linear-attention layers.

That is the point where you have built Qwen3.5 from 0 to 1.

It is not the point where you have perfect parity.

## Phase 21: What comes after milestone 1

Once the text-only model runs, do these in order:

1. implement the tokenizer properly
2. add logit parity tests against the reference model
3. add sampling
4. add beam search
5. then add multimodal support

Multimodal support means adding:

- vision patch embedding
- vision transformer
- patch merger
- multimodal token replacement
- 3D position ids
- MRoPE
- `cu_seqlens` handling

That is a second project, not a detail.

## Final advice

The correct way to build this is:

1. get the text path running
2. get the weights loading
3. get greedy generation working
4. only then chase exact parity

If you try to build the whole official feature set in one pass, you will end up debugging ten interacting problems instead of one.

If you want the next step, I can turn this guide into actual starter code files in this repo so you can begin implementing directly beside the guide.
