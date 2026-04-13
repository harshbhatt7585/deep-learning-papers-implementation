from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import IHAConfig


class Linear(nn.Linear):
    """Match nanochat's explicit compute-dtype behavior."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


class InterleavedHeadAttention(nn.Module):
    """
    Interleaved Head Attention (IHA) as described in arXiv:2602.21371.

    This implementation uses the global H * P -> H collapse from Definition 3.
    """

    def __init__(self, config: IHAConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_pseudo_heads = (
            config.num_pseudo_heads if config.num_pseudo_heads is not None else self.num_heads
        )
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.k_proj = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.v_proj = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=config.bias)

        identity = torch.eye(self.num_heads).unsqueeze(-1).expand(
            -1,
            -1,
            self.num_pseudo_heads,
        ).clone()
        self.alpha_q = nn.Parameter(identity.clone())
        self.alpha_k = nn.Parameter(identity.clone())
        self.alpha_v = nn.Parameter(identity.clone())

        collapse = torch.zeros(self.num_heads, self.num_heads * self.num_pseudo_heads)
        head_idx = torch.arange(self.num_heads)
        collapse[head_idx, head_idx * self.num_pseudo_heads] = 1.0
        self.collapse = nn.Parameter(collapse)

    def expand_position_ids(
        self,
        position_ids: torch.Tensor | None,
        *,
        batch_size: int | None = None,
        seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            if batch_size is None or seq_len is None or device is None:
                raise ValueError(
                    "batch_size, seq_len, and device are required when position_ids is None"
                )
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        if position_ids.dim() != 2:
            raise ValueError(f"position_ids must be 2D, got shape {tuple(position_ids.shape)}")

        pseudo_offsets = torch.arange(
            self.num_pseudo_heads,
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
        expanded = position_ids.unsqueeze(-1) * self.num_pseudo_heads + pseudo_offsets
        return expanded.reshape(position_ids.shape[0], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must have shape (batch, seq_len, hidden_size), got {tuple(hidden_states.shape)}"
            )

        normalized_attention_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"attention_mask must have shape (batch, seq_len), got {tuple(attention_mask.shape)}"
                )
            normalized_attention_mask = attention_mask.bool()

        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = torch.einsum("mhp,bmnd->bhpnd", self.alpha_q.to(query_states.dtype), query_states)
        key_states = torch.einsum("mhp,bmnd->bhpnd", self.alpha_k.to(key_states.dtype), key_states)
        value_states = torch.einsum("mhp,bmnd->bhpnd", self.alpha_v.to(value_states.dtype), value_states)

        query_states = self._merge_pseudo(query_states)
        key_states = self._merge_pseudo(key_states)
        value_states = self._merge_pseudo(value_states)

        attn_mask, query_keep = self._prepare_masks(
            seq_len=seq_len,
            batch_size=batch_size,
            attention_mask=normalized_attention_mask,
            device=hidden_states.device,
        )
        dropout_p = self.config.attention_dropout if self.training else 0.0

        if attn_mask is None and self._can_use_builtin_causal():
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            sdpa_mask = None if attn_mask is None else attn_mask[:, None, :, :]
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        if query_keep is not None:
            attn_output = attn_output * query_keep[:, None, :, None].to(attn_output.dtype)

        attn_output = attn_output.view(
            batch_size,
            self.num_heads,
            seq_len,
            self.num_pseudo_heads,
            self.head_dim,
        )
        attn_output = self._collapse_pseudo(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if normalized_attention_mask is not None:
            token_keep = normalized_attention_mask.to(attn_output.dtype)
            attn_output = attn_output * token_keep.unsqueeze(-1)

        return attn_output


    def _merge_pseudo(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_pseudo, seq_len, head_dim = states.shape
        return (
            states.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch_size, num_heads, seq_len * num_pseudo, head_dim)
        )

    def _collapse_pseudo(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _, head_dim = states.shape
        flat_states = states.permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size,
            seq_len,
            self.num_heads * self.num_pseudo_heads,
            head_dim,
        )
        return torch.einsum("ho,bnod->bhnd", self.collapse.to(flat_states.dtype), flat_states)

    def _can_use_builtin_causal(self) -> bool:
        return self.config.causal and self.config.window_size is None

    def _prepare_masks(
        self,
        *,
        seq_len: int,
        batch_size: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        expanded_keep = None
        if attention_mask is not None:
            expanded_keep = attention_mask.unsqueeze(-1).expand(
                batch_size,
                seq_len,
                self.num_pseudo_heads,
            ).reshape(
                batch_size,
                seq_len * self.num_pseudo_heads,
            )

        if attention_mask is None and self._can_use_builtin_causal():
            return None, expanded_keep

        total_seq = seq_len * self.num_pseudo_heads
        if not self.config.causal and self.config.window_size is None:
            base_mask = torch.ones(total_seq, total_seq, dtype=torch.bool, device=device)
        else:
            virtual_positions = torch.arange(total_seq, device=device)
            query_tokens = virtual_positions // self.num_pseudo_heads
            if self.config.causal:
                base_mask = virtual_positions[:, None] >= virtual_positions[None, :]
            else:
                base_mask = torch.ones(total_seq, total_seq, dtype=torch.bool, device=device)

            if self.config.window_size is not None:
                token_delta = query_tokens[:, None] - query_tokens[None, :]
                if self.config.causal:
                    base_mask = base_mask & (token_delta >= 0) & (token_delta < self.config.window_size)
                else:
                    base_mask = base_mask & (token_delta.abs() < self.config.window_size)

        if expanded_keep is None:
            return base_mask.unsqueeze(0).expand(batch_size, -1, -1), None

        key_mask = expanded_keep[:, None, :]
        return base_mask.unsqueeze(0) & key_mask, expanded_keep

if __name__ == "__main__":
    torch.manual_seed(0)

    config = IHAConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_pseudo_heads=2,
        attention_dropout=0.0,
        causal=True,
    )
    layer = InterleavedHeadAttention(config).eval()

    hidden_states = torch.randn(2, 5, 32)
    output = layer(hidden_states)

    print("Input shape:", tuple(hidden_states.shape))
    print("Output shape:", tuple(output.shape))
