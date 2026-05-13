"""DFlash drafter — block-diffusion speculative-decoding adapter.

Paper: Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative
Decoding" (arXiv:2602.06036, 2026). Reference implementation:
https://github.com/z-lab/dflash

This module is a faithful port of the DFlash drafter architecture to the
``TextDiffusionModel`` backbone in this repo. Compared to the reference:

* Same algorithmic core: cross-attention K/V is the concatenation of
  ``(target_hidden, drafter_self)``; drafter has no token embeddings or
  ``lm_head`` of its own — both are bound from the target at load time
  via :meth:`DFlashDraftModel.bind`.
* Same context feature: pick a handful of intermediate target layer outputs,
  concatenate along the feature dim, project through ``fc`` back to ``d_model``.
* Same training contract (see :func:`dflash_loss`): freeze the target, pick a
  block of ``block_size`` consecutive positions, replace positions ``1..bs-1``
  with the mask token, optimize the drafter to predict the masked tokens from
  ``(target_hidden_before_block, masked_block_embedded)``.
* Same inference contract (see ``spec_decode.speculate_dflash``): per block,
  one drafter forward + one target verify forward, accept by argmax-matching.

We don't implement the per-layer SWA from the reference yet; that's a
performance-only optimization for the drafter's self-attention component.
KV caching for both target and drafter is also deferred — Landing 1 prioritizes
correctness of the algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import (
    Linear,
    SDPA_BACKENDS,
    TextDiffusionConfig,
    TextDiffusionModel,
    apply_rotary_emb,
    norm,
)


# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------


@dataclass
class DFlashConfig:
    """Drafter architecture spec. Many fields mirror the target's config because
    the drafter inherits the target's ``d_model``, ``vocab_size``, and
    ``mask_token_id`` to keep ``embed_tokens`` and ``lm_head`` compatible.
    """

    # --- target-aligned (must match the bound target) ---
    target_d_model: int
    target_vocab_size: int
    target_n_layers: int
    target_n_heads: int
    target_pad_token_id: int
    mask_token_id: int

    # --- drafter-specific ---
    block_size: int = 16
    n_draft_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None
    ff_mult: int = 4
    gated_mlp: bool = False
    dropout: float = 0.0
    max_seq_len: int = 4096
    target_layer_ids: tuple[int, ...] | None = None  # which target layers feed h_ctx

    def __post_init__(self) -> None:
        if self.target_layer_ids is None:
            self.target_layer_ids = tuple(
                build_target_layer_ids(self.target_n_layers, self.n_draft_layers)
            )
        if len(self.target_layer_ids) == 0:
            raise ValueError("target_layer_ids must be non-empty")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.target_d_model % self.n_heads != 0:
            raise ValueError("target_d_model must be divisible by drafter n_heads")
        if self.n_kv_heads > self.n_heads or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if (self.target_d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head_dim must be even for rotary embeddings")
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2 (1 anchor token + >=1 draft slot)")

    @property
    def d_model(self) -> int:
        # Drafter shares d_model with target so the bound embed_tokens and
        # lm_head line up dimensionally.
        return self.target_d_model

    @property
    def context_feature_dim(self) -> int:
        return len(self.target_layer_ids) * self.d_model


def build_target_layer_ids(n_target_layers: int, n_draft_layers: int) -> list[int]:
    """Pick ``n_draft_layers`` intermediate layers from a ``n_target_layers``-deep
    target, spaced roughly evenly. Matches the heuristic in DFlash reference
    ``model.py``.
    """
    if n_target_layers <= 0:
        raise ValueError("n_target_layers must be positive")
    if n_draft_layers <= 0:
        raise ValueError("n_draft_layers must be positive")
    if n_draft_layers == 1:
        return [n_target_layers // 2]
    start = 1
    end = max(start + 1, n_target_layers - 3)
    span = end - start
    return [
        int(round(start + (i * span) / (n_draft_layers - 1)))
        for i in range(n_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Concatenate hidden states across the selected target layers (feature dim).

    ``hidden_states[0]`` is the embedding output and ``hidden_states[i]`` for
    ``i >= 1`` is the output of target block ``i - 1`` — same convention as the
    DFlash reference (which uses ``hidden_states[layer_id + 1]``).
    """
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return torch.cat(selected, dim=-1)


# ---------------------------------------------------------------------------
# Cross-attention + decoder layer
# ---------------------------------------------------------------------------


class DFlashCrossAttention(nn.Module):
    """Drafter attention with a context stream from the target.

    For each layer call we have two inputs:

    * ``x`` of shape ``(B, L_q, D)`` — the drafter's own hidden stream (i.e.
      the masked block's embeddings, evolving through the drafter's layers).
    * ``x_ctx`` of shape ``(B, L_ctx, D)`` — the target's intermediate hidden
      states for the prefix, after being projected through ``fc`` and normed.

    Q is computed from ``x`` only. K and V are computed independently for
    ``x`` and ``x_ctx`` and then concatenated along the sequence dim, so each
    drafter Q can attend to *both* the target's prefix representation and the
    drafter's current block (causal-free, bidirectional within the block).

    Position embeddings are applied per-stream with the right absolute offset:
    the prefix gets rotary positions ``0 .. L_ctx-1`` and the drafter block
    gets ``L_ctx .. L_ctx + L_q - 1`` so its tokens see the prefix as preceding.
    """

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        x_ctx: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, L_q, _ = x.shape
        L_ctx = x_ctx.shape[1]

        # Project Q from drafter stream only.
        q = self.q_proj(x).view(B, L_q, self.n_heads, self.head_dim)

        # K/V from both streams, projected independently then concatenated.
        k_ctx = self.k_proj(x_ctx).view(B, L_ctx, self.n_kv_heads, self.head_dim)
        v_ctx = self.v_proj(x_ctx).view(B, L_ctx, self.n_kv_heads, self.head_dim)
        k_q = self.k_proj(x).view(B, L_q, self.n_kv_heads, self.head_dim)
        v_q = self.v_proj(x).view(B, L_q, self.n_kv_heads, self.head_dim)

        # Rotary embeddings, per-stream with the correct absolute position.
        cos, sin = cos_sin
        # Drafter block lives at positions [L_ctx, L_ctx + L_q) so its tokens
        # see the prefix as coming before them in the rotary basis.
        q = apply_rotary_emb(q, cos[:, L_ctx : L_ctx + L_q], sin[:, L_ctx : L_ctx + L_q])
        k_ctx = apply_rotary_emb(k_ctx, cos[:, :L_ctx], sin[:, :L_ctx])
        k_q = apply_rotary_emb(k_q, cos[:, L_ctx : L_ctx + L_q], sin[:, L_ctx : L_ctx + L_q])

        # Q/K norm (Qwen3 / nanochat style — keeps attention stable at init).
        q = norm(q) * 1.2
        k_ctx = norm(k_ctx) * 1.2
        k_q = norm(k_q) * 1.2

        # Concatenate K and V across the sequence axis.
        k = torch.cat([k_ctx, k_q], dim=1)  # (B, L_ctx + L_q, n_kv_heads, head_dim)
        v = torch.cat([v_ctx, v_q], dim=1)

        # SDPA path (non-causal, fully bidirectional over the joint sequence).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        with sdpa_kernel(SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.o_proj(y)


class _MLP(nn.Module):
    """Same as model.MLP / GatedMLP, dispatched by config.gated_mlp."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        hidden = config.ff_mult * config.d_model
        self.gated = config.gated_mlp
        if self.gated:
            self.c_gate = Linear(config.d_model, hidden, bias=False)
            self.c_up = Linear(config.d_model, hidden, bias=False)
            self.c_proj = Linear(hidden, config.d_model, bias=False)
        else:
            self.c_fc = Linear(config.d_model, hidden, bias=False)
            self.c_proj = Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))
        x = self.c_fc(x)
        return self.c_proj(F.relu(x).square())


class DFlashDecoderLayer(nn.Module):
    """One drafter layer: pre-norm cross-attention + pre-norm MLP, residual on both."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.attn = DFlashCrossAttention(config)
        self.mlp = _MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        x_ctx: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), x_ctx, cos_sin=cos_sin)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# DFlashDraftModel
# ---------------------------------------------------------------------------


class DFlashDraftModel(nn.Module):
    """Block-diffusion drafter for DFlash speculative decoding.

    Owned weights:
        * ``fc`` — projects concatenated target hidden features (``len(target_layer_ids) * d``)
          down to ``d_model``.
        * ``hidden_norm`` — RMSNorm applied via the shared ``model.norm`` helper.
        * ``layers`` — stack of ``DFlashDecoderLayer``.
        * Rotary cos/sin buffers covering ``max_seq_len + block_size``.

    Borrowed weights (must call :meth:`bind` before any forward):
        * ``embed_tokens`` — target's ``token_emb``.
        * ``lm_head`` — target's ``lm_head``.
    """

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.fc = Linear(config.context_feature_dim, d, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(config.n_draft_layers)]
        )

        head_dim = d // config.n_heads
        # Allow rotary over the full prefix plus a block.
        rotary_len = config.max_seq_len + config.block_size
        cos, sin = self._precompute_rotary(rotary_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Borrowed at bind time. We assign these directly to ``self.__dict__``
        # in :meth:`bind` so nn.Module's ``__setattr__`` does NOT register them
        # as submodules — keeping the target's full-vocab matrices out of the
        # drafter's ``state_dict()`` and parameter list.
        self.__dict__["embed_tokens"] = None
        self.__dict__["lm_head"] = None

        self._init_weights()

    @property
    def device(self) -> torch.device:
        return self.fc.weight.device

    @torch.no_grad()
    def _init_weights(self) -> None:
        scale = 3**0.5 * self.config.d_model**-0.5
        nn.init.uniform_(self.fc.weight, -scale, scale)
        for layer in self.layers:
            nn.init.uniform_(layer.attn.q_proj.weight, -scale, scale)
            nn.init.uniform_(layer.attn.k_proj.weight, -scale, scale)
            nn.init.uniform_(layer.attn.v_proj.weight, -scale, scale)
            nn.init.zeros_(layer.attn.o_proj.weight)
            if isinstance(layer.mlp, _MLP):
                if layer.mlp.gated:
                    nn.init.uniform_(layer.mlp.c_gate.weight, -0.4 * scale, 0.4 * scale)
                    nn.init.uniform_(layer.mlp.c_up.weight, -0.4 * scale, 0.4 * scale)
                else:
                    nn.init.uniform_(layer.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
                nn.init.zeros_(layer.mlp.c_proj.weight)

    def _precompute_rotary(
        self, seq_len: int, head_dim: int, base: int = 100_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def bind(self, target: TextDiffusionModel) -> "DFlashDraftModel":
        """Borrow the target's token embeddings and lm_head.

        The drafter does *not* own these parameters — they're aliased to the
        target's modules so that (a) the drafter's vocab and embedding space
        match the target by construction, and (b) drafter checkpoint size stays
        small (no full-vocab matrices duplicated).

        We bypass ``nn.Module.__setattr__`` by writing directly to ``__dict__``
        so PyTorch does NOT register these as drafter submodules — they won't
        appear in ``self.parameters()`` or ``self.state_dict()``.
        """
        if target.config.vocab_size != self.config.target_vocab_size:
            raise ValueError(
                f"target vocab_size {target.config.vocab_size} != drafter target_vocab_size {self.config.target_vocab_size}"
            )
        if target.config.d_model != self.config.target_d_model:
            raise ValueError(
                f"target d_model {target.config.d_model} != drafter target_d_model {self.config.target_d_model}"
            )
        self.__dict__["embed_tokens"] = target.token_emb
        self.__dict__["lm_head"] = target.lm_head
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        *,
        logits_start: int = 0,
    ) -> torch.Tensor:
        """Run the drafter over one block.

        Args:
            input_ids: ``(B, L_q)`` long tensor — the drafter's masked block. The
                first position is the unmasked "anchor" token (last accepted from
                the previous block / prefill); positions ``1..L_q-1`` are ``mask_id``.
            target_hidden: ``(B, L_ctx, len(target_layer_ids) * d_model)`` — concat
                of selected target layer outputs over the *prefix*. The drafter
                projects this through ``fc`` and normalises before using it as
                cross-attention K/V.
            logits_start: if non-zero, return logits at positions ``[logits_start:]``
                of the block (i.e. ``logits_start=1`` skips the anchor).

        Returns:
            Logits over the drafter's positions, shape ``(B, L_q - logits_start, vocab)``.
        """
        if self.embed_tokens is None or self.lm_head is None:
            raise RuntimeError("Call .bind(target) before forward()")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")

        L_q = input_ids.size(1)
        L_ctx = target_hidden.size(1)
        if L_ctx + L_q > self.cos.size(1):
            raise ValueError(
                f"L_ctx + L_q ({L_ctx + L_q}) exceeds rotary table size {self.cos.size(1)}; "
                "increase config.max_seq_len."
            )

        # Match our target's convention: TextDiffusionModel (nanochat-style)
        # applies RMSNorm to the embedding before the stack so the residual
        # stream operates at unit magnitude end-to-end. Because the drafter
        # shares ``lm_head`` with the target via :meth:`bind`, the drafter's
        # final hidden state must live in the same scale the lm_head was
        # trained on -- which is unit-norm. Skipping this norm makes the
        # residual stream ~sqrt(d_model) larger than the layers' deltas, so
        # the drafter degenerates into "small corrections on a pass-through
        # embedding" and val_loss saturates ~0.4 nats higher (see BLOG entry
        # "tried no-embed-norm, reverted").
        x = norm(self.embed_tokens(input_ids))
        x = self.drop(x)

        # Project concatenated target features to drafter dim, then RMSNorm.
        x_ctx = norm(self.fc(target_hidden))

        cos_sin = (
            self.cos[:, : L_ctx + L_q],
            self.sin[:, : L_ctx + L_q],
        )

        for layer in self.layers:
            x = layer(x, x_ctx, cos_sin=cos_sin)

        x = norm(x)
        if logits_start:
            x = x[:, logits_start:, :]
        return self.lm_head(x)

    # ---------------------------------------------------------------------
    # Param count helpers (handy for logging)
    # ---------------------------------------------------------------------

    def owned_parameters(self) -> list[nn.Parameter]:
        """Drafter's own parameters (the borrowed ``embed_tokens`` / ``lm_head``
        are stored in ``self.__dict__`` and don't appear in ``self.parameters()``)."""
        return list(self.parameters())

    def num_owned_parameters(self) -> int:
        return sum(p.numel() for p in self.owned_parameters())


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------


def dflash_loss(
    drafter: DFlashDraftModel,
    target: TextDiffusionModel,
    input_ids: torch.Tensor,
    *,
    block_size: int | None = None,
    block_start: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Training objective for the DFlash drafter.

    Procedure (online, target frozen):

    1. Run the target on ``input_ids`` with ``output_hidden_states=True`` under
       ``torch.no_grad()``. The drafter does not learn the target.
    2. Concatenate target hidden states across ``target_layer_ids`` -> ``target_hidden``.
    3. Pick a block start position ``t`` (default: random). The block spans
       positions ``[t, t + bs)``. Position ``t`` is the unmasked "anchor";
       positions ``[t+1, t+bs)`` are replaced by ``mask_token_id``.
    4. Drafter forward consumes ``(masked_block, target_hidden[:, :t, :])`` and
       returns logits at the ``bs - 1`` masked positions.
    5. Loss is mean cross-entropy against the ground-truth tokens at those
       positions. Padding tokens are excluded via ``ignore_index``.

    Args:
        drafter: ``DFlashDraftModel``; must already be bound to ``target``.
        target: frozen ``TextDiffusionModel`` (eval mode, no_grad).
        input_ids: ``(B, T)`` long tensor.
        block_size: override drafter's default ``block_size``.
        block_start: deterministic block start (useful for tests). If None,
            a single random ``t`` is sampled per call and shared across the batch.

    Returns:
        (loss, metrics): loss is a scalar; metrics holds per-position acceptance
        proxies useful for W&B logging.
    """
    if drafter.embed_tokens is None or drafter.lm_head is None:
        raise RuntimeError("dflash_loss: drafter is not bound to target. Call drafter.bind(target).")

    bs = block_size or drafter.config.block_size
    if input_ids.size(1) < bs + 1:
        raise ValueError(
            f"dflash_loss: input_ids seq_len {input_ids.size(1)} too short for block_size {bs}; "
            "need at least bs + 1 tokens (1 prefix anchor + bs block positions)."
        )

    pad_id = drafter.config.target_pad_token_id
    mask_id = drafter.config.mask_token_id
    layer_ids = drafter.config.target_layer_ids
    T = input_ids.size(1)

    # Block start: leave at least 1 token of prefix and ensure full block fits.
    if block_start is None:
        # Use torch.randint on the same device to keep things deterministic w/
        # global seeding; one shared t per batch matches the reference loop.
        block_start = int(torch.randint(1, T - bs + 1, (1,)).item())
    else:
        if block_start < 1 or block_start + bs > T:
            raise ValueError(
                f"block_start {block_start} out of range for T={T}, bs={bs}"
            )

    # --- 1-2. Target forward (frozen, no grad) -> per-layer hidden states ----
    target.eval()
    with torch.no_grad():
        _, hidden_states = target(
            input_ids,
            attention_mask=None,
            causal=True,
            output_hidden_states=True,
        )
        target_hidden_full = extract_context_feature(hidden_states, layer_ids)
        # Slice to context-before-block; the anchor token will be embedded
        # directly by the drafter via embed_tokens.
        target_hidden = target_hidden_full[:, :block_start, :].contiguous()

    # --- 3. Build masked block ----------------------------------------------
    block_ids = input_ids[:, block_start : block_start + bs].clone()
    block_targets = input_ids[:, block_start + 1 : block_start + bs].clone()
    block_ids[:, 1:] = mask_id

    # --- 4. Drafter forward -------------------------------------------------
    logits = drafter(block_ids, target_hidden=target_hidden, logits_start=1)
    # logits shape (B, bs-1, vocab); each position predicts the token at that
    # masked slot of the block.

    # --- 5. Cross-entropy loss with padding ignored -------------------------
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        block_targets.reshape(-1),
        ignore_index=pad_id,
        reduction="mean",
    )

    # Acceptance proxy: fraction of (non-pad) masked positions where the
    # drafter's argmax matches the ground-truth token. This is a strong
    # upper-bound on inference-time acceptance at temperature=0.
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        valid = block_targets != pad_id
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            acceptance_proxy = (pred[valid] == block_targets[valid]).float().mean()
        else:
            acceptance_proxy = logits.new_tensor(0.0)

    metrics = {
        "dflash_loss": loss.detach(),
        "dflash_acceptance_proxy": acceptance_proxy.detach(),
        "dflash_block_start": logits.new_tensor(float(block_start)),
        "dflash_valid_positions": logits.new_tensor(float(n_valid)),
    }
    return loss, metrics
