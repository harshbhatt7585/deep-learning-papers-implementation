from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

from flash_attention import flash_attn


KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TextDiffusionConfig:
    vocab_size: int
    max_seq_len: int
    mask_token_id: int
    pad_token_id: int
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int | None = None
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4
    n_mtp_heads: int = 0
    mtp_arch: str = "linear"
    gated_mlp: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_kv_heads > self.n_heads or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head_dim must be even for rotary embeddings")
        if self.n_mtp_heads < 0:
            raise ValueError("n_mtp_heads must be non-negative")
        if self.mtp_arch not in {"linear", "deepseek"}:
            raise ValueError("mtp_arch must be 'linear' or 'deepseek'")


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Cast cos/sin to x.dtype so the result stays in the same dtype as x.
    # Otherwise fp32 cos/sin promotes bf16 q/k to fp32 and FA3 falls back to SDPA.
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    dim = x.shape[-1] // 2
    x1, x2 = x[..., :dim], x[..., dim:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class Linear(nn.Linear):
    """Bias-free Linear with fp32 master weights.

    The forward pass intentionally uses the inherited ``nn.Linear.forward`` so
    that ``torch.autocast`` can cache the bf16/fp16 cast of the weight across
    calls inside the same autocast region. Manually casting on every call
    (``weight.to(dtype=x.dtype)``) defeats that cache and allocates a fresh
    low-precision tensor for every Linear, every step.
    """


class SelfAttention(nn.Module):
    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.c_q = Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.c_k = Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.c_v = Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.c_proj = Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = cos_sin
        past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        q = apply_rotary_emb(q, cos[:, past_len : past_len + seq_len], sin[:, past_len : past_len + seq_len])
        k = apply_rotary_emb(k, cos[:, past_len : past_len + seq_len], sin[:, past_len : past_len + seq_len])

        q = norm(q) * 1.2
        k = norm(k) * 1.2

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        present = (k, v) if use_cache else None

        if attention_mask is None and past_key_value is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=causal)
            y = y.contiguous().view(batch_size, seq_len, self.d_model)
            y = self.c_proj(y)
            if use_cache:
                return y, present
            return y

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim == 3:
                attn_mask = attention_mask[:, None, :, :].bool()
            else:
                raise ValueError("attention_mask must be 2D or 3D")
            attn_mask = attn_mask.to(device=x.device)
        total_len = k.size(-2)
        needs_causal_mask = causal and not (past_len > 0 and seq_len == 1)
        if needs_causal_mask:
            query_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
            key_pos = torch.arange(total_len, device=x.device)
            causal_mask = key_pos[None, :] <= query_pos[:, None]
            causal_mask = causal_mask[None, None, :, :]
            attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask

        with sdpa_kernel(SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=needs_causal_mask and attn_mask is None,
            )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        y = self.c_proj(y)
        if use_cache:
            return y, present
        return y


class MLP(nn.Module):
    """ReLU-squared MLP (nanochat-style): 2 projections, no gating."""

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.d_model, config.ff_mult * config.d_model, bias=False)
        self.c_proj = Linear(config.ff_mult * config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP (LLaMA/Mixtral/Qwen-style): SiLU(gate) * up, then down."""

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        hidden = config.ff_mult * config.d_model
        self.c_gate = Linear(config.d_model, hidden, bias=False)
        self.c_up = Linear(config.d_model, hidden, bias=False)
        self.c_proj = Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = GatedMLP(config) if config.gated_mlp else MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attn(
            norm(x),
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            causal=causal,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            attn_out, present = attn_out
        x = x + attn_out
        x = x + self.mlp(norm(x))
        if use_cache:
            return x, present
        return x


class DeepSeekMTPModule(nn.Module):
    """DeepSeek-style MTP depth.

    Each depth consumes the previous hidden state plus the embedding of the
    previous token on the drafted path, projects them back to d_model, then runs
    one causal transformer block. The shared lm_head outside this module turns
    the returned hidden state into next-token logits.
    """

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.eh_proj = Linear(2 * config.d_model, config.d_model, bias=False)
        self.block = TransformerBlock(config)

    def forward(
        self,
        previous_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        token_emb: nn.Embedding,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        token_hidden = norm(token_emb(token_ids))
        x = self.eh_proj(torch.cat([norm(previous_hidden), token_hidden], dim=-1))
        x = self.block(x, cos_sin=cos_sin, causal=True)
        return norm(x)


class TextDiffusionModel(nn.Module):
    """Small bidirectional Transformer denoiser for masked text diffusion."""

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        if config.mtp_arch == "deepseek":
            self.mtp_heads = nn.ModuleList(
                [DeepSeekMTPModule(config) for _ in range(config.n_mtp_heads)]
            )
        else:
            # Medusa-style shared-vocab future heads: each MTP head is a small
            # d_model -> d_model projection whose output is normed and then
            # projected to vocab by the shared lm_head.
            self.mtp_heads = nn.ModuleList(
                [Linear(config.d_model, config.d_model, bias=False) for _ in range(config.n_mtp_heads)]
            )
        cos, sin = self._precompute_rotary_embeddings(config.max_seq_len, config.d_model // config.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.init_weights()

    @property
    def device(self) -> torch.device:
        return self.token_emb.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        causal: bool = False,
        return_hidden: bool = False,
        output_hidden_states: bool = False,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, KVCache]
        | tuple[torch.Tensor, torch.Tensor, KVCache]
        | tuple[torch.Tensor, list[torch.Tensor], KVCache]
    ):
        """Standard forward.

        ``return_hidden=True`` adds the final post-norm hidden state to the return tuple.
        ``output_hidden_states=True`` adds a *list* of per-layer hidden states (one per
        TransformerBlock output) instead — used by DFlash to capture intermediate
        target features for the drafter. The two flags are mutually exclusive; if both
        are set, the per-layer list takes precedence.
        """
        if input_ids.ndim not in (2, 3):
            raise ValueError(f"input_ids must be 2D or 3D, got shape {tuple(input_ids.shape)}")

        if input_ids.ndim == 3:
            if past_key_values is not None:
                raise ValueError("3D bagged input_ids cannot be used with past_key_values")
            batch_size, seq_len, bag_size = input_ids.shape
        else:
            batch_size, seq_len = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(1)
        total_len = past_len + seq_len
        if total_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {total_len} exceeds max_seq_len {self.config.max_seq_len}")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                f"past_key_values has {len(past_key_values)} layers, expected {len(self.blocks)}"
            )

        if input_ids.ndim == 3:
            # TST: average bag embeddings in fp32 (paper Appendix A) then cast back.
            emb = self.token_emb(input_ids)
            x = norm(emb.float().mean(dim=2).to(emb.dtype))
        else:
            x = norm(self.token_emb(input_ids))
        x = self.drop(x)
        cos_sin = (self.cos[:, :total_len], self.sin[:, :total_len])

        hidden_states_list: list[torch.Tensor] | None = None
        if output_hidden_states:
            # Match DFlash's convention: index 0 is the post-embedding state,
            # index i (i>=1) is the output of block i-1.
            hidden_states_list = [x]

        present_key_values: KVCache | None = [] if use_cache else None
        for idx, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[idx]
            block_out = block(
                x,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                causal=causal,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                x, present = block_out
                present_key_values.append(present)
            else:
                x = block_out
            if output_hidden_states:
                hidden_states_list.append(x)

        x = norm(x)
        logits = self.lm_head(x)
        if output_hidden_states:
            if use_cache:
                return logits, hidden_states_list, present_key_values
            return logits, hidden_states_list
        if return_hidden:
            if use_cache:
                return logits, x, present_key_values
            return logits, x
        if use_cache:
            return logits, present_key_values
        return logits

    @torch.no_grad()
    def init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        scale = 3**0.5 * self.config.d_model**-0.5

        def init_block(block: TransformerBlock) -> None:
            nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            nn.init.zeros_(block.attn.c_proj.weight)
            if isinstance(block.mlp, GatedMLP):
                nn.init.uniform_(block.mlp.c_gate.weight, -0.4 * scale, 0.4 * scale)
                nn.init.uniform_(block.mlp.c_up.weight, -0.4 * scale, 0.4 * scale)
            else:
                nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
            nn.init.zeros_(block.mlp.c_proj.weight)

        for head in self.mtp_heads:
            if isinstance(head, DeepSeekMTPModule):
                nn.init.uniform_(head.eh_proj.weight, -scale, scale)
                init_block(head.block)
            else:
                nn.init.uniform_(head.weight, -scale, scale)

        for block in self.blocks:
            init_block(block)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 100_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]


def make_masked_inputs(
    input_ids: torch.Tensor,
    *,
    mask_token_id: int,
    pad_token_id: int,
    mask_prob: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be between 0 and 1")

    valid_tokens = input_ids != pad_token_id
    random_mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_prob
    mask_positions = random_mask & valid_tokens

    noised = input_ids.clone()
    noised[mask_positions] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    return noised, labels


def diffusion_loss(
    model: TextDiffusionModel,
    input_ids: torch.Tensor,
    *,
    mask_prob: float = 0.3,
) -> torch.Tensor:
    config = model.config
    noised, labels = make_masked_inputs(
        input_ids,
        mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
        mask_prob=mask_prob,
    )
    attention_mask = noised != config.pad_token_id
    if bool(attention_mask.all()):
        attention_mask = None
    logits = model(noised, attention_mask=attention_mask)
    masked = labels != -100
    return F.cross_entropy(logits[masked], labels[masked])


def causal_mtp_loss(
    model: TextDiffusionModel,
    input_ids: torch.Tensor,
    *,
    mtp_loss_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if input_ids.size(1) < 2:
        raise ValueError("causal_mtp_loss requires seq_len >= 2")

    logits, hidden = model(input_ids, attention_mask=None, causal=True, return_hidden=True)
    main_loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
    )
    loss = main_loss
    aux_losses = []
    mtp_hidden: torch.Tensor | None = None
    for depth, head in enumerate(model.mtp_heads, start=1):
        offset = depth + 1
        if input_ids.size(1) <= offset:
            break
        if model.config.mtp_arch == "deepseek":
            if depth == 1:
                previous_hidden = hidden[:, :-offset, :]
            else:
                previous_hidden = mtp_hidden[:, :-1, :]
            token_ids = input_ids[:, depth:-1]
            cos_sin = (
                model.cos[:, : previous_hidden.size(1)],
                model.sin[:, : previous_hidden.size(1)],
            )
            mtp_hidden = head(
                previous_hidden,
                token_ids,
                token_emb=model.token_emb,
                cos_sin=cos_sin,
            )
            h_offset = mtp_hidden
        else:
            h_offset = norm(head(hidden[:, :-offset, :]))
        aux_logits = model.lm_head(h_offset)
        aux_loss = F.cross_entropy(
            aux_logits.contiguous().view(-1, aux_logits.size(-1)),
            input_ids[:, offset:].contiguous().view(-1),
        )
        aux_losses.append(aux_loss)
        loss = loss + mtp_loss_weight * aux_loss / max(1, len(model.mtp_heads))

    metrics = {"main_loss": main_loss.detach()}
    if aux_losses:
        metrics["mtp_loss"] = torch.stack([aux.detach() for aux in aux_losses]).mean()
    return loss, metrics


def build_block_diffusion_attention_mask(
    *,
    seq_len: int,
    block_length: int,
    device: torch.device,
) -> torch.Tensor:

    block_ids = torch.arange(seq_len, device=device) // block_length
    return block_ids[:, None] >= block_ids[None, :]


def _sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature == 0.0:
        probs = F.softmax(logits, dim=-1)
        confidence, tokens = probs.max(dim=-1)
        return tokens, confidence

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    scaled = logits / temperature
    if top_k is not None:
        kth_values = torch.topk(
            scaled,
            k=min(top_k, scaled.shape[-1]),
            dim=-1,
        ).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < kth_values, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(scaled, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
    tokens = tokens.view(probs.shape[:-1])
    confidence = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, confidence


@torch.no_grad()
def generate_causal(
    model: TextDiffusionModel,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressive generation for causal and causal+MTP checkpoints."""

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    model.eval()
    x = prompt_ids.to(model.device)
    requested_len = x.size(1) + gen_length
    max_len = min(requested_len, model.config.max_seq_len)

    while x.size(1) < max_len:
        logits = model(x, attention_mask=None, causal=True)
        next_token, _ = _sample_tokens(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        x = torch.cat([x, next_token[:, None]], dim=1)
        if eos_token_id is not None and int(next_token[0].item()) == eos_token_id:
            break

    return x[0]


@torch.no_grad()
def generate(
    model: TextDiffusionModel,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    block_length: int = 8,
    steps: int = 8,
    threshold: float = 0.7,
    editing_threshold: float | None = 0.9,
    max_post_steps: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Block-wise diffusion generation: fill masks from left blocks to right blocks."""

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    model.eval()
    config = model.config
    prompt_ids = prompt_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    requested_len = prompt_len + gen_length
    num_blocks = math.ceil(requested_len / block_length)
    total_len = num_blocks * block_length

    transfer_count = max(1, math.ceil(block_length / steps))
    full_attention_mask = build_block_diffusion_attention_mask(
        seq_len=total_len,
        block_length=block_length,
        device=model.device,
    ).unsqueeze(0)

    x = torch.full((1, total_len), config.mask_token_id, dtype=torch.long, device=model.device)
    x[:, :prompt_len] = prompt_ids
    first_generation_block = prompt_len // block_length

    for block_idx in range(first_generation_block, num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        active_slice = slice(block_start, block_end)
        prompt_in_block = torch.zeros(block_length, dtype=torch.bool, device=model.device)
        if block_start < prompt_len:
            prompt_in_block[: prompt_len - block_start] = True

        for step in range(steps + max_post_steps):
            old_block = x[:, active_slice].clone()
            active_masks = old_block == config.mask_token_id

            attention_mask = full_attention_mask[:, :block_end, :block_end]
            logits = model(x[:, :block_end], attention_mask=attention_mask)
            block_logits = logits[:, active_slice, :]
            candidates, confidence = _sample_tokens(
                block_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            accept = torch.zeros_like(active_masks)
            if active_masks.any():
                high_confidence = (confidence > threshold) & active_masks
                accept |= high_confidence
                if accept.sum() == 0:
                    masked_confidence = confidence.masked_fill(~active_masks, float("-inf"))
                    _, idx = torch.topk(
                        masked_confidence[0],
                        k=min(transfer_count, int(active_masks.sum().item())),
                    )
                    accept[0, idx] = True

            if editing_threshold is not None:
                editable = ~active_masks & ~prompt_in_block.unsqueeze(0)
                changed = candidates != old_block
                accept |= editable & changed & (confidence > editing_threshold)

            if accept.any():
                updated_block = x[:, active_slice].clone()
                updated_block[accept] = candidates[accept]
                x[:, active_slice] = updated_block

            if not active_masks.any():
                break
            if step >= steps - 1 and editing_threshold is None:
                break

        if eos_token_id is not None:
            generated = x[0, prompt_len:min(block_end, requested_len)]
            eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                return x[0, : prompt_len + int(eos_positions[0]) + 1]

    return x[0, :requested_len]


def main() -> None:
    raise SystemExit("Use train.py and sample.py; this project now supports only the LLaDA2.1 tokenizer.")


if __name__ == "__main__":
    main()
