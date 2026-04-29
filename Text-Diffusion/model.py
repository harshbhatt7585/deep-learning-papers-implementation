from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F


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

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_kv_heads > self.n_heads or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head_dim must be even for rotary embeddings")


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
    """Nanochat-style bias-free Linear with fp32 master weights."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


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
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos[:, :seq_len], sin[:, :seq_len])
        k = apply_rotary_emb(k, cos[:, :seq_len], sin[:, :seq_len])

        q = norm(q).transpose(1, 2) * 1.2
        k = norm(k).transpose(1, 2) * 1.2
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

        with sdpa_kernel(SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.d_model, config.ff_mult * config.d_model, bias=False)
        self.c_proj = Linear(config.ff_mult * config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), cos_sin=cos_sin, attention_mask=attention_mask)
        x = x + self.mlp(norm(x))
        return x


class TextDiffusionModel(nn.Module):
    """Small bidirectional Transformer denoiser for masked text diffusion."""

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
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
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        x = norm(self.token_emb(input_ids))
        x = self.drop(x)
        cos_sin = (self.cos[:, :seq_len], self.sin[:, :seq_len])

        for block in self.blocks:
            x = block(x, cos_sin=cos_sin, attention_mask=attention_mask)

        x = norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        scale = 3**0.5 * self.config.d_model**-0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
            nn.init.zeros_(block.mlp.c_proj.weight)

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
