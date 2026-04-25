from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class SimpleCharTokenizer:
    """Tiny character tokenizer for learning the algorithm without extra deps."""

    pad_token = "<pad>"
    mask_token = "<mask>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, chars: Iterable[str]) -> None:
        special_tokens = [
            self.pad_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        vocab = special_tokens + sorted(set(chars))
        self.id_to_token = list(dict.fromkeys(vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SimpleCharTokenizer":
        chars = set()
        for text in texts:
            chars.update(text)
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.mask_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        pieces = []
        special = {
            self.pad_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        }
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)


@dataclass
class TextDiffusionConfig:
    vocab_size: int
    max_seq_len: int
    mask_token_id: int
    pad_token_id: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4


class TransformerBlock(nn.Module):
    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_mult * config.d_model),
            nn.GELU(),
            nn.Linear(config.ff_mult * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                key_padding_mask = ~attention_mask.bool()
            elif attention_mask.ndim in {3, 4}:
                keep_mask = attention_mask
                if keep_mask.ndim == 4:
                    if keep_mask.shape[1] != 1:
                        raise ValueError("4D attention_mask must have shape (batch, 1, query, key)")
                    keep_mask = keep_mask[:, 0]
                keep_mask = keep_mask.to(dtype=torch.bool, device=x.device)
                batch_size = keep_mask.shape[0]
                attn_mask = ~keep_mask
                attn_mask = attn_mask[:, None].expand(
                    batch_size,
                    self.attn.num_heads,
                    keep_mask.shape[-2],
                    keep_mask.shape[-1],
                )
                attn_mask = attn_mask.reshape(
                    batch_size * self.attn.num_heads,
                    keep_mask.shape[-2],
                    keep_mask.shape[-1],
                )
            else:
                raise ValueError(
                    "attention_mask must be 2D padding mask, 3D keep mask, or 4D keep mask"
                )

        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TextDiffusionModel(nn.Module):
    """Small bidirectional Transformer denoiser for masked text diffusion."""

    def __init__(self, config: TextDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

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

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        return self.lm_head(x)


def pad_sequences(sequences: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


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

    # Guarantee at least one supervised token per row.
    missing = ~mask_positions.any(dim=1)
    if missing.any():
        first_valid = valid_tokens.float().argmax(dim=1)
        rows = torch.where(missing)[0]
        mask_positions[rows, first_valid[rows]] = True

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
    logits = model(noised, attention_mask=attention_mask)
    return F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))


def build_block_diffusion_attention_mask(
    *,
    seq_len: int,
    block_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a LLaDA-style block-causal keep mask with full attention inside each block."""

    if block_length <= 0:
        raise ValueError("block_length must be positive")
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

    scaled = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(scaled, k=min(top_k, scaled.shape[-1]), dim=-1)
        cutoff = values[..., -1, None]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)
        scaled = scaled.masked_fill(remove, float("-inf"))

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
    minimal_topk: int = 1,
    threshold: float = 0.7,
    editing_threshold: float | None = 0.9,
    max_post_steps: int = 16,
    num_to_transfer: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """LLaDA-style block diffusion generation without MoE or pretrained weights."""

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.shape[0] != 1:
        raise ValueError("This educational generator supports batch size 1 only")
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if minimal_topk <= 0:
        raise ValueError("minimal_topk must be positive")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    model.eval()
    config = model.config
    prompt_ids = prompt_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    requested_len = prompt_len + gen_length
    num_blocks = math.ceil(requested_len / block_length)
    total_len = num_blocks * block_length
    if total_len > config.max_seq_len:
        raise ValueError(f"requested length {total_len} exceeds max_seq_len {config.max_seq_len}")

    steps = min(steps, max(1, gen_length // max(minimal_topk, 1)))
    transfer_count = num_to_transfer if num_to_transfer is not None else max(1, math.ceil(block_length / steps))
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
        post_steps = 0

        while True:
            active_slice = slice(block_start, block_end)
            old_block = x[:, active_slice].clone()
            active_masks = old_block == config.mask_token_id

            if not active_masks.any():
                post_steps += 1
                if editing_threshold is None or post_steps > max_post_steps:
                    break

            attention_mask = full_attention_mask[:, :block_end, :block_end]
            logits = model(x[:, :block_end], attention_mask=attention_mask)
            block_logits = logits[:, active_slice, :]
            candidates, confidence = _sample_tokens(
                block_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            prompt_in_block = torch.zeros(block_length, dtype=torch.bool, device=model.device)
            if block_start < prompt_len:
                prompt_in_block[: prompt_len - block_start] = True

            accept = torch.zeros_like(active_masks)
            if active_masks.any():
                high_confidence = (confidence > threshold) & active_masks
                if high_confidence.sum() >= transfer_count:
                    accept |= high_confidence
                else:
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

            if not active_masks.any() and not accept.any():
                break

        if eos_token_id is not None:
            generated = x[0, prompt_len:min(block_end, requested_len)]
            eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                return x[0, : prompt_len + int(eos_positions[0]) + 1]

    return x[0, :requested_len]


def main() -> None:
    torch.manual_seed(0)

    texts = [
        "hello world",
        "hello diffusion",
        "text diffusion fills masks",
        "masked tokens are denoised",
    ]
    tokenizer = SimpleCharTokenizer.from_texts(texts)
    sequences = [tokenizer.encode(text, add_eos=True) for text in texts]
    batch = pad_sequences(sequences, tokenizer.pad_token_id)

    config = TextDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    model = TextDiffusionModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, batch, mask_prob=0.35)
        loss.backward()
        optimizer.step()
        if step in {0, 19}:
            print(f"step {step:02d} loss {loss.item():.4f}")

    prompt = torch.tensor(tokenizer.encode("hello "), dtype=torch.long)
    output = generate(
        model,
        prompt,
        gen_length=24,
        block_length=8,
        steps=8,
        threshold=0.45,
        editing_threshold=0.80,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("generated ids:", output.tolist())
    print("generated text:", repr(tokenizer.decode(output)))


if __name__ == "__main__":
    main()
