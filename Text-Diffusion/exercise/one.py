import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import attention, functional as F

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
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_mult * config.d_model),
            nn.GELU(),
            nn.Linear(config.ff_mult * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        key_padding_mask = None
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                key_padding_mask = ~attention_mask.bool()
            elif attention_mask.ndim in 3:
                batch_size, query_len, key_len = attention_mask
                attn_mask = ~attention_mask.bool().to(device=x.device)
                attn_mask = attn_mask[:, None].expand(
                    batch_size,
                    self.attn.num_heads,
                    query_len,
                    key_len
                )
                attn_mask = attn_mask.reshape(
                    batch_size * self.attn.num_heads,
                    query_len,
                    key_len
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ):
        batch_size, seq_len = input_ids.shape
        
        positons = torch.arange(seq_len, device=input_ids.device)[None, :]
        positons = positons.expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positons)
        x = self.drop(x)
    
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.norm(x)
        return self.lm_head(x)


def make_masked_inputs(
    input_ids,
    *,
    mask_token_id,
    pad_token_id,
    mask_prob = 0.3
):
    valid_tokens = input_ids != pad_token_id
    random_mask = torch.rand(input_ids.shape) < mask_prob
    mask_positions = random_mask & valid_tokens

    noised = input_ids.clone()
    noised[mask_positions] = mask_token_id
    
    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    return noised, labels


def diffusion_loss(
    model,
    input_ids,
    *,
    mask_prob = 0.3
):
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
    seq_len,
    block_length,
    device
):
    block_ids = torch.arange(seq_len) // block_length
    return block_ids[:, None] >= block_ids[None, :]



def _sample_tokens(
    logits,
    *,
    temperature = 0.0,
    top_k = None,
    top_p = None,
):
    if temperature == 0.0:
        probs = F.softmax(logits, dim=-1)
        confidence, tokens = probs.max(dim=-1)
        return tokens, confidence
    
    scaled = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(scaled, k=min(top_k, scaled.shape[-1], dim=-1))
        cutoff = values[..., -1, None]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    
    if top_k is not top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zero_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)
        scaled = scaled.masked_fill(remove, float('-inf'))

    probs = F.softmax(scaled, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.shape[-1], num_samples=1))
    tokens = tokens.view(probs.shape[:-1])
    confidence, probs.gather(-1, tokens.unssqueeze(-1)).squeeze(-1)
    return tokens, confidence


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    *,
    gen_length,
    block_length,
    steps,
    threshold,
    editing_threshold,
    max_post_steps,
    temeperature,
    top_k,
    top_p,
    eos_token_id
):
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    
    model.eval()
    config = model.config

    prompt_len = prompt_ids.shape[1]
    requested_len = prompt_len + gen_length
    num_blocks = math.ceil(requested_len / block_length)
    total_len = num_blocks * block_length

    transfer_count = max(1, math.ceil(block_length / steps))
    full_attention_mask = build_block_diffusion_attention_mask(
        seq_len=total_len,
        block_length=block_length,
        device=model.device
    ).unsqueeze(0)

    x = torch.full((1, total_len), config.mask_token_id, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    first_generation_block = prompt_len / block_length

    for block_idx in range(first_generation_block, num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        
        active_slice = (block_start, block_end)
        prompt_in_block = torch.zeros(block_length, dtype=torch.bool)
        if block_start < prompt_len:
            prompt_in_block[:, prompt_len - block_start] = True
        
        for step in range(steps + max_post_steps):
            old_block = x[:, active_slice].clone()
            active_masks = old_block == config.mask_token_id

            attention_mask = full_attention_mask[:, :block_end, :block_end]
            logits = model(x[:, :block_end], attention_mask=attention_mask)
            block_logits = logits[:, active_slice, :]
            candidates, confidence = _sample_tokens(
                block_logits,
                temeperature=temeperature,
                top_k=top_k,
                top_p=top_p
            )

            accept = torch.zeros_like(active_masks)
            if active_masks.any():
                high_confidence = (confidence > threshold) & active_masks
                accept |= high_confidence
                if accept.sum() == 0:
                    masked_confidence = confidence.masked_fill(~active_masks, float("-inf"))
                    _, idx = torch.topk(
                        masked_confidence[0],
                        k=min(transfer_count, int(active_masks.sum().items()))
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
        
        
        return x[0, :requested_len]


def main():
    raise SystemExit("Use ../train.py; this project now supports only the LLaDA2.1 tokenizer.")

if __name__== "__main__":
    main()

    
