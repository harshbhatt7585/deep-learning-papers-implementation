
import math
import torch
import torch.nn.functional as F


def sample_tokens(
    logits,
    *,
    temperature,
):
    probs = F.softmax(logits, dim=-1)
    confidence, tokens = probs.max(dim=-1)
    return tokens, confidence


def build_block_diffusion_attention_mask(
    *,
    seq_len,
    block_length,
    device,
):
    block_ids = torch.arange(seq_len, device=device) // block_length
    return block_ids[:, None] >= block_ids[None, :]



def generate(
    model,
    prompt_ids,
    *,
    gen_length = 16,
    block_length = 8,
    steps = 4,
    threshold = 0.7,
    editing_threshold = 0.9,
    max_post_steps = 16,
    temperatrue = 0.0,
):
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    
    model.eval()
    config = model.config
    prompt_ids = prompt_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    requested_len = prompt_len + gen_length
    num_blocks = math.ciel(requested_len / block_length)
    total_len = num_blocks * block_length

    transfer_count = max(1, math.ceil(block_length / steps))
    full_attention_mask = build_block_diffusion_attention_mask(
        seq_len=total_len,
        block_length=block_length,
        device=model.device
    ).unsqueeze(0)

    x = torch.full((1, total_len), config.mask_token_id, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    first_generation_block = prompt_len // block_length

    for block_idx in range(first_generation_block, num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        active_slice = (block_start, block_end)
        prompt_in_block = torch.zeros(block_length, dtype=torch.bool, device=model.device)
        if block_start < prompt_len:
            prompt_in_block[:, prompt_len - block_start] = True
        
        for step in range(steps + max_post_steps):
            old_block = x[:, active_slice].clone()
            active_masks = old_block == config.mask_token_id

            attention_mask = full_attention_mask[:, :block_end, :block_end]
            logits = model(x[:, :block_end], attention_mask=attention_mask)
            block_logits = logits[:, active_slice, :]
            candiates, confidence = sample_tokens(
                block_logits,
                temperatrue=temperatrue,
            )
            accept=torch.zeros_like(active_masks)
            if active_masks.any():
                high_confidence = (confidence > threshold) & active_masks
                accept |= high_confidence
                if accept.sum() == 0:
                    masked_confidence, = confidence.masked_fill(
                        ~active_masks,
                        float('-inf'),
                    )
                    for b in range(accept.shape[0]):
                        _, idx = torch.topk(masked_confidence[b],
                        k=min(transfer_count, int(active_masks.sum().item())))
                        accept[b, idx] = True

        
        if editing_threshold is not None:
            editable = ~active_masks & ~prompt_in_block.unsqueeze(0)
            changed = candiates != old_block
            accept |= editable & changed & (confidence > editing_threshold)
        
        if accept.any():
            updated_block = x[:, active_slice].clone()
            updated_block[accept] = candiates[accept]
            x[:, active_slice] = updated_block
        
        if not active_masks.any():
            break

        if step >= steps - 1 and editing_threshold is None:
            break

 
    return x[0, :requested_len]



