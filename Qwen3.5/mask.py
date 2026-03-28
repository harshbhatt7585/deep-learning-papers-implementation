import torch

def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype
)-> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal= 1 + kv_length - query_length)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(
        batch_size,
        1,
        query_length,
        kv_length
    )
    if attention_mask is None:
        return causal
    
    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype) * min_value)
    return causal + padding_mask


if __name__ == "__main__":
    print(build_causal_mask(
        torch.randn(4,20),
        4,
        20,
        20,
        'cpu',
        torch.float32
    ))
