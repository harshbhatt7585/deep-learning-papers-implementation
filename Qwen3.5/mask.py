import torch

def build_casual_mask(
    attenton_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype
)-> torch.Tensor:
    min_value = torch.finfo(dtype).min
    casual = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    casual = torch.triu(casual, diagonal= 1 + kv_length - query_length)
    casual = casual.unsqueeze(0).unsqueeze(0).expand(
        batch_size,
        1,
        query_length,
        kv_length
    )
    if attenton_mask is None:
        return casual
    
    padding_mask = (1.0 - attenton_mask[:, None, None, :].to(dtype) * min_value)
    return casual + padding_mask

