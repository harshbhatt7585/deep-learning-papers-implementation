from __future__ import annotations

import torch


KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class StaticKVCache:
    """Pre-allocated KV buffers for autoregressive decoding.

    The list-based :data:`KVCache` re-runs ``torch.cat`` on the full key/value
    tensors every decode step, reallocating and copying the whole cache per token
    (O(T^2) memory traffic over a rollout). This cache instead writes new
    keys/values into a fixed ``(batch, max_len, n_kv_heads, head_dim)`` buffer and
    reads back only the filled prefix.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_len: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        # Zero-initialised: flash decode kernels can read block-padded regions
        # past the written prefix, and uninitialised bytes can propagate NaNs.
        shape = (batch_size, max_len, n_kv_heads, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.batch_size = batch_size
        self.max_len = max_len
        self.pos = 0

    def __len__(self) -> int:
        return len(self.k)

    def reset(self) -> None:
        self.pos = 0
