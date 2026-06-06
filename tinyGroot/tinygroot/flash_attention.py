from __future__ import annotations

import os
import sys
import importlib
from types import SimpleNamespace

import torch
from torch.nn import functional as F


def _load_flash_attention_3():
    """Load FA3 on Hopper GPUs when the optional kernels package is available."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        from kernels import get_kernel

        return get_kernel("kernels-community/flash-attn3")
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_override_impl = os.environ.get("TEXT_DIFFUSION_ATTENTION_IMPL")
_logged_attention_impl = False


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _disable_dynamo(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = importlib.import_module("torch._dynamo")
    return dynamo.disable(fn)


def _log_attention_impl(impl: str, reason: str) -> None:
    global _logged_attention_impl
    if _logged_attention_impl:
        return
    if _is_rank_zero():
        print(f"[flash_attention] using {impl}: {reason}", file=sys.stderr, flush=True)
    _logged_attention_impl = True


def describe_attention_backend(*, masked: bool, dtype: torch.dtype | None = None) -> str:
    if masked:
        fa3_status = "available" if HAS_FA3 else "unavailable"
        return f"SDPA (masked attention path; FA3 {fa3_status})"
    if _override_impl == "sdpa":
        return "SDPA (TEXT_DIFFUSION_ATTENTION_IMPL=sdpa)"
    if _override_impl == "fa3":
        return "FA3 (TEXT_DIFFUSION_ATTENTION_IMPL=fa3)"
    if not HAS_FA3:
        return "SDPA (FA3 unavailable)"
    if dtype in (None, torch.bfloat16):
        return "FA3 (Hopper GPU with bf16 tensors)"
    return f"SDPA (FA3 requires bf16 tensors, got {str(dtype).replace('torch.', '')})"


def _use_fa3(*tensors: torch.Tensor) -> bool:
    if _override_impl == "sdpa":
        _log_attention_impl("SDPA", "TEXT_DIFFUSION_ATTENTION_IMPL=sdpa")
        return False
    if _override_impl == "fa3":
        if not HAS_FA3:
            raise RuntimeError("TEXT_DIFFUSION_ATTENTION_IMPL=fa3, but FA3 is not available")
        _log_attention_impl("FA3", "TEXT_DIFFUSION_ATTENTION_IMPL=fa3")
        return True
    if not HAS_FA3:
        _log_attention_impl("SDPA", "FA3 is unavailable on this device or kernels could not be loaded")
        return False
    if all(t.dtype == torch.bfloat16 for t in tensors):
        _log_attention_impl("FA3", "Hopper GPU with bf16 tensors")
        return True
    dtypes = ", ".join(str(t.dtype).replace("torch.", "") for t in tensors)
    _log_attention_impl("SDPA", f"FA3 requires bf16 tensors, got {dtypes}")
    return False


def _unpack_fa3_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


@_disable_dynamo
def _call_fa3_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> torch.Tensor:
    return _unpack_fa3_output(_fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size))


@_disable_dynamo
def _call_fa3_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    cache_seqlens: torch.Tensor | int | None,
    causal: bool,
    window_size: tuple[int, int],
) -> torch.Tensor:
    return _unpack_fa3_output(
        _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )
    )


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
    enable_gqa: bool,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPA attention for tensors in (B, H, T, D) layout.

    ``key_mask`` is an optional ``(B, t_k)`` boolean/float tensor (True/1 = attend)
    used to drop padding keys during batched, variable-length cached decoding.
    """
    t_q = q.size(2)
    t_k = k.size(2)
    left_window, right_window = window_size

    # ``is_causal=True`` only builds the intended mask for a square attention
    # matrix. During cached decode (t_q=1, t_k>1) it would top-left-align a (1, t_k)
    # tril and let the lone query attend to key 0 only, so restrict the shortcut to
    # the non-causal or square cases and let the explicit paths below handle the rest.
    if key_mask is None and left_window < 0 and right_window < 0 and (not causal or t_q == t_k):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)

    if key_mask is None and causal and t_q == 1:
        if left_window >= 0 and left_window + 1 < t_k:
            start = max(0, t_k - left_window - 1)
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    row_idx = (t_k - t_q) + torch.arange(t_q, device=device).unsqueeze(1)
    col_idx = torch.arange(t_k, device=device).unsqueeze(0)
    mask = torch.ones((t_q, t_k), dtype=torch.bool, device=device)

    if causal:
        mask = col_idx <= row_idx
    if left_window >= 0:
        mask = mask & ((row_idx - col_idx) <= left_window)
    if right_window >= 0:
        mask = mask & ((col_idx - row_idx) <= right_window)

    if key_mask is not None:
        mask = (mask[None, None, :, :] & key_mask[:, None, None, :].to(torch.bool)).clone()
        # A left-pad query position can have every key masked, which would NaN the
        # softmax. Let every query attend to its own position (output is discarded
        # for pad rows; for real queries this is already allowed).
        q_idx = torch.arange(t_q, device=device)
        k_idx = (t_k - t_q) + q_idx
        mask[:, :, q_idx, k_idx] = True

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA3-compatible attention for training tensors in (B, T, H, D) layout."""
    # FA3 can't take an arbitrary key-padding mask; fall back to SDPA when one is set.
    if key_mask is None and _use_fa3(q, k, v):
        return _call_fa3_flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = _sdpa_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        causal=causal,
        window_size=window_size,
        enable_gqa=enable_gqa,
        key_mask=key_mask,
    )
    return y.transpose(1, 2)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | int | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA3-compatible KV-cache attention with SDPA fallback."""
    # FA3 can't take an arbitrary key-padding mask; fall back to SDPA when one is set.
    if key_mask is None and _use_fa3(q, k_cache, v_cache):
        return _call_fa3_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required for SDPA KV-cache attention")

    batch_size, t_new, _, _ = q.shape
    if isinstance(cache_seqlens, torch.Tensor):
        if not bool((cache_seqlens == cache_seqlens[0]).all()):
            raise ValueError("SDPA KV-cache fallback requires uniform cache_seqlens")
        pos = int(cache_seqlens[0].item())
    else:
        pos = int(cache_seqlens)

    if k is not None and v is not None:
        k_cache[:batch_size, pos : pos + t_new, :, :] = k
        v_cache[:batch_size, pos : pos + t_new, :, :] = v

    end_pos = pos + t_new
    k_full = k_cache[:batch_size, :end_pos, :, :]
    v_full = v_cache[:batch_size, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = _sdpa_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        causal=causal,
        window_size=window_size,
        enable_gqa=enable_gqa,
        key_mask=key_mask,
    )
    return y.transpose(1, 2)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
