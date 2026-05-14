"""Speculative decoding for TextDiffusionModel — two drafter strategies.

Two independent algorithms share this module:

### Mode 1: ``speculate_mtp`` — MTP heads as drafters (Medusa-style)

Drafts come from the *target's own* auxiliary MTP heads. Acceptance uses the
Leviathan ratio test, so output is bit-identical to plain AR at any temperature.
K = number of MTP heads, fixed at training time. Each outer step does exactly
two target forwards (draft + verify). See ``speculate_mtp`` below.

### Mode 2: ``speculate_dflash`` — block-diffusion drafter (faithful to Chen et al. 2026)

Drafts come from a small block-diffusion drafter (``DFlashDraftModel``) that
shares the target's ``token_emb`` and ``lm_head`` and is conditioned on the
target's intermediate hidden states via cross-attention. The drafter has its
own small stack of transformer layers and a single ``fc`` projecting
concatenated target features back to ``d_model``.

Per outer step:

  1. Build the drafter block: ``[last_token, mask, mask, ..., mask]`` of length
     ``block_size``.
  2. One drafter forward, returning predictions for the ``block_size - 1`` mask
     slots. Sample (argmax at T=0).
  3. One target verify forward over ``[last_token, draft_0, ..., draft_{B-2}]``.
  4. Accept by left-to-right argmax matching: accept the longest prefix of
     drafts whose token id equals the target's argmax at the corresponding
     position. Always commit one extra "bonus/correction" token from the
     target's argmax at position ``acceptance_length`` of the verify forward.

This is the same algorithm as the reference implementation at
https://github.com/z-lab/dflash (``dflash/model.py::dflash_generate``).

**Correctness invariant**: at temperature=0, ``speculate_dflash`` produces the
same output token sequence as plain AR token-for-token. At T>0, acceptance is
argmax-matching (not Leviathan), so the output is an approximation of the
target distribution — same trade-off as the reference.

### Training a dflash drafter

Use ``speed_run.sh draft`` (parallel to ``speed_run.sh train``). The ``draft``
mode wires up ``--objective dflash`` with sensible drafter defaults and
requires a frozen target checkpoint via ``TARGET_CHECKPOINT``. The drafter
inherits the target's ``d_model`` and ``vocab_size`` by construction.

Example::

    TARGET_CHECKPOINT=runs/<TARGET>/checkpoint.pt bash speed_run.sh draft 4gpu

### Smoke test CLI::

    # MTP-only:
    python -m spec_decode --checkpoint runs/<RUN>/checkpoint.pt --prompt "..." --gen-length 64

    # dflash-only:
    python -m spec_decode --checkpoint runs/<TARGET>/checkpoint.pt \\
        --drafter-checkpoint runs/<DRAFTER>/checkpoint.pt --mode dflash --block-size 16

    # both (target must have MTP heads, drafter is a DFlash checkpoint):
    python -m spec_decode --checkpoint runs/<TARGET>/checkpoint.pt \\
        --drafter-checkpoint runs/<DRAFTER>/checkpoint.pt --block-size 16

The script verifies at temperature=0 that every mode produces the same output
as plain AR (correctness) and reports wall-clock speedup + acceptance rate.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from model import TextDiffusionConfig, TextDiffusionModel, generate_causal, norm


@dataclass
class SpecStats:
    """Counters collected during a speculative-decoding run.

    ``mode`` is "mtp" or "dflash" so the smoke-test report can label runs.
    ``drafter_forwards`` is 0 for MTP (the draft pass is also a target pass) and
    equals N * outer_steps for dflash, where N is the denoising-step budget.
    """

    target_forwards: int = 0
    drafter_forwards: int = 0
    drafts_proposed: int = 0
    drafts_accepted: int = 0
    tokens_generated: int = 0
    mtp_heads: int = 0
    draft_k: int = 0
    mode: str = "mtp"
    per_step_accepted: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.drafts_proposed == 0:
            return 0.0
        return self.drafts_accepted / self.drafts_proposed

    @property
    def tokens_per_forward(self) -> float:
        if self.target_forwards == 0:
            return 0.0
        return self.tokens_generated / self.target_forwards

    def as_dict(self) -> dict[str, float]:
        return {
            "mode": self.mode,
            "target_forwards": self.target_forwards,
            "drafter_forwards": self.drafter_forwards,
            "drafts_proposed": self.drafts_proposed,
            "drafts_accepted": self.drafts_accepted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_generated": self.tokens_generated,
            "tokens_per_forward": self.tokens_per_forward,
            "mtp_heads": self.mtp_heads,
            "draft_k": self.draft_k,
        }


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        kth_values = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_values, float("-inf"))
    if top_p is not None and top_p < 1.0:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits


def _probs_from_logits(
    logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Convert (..., V) logits into (..., V) probabilities.

    At temperature=0 we return a one-hot at argmax, which makes the Leviathan
    accept/reject rule collapse to deterministic argmax-matching (see module docstring).
    """
    if temperature == 0.0:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    filtered = _filter_logits(logits.float() / temperature, top_k=top_k, top_p=top_p)
    return F.softmax(filtered, dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token id per row from (..., V) probabilities.

    Returns a tensor with shape `probs.shape[:-1]` and dtype long.
    """
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(probs.shape[:-1])


def _mtp_draft_dists(
    model: TextDiffusionModel,
    hidden_last: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> list[torch.Tensor]:
    """Run each MTP head on the last-position hidden state and return draft probs.

    Args:
        hidden_last: (B, 1, D) hidden state from the target's final norm at the
            last token of the prefix.

    Returns:
        list of (B, V) probability tensors, one per MTP head.
    """
    draft_probs: list[torch.Tensor] = []
    for mtp in model.mtp_heads:
        h = norm(mtp(hidden_last))
        logits = model.lm_head(h)[:, 0, :]
        draft_probs.append(_probs_from_logits(logits, temperature, top_k=top_k, top_p=top_p))
    return draft_probs


@torch.no_grad()
def speculate_mtp(
    model: TextDiffusionModel,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Generate ``gen_length`` tokens using the target model's MTP heads as drafters.

    Args:
        model: trained TextDiffusionModel with ``mtp_heads`` (Medusa-style).
        prompt_ids: 1D or 2D (B=1) tensor of prompt token ids.
        gen_length: number of new tokens to produce (cap; EOS may stop early).
        temperature: 0.0 = greedy. At T=0 the output is identical to plain AR.
        eos_token_id: optional; stop at first occurrence in generated tokens.

    Returns:
        (generated_ids, stats) where generated_ids is 1D (prompt + generated)
        and stats summarizes target forwards, acceptance, and throughput.
    """
    model.eval()
    config = model.config
    num_heads = len(model.mtp_heads)
    stats = SpecStats(mode="mtp", mtp_heads=num_heads, draft_k=num_heads)

    if prompt_ids.ndim == 1:
        prefix = prompt_ids.unsqueeze(0).to(model.device)
    else:
        prefix = prompt_ids.to(model.device)
    if prefix.size(0) != 1:
        raise ValueError("speculate_mtp currently supports batch_size=1 only")

    target_length = min(prefix.size(1) + gen_length, config.max_seq_len)

    if num_heads == 0:
        # No drafters: just run plain AR but still report stats.
        out = generate_causal(
            model,
            prefix[0],
            gen_length=gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
        stats.tokens_generated = int(out.size(0) - prefix.size(1))
        stats.target_forwards = stats.tokens_generated
        return out, stats

    finished = False
    while prefix.size(1) < target_length and not finished:
        accepted_this_step = 0

        # --- 1. Draft forward ---------------------------------------------------
        logits, hidden = model(prefix, attention_mask=None, causal=True, return_hidden=True)
        stats.target_forwards += 1

        p_main = _probs_from_logits(logits[:, -1, :], temperature, top_k=top_k, top_p=top_p)  # (1, V)
        y_main = _sample_from_probs(p_main)  # (1,)

        draft_probs = _mtp_draft_dists(
            model, hidden[:, -1:, :], temperature, top_k=top_k, top_p=top_p
        )  # list of (1, V)
        draft_tokens = [_sample_from_probs(q) for q in draft_probs]  # list of (1,)

        # --- 2. Verify forward --------------------------------------------------
        extension = torch.cat(
            [y_main.unsqueeze(-1)] + [tok.unsqueeze(-1) for tok in draft_tokens],
            dim=-1,
        )  # (1, K+1)

        max_extension = config.max_seq_len - prefix.size(1)
        if max_extension <= 0:
            break
        if extension.size(1) > max_extension:
            extension = extension[:, :max_extension]
            draft_tokens = draft_tokens[: extension.size(1) - 1]
            draft_probs = draft_probs[: extension.size(1) - 1]

        extended = torch.cat([prefix, extension], dim=1)
        logits_v = model(extended, attention_mask=None, causal=True)
        stats.target_forwards += 1

        prefix_end = prefix.size(1)

        # Commit y_main: it was sampled from the target's true main-head distribution.
        prefix = torch.cat([prefix, y_main.unsqueeze(-1)], dim=1)
        stats.tokens_generated += 1
        if eos_token_id is not None and int(y_main[0].item()) == eos_token_id:
            stats.per_step_accepted.append(accepted_this_step)
            break
        if prefix.size(1) >= target_length:
            stats.per_step_accepted.append(accepted_this_step)
            break

        # Verify drafts left-to-right.
        all_accepted = True
        for k, (y_k, q_k) in enumerate(zip(draft_tokens, draft_probs)):
            target_pos = prefix_end + k
            p_true = _probs_from_logits(logits_v[:, target_pos, :], temperature, top_k=top_k, top_p=top_p)
            stats.drafts_proposed += 1

            q_y = q_k.gather(-1, y_k.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
            p_y = p_true.gather(-1, y_k.unsqueeze(-1)).squeeze(-1)
            accept_prob = (p_y / q_y).clamp(max=1.0)
            u = torch.rand_like(accept_prob)

            if bool((u < accept_prob).item()):
                prefix = torch.cat([prefix, y_k.unsqueeze(-1)], dim=1)
                stats.tokens_generated += 1
                stats.drafts_accepted += 1
                accepted_this_step += 1
                if eos_token_id is not None and int(y_k[0].item()) == eos_token_id:
                    finished = True
                    break
                if prefix.size(1) >= target_length:
                    finished = True
                    break
            else:
                residual = (p_true - q_k).clamp(min=0)
                residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                residual = residual / residual_sum
                y_corrected = _sample_from_probs(residual)
                prefix = torch.cat([prefix, y_corrected.unsqueeze(-1)], dim=1)
                stats.tokens_generated += 1
                all_accepted = False
                if eos_token_id is not None and int(y_corrected[0].item()) == eos_token_id:
                    finished = True
                break

        # Bonus token if every draft passed.
        if all_accepted and not finished and prefix.size(1) < target_length:
            bonus_pos = prefix_end + len(draft_tokens)
            if bonus_pos < logits_v.size(1):
                p_bonus = _probs_from_logits(logits_v[:, bonus_pos, :], temperature, top_k=top_k, top_p=top_p)
                y_bonus = _sample_from_probs(p_bonus)
                prefix = torch.cat([prefix, y_bonus.unsqueeze(-1)], dim=1)
                stats.tokens_generated += 1
                if eos_token_id is not None and int(y_bonus[0].item()) == eos_token_id:
                    finished = True

        stats.per_step_accepted.append(accepted_this_step)

    return prefix[0], stats


# ---------------------------------------------------------------------------
# dflash: separate diffusion drafter
# ---------------------------------------------------------------------------


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Greedy at T=0, multinomial otherwise. ``logits`` is ``(..., V)``; returns
    ``logits.shape[:-1]`` long tensor."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    filtered = _filter_logits(logits.float() / temperature, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(probs.shape[:-1])


def _cache_seq_len(cache: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    return 0 if not cache else cache[0][0].size(1)


def _crop_cache(
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    seq_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k[:, :seq_len, :, :].contiguous(), v[:, :seq_len, :, :].contiguous()) for k, v in cache]


@torch.no_grad()
def generate_causal_cached(
    model: TextDiffusionModel,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressive generation using the target KV cache."""
    if prompt_ids.ndim == 1:
        x = prompt_ids.unsqueeze(0).to(model.device)
    else:
        x = prompt_ids.to(model.device)

    model.eval()
    target_length = min(x.size(1) + gen_length, model.config.max_seq_len)
    logits, cache = model(x, attention_mask=None, causal=True, use_cache=True)
    while x.size(1) < target_length:
        next_token = _sample_token(logits[:, -1, :], temperature, top_k=top_k, top_p=top_p).unsqueeze(-1)
        x = torch.cat([x, next_token], dim=1)
        if eos_token_id is not None and int(next_token[0, 0].item()) == eos_token_id:
            break
        if x.size(1) >= target_length:
            break
        logits, cache = model(
            next_token,
            attention_mask=None,
            causal=True,
            past_key_values=cache,
            use_cache=True,
        )
    return x[0]


@torch.no_grad()
def speculate_dflash(
    target: TextDiffusionModel,
    drafter,  # DFlashDraftModel — typed loosely to avoid a hard import cycle.
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    block_size: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    """Block-diffusion speculative decoding (DFlash, Chen et al. 2026).

    Per outer step:

      1. Build block ``[last_token, mask × (B-1)]`` where ``last_token`` is the
         most recently committed token and ``B = block_size``.
      2. One drafter forward conditioned on the accumulated target hidden
         states; returns logits at the ``B-1`` mask positions.
      3. One target verify forward on ``[last_token, drafts...]``; capture
         ``output_hidden_states`` so we can grow the target-hidden context.
      4. Accept-by-argmax-matching left-to-right; commit ``acc + 1`` new tokens
         (``acc`` matched drafts + 1 correction/bonus from target argmax).

    At ``temperature=0``, output is token-for-token identical to plain AR from
    the target — exactly the correctness invariant the reference DFlash
    implementation guarantees.

    Args:
        target: trained ``TextDiffusionModel`` (the model we're preserving).
        drafter: ``DFlashDraftModel`` bound to ``target`` via ``drafter.bind(target)``.
        prompt_ids: 1D or 2D (B=1) prompt ids.
        gen_length: max number of new tokens to produce (cap).
        block_size: ``B`` from the paper (default 16).
        temperature: 0.0 = greedy.
        eos_token_id: optional early-stop trigger.

    Returns:
        (generated_ids, stats): 1D long tensor (prompt + generated) and stats.
    """
    from dflash_model import extract_context_feature, DFlashDraftModel  # local import

    target.eval()
    drafter.eval()
    if not isinstance(drafter, DFlashDraftModel):
        raise TypeError(
            f"speculate_dflash expects a DFlashDraftModel drafter, got {type(drafter).__name__}"
        )
    if drafter.embed_tokens is None or drafter.lm_head is None:
        raise RuntimeError("drafter is not bound to target; call drafter.bind(target) first")

    config = target.config
    stats = SpecStats(mode="dflash", draft_k=block_size)
    layer_ids = drafter.config.target_layer_ids
    mask_id = drafter.config.mask_token_id
    if block_size < 2:
        raise ValueError("block_size must be >= 2 (1 anchor + >=1 draft slot)")

    if prompt_ids.ndim == 1:
        prefix_ids = prompt_ids.unsqueeze(0).to(target.device)
    else:
        prefix_ids = prompt_ids.to(target.device)
    if prefix_ids.size(0) != 1:
        raise ValueError("speculate_dflash currently supports batch_size=1 only")

    target_length = min(prefix_ids.size(1) + gen_length, config.max_seq_len)

    # --- Prefill: target forward over the full prompt -----------------------
    prefill_logits, prefill_hidden_list, target_cache = target(
        prefix_ids, attention_mask=None, causal=True, output_hidden_states=True, use_cache=True
    )
    stats.target_forwards += 1
    target_hidden = extract_context_feature(prefill_hidden_list, layer_ids)  # (1, L, D_ctx)

    # First sampled token comes from the prefill's last-position logits.
    first_token = _sample_token(
        prefill_logits[:, -1, :], temperature, top_k=top_k, top_p=top_p
    ).unsqueeze(-1)  # (1, 1)
    tokens = torch.cat([prefix_ids, first_token], dim=1)
    stats.tokens_generated += 1
    if eos_token_id is not None and int(first_token[0, 0].item()) == eos_token_id:
        return tokens[0], stats

    while tokens.size(1) < target_length:
        # Effective block: at most block_size positions; clamp to remaining
        # budget so we don't propose drafts we can't commit anyway.
        remaining = target_length - tokens.size(1)
        bs = min(block_size, remaining + 1)
        if bs < 2:
            break

        last_token = tokens[:, -1:]  # (1, 1)

        # --- 1. Drafter forward ----------------------------------------------
        block_input = torch.cat(
            [last_token, torch.full((1, bs - 1), mask_id, dtype=torch.long, device=tokens.device)],
            dim=1,
        )  # (1, bs)
        draft_logits = drafter(block_input, target_hidden=target_hidden, logits_start=1)
        # (1, bs-1, V) — predictions at the masked positions only.
        stats.drafter_forwards += 1
        draft_tokens = _sample_token(draft_logits, temperature, top_k=top_k, top_p=top_p)  # (1, bs-1)

        # --- 2. Target verify forward ----------------------------------------
        verify_input = torch.cat([last_token, draft_tokens], dim=1)  # (1, bs)
        if tokens.size(1) - 1 + verify_input.size(1) > config.max_seq_len:
            # Target's context window would overflow; truncate to fit.
            keep = config.max_seq_len - (tokens.size(1) - 1)
            verify_input = verify_input[:, :keep]
            draft_tokens = draft_tokens[:, : keep - 1]
            if verify_input.size(1) < 2:
                break

        cache_len_before = _cache_seq_len(target_cache)
        target_logits, target_hidden_list_new, verify_cache = target(
            verify_input,
            attention_mask=None,
            causal=True,
            output_hidden_states=True,
            past_key_values=target_cache,
            use_cache=True,
        )
        stats.target_forwards += 1
        verify_hidden_full = extract_context_feature(target_hidden_list_new, layer_ids)

        # The cached verify forward returns only the new block positions:
        # position 0 is the anchor, positions 1.. are draft tokens.
        verify_logits = target_logits
        verify_hidden = verify_hidden_full

        # Target's argmax at each verify position; argmax-match against drafts.
        target_argmax = _sample_token(verify_logits, temperature, top_k=top_k, top_p=top_p)  # (1, bs)

        # Compare drafts (positions 1..bs-1 of verify_input) with target's argmax
        # at positions 0..bs-2 of verify_logits. cumprod gives the longest
        # accepted prefix.
        n_drafts = draft_tokens.size(1)
        if n_drafts == 0:
            break
        matches = (draft_tokens == target_argmax[:, :n_drafts]).long()  # (1, n_drafts)
        acceptance_length = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
        stats.drafts_proposed += n_drafts
        stats.drafts_accepted += acceptance_length

        # Commit accepted drafts (positions 1..acceptance_length of block) +
        # 1 correction/bonus token from target.argmax at position acceptance_length.
        accepted_drafts = draft_tokens[:, :acceptance_length]  # (1, acc)
        # Target's argmax at position acceptance_length predicts what should
        # follow [last_token, drafts[0..acc-1]] — that's either a correction
        # (if a draft mismatched) or the bonus after a fully-accepted block.
        bonus_token = target_argmax[:, acceptance_length : acceptance_length + 1]  # (1, 1)
        new_tokens = torch.cat([accepted_drafts, bonus_token], dim=1)  # (1, acc + 1)
        tokens = torch.cat([tokens, new_tokens], dim=1)
        stats.tokens_generated += new_tokens.size(1)
        stats.per_step_accepted.append(acceptance_length + 1)

        # Grow target_hidden by every accepted position's hidden state.
        # The verify forward produced hidden states for positions 0..bs-1 of
        # the block. We keep positions 0..acceptance_length:
        #   - position 0 = anchor token (this iter's "last_token", which is
        #     the previous iter's bonus — its hidden state was NOT in
        #     target_hidden before, so we add it now)
        #   - positions 1..acceptance_length = accepted drafts
        # The current iter's bonus token (target.argmax at position
        # acceptance_length) does NOT have a target hidden state yet; the
        # next iter's verify forward will produce it (as that block's anchor).
        new_hidden = verify_hidden[:, : acceptance_length + 1, :]
        target_hidden = torch.cat([target_hidden, new_hidden], dim=1)
        target_cache = _crop_cache(verify_cache, cache_len_before + acceptance_length + 1)

        if eos_token_id is not None:
            if int(bonus_token[0, 0].item()) == eos_token_id:
                break
            if acceptance_length > 0 and (accepted_drafts == eos_token_id).any():
                break

    # Reference DFlash truncates at the end so we don't return more tokens than
    # the caller asked for. We may overshoot by 1 token because each block
    # commits acceptance_length + 1 atomically.
    if tokens.size(1) > target_length:
        tokens = tokens[:, :target_length]
    return tokens[0], stats


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------


def _clean_state_dict(state: dict) -> dict:
    """Strip any DDP/compile prefixes that ``save_checkpoint`` may have left in."""
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TextDiffusionModel, dict]:
    """Load a training checkpoint and rebuild the (target) model on ``device``."""
    if not Path(checkpoint_path).exists():
        raise SystemExit(
            f"[spec_decode] target checkpoint not found: {checkpoint_path}\n"
            f"  -> If you're on a Modal container, run 'modal run modal_train.py::list_runs' from your Mac\n"
            f"     to discover the actual checkpoint paths on the runs volume.\n"
            f"  -> If you're running locally, download the checkpoint first via\n"
            f"     'modal volume get text-diffusion-runs <run-name>/checkpoint.pt'."
        )
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in blob:
        raise KeyError(
            f"checkpoint at {checkpoint_path} has no 'config' key. "
            "Use a checkpoint produced by train.py (which saves the config alongside the weights)."
        )
    cfg_dict = blob["config"]
    if isinstance(cfg_dict, TextDiffusionConfig):
        config = cfg_dict
    else:
        config = TextDiffusionConfig(**cfg_dict)
    cleaned = _clean_state_dict(blob["model_state"])
    mtp_weight = cleaned.get("mtp_heads.0.weight")
    if (
        mtp_weight is not None
        and tuple(mtp_weight.shape) != (config.d_model, config.d_model)
    ):
        print(
            "[spec_decode] target checkpoint uses legacy full-vocab MTP heads; "
            "ignoring MTP heads for target loading"
        )
        config.n_mtp_heads = 0

    model = TextDiffusionModel(config).to(device)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[spec_decode] missing keys when loading checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[spec_decode] unexpected keys when loading checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model, blob


def _build_dflash_drafter_from_checkpoint(
    checkpoint_path: Path,
    target: TextDiffusionModel,
    device: torch.device,
):
    """Load a DFlash drafter checkpoint, build the drafter, bind it to ``target``."""
    from dflash_model import DFlashConfig, DFlashDraftModel

    if not Path(checkpoint_path).exists():
        raise SystemExit(
            f"[spec_decode] drafter checkpoint not found: {checkpoint_path}\n"
            f"  -> Make sure you've trained a DFlash drafter via\n"
            f"     'TARGET_CHECKPOINT=<path> bash speed_run.sh draft 4gpu' first.\n"
            f"  -> Run 'modal run modal_train.py::list_runs' to discover existing drafter paths."
        )
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in blob:
        raise KeyError(
            f"drafter checkpoint at {checkpoint_path} has no 'config' key."
        )
    cfg_dict = blob["config"]
    if isinstance(cfg_dict, DFlashConfig):
        cfg = cfg_dict
    elif isinstance(cfg_dict, dict) and "target_d_model" in cfg_dict:
        # tuple-typed fields come back as lists/tuples via asdict; ensure both work
        if "target_layer_ids" in cfg_dict and cfg_dict["target_layer_ids"] is not None:
            cfg_dict = {**cfg_dict, "target_layer_ids": tuple(cfg_dict["target_layer_ids"])}
        cfg = DFlashConfig(**cfg_dict)
    else:
        raise SystemExit(
            f"drafter checkpoint at {checkpoint_path} does not look like a DFlash drafter "
            f"(config keys: {list(cfg_dict.keys()) if isinstance(cfg_dict, dict) else type(cfg_dict)}). "
            "If you trained a standalone-diffusion drafter, re-train with --objective dflash."
        )

    if cfg.target_d_model != target.config.d_model:
        raise SystemExit(
            f"drafter target_d_model={cfg.target_d_model} != target d_model={target.config.d_model}. "
            "Drafter must be paired with the target it was trained against."
        )
    if cfg.target_vocab_size != target.config.vocab_size:
        raise SystemExit(
            f"drafter target_vocab_size={cfg.target_vocab_size} != target vocab_size={target.config.vocab_size}."
        )

    drafter = DFlashDraftModel(cfg).to(device)
    drafter.bind(target)
    cleaned = _clean_state_dict(blob["model_state"])
    missing, unexpected = drafter.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[spec_decode] missing keys when loading drafter: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[spec_decode] unexpected keys when loading drafter: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    drafter.eval()
    return drafter, blob, cfg


def _load_tokenizer(checkpoint_blob: dict, args: argparse.Namespace):
    """Find and load the tokenizer that matches this checkpoint.

    Resolution order:
      1. --tokenizer-dir flag if given,
      2. ``<checkpoint_dir>/tokenizer_hf`` (this is where ``save_checkpoint`` puts it),
      3. ``args.nanochat_tokenizer_cache_dir`` stored inside the checkpoint blob,
      4. give up and print raw ids.
    """
    try:
        from tokenizer import NanochatTokenizer
    except Exception as exc:
        print(f"[spec_decode] could not import NanochatTokenizer ({exc}); using raw id printing only.")
        return None

    candidates: list[Path] = []
    if args.tokenizer_dir is not None:
        candidates.append(Path(args.tokenizer_dir))
    candidates.append(args.checkpoint.parent / "tokenizer_hf")
    saved_args = checkpoint_blob.get("args") or {}
    cached = saved_args.get("nanochat_tokenizer_cache_dir")
    if cached:
        candidates.append(Path(cached))

    for candidate in candidates:
        if candidate.exists():
            print(f"[spec_decode] loading tokenizer from {candidate}")
            return NanochatTokenizer.load(candidate)

    raise SystemExit(
        "no tokenizer found. Pass --tokenizer-dir <path>. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def _format_ids(ids: torch.Tensor, tokenizer) -> str:
    if tokenizer is None:
        return " ".join(str(int(t)) for t in ids.tolist())
    try:
        return tokenizer.decode(ids.detach().cpu().tolist())
    except Exception as exc:
        return f"<decode error: {exc}; raw ids: {ids[:16].tolist()}...>"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speculative-decoding smoke test for TextDiffusionModel — supports MTP-Medusa and dflash."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to target checkpoint.pt produced by train.py")
    parser.add_argument(
        "--drafter-checkpoint",
        type=Path,
        default=None,
        help="path to a separate diffusion-trained drafter checkpoint. Enables dflash mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["mtp", "dflash", "both", "auto"],
        default="auto",
        help="auto = run dflash if --drafter-checkpoint is given else mtp; "
        "both = run mtp + dflash + AR (requires --drafter-checkpoint).",
    )
    parser.add_argument(
        "--block-size", type=int, default=16,
        help="dflash: block size B (1 anchor + B-1 drafted tokens). DFlash paper default is 15-16.",
    )
    parser.add_argument("--drafter-tokenizer-dir", type=Path, default=None, help="optional override for drafter tokenizer dir")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="optional override for the tokenizer cache dir")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy (output must match AR exactly)")
    parser.add_argument("--top-k", type=int, default=None, help="optional top-k filter for temperature sampling")
    parser.add_argument("--top-p", type=float, default=None, help="optional nucleus filter for temperature sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="number of warmup runs before timing")
    return parser.parse_args()


def _resolve_modes(args: argparse.Namespace) -> list[str]:
    """Decide which spec modes to run given the CLI args."""
    if args.mode == "auto":
        return ["dflash"] if args.drafter_checkpoint is not None else ["mtp"]
    if args.mode == "both":
        if args.drafter_checkpoint is None:
            raise SystemExit("--mode both requires --drafter-checkpoint")
        return ["mtp", "dflash"]
    if args.mode == "dflash" and args.drafter_checkpoint is None:
        raise SystemExit("--mode dflash requires --drafter-checkpoint")
    return [args.mode]


def _time_run(callable_fn, args, device) -> tuple[object, float]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = callable_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _print_spec_report(stats: SpecStats, ids: torch.Tensor, wall_s: float, ar_time_s: float, tokenizer) -> None:
    if stats.mode == "mtp":
        label = "MTP-Medusa"
        detail = f"MTP heads = {stats.mtp_heads}"
    else:
        label = "DFlash"
        detail = f"block_size = {stats.draft_k}"
    print()
    print(f"== speculative · {label} ({detail}) ==")
    print(f"  tokens generated: {stats.tokens_generated}")
    print(f"  target forwards:  {stats.target_forwards}")
    if stats.mode == "dflash":
        print(f"  drafter forwards: {stats.drafter_forwards}")
    print(f"  drafts proposed:  {stats.drafts_proposed}")
    print(f"  drafts accepted:  {stats.drafts_accepted}")
    print(f"  acceptance rate:  {stats.acceptance_rate*100:.1f}%")
    if stats.per_step_accepted:
        avg = sum(stats.per_step_accepted) / len(stats.per_step_accepted)
        print(f"  avg accepted/block: {avg:.2f} (over {len(stats.per_step_accepted)} blocks)")
    print(f"  tokens/forward:   {stats.tokens_per_forward:.2f}")
    print(f"  wall time:        {wall_s*1000:.1f} ms ({stats.tokens_generated/wall_s:.1f} tok/s)")
    print(f"  speedup vs AR:    {ar_time_s/wall_s:.2f}x")
    print(f"  output: {_format_ids(ids, tokenizer)}")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[spec_decode] loading target checkpoint {args.checkpoint} onto {device}")
    model, blob = _build_model_from_checkpoint(args.checkpoint, device)
    tokenizer = _load_tokenizer(blob, args)
    print(
        f"[spec_decode] target: d_model={model.config.d_model} "
        f"n_layers={model.config.n_layers} "
        f"n_heads={model.config.n_heads} "
        f"mtp_heads={len(model.mtp_heads)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    modes = _resolve_modes(args)

    drafter = None
    drafter_cfg = None
    if args.drafter_checkpoint is not None:
        print(f"[spec_decode] loading drafter checkpoint {args.drafter_checkpoint}")
        drafter, _drafter_blob, drafter_cfg = _build_dflash_drafter_from_checkpoint(
            args.drafter_checkpoint, model, device
        )
        owned = drafter.num_owned_parameters()
        target_total = sum(p.numel() for p in model.parameters())
        print(
            f"[spec_decode] drafter (DFlash): "
            f"d_model={drafter_cfg.d_model} "
            f"n_draft_layers={drafter_cfg.n_draft_layers} "
            f"n_heads={drafter_cfg.n_heads} "
            f"block_size={drafter_cfg.block_size} "
            f"target_layer_ids={drafter_cfg.target_layer_ids} "
            f"owned_params={owned:,} "
            f"(={100.0 * owned / target_total:.1f}% of target trunk; "
            f"shares embed/lm_head with target)"
        )

    if "mtp" in modes and len(model.mtp_heads) == 0:
        print(
            "[spec_decode] WARNING: target has 0 MTP heads. MTP-Medusa will fall back "
            "to plain AR. Re-train with --mtp-heads >= 1 to see a speedup, or use --mode dflash."
        )

    if tokenizer is not None:
        prompt_ids = torch.tensor(tokenizer.encode(args.prompt, add_bos=True), dtype=torch.long)
    else:
        # No tokenizer: assume the user passed space-separated integer ids.
        prompt_ids = torch.tensor([int(x) for x in args.prompt.split()], dtype=torch.long)

    print(f"[spec_decode] prompt: {args.prompt!r} ({prompt_ids.numel()} tokens)")

    # --- Warmup ---------------------------------------------------------------
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        _ = generate_causal_cached(
            model, prompt_ids, gen_length=8,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        if "mtp" in modes:
            torch.manual_seed(args.seed)
            _ = speculate_mtp(
                model, prompt_ids, gen_length=8,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            )
        if "dflash" in modes and drafter is not None:
            torch.manual_seed(args.seed)
            _ = speculate_dflash(
                model, drafter, prompt_ids,
                gen_length=8, block_size=min(args.block_size, 8),
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            )

    # --- Baseline: plain AR ---------------------------------------------------
    ar_ids, ar_time = _time_run(
        lambda: generate_causal_cached(
            model, prompt_ids, gen_length=args.gen_length,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        ),
        args, device,
    )
    ar_generated = int(ar_ids.size(0) - prompt_ids.size(0))

    print()
    print(f"== plain AR ==")
    print(f"  tokens generated: {ar_generated}")
    print(f"  wall time:        {ar_time*1000:.1f} ms ({ar_generated/ar_time:.1f} tok/s)")
    print(f"  output: {_format_ids(ar_ids, tokenizer)}")

    # --- Speculative runs -----------------------------------------------------
    results: dict[str, tuple[torch.Tensor, SpecStats, float]] = {}

    if "mtp" in modes:
        (spec_ids, stats), spec_time = _time_run(
            lambda: speculate_mtp(
                model, prompt_ids, gen_length=args.gen_length,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            ),
            args, device,
        )
        results["mtp"] = (spec_ids, stats, spec_time)
        _print_spec_report(stats, spec_ids, spec_time, ar_time, tokenizer)

    if "dflash" in modes and drafter is not None:
        (spec_ids, stats), spec_time = _time_run(
            lambda: speculate_dflash(
                model, drafter, prompt_ids,
                gen_length=args.gen_length, block_size=args.block_size,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            ),
            args, device,
        )
        results["dflash"] = (spec_ids, stats, spec_time)
        _print_spec_report(stats, spec_ids, spec_time, ar_time, tokenizer)

    # --- Correctness check at T=0 --------------------------------------------
    if args.temperature == 0.0:
        print()
        for name, (spec_ids, _stats, _wall) in results.items():
            common = min(ar_ids.numel(), spec_ids.numel())
            match = bool(torch.equal(ar_ids[:common], spec_ids[:common]))
            if match:
                print(f"[spec_decode] OK ({name}): matches plain AR token-for-token on the first {common} tokens.")
            else:
                first_diff = int((ar_ids[:common] != spec_ids[:common]).nonzero()[0, 0].item())
                print(
                    f"[spec_decode] FAIL ({name}): outputs diverge at token index {first_diff}. "
                    f"AR={int(ar_ids[first_diff].item())} spec={int(spec_ids[first_diff].item())}. "
                    "This indicates a bug in the verify/accept logic."
                )


if __name__ == "__main__":
    main()
