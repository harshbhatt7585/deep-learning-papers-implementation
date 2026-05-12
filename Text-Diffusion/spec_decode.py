"""Medusa-style speculative decoding using the model's own MTP heads as drafters.

Algorithm (Leviathan et al. 2023, "Fast Inference from Transformers via Speculative
Decoding"; Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration"):

Each outer step does exactly two target forwards:

  1. Draft forward on `prefix`:
       - main lm_head           -> p(t+1 | prefix)     -> sample y_main  (auto-accepted)
       - each mtp_head[k]       -> q_k(t+2+k | prefix) -> sample y_k     (drafted)

  2. Verify forward on `prefix || y_main || y_0 || ... || y_{K-1}`:
       - logits at position p+k give the *true* target distribution
         p_true_k(token at p+k+1 | prefix, y_main, y_0..y_{k-1})
       - For each draft y_k, accept with prob min(1, p_true_k(y_k) / q_k(y_k)).
       - On first rejection, sample one corrected token from
         normalize(max(0, p_true_k - q_k)) and stop verifying drafts for this step.
       - If all drafts accept, sample one bonus token from
         p_true_{K}(token at p+K+1 | prefix, y_main, all drafts).

This produces between 2 (y_main + 1 corrected) and K+2 (y_main + K drafts + 1 bonus)
accepted tokens per outer iteration. Output distribution is identical to plain AR
sampling from the target (proof: see Leviathan et al. Theorem 1). At temperature=0
the output tokens are bit-identical to ``generate_causal``.

Run as a script for a smoke test:

    python -m spec_decode --checkpoint runs/<RUN>/checkpoint.pt \\
        --prompt "The capital of France is" --gen-length 64

It will run both ``generate_causal`` and ``speculate_mtp`` and:
  - verify they produce the same tokens at temperature=0 (correctness),
  - report wall-clock speedup and acceptance rate.
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
    """Counters collected during a speculative-decoding run."""

    target_forwards: int = 0
    drafts_proposed: int = 0
    drafts_accepted: int = 0
    tokens_generated: int = 0
    mtp_heads: int = 0
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
            "target_forwards": self.target_forwards,
            "drafts_proposed": self.drafts_proposed,
            "drafts_accepted": self.drafts_accepted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_generated": self.tokens_generated,
            "tokens_per_forward": self.tokens_per_forward,
            "mtp_heads": self.mtp_heads,
        }


def _probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
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
    return F.softmax(logits.float() / temperature, dim=-1)


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
        draft_probs.append(_probs_from_logits(logits, temperature))
    return draft_probs


@torch.no_grad()
def speculate_mtp(
    model: TextDiffusionModel,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
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
    stats = SpecStats(mtp_heads=num_heads)

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

        p_main = _probs_from_logits(logits[:, -1, :], temperature)  # (1, V)
        y_main = _sample_from_probs(p_main)  # (1,)

        draft_probs = _mtp_draft_dists(model, hidden[:, -1:, :], temperature)  # list of (1, V)
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
            p_true = _probs_from_logits(logits_v[:, target_pos, :], temperature)
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
                p_bonus = _probs_from_logits(logits_v[:, bonus_pos, :], temperature)
                y_bonus = _sample_from_probs(p_bonus)
                prefix = torch.cat([prefix, y_bonus.unsqueeze(-1)], dim=1)
                stats.tokens_generated += 1
                if eos_token_id is not None and int(y_bonus[0].item()) == eos_token_id:
                    finished = True

        stats.per_step_accepted.append(accepted_this_step)

    return prefix[0], stats


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------


def _build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TextDiffusionModel, dict]:
    """Load a training checkpoint and rebuild the model on ``device``.

    The checkpoint dict contains ``model_state`` and ``config`` (a dataclass dict)
    as written by ``train.py``.
    """
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

    model = TextDiffusionModel(config).to(device)
    state = blob["model_state"]
    # Strip any DDP/compile prefixes if present.
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[spec_decode] missing keys when loading checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[spec_decode] unexpected keys when loading checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model, blob


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
    parser = argparse.ArgumentParser(description="Speculative-decoding smoke test for TextDiffusionModel + MTP heads.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to checkpoint.pt produced by train.py")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="optional override for the tokenizer cache dir")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy (output must match AR exactly)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="number of warmup runs before timing")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[spec_decode] loading checkpoint {args.checkpoint} onto {device}")
    model, blob = _build_model_from_checkpoint(args.checkpoint, device)
    tokenizer = _load_tokenizer(blob, args)
    print(
        f"[spec_decode] model: d_model={model.config.d_model} "
        f"n_layers={model.config.n_layers} "
        f"n_heads={model.config.n_heads} "
        f"mtp_heads={len(model.mtp_heads)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    if len(model.mtp_heads) == 0:
        print(
            "[spec_decode] WARNING: this checkpoint has 0 MTP heads. "
            "Speculative decoding will fall back to plain AR. "
            "Re-train with --mtp-heads >= 1 to see a speedup."
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
        _ = generate_causal(
            model, prompt_ids, gen_length=8, temperature=args.temperature,
        )
        torch.manual_seed(args.seed)
        _ = speculate_mtp(
            model, prompt_ids, gen_length=8, temperature=args.temperature,
        )

    # --- Baseline: plain AR ---------------------------------------------------
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ar_ids = generate_causal(
        model, prompt_ids, gen_length=args.gen_length, temperature=args.temperature,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    ar_time = time.perf_counter() - t0
    ar_generated = int(ar_ids.size(0) - prompt_ids.size(0))

    # --- Speculative: MTP-drafted --------------------------------------------
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec_ids, stats = speculate_mtp(
        model, prompt_ids, gen_length=args.gen_length, temperature=args.temperature,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    spec_time = time.perf_counter() - t0

    # --- Report ---------------------------------------------------------------
    print()
    print(f"== plain AR ==")
    print(f"  tokens generated: {ar_generated}")
    print(f"  wall time:        {ar_time*1000:.1f} ms ({ar_generated/ar_time:.1f} tok/s)")
    print(f"  output: {_format_ids(ar_ids, tokenizer)}")
    print()
    print(f"== speculative (MTP heads = {stats.mtp_heads}) ==")
    print(f"  tokens generated: {stats.tokens_generated}")
    print(f"  target forwards:  {stats.target_forwards}")
    print(f"  drafts proposed:  {stats.drafts_proposed}")
    print(f"  drafts accepted:  {stats.drafts_accepted}")
    print(f"  acceptance rate:  {stats.acceptance_rate*100:.1f}%")
    print(f"  tokens/forward:   {stats.tokens_per_forward:.2f}")
    print(f"  wall time:        {spec_time*1000:.1f} ms ({stats.tokens_generated/spec_time:.1f} tok/s)")
    print(f"  speedup:          {ar_time/spec_time:.2f}x")
    print(f"  output: {_format_ids(spec_ids, tokenizer)}")

    # --- Correctness check at T=0 --------------------------------------------
    if args.temperature == 0.0:
        common = min(ar_ids.numel(), spec_ids.numel())
        match = bool(torch.equal(ar_ids[:common], spec_ids[:common]))
        print()
        if match:
            print(f"[spec_decode] OK: speculative output matches plain AR token-for-token on the first {common} tokens.")
        else:
            first_diff = int((ar_ids[:common] != spec_ids[:common]).nonzero()[0, 0].item())
            print(
                f"[spec_decode] FAIL: outputs diverge at token index {first_diff}. "
                f"AR={int(ar_ids[first_diff].item())} spec={int(spec_ids[first_diff].item())}. "
                "This indicates a bug in the verify/accept logic."
            )


if __name__ == "__main__":
    main()
