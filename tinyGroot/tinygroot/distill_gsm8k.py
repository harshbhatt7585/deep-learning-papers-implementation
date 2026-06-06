"""GSM8K rejection-sampling distillation pipeline.

A teacher model generates chain-of-thought solutions; we convert each arithmetic
step ``a op b = c`` into a verified calculator tool call (computing the result
ourselves so tool outputs are always correct), keep only traces whose final
answer matches the gold answer, and write them as conversations that ``CustomJSON``
can load for SFT. The generation backend is abstracted behind a ``generate_fn``
callable so the deterministic core (convert/verify/filter) is unit-testable
without a GPU, and the same core serves teacher-distillation and self-distillation
(rejection-sampling fine-tuning).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from tinygroot.chat_core_eval import extract_gsm_answer, use_calculator

# a op b = c  (numbers may have commas/decimals; op in + - * /)
_ARITH_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([-+*/])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)"
)


def _norm_num(s: str) -> str | None:
    """Normalise a numeric string the way extract_gsm_answer does (drop commas,
    integral floats -> ints) so computed and written outputs compare cleanly."""
    return extract_gsm_answer(f"#### {s}")


def cot_to_tool_parts(cot: str) -> list[dict[str, str]] | None:
    """Convert a plain CoT string into interleaved text / python / python_output
    parts, replacing each ``a op b = c`` with a tool call whose output we compute
    ourselves via ``use_calculator``. Returns None if no arithmetic step is found
    (a trace with no computation is not useful as a tool-use example)."""
    parts: list[dict[str, str]] = []
    cursor = 0
    found = False
    for m in _ARITH_RE.finditer(cot):
        a, op, b, _claimed = m.groups()
        expr = f"{a.replace(',', '')}{op}{b.replace(',', '')}"
        computed = use_calculator(expr)
        if computed is None:
            continue  # leave unparseable arithmetic as plain text
        result = _norm_num(str(computed))
        if result is None:
            continue
        lead = cot[cursor : m.start()]
        if lead:
            parts.append({"type": "text", "text": lead})
        parts.append({"type": "python", "text": expr})
        parts.append({"type": "python_output", "text": result})
        cursor = m.end()
        found = True
    if not found:
        return None
    tail = cot[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts


def make_example(question: str, cot: str, gold: str) -> list[dict[str, Any]] | None:
    """Build a [user, assistant] conversation from a teacher CoT, or None if the
    trace is wrong (final answer != gold) or has no usable arithmetic."""
    pred = extract_gsm_answer(cot)
    if pred is None or pred != gold:
        return None  # rejection sampling: keep only correct final answers
    parts = cot_to_tool_parts(cot)
    if parts is None:
        return None
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": parts},
    ]


def run_distillation(
    problems: list[dict[str, str]],
    generate_fn: Callable[[list[str], int], list[list[str]]],
    *,
    samples_per_problem: int,
    out_path: Path,
    max_keep_per_problem: int = 4,
) -> dict[str, int]:
    """Generate, filter, and write distilled SFT conversations.

    ``problems``: list of {"question": str, "gold": str}.
    ``generate_fn(questions, n)``: returns, for each question, a list of n CoT strings.
    Writes one JSON conversation per line (CustomJSON format) and returns stats.
    """
    questions = [p["question"] for p in problems]
    golds = [p["gold"] for p in problems]
    completions = generate_fn(questions, samples_per_problem)
    if len(completions) != len(problems):
        raise ValueError("generate_fn must return one completion list per problem")

    kept = 0
    covered = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for question, gold, samples in zip(questions, golds, completions):
            kept_this = 0
            seen: set[str] = set()
            for cot in samples:
                if kept_this >= max_keep_per_problem:
                    break
                convo = make_example(question, cot, gold)
                if convo is None:
                    continue
                key = json.dumps(convo[1]["content"], sort_keys=True)
                if key in seen:
                    continue  # dedup identical traces for the same problem
                seen.add(key)
                f.write(json.dumps(convo) + "\n")
                kept += 1
                kept_this += 1
            if kept_this > 0:
                covered += 1
    return {
        "problems": len(problems),
        "problems_covered": covered,   # >=1 correct trace
        "traces_written": kept,
    }
