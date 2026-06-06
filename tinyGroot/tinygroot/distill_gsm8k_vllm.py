"""Offline GSM8K trace generation with a vLLM teacher, for rejection-sampling
distillation into the tinyGroot student.

Run on a GPU box (needs vLLM + the teacher weights):

    python -m tinygroot.distill_gsm8k_vllm \
        --teacher Qwen/Qwen2.5-0.5B-Instruct \
        --samples 4 --max-problems 200 \
        --out runs/distill_gsm8k.jsonl

Then SFT the student on the result by injecting it as a CustomJSON dataset:

    torchrun ... tinygroot/training/chat_sft.py \
        --identity-jsonl runs/distill_gsm8k.jsonl --identity-weight 3 \
        --gsm8k-epochs 1 ...

Start small (--max-problems 200) to confirm the teacher's format/quality and the
keep-rate before generating the full 7473-problem set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from tinygroot.chat_core_eval import extract_gsm_answer
from tinygroot.distill_gsm8k import run_distillation

# Few-shot exemplars that lock the teacher into inline "a op b = c" arithmetic and a
# final "#### <answer>" line -- exactly what the CoT->tool converter keys on.
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. Write every "
    "calculation inline in the exact form `a op b = c` (op is one of + - * /), using "
    "only two numbers per step. End with a final line `#### <answer>` containing only "
    "the numeric answer."
)
FEWSHOT = [
    (
        "Natalia sold clips to 48 friends in April, and half as many in May. "
        "How many clips did she sell altogether?",
        "In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total?",
        "White fiber needed is 2 / 2 = 1 bolt. In total that is 2 + 1 = 3 bolts.\n#### 3",
    ),
]


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM GSM8K teacher trace generation.")
    p.add_argument("--teacher", required=True, help="HF model id or local path of the teacher.")
    p.add_argument("--out", type=Path, required=True, help="Output jsonl (CustomJSON format).")
    p.add_argument("--split", default="train")
    p.add_argument("--max-problems", type=int, default=-1, help="<=0 uses the full split.")
    p.add_argument("--samples", type=int, default=4, help="Teacher samples per problem.")
    p.add_argument("--max-keep-per-problem", type=int, default=2, help="Distinct correct traces kept per problem.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    problems = [
        {"question": row["question"], "gold": extract_gsm_answer(row["answer"])}
        for row in ds
    ]
    problems = [p for p in problems if p["gold"] is not None]
    if args.max_problems > 0:
        problems = problems[: args.max_problems]
    print(f"loaded {len(problems)} GSM8K problems from split={args.split}")

    llm = LLM(
        model=args.teacher,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    def generate_fn(questions: list[str], n: int) -> list[list[str]]:
        prompts = [build_prompt(tokenizer, q) for q in questions]
        outputs = llm.generate(prompts, sampling)  # vLLM batches internally
        return [[o.text for o in out.outputs] for out in outputs]

    stats = run_distillation(
        problems,
        generate_fn,
        samples_per_problem=args.samples,
        out_path=args.out,
        max_keep_per_problem=args.max_keep_per_problem,
    )
    keep_rate = stats["problems_covered"] / max(1, stats["problems"])
    print(
        f"distillation: problems={stats['problems']} "
        f"covered={stats['problems_covered']} ({keep_rate:.1%}) "
        f"traces_written={stats['traces_written']} -> {args.out}"
    )
    if keep_rate < 0.15:
        print(
            "WARNING: low coverage -- the teacher rarely solves GSM8K in the required "
            "format. Check a few raw completions, raise --samples, or use a stronger teacher."
        )


if __name__ == "__main__":
    main()
