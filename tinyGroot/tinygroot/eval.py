from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinygroot.chat_core_eval import execute_code, extract_gsm_answer, extract_imports, extract_program
from tinygroot.engine import Engine
from tinygroot.fp8 import disable_fp8
from tinygroot.model import TinyGrootConfig, TinyGrootModel, infer_arch_from_state_dict
from tinygroot.sft_chat import generate_with_tools, render_prompt_for_completion
from tinygroot.sft_data import ARC, GSM8K, HumanEval, MMLU, SpellingBee, Task, ensure_words
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.utils import (
    Runtime,
    cleanup_distributed,
    create_runtime,
    is_dist,
    load_meta,
    load_model_state,
    log,
    rank,
    resolve_checkpoint_dir,
    unwrap_model,
    world_size,
)


CHATCORE_CATEGORICAL = ("ARC-Easy", "ARC-Challenge", "MMLU")
CHATCORE_GENERATIVE = ("GSM8K", "HumanEval", "SpellingBee")
CHATCORE_BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_checkpoint_for_eval(
    checkpoint: Path,
    runtime: Runtime,
    *,
    tokenizer_dir: Path | None = None,
) -> tuple[TinyGrootModel, NanochatTokenizer, dict[str, Any]]:
    meta = load_meta(checkpoint)
    state = clean_state_dict(load_model_state(checkpoint, map_location=runtime.device))
    cfg = dict(meta["config"])
    cfg["arch"] = infer_arch_from_state_dict(state)
    config = TinyGrootConfig(**cfg)
    tokenizer_path = tokenizer_dir or resolve_checkpoint_dir(checkpoint) / "tokenizer_hf"
    tokenizer = NanochatTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config).to(runtime.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, meta


def _assistant_ref_text(conversation: dict[str, Any]) -> str:
    content = conversation["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return ""


def evaluate_gsm_completion(conversation: dict[str, Any], completion: str) -> bool:
    ref = extract_gsm_answer(_assistant_ref_text(conversation))
    pred = extract_gsm_answer(completion)
    return pred is not None and pred == ref


def evaluate_humaneval_completion(conversation: dict[str, Any], completion: str) -> bool:
    imports = extract_imports(conversation["messages"][0]["content"])
    code = extract_program(completion)
    program = (
        f"{imports}\n\n{code}\n\n{conversation['test']}\n"
        f"check({conversation['entry_point']})"
    )
    return execute_code(program).success


def _reduce_pass_counts(num_passed: int, total: int, device: torch.device) -> tuple[int, int]:
    counts = torch.tensor([num_passed, total], dtype=torch.long, device=device)
    if is_dist():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


def _build_chatcore_task(name: str, words_path: Path) -> Task:
    if name == "ARC-Easy":
        return ARC("ARC-Easy", "test")
    if name == "ARC-Challenge":
        return ARC("ARC-Challenge", "test")
    if name == "MMLU":
        return MMLU("test")
    if name == "GSM8K":
        return GSM8K("test")
    if name == "HumanEval":
        return HumanEval()
    if name == "SpellingBee":
        return SpellingBee(ensure_words(words_path), size=256, split="test")
    raise KeyError(name)


@torch.no_grad()
def _chatcore_categorical(
    model: TinyGrootModel,
    tokenizer: NanochatTokenizer,
    runtime: Runtime,
    task: Task,
    *,
    batch_size: int,
    max_problems: int,
    prompt_max_tokens: int,
) -> float:
    bos = tokenizer.bos_token_id
    num_problems = len(task) if max_problems <= 0 else min(len(task), max_problems)
    if num_problems <= 0:
        return 0.0
    num_batches = -(-num_problems // batch_size)
    letter_to_id: dict[str, int] = {}
    num_passed = 0
    total = 0
    for batch_idx in range(rank(), num_batches, world_size()):
        i0 = batch_idx * batch_size
        i1 = min(i0 + batch_size, num_problems)
        conversations = [task[i] for i in range(i0, i1)]
        prompt_ids = [render_prompt_for_completion(tokenizer, c, max_tokens=prompt_max_tokens) for c in conversations]
        max_len = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_len - len(ids)) for ids in prompt_ids]
        x = torch.tensor(padded, dtype=torch.long, device=runtime.device)
        logits = model(x, attention_mask=None, causal=True)
        for idx, conv in enumerate(conversations):
            letters = conv["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id:
                    encoded = tokenizer.encode(letter)
                    if len(encoded) != 1:
                        raise ValueError(f"letter {letter!r} must encode to a single token, got {encoded}")
                    letter_to_id[letter] = encoded[0]
                letter_ids.append(letter_to_id[letter])
            focus = logits[idx, answer_positions[idx], letter_ids]
            predicted = letters[int(focus.argmax(dim=-1).item())]
            gold = conv["messages"][-1]["content"]
            num_passed += int(predicted == gold)
            total += 1
    num_passed, total = _reduce_pass_counts(num_passed, total, runtime.device)
    return num_passed / max(1, total)


@torch.no_grad()
def _chatcore_generative(
    model: TinyGrootModel,
    tokenizer: NanochatTokenizer,
    runtime: Runtime,
    task: Task,
    evaluate_fn: Callable[[dict[str, Any], str], bool],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    max_problems: int,
    prompt_max_tokens: int,
) -> float:
    num_problems = len(task) if max_problems <= 0 else min(len(task), max_problems)
    if num_problems <= 0:
        return 0.0
    num_passed = 0
    total = 0
    for i in range(rank(), num_problems, world_size()):
        conv = task[i]
        prompt_ids = render_prompt_for_completion(tokenizer, conv, max_tokens=prompt_max_tokens)
        gen_tokens = generate_with_tools(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        completion = tokenizer.decode(gen_tokens, skip_special=True)
        num_passed += int(bool(evaluate_fn(conv, completion)))
        total += 1
    num_passed, total = _reduce_pass_counts(num_passed, total, runtime.device)
    return num_passed / max(1, total)


@torch.no_grad()
def evaluate_chatcore(model: torch.nn.Module, tokenizer: NanochatTokenizer, args: Any, runtime: Runtime) -> dict[str, float]:
    if getattr(args, "chatcore_every", 1) <= 0:
        return {}
    source = unwrap_model(model)
    was_training = source.training
    source.eval()
    prompt_max_tokens = max(64, source.config.max_seq_len - args.chatcore_max_new_tokens - 8)
    words_path = getattr(args, "words_path", Path("/data/words_alpha.txt"))
    results: dict[str, float] = {}
    with disable_fp8(source):
        for name in CHATCORE_CATEGORICAL:
            task = _build_chatcore_task(name, words_path)
            acc = _chatcore_categorical(
                source,
                tokenizer,
                runtime,
                task,
                batch_size=args.chatcore_batch_size,
                max_problems=args.chatcore_max_cat,
                prompt_max_tokens=prompt_max_tokens,
            )
            results[name] = acc
            log(f"chatcore {name}: {100 * acc:.2f}%")

        generative_evaluators = {
            "GSM8K": evaluate_gsm_completion,
            "HumanEval": evaluate_humaneval_completion,
            "SpellingBee": evaluate_gsm_completion,
        }
        for name in CHATCORE_GENERATIVE:
            task = _build_chatcore_task(name, words_path)
            acc = _chatcore_generative(
                source,
                tokenizer,
                runtime,
                task,
                generative_evaluators[name],
                max_new_tokens=args.chatcore_max_new_tokens,
                temperature=args.chatcore_temperature,
                top_k=args.chatcore_top_k if args.chatcore_top_k > 0 else None,
                max_problems=args.chatcore_max_sample,
                prompt_max_tokens=prompt_max_tokens,
            )
            results[name] = acc
            log(f"chatcore {name}: {100 * acc:.2f}%")
    if was_training:
        source.train()

    all_tasks = CHATCORE_CATEGORICAL + CHATCORE_GENERATIVE
    centered = sum((results[n] - CHATCORE_BASELINES[n]) / (1.0 - CHATCORE_BASELINES[n]) for n in all_tasks)
    centered /= len(all_tasks)
    cat_centered = sum(
        (results[n] - CHATCORE_BASELINES[n]) / (1.0 - CHATCORE_BASELINES[n])
        for n in CHATCORE_CATEGORICAL
    ) / len(CHATCORE_CATEGORICAL)
    metrics: dict[str, float] = {"chatcore": centered, "chatcore_cat": cat_centered}
    for name, acc in results.items():
        metrics[f"chatcore_{name}"] = acc
    return metrics


@torch.no_grad()
def evaluate_gsm8k_passk(
    model: torch.nn.Module,
    tokenizer: NanochatTokenizer,
    runtime: Runtime,
    *,
    task: GSM8K | None = None,
    max_examples: int = 400,
    num_samples: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = 50,
    seed: int = 0,
) -> dict[str, float]:
    source = unwrap_model(model)
    was_training = source.training
    source.eval()
    task = task or GSM8K("test")
    num_samples = max(1, num_samples)
    pass_counts = torch.zeros(num_samples, dtype=torch.float32, device=runtime.device)
    total = 0
    max_examples = min(max_examples, len(task))
    engine = Engine(source, tokenizer)
    with disable_fp8(source):
        for idx in range(rank(), max_examples, world_size()):
            conversation = task[idx]
            prompt_ids = render_prompt_for_completion(
                tokenizer,
                conversation,
                max_tokens=max(1, source.config.max_seq_len - max_new_tokens),
            )
            sequences, _masks = engine.generate_batch(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=min(max_new_tokens, max(0, source.config.max_seq_len - len(prompt_ids))),
                temperature=temperature,
                top_k=top_k,
                seed=(seed + idx) & 0x7FFFFFFF,
            )
            outcomes = []
            for seq in sequences:
                completion = tokenizer.decode(seq[len(prompt_ids):], skip_special=True)
                outcomes.append(evaluate_gsm_completion(conversation, completion))
            for k in range(1, num_samples + 1):
                pass_counts[k - 1] += float(any(outcomes[:k]))
            total += 1
    total_tensor = torch.tensor(total, dtype=torch.float32, device=runtime.device)
    if is_dist():
        dist.all_reduce(pass_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    if was_training:
        source.train()
    pass_rates = pass_counts / total_tensor.clamp(min=1)
    return {f"pass@{k}": float(pass_rates[k - 1].item()) for k in range(1, num_samples + 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a tinyGroot chat checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint.pt or its run directory.")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Defaults to <checkpoint-dir>/tokenizer_hf.")
    parser.add_argument("--suite", choices=["chatcore", "gsm8k-passk", "both"], default="chatcore")
    parser.add_argument("--device-type", type=str, default="", help="Reserved for compatibility; runtime autodetects today.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="Max categorical examples per task (-1 = all).")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="Max generative examples per task.")
    parser.add_argument("--chatcore-max-new-tokens", type=int, default=512)
    parser.add_argument("--chatcore-temperature", type=float, default=0.0)
    parser.add_argument("--chatcore-top-k", type=int, default=50)
    parser.add_argument("--chatcore-batch-size", type=int, default=8)
    parser.add_argument("--words-path", type=Path, default=Path("/data/words_alpha.txt"))

    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--eval-num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        model, tokenizer, meta = load_checkpoint_for_eval(args.checkpoint, runtime, tokenizer_dir=args.tokenizer_dir)
        log(f"loaded checkpoint: {args.checkpoint} step={meta.get('step')}")
        metrics: dict[str, float] = {}
        start = time.time()
        if args.suite in ("chatcore", "both"):
            metrics.update(evaluate_chatcore(model, tokenizer, args, runtime))
        if args.suite in ("gsm8k-passk", "both"):
            metrics.update(
                {
                    f"gsm8k_{key}": value
                    for key, value in evaluate_gsm8k_passk(
                        model,
                        tokenizer,
                        runtime,
                        max_examples=args.eval_examples,
                        num_samples=args.eval_num_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k if args.top_k > 0 else None,
                        seed=args.seed,
                    ).items()
                }
            )
        log(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()) + f" eval_time={time.time() - start:.1f}s")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
