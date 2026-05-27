from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from chat_core_eval import (
    execute_code,
    extract_gsm_answer,
    extract_imports,
    extract_program,
    use_calculator,
)
from fp8 import disable_fp8
from model import TextDiffusionModel, _sample_tokens
from sft_data import ARC, GSM8K, HumanEval, MMLU, SpellingBee, Task, ensure_words
from tokenizer import NanochatTokenizer
from utils import Runtime, is_dist, log, rank, unwrap_model, world_size


SAMPLE_USER_PROMPTS = [
    "Write a short polite reply to: I am feeling nervous about an exam.",
    "Explain photosynthesis in two sentences.",
    (
        "Multiple Choice question: What gas do plants absorb during photosynthesis?\n"
        "- Oxygen=A\n"
        "- Carbon dioxide=B\n"
        "- Nitrogen=C\n"
        "- Helium=D\n\n"
        "Respond only with the letter of the correct answer."
    ),
    "Sarah has 3 bags with 4 apples each. She eats 2 apples. How many apples are left?",
    "Spell the word: strawberry",
    "How many r are in the word strawberry?",
]


def special_id(tokenizer: NanochatTokenizer, text: str) -> int:
    token_id = tokenizer.tokenizer.token_to_id(text)
    if token_id is None:
        raise KeyError(f"Tokenizer is missing special token {text!r}")
    return int(token_id)


@dataclass(frozen=True)
class ChatSpecialIds:
    user_start: int
    user_end: int
    assistant_start: int
    assistant_end: int
    python_start: int
    python_end: int
    output_start: int
    output_end: int

    @classmethod
    def from_tokenizer(cls, tokenizer: NanochatTokenizer) -> "ChatSpecialIds":
        return cls(
            user_start=special_id(tokenizer, "<|user_start|>"),
            user_end=special_id(tokenizer, "<|user_end|>"),
            assistant_start=special_id(tokenizer, "<|assistant_start|>"),
            assistant_end=special_id(tokenizer, "<|assistant_end|>"),
            python_start=special_id(tokenizer, "<|python_start|>"),
            python_end=special_id(tokenizer, "<|python_end|>"),
            output_start=special_id(tokenizer, "<|output_start|>"),
            output_end=special_id(tokenizer, "<|output_end|>"),
        )


def render_conversation(
    tokenizer: NanochatTokenizer,
    conversation: dict[str, Any],
    *,
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []

    def add(token_ids: int | list[int], mask_val: int) -> None:
        values = [token_ids] if isinstance(token_ids, int) else token_ids
        ids.extend(values)
        mask.extend([mask_val] * len(values))

    messages = conversation["messages"]
    if messages and messages[0]["role"] == "system":
        messages = copy.deepcopy(messages)
        if len(messages) < 2 or messages[1]["role"] != "user":
            return [tokenizer.bos_token_id], [0]
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]

    add(tokenizer.bos_token_id, 0)
    specials = ChatSpecialIds.from_tokenizer(tokenizer)

    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != expected:
            break
        content = message["content"]
        if expected == "user":
            if not isinstance(content, str):
                break
            add(specials.user_start, 0)
            add(tokenizer.encode(content), 0)
            add(specials.user_end, 0)
        else:
            add(specials.assistant_start, 0)
            if isinstance(content, str):
                add(tokenizer.encode(content), 1)
            elif isinstance(content, list):
                for part in content:
                    text_ids = tokenizer.encode(part["text"])
                    if part["type"] == "text":
                        add(text_ids, 1)
                    elif part["type"] == "python":
                        add(specials.python_start, 1)
                        add(text_ids, 1)
                        add(specials.python_end, 1)
                    elif part["type"] == "python_output":
                        add(specials.output_start, 0)
                        add(text_ids, 0)
                        add(specials.output_end, 0)
            add(specials.assistant_end, 1)
        if len(ids) >= max_tokens:
            break
    return ids[:max_tokens], mask[:max_tokens]


def render_prompt_for_completion(
    tokenizer: NanochatTokenizer,
    conversation: dict[str, Any],
    *,
    max_tokens: int,
) -> list[int]:
    messages = list(conversation["messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    ids, _ = render_conversation(tokenizer, {"messages": messages}, max_tokens=max_tokens)
    ids = list(ids)
    ids.append(ChatSpecialIds.from_tokenizer(tokenizer).assistant_start)
    return ids


@torch.no_grad()
def generate_with_tools(
    model: TextDiffusionModel,
    tokenizer: NanochatTokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> list[int]:
    bos = tokenizer.bos_token_id
    specials = ChatSpecialIds.from_tokenizer(tokenizer)

    device = model.device
    max_seq_len = model.config.max_seq_len
    seq = list(prompt_ids)
    prompt_len = len(seq)
    forced: list[int] = []
    in_python = False
    python_expr_tokens: list[int] = []

    for _ in range(max_new_tokens):
        if forced:
            next_token = forced.pop(0)
        else:
            if len(seq) >= max_seq_len:
                break
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x, attention_mask=None, causal=True)
            sampled, _ = _sample_tokens(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=None)
            next_token = int(sampled.item())
        seq.append(next_token)
        if next_token == specials.assistant_end or next_token == bos:
            break
        if next_token == specials.python_start:
            in_python = True
            python_expr_tokens = []
        elif next_token == specials.python_end and in_python:
            in_python = False
            expr = tokenizer.decode(python_expr_tokens, skip_special=True)
            result = use_calculator(expr)
            if result is not None:
                result_tokens = tokenizer.encode(str(result))
                forced.append(specials.output_start)
                forced.extend(result_tokens)
                forced.append(specials.output_end)
            python_expr_tokens = []
        elif in_python:
            python_expr_tokens.append(next_token)

    return seq[prompt_len:]


def _assistant_ref_text(conversation: dict[str, Any]) -> str:
    content = conversation["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        return content[-1].get("text", "")
    return ""


def _evaluate_gsm_completion(conversation: dict[str, Any], completion: str) -> bool:
    ref = extract_gsm_answer(_assistant_ref_text(conversation))
    pred = extract_gsm_answer(completion)
    return pred is not None and pred == ref


def _evaluate_humaneval_completion(conversation: dict[str, Any], completion: str) -> bool:
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


def _chatcore_categorical(
    model: TextDiffusionModel,
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
                    assert len(encoded) == 1, f"letter {letter!r} must encode to a single token (got {encoded})"
                    letter_to_id[letter] = encoded[0]
                letter_ids.append(letter_to_id[letter])
            focus = logits[idx, answer_positions[idx], letter_ids]
            predicted = letters[int(focus.argmax(dim=-1).item())]
            gold = conv["messages"][-1]["content"]
            num_passed += int(predicted == gold)
            total += 1
    num_passed, total = _reduce_pass_counts(num_passed, total, runtime.device)
    return num_passed / max(1, total)


def _chatcore_generative(
    model: TextDiffusionModel,
    tokenizer: NanochatTokenizer,
    runtime: Runtime,
    task: Task,
    evaluate_fn,
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
            model, tokenizer, prompt_ids,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        )
        completion = tokenizer.decode(gen_tokens, skip_special=True)
        num_passed += int(bool(evaluate_fn(conv, completion)))
        total += 1
    num_passed, total = _reduce_pass_counts(num_passed, total, runtime.device)
    return num_passed / max(1, total)


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


def _build_chatcore_task(name: str, args) -> Task:
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
        words = ensure_words(args.words_path)
        return SpellingBee(words, size=256, split="test")
    raise KeyError(name)


@torch.no_grad()
def evaluate_chatcore(model: torch.nn.Module, tokenizer: NanochatTokenizer, args, runtime: Runtime) -> dict[str, float]:
    if args.chatcore_every <= 0:
        return {}
    model.eval()
    source = unwrap_model(model)
    prompt_max_tokens = max(64, source.config.max_seq_len - args.chatcore_max_new_tokens - 8)
    results: dict[str, float] = {}
    with disable_fp8(source):
        for name in CHATCORE_CATEGORICAL:
            task = _build_chatcore_task(name, args)
            acc = _chatcore_categorical(
                source, tokenizer, runtime, task,
                batch_size=args.chatcore_batch_size,
                max_problems=args.chatcore_max_cat,
                prompt_max_tokens=prompt_max_tokens,
            )
            results[name] = acc
            log(f"chatcore {name}: {100 * acc:.2f}%")
        generative_evaluators = {
            "GSM8K": _evaluate_gsm_completion,
            "HumanEval": _evaluate_humaneval_completion,
            "SpellingBee": _evaluate_gsm_completion,
        }
        for name in CHATCORE_GENERATIVE:
            task = _build_chatcore_task(name, args)
            acc = _chatcore_generative(
                source, tokenizer, runtime, task, generative_evaluators[name],
                max_new_tokens=args.chatcore_max_new_tokens,
                temperature=args.chatcore_temperature,
                top_k=args.chatcore_top_k,
                max_problems=args.chatcore_max_sample,
                prompt_max_tokens=prompt_max_tokens,
            )
            results[name] = acc
            log(f"chatcore {name}: {100 * acc:.2f}%")
    model.train()
    all_tasks = CHATCORE_CATEGORICAL + CHATCORE_GENERATIVE
    centered = sum(
        (results[n] - CHATCORE_BASELINES[n]) / (1.0 - CHATCORE_BASELINES[n])
        for n in all_tasks
    ) / len(all_tasks)
    cat_centered = sum(
        (results[n] - CHATCORE_BASELINES[n]) / (1.0 - CHATCORE_BASELINES[n])
        for n in CHATCORE_CATEGORICAL
    ) / len(CHATCORE_CATEGORICAL)
    metrics: dict[str, float] = {"chatcore": centered, "chatcore_cat": cat_centered}
    for n, acc in results.items():
        metrics[f"chatcore_{n}"] = acc
    return metrics


@torch.no_grad()
def sample_text(model: torch.nn.Module, tokenizer: NanochatTokenizer, args, runtime: Runtime) -> str:
    source = unwrap_model(model)
    source.eval()
    samples = []
    with disable_fp8(source):
        for prompt in SAMPLE_USER_PROMPTS:
            prompt_ids = render_prompt_for_completion(
                tokenizer,
                {"messages": [{"role": "user", "content": prompt}]},
                max_tokens=max(1, source.config.max_seq_len - args.sample_length),
            )
            output = generate_with_tools(
                source,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.sample_length,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
            )
            decoded = tokenizer.decode(output, skip_special=True).strip()
            samples.append(f"USER:\n{prompt}\n\nMODEL:\n{decoded}")
    sample = "\n\n---\n\n".join(samples)
    log("samples:\n" + sample)
    source.train()
    return sample
