from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from jinja2 import Template

from model import TinyGrootModel
from tokenizer import NanochatTokenizer


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def ensure_eval_bundle(cache_dir: Path) -> Path:
    eval_bundle_dir = cache_dir / "eval_bundle"
    if eval_bundle_dir.exists():
        return eval_bundle_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "eval_bundle.zip"
    tmp_path = zip_path.with_suffix(".zip.tmp")
    print(f"downloading CORE eval bundle: {EVAL_BUNDLE_URL}", flush=True)
    with urllib.request.urlopen(EVAL_BUNDLE_URL, timeout=60) as response:
        with tmp_path.open("wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
    tmp_path.replace(zip_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(str(Path(tmpdir) / "eval_bundle"), eval_bundle_dir)
    return eval_bundle_dir


def render_prompts_mc(item: dict[str, Any], continuation_delimiter: str, fewshot_examples: list[dict[str, Any]]) -> list[str]:
    template = Template(
        """
        {%- for example in fewshot_examples -%}
        {{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}
        {% endfor -%}
        {{ item.query }}{{ continuation_delimiter }}{{ choice }}
        """.strip()
    )
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item: dict[str, Any], continuation_delimiter: str, fewshot_examples: list[dict[str, Any]]) -> list[str]:
    template = Template(
        """
        {%- for example in fewshot_examples -%}
        {{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}
        {% endfor -%}
        {{ context }}{{ continuation_delimiter }}{{ item.continuation }}
        """.strip()
    )
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(context=context_option, **context) for context_option in item["context_options"]]


def render_prompts_lm(item: dict[str, Any], continuation_delimiter: str, fewshot_examples: list[dict[str, Any]]) -> list[str]:
    template = Template(
        """
        {%- for example in fewshot_examples -%}
        {{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}
        {% endfor -%}
        {{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}
        """.strip()
    )
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences: list[list[int]], *, direction: str) -> int:
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def encode_prompts(tokenizer: NanochatTokenizer, prompts: list[str]) -> list[list[int]]:
    return [tokenizer.encode(prompt, add_bos=True) for prompt in prompts]


def batch_sequences_mc(tokenizer: NanochatTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = encode_prompts(tokenizer, prompts)
    answer_start_idx = find_common_length(tokens, direction="left")
    return tokens, [answer_start_idx] * len(prompts), [len(seq) for seq in tokens]


def batch_sequences_schema(tokenizer: NanochatTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = encode_prompts(tokenizer, prompts)
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(seq) for seq in tokens]
    start_indices = [end_idx - suffix_length for end_idx in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer: NanochatTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens_without, tokens_with = encode_prompts(tokenizer, prompts)
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    if start_idx >= end_idx or tokens_without != tokens_with[:start_idx]:
        raise ValueError("language modeling prompt without continuation is not a token prefix")
    return [tokens_with], [start_idx], [end_idx]


def crop_to_model_context(
    model: TinyGrootModel,
    tokens: list[list[int]],
    start_indices: list[int],
    end_indices: list[int],
) -> tuple[list[list[int]], list[int], list[int]] | None:
    max_tokens = model.config.max_seq_len
    cropped_tokens = []
    cropped_starts = []
    cropped_ends = []
    for seq, start_idx, end_idx in zip(tokens, start_indices, end_indices):
        if len(seq) <= max_tokens:
            cropped_tokens.append(seq)
            cropped_starts.append(start_idx)
            cropped_ends.append(end_idx)
            continue

        num_to_crop = len(seq) - max_tokens
        new_start = start_idx - num_to_crop
        new_end = end_idx - num_to_crop
        if new_start <= 0 or new_end <= new_start:
            return None
        cropped_tokens.append(seq[-max_tokens:])
        cropped_starts.append(new_start)
        cropped_ends.append(new_end)
    return cropped_tokens, cropped_starts, cropped_ends


def stack_sequences(tokens: list[list[int]], pad_token_id: int) -> torch.Tensor:
    batch_size = len(tokens)
    seq_len = max(len(seq) for seq in tokens)
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(tokens):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return input_ids


@torch.no_grad()
def forward_causal_model(
    model: TinyGrootModel,
    input_ids: torch.Tensor,
    tokenizer: NanochatTokenizer,
    start_indices: list[int],
    end_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = input_ids != tokenizer.pad_token_id
    logits = model(input_ids, attention_mask=attention_mask, causal=True)
    token_losses = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(input_ids.size(0), input_ids.size(1) - 1)

    losses = torch.zeros_like(input_ids, dtype=token_losses.dtype)
    predictions = torch.full_like(input_ids, tokenizer.pad_token_id)
    predictions[:, 1:] = logits[:, :-1, :].argmax(dim=-1)
    for row, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
        start_idx = max(1, start_idx)
        if end_idx > start_idx:
            losses[row, start_idx:end_idx] = token_losses[row, start_idx - 1 : end_idx - 1]
    return losses, predictions


@torch.no_grad()
def evaluate_example(
    idx: int,
    model: TinyGrootModel,
    tokenizer: NanochatTokenizer,
    data: list[dict[str, Any]],
    device: torch.device,
    task_meta: dict[str, Any],
) -> bool:
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported CORE task type: {task_type}")

    cropped = crop_to_model_context(model, tokens, start_indices, end_indices)
    if cropped is None:
        return False
    tokens, start_indices, end_indices = cropped

    input_ids = stack_sequences(tokens, tokenizer.pad_token_id).to(device)
    losses, predictions = forward_causal_model(model, input_ids, tokenizer, start_indices, end_indices)

    if task_type == "language_modeling":
        start_idx, end_idx = start_indices[0], end_indices[0]
        predicted_tokens = predictions[0, start_idx:end_idx]
        actual_tokens = input_ids[0, start_idx:end_idx]
        return bool(torch.all(predicted_tokens == actual_tokens).item())

    mean_losses = [
        losses[row, start_idx:end_idx].mean().item()
        for row, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
    ]
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


def evaluate_task(
    model: TinyGrootModel,
    tokenizer: NanochatTokenizer,
    data: list[dict[str, Any]],
    device: torch.device,
    task_meta: dict[str, Any],
) -> float:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    for idx in range(rank, len(data), world_size):
        try:
            is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        except ValueError:
            is_correct = False
        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core(
    model: TinyGrootModel,
    tokenizer: NanochatTokenizer,
    device: torch.device,
    *,
    cache_dir: Path,
    max_per_task: int = -1,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("CORE evaluation requires PyYAML.") from exc

    eval_bundle_dir = ensure_eval_bundle(cache_dir)
    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    meta_path = eval_bundle_dir / "eval_meta_data.csv"

    config = yaml.safe_load(config_path.read_text())
    tasks = config["icl_tasks"]

    random_baselines = {}
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ",
                end="",
                flush=True,
            )

        data_path = data_base_path / task_meta["dataset_uri"]
        with data_path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        rng = random.Random(1337)
        rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        random_baseline = random_baselines[label]
        centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        results[label] = accuracy
        centered_results[label] = centered

        if not dist.is_initialized() or dist.get_rank() == 0:
            elapsed = time.time() - start_time
            print(f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {elapsed:.2f}s", flush=True)

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
