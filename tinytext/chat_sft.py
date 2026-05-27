from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from core_eval import ensure_eval_bundle, evaluate_core
from fp8 import disable_fp8
from model import TextDiffusionConfig, TextDiffusionModel, generate_causal
from nanochat_optim import DistMuonAdamW, MuonAdamW
from tokenizer import NanochatTokenizer
from train import apply_fp8_training, fp8_module_filter
from utils import (
    Runtime,
    args_as_plain_dict,
    autocast_context,
    cleanup_distributed,
    create_runtime,
    init_wandb,
    is_dist,
    is_main_process,
    log,
    rank,
    save_checkpoint,
    set_lr,
    unwrap_model,
    world_size,
)


IGNORE_INDEX = -100
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyText chat SFT from a TinyText checkpoint.pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="TinyText pretrain checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for SFT checkpoint.pt")
    parser.add_argument("--run", "--wandb-name", dest="wandb_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="text-diffusion-sft")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/wandb"))

    parser.add_argument("--device-batch-size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--total-batch-size", type=int, default=524_288, help="Global SFT tokens per optimizer step")
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to checkpoint max_seq_len")
    parser.add_argument("--max-steps", "--num-iterations", dest="max_steps", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-tokens", type=int, default=2_097_152)
    parser.add_argument("--core-every", "--chatcore-every", dest="core_every", type=int, default=-1)
    parser.add_argument("--core-eval-max-per-task", type=int, default=500)
    parser.add_argument("--core-eval-cache-dir", type=Path, default=Path("data/core_eval"))
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--sample-temperature", type=float, default=0.6)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--sample-top-p", type=float, default=None)

    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.8)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise")
    parser.add_argument("--train-mtp-heads", action="store_true", help="Default freezes MTP heads for pure chat SFT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--identity-jsonl", type=Path, default=Path("/data/identity_conversations.jsonl"))
    parser.add_argument("--identity-weight", type=int, default=2)
    parser.add_argument("--no-smoltalk", action="store_true")
    parser.add_argument("--mmlu-epochs", type=int, default=3)
    parser.add_argument("--gsm8k-epochs", type=int, default=4)
    parser.add_argument("--simple-spelling-size", type=int, default=200_000)
    parser.add_argument("--spellingbee-size", type=int, default=80_000)
    parser.add_argument("--words-path", type=Path, default=Path("/data/words_alpha.txt"))
    return parser.parse_args()


class Task:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError


class TaskMixture(Task):
    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks
        self.lengths = [len(task) for task in tasks]
        self.index_map: list[tuple[int, int]] = []
        for task_idx, length in enumerate(self.lengths):
            self.index_map.extend((task_idx, local_idx) for local_idx in range(length))
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class SmolTalk(Task):
    def __init__(self, split: str) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"messages": self.ds[index]["messages"]}


def render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


class MMLU(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, split: str, stop: int | None = None) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("cais/mmlu", "all", split=split).shuffle(seed=42)
        self.stop = stop

    def __len__(self) -> int:
        return min(len(self.ds), self.stop) if self.stop is not None else len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": render_mc(row["question"], self.letters, row["choices"])},
                {"role": "assistant", "content": self.letters[row["answer"]]},
            ]
        }


class GSM8K(Task):
    def __init__(self, split: str, stop: int | None = None) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=42)
        self.stop = stop

    def __len__(self) -> int:
        return min(len(self.ds), self.stop) if self.stop is not None else len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import re

        row = self.ds[index]
        parts = []
        for part in re.split(r"(<<[^>]+>>)", row["answer"]):
            if part.startswith("<<") and part.endswith(">>"):
                inner = part[2:-2]
                expr, result = inner.rsplit("=", 1) if "=" in inner else (inner, "")
                parts.append({"type": "python", "text": expr})
                parts.append({"type": "python_output", "text": result})
            else:
                parts.append({"type": "text", "text": part})
        return {"messages": [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": parts}]}


class CustomJSON(Task):
    def __init__(self, path: Path) -> None:
        self.conversations: list[list[dict[str, Any]]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.conversations.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"messages": self.conversations[index]}


def ensure_words(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dist():
        if rank() == 0 and not path.exists():
            urllib.request.urlretrieve(WORD_LIST_URL, path)
        dist.barrier()
    elif not path.exists():
        urllib.request.urlretrieve(WORD_LIST_URL, path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SimpleSpelling(Task):
    def __init__(self, words: list[str], size: int, split: str) -> None:
        self.words = list(words)
        random.Random(42).shuffle(self.words)
        self.size = size
        self.seed_offset = 0 if split == "train" else 10_000_000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        word = random.Random(self.seed_offset + index).choice(self.words)
        return {"messages": [{"role": "user", "content": f"Spell the word: {word}"}, {"role": "assistant", "content": f"{word}:{','.join(word)}"}]}


class SpellingBee(Task):
    templates = [
        "How many {letter} are in the word {word}",
        "How many {letter} are in {word}",
        "Count the number of {letter} in {word}",
        "How many times does {letter} appear in {word}",
        "In the word {word}, how many {letter} are there",
    ]

    def __init__(self, words: list[str], size: int, split: str) -> None:
        self.words = words
        self.size = size
        self.seed_offset = 0 if split == "train" else 10_000_000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed_offset + index)
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice("abcdefghijklmnopqrstuvwxyz")
        count = word.count(letter)
        prompt = rng.choice(self.templates).format(letter=letter, word=word)
        spelled = ",".join(word)
        answer = (
            f"We need to count '{letter}' in '{word}'.\n\n"
            f"First spell the word:\n{word}:{spelled}\n\n"
            f"The letter '{letter}' appears {count} times.\n\n#### {count}"
        )
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def build_datasets(args: argparse.Namespace) -> tuple[TaskMixture, TaskMixture]:
    train_tasks: list[Task] = []
    if not args.no_smoltalk:
        train_tasks.append(SmolTalk("train"))
    identity = CustomJSON(args.identity_jsonl)
    if len(identity) > 0:
        train_tasks.extend(identity for _ in range(args.identity_weight))
    elif args.identity_weight > 0:
        log(f"identity jsonl not found or empty, skipping: {args.identity_jsonl}")
    train_tasks.extend(MMLU("auxiliary_train") for _ in range(args.mmlu_epochs))
    train_tasks.extend(GSM8K("train") for _ in range(args.gsm8k_epochs))

    needs_words = args.simple_spelling_size > 0 or args.spellingbee_size > 0
    words = ensure_words(args.words_path) if needs_words else []
    if args.simple_spelling_size > 0:
        train_tasks.append(SimpleSpelling(words, args.simple_spelling_size, "train"))
    if args.spellingbee_size > 0:
        train_tasks.append(SpellingBee(words, args.spellingbee_size, "train"))
    if not train_tasks:
        raise ValueError("No SFT training tasks enabled.")

    val_tasks: list[Task] = []
    if not args.no_smoltalk:
        val_tasks.append(SmolTalk("test"))
    if args.mmlu_epochs > 0:
        val_tasks.append(MMLU("test", stop=5200))
    if args.gsm8k_epochs > 0:
        val_tasks.append(GSM8K("test", stop=420))
    if args.simple_spelling_size > 0:
        val_tasks.append(SimpleSpelling(words, min(5000, args.simple_spelling_size), "test"))
    if args.spellingbee_size > 0:
        val_tasks.append(SpellingBee(words, min(2000, args.spellingbee_size), "test"))
    if not val_tasks:
        val_tasks.append(train_tasks[0])
    return TaskMixture(train_tasks), TaskMixture(val_tasks)


def special_id(tokenizer: NanochatTokenizer, text: str) -> int:
    token_id = tokenizer.tokenizer.token_to_id(text)
    if token_id is None:
        raise KeyError(f"Tokenizer is missing special token {text!r}")
    return int(token_id)


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
    user_start = special_id(tokenizer, "<|user_start|>")
    user_end = special_id(tokenizer, "<|user_end|>")
    assistant_start = special_id(tokenizer, "<|assistant_start|>")
    assistant_end = special_id(tokenizer, "<|assistant_end|>")
    python_start = special_id(tokenizer, "<|python_start|>")
    python_end = special_id(tokenizer, "<|python_end|>")
    output_start = special_id(tokenizer, "<|output_start|>")
    output_end = special_id(tokenizer, "<|output_end|>")

    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != expected:
            break
        content = message["content"]
        if expected == "user":
            if not isinstance(content, str):
                break
            add(user_start, 0)
            add(tokenizer.encode(content), 0)
            add(user_end, 0)
        else:
            add(assistant_start, 0)
            if isinstance(content, str):
                add(tokenizer.encode(content), 1)
            elif isinstance(content, list):
                for part in content:
                    text_ids = tokenizer.encode(part["text"])
                    if part["type"] == "text":
                        add(text_ids, 1)
                    elif part["type"] == "python":
                        add(python_start, 1)
                        add(text_ids, 1)
                        add(python_end, 1)
                    elif part["type"] == "python_output":
                        add(output_start, 0)
                        add(text_ids, 0)
                        add(output_end, 0)
            add(assistant_end, 1)
        if len(ids) >= max_tokens:
            break
    return ids[:max_tokens], mask[:max_tokens]


class SFTBatcher:
    def __init__(self, dataset: Task, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime, split: str, buffer_size: int = 100) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.runtime = runtime
        self.split = split
        self.buffer_size = buffer_size
        self.conv_buffer: list[tuple[list[int], list[int]]] = []
        self.cursor = rank()
        self.consumed = rank()
        self.epoch = 1
        self.it = 0
        self.last_step = False
        self.approx_progress = 0.0

    def refill(self) -> None:
        dataset_size = len(self.dataset)
        while len(self.conv_buffer) < self.buffer_size:
            conv = self.dataset[self.cursor]
            ids, mask = render_conversation(self.tokenizer, conv, max_tokens=self.args.seq_len + 1)
            if len(ids) >= 2:
                self.conv_buffer.append((ids, mask))
            self.cursor += world_size()
            if self.cursor >= dataset_size:
                self.cursor %= dataset_size
                self.epoch += 1

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        row_capacity = self.args.seq_len + 1
        rows: list[list[int]] = []
        masks: list[list[int]] = []

        for _ in range(self.args.batch_size):
            row: list[int] = []
            mask_row: list[int] = []
            while len(row) < row_capacity:
                self.refill()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv_ids, _) in enumerate(self.conv_buffer):
                    if len(conv_ids) <= remaining and len(conv_ids) > best_len:
                        best_idx = i
                        best_len = len(conv_ids)
                if best_idx < 0:
                    row.extend([self.tokenizer.bos_token_id] * remaining)
                    mask_row.extend([0] * remaining)
                    break
                conv_ids, conv_mask = self.conv_buffer.pop(best_idx)
                row.extend(conv_ids)
                mask_row.extend(conv_mask)
                self.consumed += world_size()
            rows.append(row[:row_capacity])
            masks.append(mask_row[:row_capacity])

        self.it += 1
        if self.split == "train":
            if self.args.max_steps > 0:
                self.approx_progress = self.it / self.args.max_steps
                if self.it >= self.args.max_steps:
                    self.last_step = True
            else:
                self.approx_progress = self.consumed / max(1, len(self.dataset))
                if self.consumed >= len(self.dataset):
                    self.last_step = True

        use_cuda = self.runtime.device.type == "cuda"
        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, pin_memory=use_cuda)
        x = batch[:, :-1].to(self.runtime.device, non_blocking=use_cuda).contiguous()
        y = batch[:, 1:].to(self.runtime.device, non_blocking=use_cuda).contiguous()
        target_mask = mask_tensor[:, 1:].to(self.runtime.device, non_blocking=use_cuda)
        y = y.masked_fill(~target_mask, IGNORE_INDEX)
        return x, y


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_model_and_tokenizer(args: argparse.Namespace, runtime: Runtime) -> tuple[torch.nn.Module, NanochatTokenizer, dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location=runtime.device, weights_only=False)
    config = TextDiffusionConfig(**checkpoint["config"])
    if args.seq_len is None:
        args.seq_len = config.max_seq_len
    if args.seq_len > config.max_seq_len:
        raise ValueError(f"--seq-len {args.seq_len} exceeds checkpoint max_seq_len {config.max_seq_len}")

    tokenizer_dir = args.checkpoint.parent / "tokenizer_hf"
    tokenizer = NanochatTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TextDiffusionModel(config).to(runtime.device)
    model.load_state_dict(clean_state_dict(checkpoint["model_state"]), strict=True)
    if not args.train_mtp_heads:
        for param in model.mtp_heads.parameters():
            param.requires_grad = False
        log("frozen MTP heads for chat SFT")
    model = apply_fp8_training(model, args, runtime)
    if args.compile:
        log("compiling model with torch.compile(dynamic=False)")
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    if is_dist() and args.optimizer == "muon":
        log("distributed model: replicated parameters with DistMuonAdamW gradient sync")
        return model, tokenizer, checkpoint
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    return model, tokenizer, checkpoint


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, runtime: Runtime) -> torch.optim.Optimizer:
    source = unwrap_model(model)
    trainable = lambda params: [p for p in params if p.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable(source.parameters()),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=runtime.device.type == "cuda",
        )

    model_dim = source.config.d_model
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    embedding_params = trainable(source.token_emb.parameters())
    lm_head_params = trainable(source.lm_head.parameters())
    matrix_params = trainable(source.blocks.parameters())
    if args.train_mtp_heads:
        matrix_params += trainable(source.mtp_heads.parameters())
    seen = {id(p) for p in embedding_params + lm_head_params + matrix_params}
    scalar_params = [p for p in source.parameters() if p.requires_grad and id(p) not in seen]

    groups: list[dict[str, Any]] = [
        {
            "kind": "adamw",
            "params": lm_head_params,
            "lr": args.unembedding_lr * dmodel_lr_scale,
            "lr_multiplier": args.unembedding_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.96),
            "eps": 1e-10,
            "weight_decay": 0.01,
        },
        {
            "kind": "adamw",
            "params": embedding_params,
            "lr": args.embedding_lr * dmodel_lr_scale,
            "lr_multiplier": args.embedding_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.995),
            "eps": 1e-10,
            "weight_decay": 0.001,
        },
    ]
    if scalar_params:
        groups.append(
            {
                "kind": "adamw",
                "params": scalar_params,
                "lr": args.scalar_lr,
                "lr_multiplier": args.scalar_lr / args.lr,
                "betas": (0.8, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            }
        )
    for shape in sorted({p.shape for p in matrix_params}):
        shape_params = [p for p in matrix_params if p.shape == shape]
        groups.append(
            {
                "kind": "muon",
                "params": shape_params,
                "lr": args.matrix_lr,
                "lr_multiplier": args.matrix_lr / args.lr,
                "momentum": args.muon_momentum,
                "weight_decay": args.weight_decay,
                "ns_steps": args.muon_ns_steps,
                "beta2": 0.9,
            }
        )
    optimizer_cls = DistMuonAdamW if is_dist() else MuonAdamW
    return optimizer_cls(groups)


def lr_multiplier(progress: float, args: argparse.Namespace) -> float:
    progress = max(0.0, min(1.0, progress))
    if args.warmup_ratio > 0 and progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if args.warmdown_ratio <= 0 or progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1.0 - decay) + decay * args.final_lr_frac


def sft_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    logits = model(x, attention_mask=None, causal=True)
    flat_y = y.reshape(-1)
    valid = int((flat_y != IGNORE_INDEX).sum().item())
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        flat_y,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum / max(1, valid), valid


@torch.no_grad()
def evaluate_sft(model: torch.nn.Module, batcher: SFTBatcher, args: argparse.Namespace, runtime: Runtime) -> dict[str, float]:
    batcher.cursor = rank()
    batcher.consumed = rank()
    batcher.conv_buffer.clear()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = max(1, args.eval_tokens // (args.batch_size * args.seq_len * world_size()))
    for _ in range(steps):
        x, y = batcher.next()
        with autocast_context(runtime.device, args.amp_dtype):
            with disable_fp8(unwrap_model(model)):
                loss, valid = sft_loss(unwrap_model(model), x, y)
        total_loss += float(loss.item()) * valid
        total_tokens += valid
    totals = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=runtime.device)
    if is_dist():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    token_count = max(1.0, float(totals[1].item()))
    return {"loss": float(totals[0].item()) / token_count, "tokens": token_count}


@torch.no_grad()
def evaluate_core_metric(model: torch.nn.Module, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime) -> dict[str, float]:
    if args.core_every <= 0:
        return {}
    model.eval()
    source = unwrap_model(model)
    if is_dist():
        if is_main_process():
            ensure_eval_bundle(args.core_eval_cache_dir)
        dist.barrier()
    with disable_fp8(source):
        results = evaluate_core(
            source,
            tokenizer,
            runtime.device,
            cache_dir=args.core_eval_cache_dir,
            max_per_task=args.core_eval_max_per_task,
        )
    model.train()
    return {"core": float(results["core_metric"])}


@torch.no_grad()
def sample_text(model: torch.nn.Module, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime) -> str:
    source = unwrap_model(model)
    source.eval()
    samples = []
    user_start = special_id(tokenizer, "<|user_start|>")
    user_end = special_id(tokenizer, "<|user_end|>")
    assistant_start = special_id(tokenizer, "<|assistant_start|>")
    with disable_fp8(source):
        for prompt in SAMPLE_USER_PROMPTS:
            prompt_ids_list = [tokenizer.bos_token_id, user_start]
            prompt_ids_list.extend(tokenizer.encode(prompt))
            prompt_ids_list.extend([user_end, assistant_start])
            prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=runtime.device)
            output = generate_causal(
                source,
                prompt_ids,
                gen_length=args.sample_length,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                top_p=args.sample_top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.decode(output.detach().cpu().tolist(), skip_special=False)
            samples.append(f"USER:\n{prompt}\n\nMODEL:\n{decoded}")
    sample = "\n\n---\n\n".join(samples)
    log("samples:\n" + sample)
    source.train()
    return sample


def log_wandb(wandb_run: Any, metrics: dict[str, float], step: int) -> None:
    if wandb_run is not None and is_main_process():
        wandb_run.log(metrics, step=step)


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    model, tokenizer, checkpoint = load_model_and_tokenizer(args, runtime)
    train_dataset, val_dataset = build_datasets(args)
    tokens_per_micro = args.batch_size * args.seq_len * world_size()
    if args.total_batch_size % tokens_per_micro != 0:
        raise ValueError(f"--total-batch-size must be divisible by {tokens_per_micro:,}")
    args.grad_accum_steps = args.total_batch_size // tokens_per_micro
    args.tokens_per_step = args.total_batch_size
    args.wandb_project = args.wandb_project

    optimizer = build_optimizer(args, model, runtime)
    for group in optimizer.param_groups:
        group["lr"] *= args.init_lr_frac
        group["initial_lr"] = group["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=(runtime.device.type == "cuda" and args.amp_dtype == "float16"))
    wandb_run = init_wandb(args, unwrap_model(model).config, runtime)

    train_batcher = SFTBatcher(train_dataset, tokenizer, args, runtime, "train")
    val_batcher = SFTBatcher(val_dataset, tokenizer, args, runtime, "val")
    log(f"loaded checkpoint: {args.checkpoint} step={checkpoint.get('step')}")
    log(f"train conversations: {len(train_dataset):,}; val conversations: {len(val_dataset):,}")
    log(f"tokens/micro={tokens_per_micro:,}; grad_accum={args.grad_accum_steps}; tokens/step={args.total_batch_size:,}")

    model.train()
    smooth_loss = 0.0
    total_time = 0.0
    step = 0
    while True:
        if train_batcher.last_step:
            break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        if args.eval_every > 0 and (step == 0 or step % args.eval_every == 0):
            metrics = evaluate_sft(model, val_batcher, args, runtime)
            log(f"step {step:05d} val_loss {metrics['loss']:.4f}")
            log_wandb(wandb_run, {"eval/loss": metrics["loss"], "eval/tokens": metrics["tokens"]}, step)

        if args.core_every > 0 and step > 0 and step % args.core_every == 0:
            core_metrics = evaluate_core_metric(model, tokenizer, args, runtime)
            if core_metrics:
                log(f"step {step:05d} core {core_metrics['core']:.4f}")
                log_wandb(wandb_run, {"eval/core": core_metrics["core"]}, step)

        progress = (step / args.max_steps) if args.max_steps > 0 else train_batcher.approx_progress
        lr = args.lr * lr_multiplier(progress, args)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        train_loss = 0.0
        valid_tokens = 0

        for micro_step in range(args.grad_accum_steps):
            x, y = train_batcher.next()
            sync_context = model.no_sync() if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1 else nullcontext()
            with sync_context:
                with autocast_context(runtime.device, args.amp_dtype):
                    loss, valid = sft_loss(model, x, y)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            train_loss += float(loss.detach().item()) * valid
            valid_tokens += valid

        if is_dist():
            last_step_tensor = torch.tensor(int(train_batcher.last_step), dtype=torch.int32, device=runtime.device)
            dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
            train_batcher.last_step = bool(last_step_tensor.item())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in unwrap_model(model).parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        elapsed = time.time() - start
        total_time += elapsed
        step += 1
        train_loss /= max(1, valid_tokens)
        smooth_loss = train_loss if step == 1 else 0.9 * smooth_loss + 0.1 * train_loss
        log(f"step {step:05d} train_loss {train_loss:.4f} smooth {smooth_loss:.4f} lr {lr:.2e} tok/s {valid_tokens / max(elapsed, 1e-9):,.0f}")
        log_wandb(wandb_run, {"train/loss": train_loss, "train/smooth_loss": smooth_loss, "train/lr": lr, "train/tokens": step * args.total_batch_size}, step)

        if args.sample_every > 0 and step % args.sample_every == 0 and is_main_process():
            sample = sample_text(model, tokenizer, args, runtime)
            if wandb_run is not None:
                import wandb

                wandb_run.log({"sample/text": wandb.Html(f"<pre>{sample}</pre>")}, step=step)

        if is_main_process() and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=step, args=args)
            log(f"saved checkpoint: {args.out_dir / 'checkpoint.pt'}")

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

    if is_main_process():
        save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=step, args=args)
        log(f"saved final checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = step
            wandb_run.summary["total_training_time"] = total_time
            wandb_run.finish()


def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        train(args, runtime)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
