from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import json
import random
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from chat_core_eval import (
    execute_code,
    extract_gsm_answer,
    extract_imports,
    extract_program,
    use_calculator,
)
from fp8 import disable_fp8
from model import TextDiffusionConfig, TextDiffusionModel, _sample_tokens, norm
from nanochat_optim import DistMuonAdamW, MuonAdamW
from tokenizer import NanochatTokenizer
from train import apply_fp8_training, fp8_module_filter
from utils import (
    Runtime,
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
    parser.add_argument("--wandb-project", type=str, default="tinyGroot-sft")
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
    parser.add_argument("--chatcore-every", type=int, default=200, help="-1 disables ChatCORE eval")
    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task (-1 = all)")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task")
    parser.add_argument("--chatcore-max-new-tokens", type=int, default=512)
    parser.add_argument("--chatcore-temperature", type=float, default=0.0)
    parser.add_argument("--chatcore-top-k", type=int, default=50)
    parser.add_argument("--chatcore-batch-size", type=int, default=8, help="batch size for categorical ChatCORE eval")
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
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
    parser.add_argument("--train-mtp-heads", action=argparse.BooleanOptionalAction, default=True, help="Train MTP heads with an assistant-token auxiliary loss")
    parser.add_argument("--mtp-loss-weight", type=float, default=0.15)
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


class HuggingFaceTask(Task):
    def __init__(self, path: str, *dataset_args: str, split: str, stop: int | None = None) -> None:
        from datasets import load_dataset

        self.ds = load_dataset(path, *dataset_args, split=split).shuffle(seed=42)
        self.stop = stop

    def __len__(self) -> int:
        return min(len(self.ds), self.stop) if self.stop is not None else len(self.ds)


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


class SmolTalk(HuggingFaceTask):
    def __init__(self, split: str) -> None:
        super().__init__("HuggingFaceTB/smol-smoltalk", split=split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"messages": self.ds[index]["messages"]}


def render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


class MMLU(HuggingFaceTask):
    letters = ("A", "B", "C", "D")

    def __init__(self, split: str, stop: int | None = None) -> None:
        super().__init__("cais/mmlu", "all", split=split, stop=stop)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": render_mc(row["question"], self.letters, row["choices"])},
                {"role": "assistant", "content": self.letters[row["answer"]]},
            ],
            "letters": self.letters,
        }


class ARC(HuggingFaceTask):
    def __init__(self, subset: str, split: str, stop: int | None = None) -> None:
        assert subset in ("ARC-Easy", "ARC-Challenge")
        super().__init__("allenai/ai2_arc", subset, split=split, stop=stop)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        letters = tuple(row["choices"]["label"])
        return {
            "messages": [
                {"role": "user", "content": render_mc(row["question"], letters, row["choices"]["text"])},
                {"role": "assistant", "content": row["answerKey"]},
            ],
            "letters": letters,
        }


class HumanEval(HuggingFaceTask):
    def __init__(self, stop: int | None = None) -> None:
        super().__init__("openai/openai_humaneval", split="test", stop=stop)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": f"{row['prompt']}\n{row['canonical_solution']}"},
            ],
            "entry_point": row["entry_point"],
            "test": row["test"],
        }


class GSM8K(HuggingFaceTask):
    def __init__(self, split: str, stop: int | None = None) -> None:
        super().__init__("openai/gsm8k", "main", split=split, stop=stop)

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
    elif len(model.mtp_heads) > 0:
        log(f"training MTP heads for chat SFT: heads={len(model.mtp_heads)} loss_weight={args.mtp_loss_weight:g}")
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


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_pass_counts(num_passed: int, total: int, device: torch.device) -> tuple[int, int]:
    counts = torch.tensor([num_passed, total], dtype=torch.long, device=device)
    all_reduce_sum(counts)
    return int(counts[0].item()), int(counts[1].item())


@dataclass
class LossAccumulator:
    total: torch.Tensor
    main: torch.Tensor
    mtp: torch.Tensor
    tokens: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device, dtype: torch.dtype) -> "LossAccumulator":
        zero = torch.zeros((), device=device, dtype=dtype)
        return cls(total=zero.clone(), main=zero.clone(), mtp=zero.clone(), tokens=zero.clone())

    def add(
        self,
        total_loss: torch.Tensor,
        main_loss: torch.Tensor,
        mtp_loss: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        valid = valid_tokens.to(self.tokens.dtype)
        self.total += total_loss.detach().to(self.total.dtype) * valid
        self.main += main_loss.detach().to(self.main.dtype) * valid
        self.mtp += mtp_loss.detach().to(self.mtp.dtype) * valid
        self.tokens += valid

    def values(self, *, reduce: bool = False) -> tuple[float, float, float, int]:
        packed = torch.stack([self.total, self.main, self.mtp, self.tokens])
        if reduce:
            all_reduce_sum(packed)
        total, main, mtp, tokens = packed.tolist()
        return total, main, mtp, int(tokens)


def sft_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss_for_backward, main_loss_detached, mtp_loss_detached, valid_tokens_tensor).

    All returned tensors stay on-device; the caller can defer
    a single sync to log time by accumulating these into tensor buffers.
    """
    source = unwrap_model(model)
    use_mtp = args.train_mtp_heads and len(source.mtp_heads) > 0
    if use_mtp:
        logits, hidden = model(x, attention_mask=None, causal=True, return_hidden=True)
    else:
        logits = model(x, attention_mask=None, causal=True)
        hidden = None
    flat_y = y.reshape(-1)
    valid = (flat_y != IGNORE_INDEX).sum()
    main_loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        flat_y,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    main_loss = main_loss_sum / valid.clamp(min=1)

    mtp_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)
    if use_mtp and hidden is not None:
        mtp_hidden: torch.Tensor | None = None
        aux_terms: list[torch.Tensor] = []
        for depth, head in enumerate(source.mtp_heads, start=1):
            target_shift = depth
            if y.size(1) <= target_shift:
                break
            target = y[:, target_shift:]
            if source.config.mtp_arch == "deepseek":
                if depth == 1:
                    previous_hidden = hidden[:, :-target_shift, :]
                else:
                    if mtp_hidden is None:
                        break
                    previous_hidden = mtp_hidden[:, :-1, :]
                token_ids = x[:, depth:]
                cos_sin = (
                    source.cos[:, : previous_hidden.size(1)],
                    source.sin[:, : previous_hidden.size(1)],
                )
                mtp_hidden = head(
                    previous_hidden,
                    token_ids,
                    token_emb=source.token_emb,
                    cos_sin=cos_sin,
                )
                h_offset = mtp_hidden
            else:
                h_offset = norm(head(hidden[:, :-target_shift, :]))
            aux_logits = source.lm_head(h_offset)
            target_valid = (target != IGNORE_INDEX).sum()
            mtp_loss_sum = F.cross_entropy(
                aux_logits.reshape(-1, aux_logits.size(-1)),
                target.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            aux_terms.append(mtp_loss_sum / target_valid.clamp(min=1))
        if aux_terms:
            mtp_loss = torch.stack(aux_terms).mean()

    total = main_loss + args.mtp_loss_weight * mtp_loss
    return total, main_loss.detach(), mtp_loss.detach(), valid


@torch.no_grad()
def evaluate_sft(model: torch.nn.Module, batcher: SFTBatcher, args: argparse.Namespace, runtime: Runtime) -> dict[str, float]:
    batcher.cursor = rank()
    batcher.consumed = rank()
    batcher.conv_buffer.clear()
    model.eval()
    losses = LossAccumulator.empty(runtime.device, torch.float64)
    steps = max(1, args.eval_tokens // (args.batch_size * args.seq_len * world_size()))
    for _ in range(steps):
        x, y = batcher.next()
        with autocast_context(runtime.device, args.amp_dtype):
            with disable_fp8(unwrap_model(model)):
                total_loss, main_loss, mtp_loss, valid = sft_loss(unwrap_model(model), x, y, args)
        losses.add(total_loss, main_loss, mtp_loss, valid)
    model.train()
    _, main_sum, mtp_sum, tokens = losses.values(reduce=True)
    denom = max(1.0, float(tokens))
    return {
        "loss": main_sum / denom,
        "mtp_loss": mtp_sum / denom,
        "tokens": tokens,
    }


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
    """Causal generation that stops on <|assistant_end|>/<|bos|> and runs
    use_calculator on <|python_start|>...<|python_end|> spans, force-inserting
    the result inside <|output_start|>...<|output_end|>."""
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
    num_passed, total = reduce_pass_counts(num_passed, total, runtime.device)
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
    num_passed, total = reduce_pass_counts(num_passed, total, runtime.device)
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


def _build_chatcore_task(name: str, args: argparse.Namespace) -> Task:
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
def evaluate_chatcore(model: torch.nn.Module, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime) -> dict[str, float]:
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
def sample_text(model: torch.nn.Module, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime) -> str:
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
            log(
                f"step {step:05d} val_loss {metrics['loss']:.4f} "
                f"mtp_loss {metrics['mtp_loss']:.4f}"
            )
            log_wandb(
                wandb_run,
                {
                    "eval/loss": metrics["loss"],
                    "eval/mtp_loss": metrics["mtp_loss"],
                    "eval/tokens": metrics["tokens"],
                },
                step,
            )

        if args.chatcore_every > 0 and step > 0 and step % args.chatcore_every == 0:
            chatcore_metrics = evaluate_chatcore(model, tokenizer, args, runtime)
            if chatcore_metrics:
                log(
                    f"step {step:05d} chatcore {chatcore_metrics['chatcore']:.4f} "
                    f"cat {chatcore_metrics['chatcore_cat']:.4f}"
                )
                log_wandb(wandb_run, {f"eval/{k}": v for k, v in chatcore_metrics.items()}, step)

        progress = (step / args.max_steps) if args.max_steps > 0 else train_batcher.approx_progress
        lr = args.lr * lr_multiplier(progress, args)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        losses = LossAccumulator.empty(runtime.device, torch.float32)

        for micro_step in range(args.grad_accum_steps):
            x, y = train_batcher.next()
            sync_context = model.no_sync() if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1 else nullcontext()
            with sync_context:
                with autocast_context(runtime.device, args.amp_dtype):
                    loss, main_loss, mtp_loss, valid = sft_loss(model, x, y, args)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            losses.add(loss, main_loss, mtp_loss, valid)

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
        total_loss_sum, main_loss_sum_v, mtp_loss_sum_v, valid_tokens = losses.values()
        denom = max(1, valid_tokens)
        train_loss = total_loss_sum / denom
        main_loss_val = main_loss_sum_v / denom
        mtp_loss_val = mtp_loss_sum_v / denom
        smooth_loss = main_loss_val if step == 1 else 0.9 * smooth_loss + 0.1 * main_loss_val
        log(
            f"step {step:05d} train_loss {train_loss:.4f} main {main_loss_val:.4f} "
            f"mtp {mtp_loss_val:.4f} smooth {smooth_loss:.4f} lr {lr:.2e} "
            f"tok/s {valid_tokens / max(elapsed, 1e-9):,.0f}"
        )
        log_wandb(
            wandb_run,
            {
                "train/loss": train_loss,
                "train/main_loss": main_loss_val,
                "train/mtp_loss": mtp_loss_val,
                "train/smooth_loss": smooth_loss,
                "train/lr": lr,
                "train/tokens": step * args.total_batch_size,
            },
            step,
        )

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
