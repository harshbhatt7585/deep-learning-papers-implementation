from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any

import torch.distributed as dist

from tinygroot.utils import is_dist, log, rank


WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"


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


def build_datasets(args) -> tuple[TaskMixture, TaskMixture]:
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
