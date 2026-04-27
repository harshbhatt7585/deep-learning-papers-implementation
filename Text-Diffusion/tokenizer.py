from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

LLADA21_TOKENIZER_NAME = "inclusionAI/LLaDA2.1-mini"


class SimpleCharTokenizer:
    """Tiny character tokenizer for learning the algorithm without extra deps."""

    pad_token = "<pad>"
    mask_token = "<mask>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, chars: Iterable[str]) -> None:
        special_tokens = [
            self.pad_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        vocab = special_tokens + sorted(set(chars))
        self.id_to_token = list(dict.fromkeys(vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SimpleCharTokenizer":
        chars = set()
        for text in texts:
            chars.update(text)
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.mask_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        pieces = []
        special = {
            self.pad_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        }
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"id_to_token": self.id_to_token}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SimpleCharTokenizer":
        path = Path(path)
        data = json.loads(path.read_text())
        tokenizer = cls(chars=[])
        tokenizer.id_to_token = data["id_to_token"]
        tokenizer.token_to_id = {
            token: idx for idx, token in enumerate(tokenizer.id_to_token)
        }
        return tokenizer


def pad_sequences(sequences: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


class LLaDA21Tokenizer:
    """Small wrapper around LLaDA2.1-mini's Hugging Face tokenizer."""

    model_name = LLADA21_TOKENIZER_NAME

    def __init__(self, hf_tokenizer) -> None:
        self.hf_tokenizer = hf_tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = LLADA21_TOKENIZER_NAME,
        *,
        local_files_only: bool = False,
    ) -> "LLaDA21Tokenizer":
        from huggingface_hub import snapshot_download
        from transformers import PreTrainedTokenizerFast

        model_path = snapshot_download(
            model_name,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            local_files_only=local_files_only,
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return max(self.hf_tokenizer.get_vocab().values()) + 1

    @property
    def pad_token_id(self) -> int:
        return int(self.hf_tokenizer.pad_token_id)

    @property
    def mask_token_id(self) -> int:
        return int(self.hf_tokenizer.mask_token_id)

    @property
    def bos_token_id(self) -> int:
        return int(self.hf_tokenizer.bos_token_id)

    @property
    def eos_token_id(self) -> int:
        return int(self.hf_tokenizer.eos_token_id)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self.hf_tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.hf_tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path) -> "LLaDA21Tokenizer":
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            path,
            trust_remote_code=True,
        )
        return cls(tokenizer)
