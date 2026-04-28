from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

LLADA21_TOKENIZER_NAME = "inclusionAI/LLaDA2.1-mini"


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
        tokenizer.model_max_length = int(1e12)
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
        tokenizer.model_max_length = int(1e12)
        return cls(tokenizer)
