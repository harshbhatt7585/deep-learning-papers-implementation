from __future__ import annotations

from typing import Iterable

import torch


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


def pad_sequences(sequences: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch
