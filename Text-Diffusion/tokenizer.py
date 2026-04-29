from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NANOCHAT_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
NANOCHAT_SPECIAL_TOKENS = [
    "<|bos|>",
    "<|pad|>",
    "<|mask|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


@dataclass
class NanochatTokenizer:
    """Nanochat-style 32K byte-level BPE tokenizer with diffusion specials."""

    tokenizer: object

    tokenizer_type = "nanochat"
    bos_token = "<|bos|>"
    eos_token = "<|bos|>"
    pad_token = "<|pad|>"
    mask_token = "<|mask|>"

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: str | Path,
        *,
        train_text: str | None = None,
        vocab_size: int = 32_768,
        train_chars: int = 2_000_000_000,
        doc_cap: int = 10_000,
        local_files_only: bool = False,
    ) -> "NanochatTokenizer":
        cache_dir = Path(cache_dir)
        tokenizer_path = cache_dir / "tokenizer.json"
        if tokenizer_path.exists():
            return cls.load(cache_dir)
        if local_files_only:
            raise FileNotFoundError(f"nanochat tokenizer not found at {tokenizer_path}")
        if train_text is None:
            raise ValueError("train_text is required to train the nanochat tokenizer")

        tokenizer = cls.train_from_text(
            train_text,
            vocab_size=vocab_size,
            train_chars=train_chars,
            doc_cap=doc_cap,
        )
        tokenizer.save(cache_dir)
        return tokenizer

    @classmethod
    def train_from_text(
        cls,
        text: str,
        *,
        vocab_size: int = 32_768,
        train_chars: int = 2_000_000_000,
        doc_cap: int = 10_000,
    ) -> "NanochatTokenizer":
        from tokenizers import Regex, Tokenizer
        from tokenizers import decoders, pre_tokenizers
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=Regex(NANOCHAT_SPLIT_PATTERN),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=NANOCHAT_SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(
            cls._training_docs(text, max_chars=train_chars, doc_cap=doc_cap),
            trainer,
        )
        return cls(tokenizer)

    @staticmethod
    def _training_docs(text: str, *, max_chars: int, doc_cap: int) -> Iterable[str]:
        seen = 0
        for doc in text.splitlines():
            if seen >= max_chars:
                break
            if not doc:
                continue
            doc = doc[:doc_cap]
            remaining = max_chars - seen
            if len(doc) > remaining:
                doc = doc[:remaining]
            seen += len(doc)
            yield doc

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.token_to_id(self.pad_token))

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.token_to_id(self.mask_token))

    @property
    def bos_token_id(self) -> int:
        return int(self.tokenizer.token_to_id(self.bos_token))

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.token_to_id(self.eos_token))

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))

    @classmethod
    def load(cls, path: str | Path) -> "NanochatTokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(Path(path) / "tokenizer.json")))
