from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tokenizer import NanochatTokenizer
from utils import NANOCHAT_MAX_SHARD, download_nanochat_shard, read_parquet_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize text into mmap-friendly token shards.")
    parser.add_argument("--data", type=Path, default=None, help="Optional local text file for smoke tests.")
    parser.add_argument("--nanochat-cache-dir", type=Path, default=Path("data/nanochat_climbmix"))
    parser.add_argument("--nanochat-tokenizer-cache-dir", type=Path, default=Path("data/nanochat_tokenizer_32k"))
    parser.add_argument("--token-shards-dir", type=Path, default=Path("data/nanochat_tokens_32k"))
    parser.add_argument("--train-shards", type=int, default=170)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--max-val-chars", type=int, default=1_000_000)
    parser.add_argument("--tokenizer-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--doc-batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_or_train_tokenizer(tokenizer_dir: Path, train_text: str) -> NanochatTokenizer:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"loading tokenizer: {tokenizer_dir}", flush=True)
        return NanochatTokenizer.load(tokenizer_dir)

    print(f"training tokenizer: {tokenizer_dir}", flush=True)
    return NanochatTokenizer.from_pretrained(tokenizer_dir, train_text=train_text)


def configure_tokenizer_parallelism(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.tokenizer_threads))
    print(
        "tokenizer parallelism: "
        f"TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']} "
        f"RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']} "
        f"doc_batch_size={args.doc_batch_size}",
        flush=True,
    )


def text_documents(text: str):
    start = 0
    while start < len(text):
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        line = text[start:end]
        if line.endswith("\r"):
            line = line[:-1]
        if line:
            yield line
        start = end + 1


def encode_text(tokenizer: NanochatTokenizer, text: str, args: argparse.Namespace) -> list[int]:
    ids = tokenizer.encode_documents(
        text_documents(text),
        add_bos=True,
        add_eos=False,
        batch_size=args.doc_batch_size,
    )
    if ids:
        return ids
    return tokenizer.encode(text, add_eos=True)


def token_dtype(ids: list[int]) -> np.dtype:
    max_id = max(ids) if ids else 0
    if max_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


def write_tokens(path: Path, ids: list[int], *, overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        array = np.load(path, mmap_mode="r")
        print(f"using cached token shard {path.name}: {array.shape[0]:,} tokens", flush=True)
        return {"file": path.name, "tokens": int(array.shape[0]), "dtype": str(array.dtype)}

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(ids, dtype=token_dtype(ids))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, array)
    tmp_path.replace(path)
    print(f"wrote token shard {path.name}: {array.shape[0]:,} tokens", flush=True)
    return {"file": path.name, "tokens": int(array.shape[0]), "dtype": str(array.dtype)}


def pretokenize_local_file(args: argparse.Namespace) -> None:
    text = args.data.read_text()
    if args.max_train_chars is not None:
        text = text[: args.max_train_chars]
    tokenizer = load_or_train_tokenizer(args.nanochat_tokenizer_cache_dir, text)
    ids = encode_text(tokenizer, text, args)
    split = int(0.95 * len(ids))

    train_entry = write_tokens(
        args.token_shards_dir / "train_00000.npy",
        ids[:split],
        overwrite=args.overwrite,
    )
    val_entry = write_tokens(
        args.token_shards_dir / "val.npy",
        ids[split:],
        overwrite=args.overwrite,
    )
    metadata = {
        "source": str(args.data),
        "tokenizer": "nanochat",
        "tokenizer_dir": str(args.nanochat_tokenizer_cache_dir),
        "train_chars": int(0.95 * len(text)),
        "val_chars": len(text) - int(0.95 * len(text)),
        "train_tokens": train_entry["tokens"],
        "val_tokens": val_entry["tokens"],
        "train_files": [train_entry],
        "val_file": val_entry,
        "document_bos": True,
        "tokenizer_threads": args.tokenizer_threads,
        "doc_batch_size": args.doc_batch_size,
    }
    write_metadata(args.token_shards_dir, metadata)


def pretokenize_nanochat(args: argparse.Namespace) -> None:
    args.token_shards_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.token_shards_dir / "metadata.json"
    old_metadata = {}
    if metadata_path.exists() and not args.overwrite:
        old_metadata = json.loads(metadata_path.read_text())
    old_train_entries = {
        entry["file"]: entry
        for entry in old_metadata.get("train_files", [])
        if isinstance(entry, dict) and "file" in entry
    }
    tokenizer: NanochatTokenizer | None = None
    train_entries = []
    train_chars = 0
    remaining_chars = args.max_train_chars

    for shard_index in range(args.train_shards):
        out_path = args.token_shards_dir / f"train_{shard_index:05d}.npy"
        if out_path.exists() and not args.overwrite:
            entry = write_tokens(out_path, [], overwrite=False)
            entry.update(old_train_entries.get(out_path.name, {}))
            train_entries.append(entry)
            shard_chars = entry.get("chars")
            if shard_chars is not None:
                train_chars += int(shard_chars)
                if remaining_chars is not None:
                    remaining_chars -= int(shard_chars)
                    if remaining_chars <= 0:
                        break
            continue

        if remaining_chars is not None and remaining_chars <= 0:
            break

        parquet_path = download_nanochat_shard(shard_index, args.nanochat_cache_dir)
        print(f"reading train shard {shard_index + 1}/{args.train_shards}: {parquet_path.name}", flush=True)
        text = read_parquet_text(parquet_path, max_chars=remaining_chars)
        if tokenizer is None:
            tokenizer = load_or_train_tokenizer(args.nanochat_tokenizer_cache_dir, text)
        print(f"tokenizing train shard {shard_index + 1}/{args.train_shards}", flush=True)
        ids = encode_text(tokenizer, text, args)
        entry = write_tokens(out_path, ids, overwrite=args.overwrite)
        entry["chars"] = len(text)
        train_entries.append(entry)
        train_chars += len(text)

        if remaining_chars is not None:
            remaining_chars -= len(text)

    if tokenizer is None:
        tokenizer = NanochatTokenizer.load(args.nanochat_tokenizer_cache_dir)

    val_path = args.token_shards_dir / "val.npy"
    val_chars = None
    if val_path.exists() and not args.overwrite:
        val_entry = write_tokens(val_path, [], overwrite=False)
        val_entry.update(old_metadata.get("val_file", {}))
        val_chars = val_entry.get("chars")
    else:
        parquet_path = download_nanochat_shard(NANOCHAT_MAX_SHARD, args.nanochat_cache_dir)
        print(f"reading val shard: {parquet_path.name}", flush=True)
        val_text = read_parquet_text(parquet_path, max_chars=args.max_val_chars)
        val_chars = len(val_text)
        print("tokenizing val shard", flush=True)
        val_ids = encode_text(tokenizer, val_text, args)
        val_entry = write_tokens(val_path, val_ids, overwrite=args.overwrite)
        val_entry["chars"] = val_chars

    metadata = {
        "source": "nanochat/climbmix-400b-shuffle",
        "tokenizer": "nanochat",
        "tokenizer_dir": str(args.nanochat_tokenizer_cache_dir),
        "train_shards_requested": args.train_shards,
        "max_train_chars": args.max_train_chars,
        "max_val_chars": args.max_val_chars,
        "train_chars": train_chars or None,
        "val_chars": val_chars,
        "train_tokens": sum(entry["tokens"] for entry in train_entries),
        "val_tokens": val_entry["tokens"],
        "train_files": train_entries,
        "val_file": val_entry,
        "document_bos": True,
        "tokenizer_threads": args.tokenizer_threads,
        "doc_batch_size": args.doc_batch_size,
    }
    write_metadata(args.token_shards_dir, metadata)


def write_metadata(token_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = token_dir / "metadata.json"
    tmp_path = metadata_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(metadata_path)
    print(f"wrote metadata: {metadata_path}", flush=True)


def main() -> None:
    args = parse_args()
    configure_tokenizer_parallelism(args)
    if args.data is not None:
        pretokenize_local_file(args)
    else:
        pretokenize_nanochat(args)


if __name__ == "__main__":
    main()
