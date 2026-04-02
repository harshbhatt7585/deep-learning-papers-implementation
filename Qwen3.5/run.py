from __future__ import annotations

import argparse

import torch
from transformers import AutoProcessor, AutoTokenizer

from config import Qwen35Config
from generate import greedy_generate
from loader import load_config_dict, load_weights, resolve_model_path
from model import Qwen35ForCausalLM


DOWNLOAD_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "processor_config.json",
    "*.txt",
]


def load_tokenizer(model_dir):
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        if hasattr(processor, "tokenizer"):
            return processor.tokenizer
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tiny-random/qwen3.5")
    parser.add_argument("--prompt", default="Hello from Qwen3.5")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_path(args.model_id, allow_patterns=DOWNLOAD_PATTERNS)
    config = Qwen35Config.from_dict(load_config_dict(model_dir))
    model = Qwen35ForCausalLM(config).to(args.device)

    missing, unexpected = load_weights(model, model_dir)
    print("missing_keys", len(missing))
    print("unexpected_keys", len(unexpected))
    if missing:
        print("first_missing", missing[:10])
    if unexpected:
        print("first_unexpected", unexpected[:10])

    tokenizer = load_tokenizer(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    generated = greedy_generate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
