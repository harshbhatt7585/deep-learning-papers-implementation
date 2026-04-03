from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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


def compare_cache_step(model, inputs):
    with torch.no_grad():
        logits, cache = model(**inputs)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        extended_ids = torch.cat([inputs["input_ids"], next_token], dim=1)
        extended_mask = torch.cat(
            [inputs["attention_mask"], torch.ones_like(next_token, dtype=inputs["attention_mask"].dtype)],
            dim=1,
        )
        cached_logits, _ = model(input_ids=next_token, attention_mask=extended_mask, past_key_values=cache)
        full_logits, _ = model(input_ids=extended_ids, attention_mask=extended_mask)
    return float((cached_logits[:, -1, :] - full_logits[:, -1, :]).abs().max())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tiny-random/qwen3.5")
    parser.add_argument("--prompt", default="Hello from Qwen3.5")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--logit-threshold", type=float, default=0.05)
    parser.add_argument("--cache-threshold", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_path(args.model_id, allow_patterns=DOWNLOAD_PATTERNS)
    tokenizer = load_tokenizer(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}

    config = Qwen35Config.from_dict(load_config_dict(model_dir))
    ours = Qwen35ForCausalLM(config).to(args.device)
    missing, unexpected = load_weights(ours, model_dir)
    if missing or unexpected:
        raise RuntimeError(f"load mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    ours.eval()

    ref = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(args.device, dtype=torch.float32)
    ref.eval()

    with torch.no_grad():
        ours_logits, _ = ours(**inputs)
        ref_logits = ref(**inputs).logits.float()

    logit_max_abs = float((ours_logits - ref_logits).abs().max())
    logit_mean_abs = float((ours_logits - ref_logits).abs().mean())
    cache_max_abs = compare_cache_step(ours, inputs)

    ours_generated = greedy_generate(ours, inputs["input_ids"], inputs["attention_mask"], args.max_new_tokens)
    ref_generated = greedy_generate(ref, inputs["input_ids"], inputs["attention_mask"], args.max_new_tokens)

    print(
        {
            "logit_max_abs": logit_max_abs,
            "logit_mean_abs": logit_mean_abs,
            "cache_max_abs": cache_max_abs,
            "ours_text": tokenizer.decode(ours_generated[0], skip_special_tokens=False),
            "ref_text": tokenizer.decode(ref_generated[0], skip_special_tokens=False),
        }
    )

    if logit_max_abs > args.logit_threshold:
        raise SystemExit(f"logit_max_abs {logit_max_abs:.6f} exceeded threshold {args.logit_threshold:.6f}")
    if cache_max_abs > args.cache_threshold:
        raise SystemExit(f"cache_max_abs {cache_max_abs:.6f} exceeded threshold {args.cache_threshold:.6f}")


if __name__ == "__main__":
    main()
