from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F

from tinygroot.chat_core_eval import use_calculator
from tinygroot.model import StaticKVCache, TinyGrootModel
from tinygroot.tokenizer import NanochatTokenizer


def special_id(tokenizer: NanochatTokenizer, text: str) -> int:
    token_id = tokenizer.tokenizer.token_to_id(text)
    if token_id is None:
        raise KeyError(f"Tokenizer is missing special token {text!r}")
    return int(token_id)


@torch.inference_mode()
def sample_next_token(
    logits: torch.Tensor,
    rng: torch.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k, dim=-1)
        probs = F.softmax(values.float() / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return indices.gather(1, choice)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


@dataclass
class RowState:
    current_tokens: list[int]
    forced_tokens: deque[int] = field(default_factory=deque)
    in_python_block: bool = False
    python_expr_tokens: list[int] = field(default_factory=list)
    completed: bool = False


class Engine:
    """KV-cached TinyGroot generation with nanochat-style calculator tool use."""

    def __init__(self, model: TinyGrootModel, tokenizer: NanochatTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        *,
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterable[tuple[list[int], list[int]]]:
        if not tokens or not isinstance(tokens[0], int):
            raise ValueError("tokens must be a non-empty list of ints")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        device = self.model.device
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        python_start = special_id(self.tokenizer, "<|python_start|>")
        python_end = special_id(self.tokenizer, "<|python_end|>")
        output_start = special_id(self.tokenizer, "<|output_start|>")
        output_end = special_id(self.tokenizer, "<|output_end|>")
        assistant_end = special_id(self.tokenizer, "<|assistant_end|>")
        bos = self.tokenizer.bos_token_id

        config = self.model.config
        head_dim = config.d_model // config.n_heads
        prompt_len = len(tokens)
        horizon = max_tokens if max_tokens is not None else config.max_seq_len - prompt_len
        cache_len = min(config.max_seq_len, prompt_len + max(0, horizon))
        cache = StaticKVCache(
            n_layers=config.n_layers,
            batch_size=num_samples,
            max_len=cache_len,
            n_kv_heads=config.n_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=next(self.model.parameters()).dtype,
        )

        # Prefill once with the prompt broadcast across all samples, writing the
        # prompt's keys/values straight into the static buffer for every row.
        prompt = torch.tensor([tokens], dtype=torch.long, device=device).expand(num_samples, -1).contiguous()
        logits, cache = self.model(
            prompt, attention_mask=None, causal=True, past_key_values=cache, use_cache=True
        )
        logits = logits[:, -1, :].contiguous()
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        generated = 0
        while True:
            if max_tokens is not None and generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            sampled = sample_next_token(logits, rng, temperature=temperature, top_k=top_k).squeeze(1).tolist()
            token_column: list[int] = []
            token_masks: list[int] = []
            for row, state in enumerate(row_states):
                if state.completed:
                    next_token = assistant_end
                    train_mask = 0
                elif state.forced_tokens:
                    next_token = state.forced_tokens.popleft()
                    train_mask = 0
                else:
                    next_token = int(sampled[row])
                    train_mask = 1

                token_column.append(next_token)
                token_masks.append(train_mask)

                if state.completed:
                    continue
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                elif next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    expr = self.tokenizer.decode(state.python_expr_tokens, skip_special=True)
                    result = use_calculator(expr)
                    if result is not None:
                        state.forced_tokens.append(output_start)
                        state.forced_tokens.extend(self.tokenizer.encode(str(result)))
                        state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits, cache = self.model(
                ids,
                attention_mask=None,
                causal=True,
                past_key_values=cache,
                use_cache=True,
            )
            logits = logits[:, -1, :]

    @torch.inference_mode()
    def generate_batch(
        self,
        tokens: list[int],
        *,
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> tuple[list[list[int]], list[list[int]]]:
        assistant_end = special_id(self.tokenizer, "<|assistant_end|>")
        bos = self.tokenizer.bos_token_id
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(
            tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        ):
            for row, (token, mask) in enumerate(zip(token_column, token_masks)):
                if completed[row]:
                    continue
                if token == assistant_end or token == bos:
                    completed[row] = True
                    continue
                results[row].append(token)
                masks[row].append(mask)
            if all(completed):
                break
        return results, masks
