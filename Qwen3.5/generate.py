from __future__ import annotations

import torch


def _unpack_model_output(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "past_key_values", None)
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


@torch.no_grad()
def greedy_generate(model, input_ids, attention_mask=None, max_new_tokens: int = 32):
    model.eval()
    logits, cache = _unpack_model_output(model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True))
    generated = input_ids

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

        logits, cache = _unpack_model_output(
            model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
        )

    return generated
