from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygroot.hf_upload import download_checkpoint_from_hub
from tinygroot.infer.chat_infer import clean_state_dict
from tinygroot.chat_core_eval import use_calculator
from tinygroot.model import TinyGrootConfig, TinyGrootModel, _sample_tokens, infer_arch_from_state_dict
from tinygroot.sft_chat import ChatSpecialIds, generate_with_tools, render_prompt_for_completion
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.utils import load_meta, load_model_state, resolve_checkpoint_dir


DEFAULT_REPO_ID = "harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000"
DEFAULT_CACHE_DIR = ROOT / "runs" / "hf_checkpoints"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def pick_device(require_mps: bool) -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if require_mps:
        raise RuntimeError(
            "MPS is not available. Run on an Apple Silicon Mac with an MPS-enabled PyTorch build, "
            "or pass --allow-cpu-fallback."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint_cpu_then_device(path: Path, device: torch.device) -> tuple[TinyGrootModel, NanochatTokenizer, dict[str, Any]]:
    meta = load_meta(path)
    if "config" not in meta:
        raise KeyError(f"checkpoint at {path} has no 'config'")

    checkpoint_dir = resolve_checkpoint_dir(path)
    tokenizer = NanochatTokenizer.load(checkpoint_dir / "tokenizer_hf")
    state = clean_state_dict(load_model_state(path, map_location="cpu"))
    config_blob = meta["config"]
    if isinstance(config_blob, TinyGrootConfig):
        config_blob.arch = infer_arch_from_state_dict(state)
        config = config_blob
    else:
        cfg = dict(config_blob)
        cfg["arch"] = infer_arch_from_state_dict(state)
        config = TinyGrootConfig(**cfg)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, meta


class TinyGrootChat:
    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None,
        cache_dir: Path,
        device: torch.device,
        checkpoint: Path | None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.lock = threading.Lock()

        checkpoint_path = checkpoint or download_checkpoint_from_hub(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
        self.model, self.tokenizer, self.meta = load_checkpoint_cpu_then_device(self.checkpoint_dir, device)

    @property
    def info(self) -> dict[str, Any]:
        config = self.meta.get("config", {})
        return {
            "repo_id": self.repo_id,
            "revision": self.revision or "main",
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": str(self.device),
            "step": self.meta.get("step"),
            "max_seq_len": getattr(self.model.config, "max_seq_len", None),
            "layers": config.get("n_layers") if isinstance(config, dict) else None,
            "hidden": config.get("d_model") if isinstance(config, dict) else None,
            "heads": config.get("n_heads") if isinstance(config, dict) else None,
            "mtp_heads": config.get("n_mtp_heads") if isinstance(config, dict) else None,
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages cannot be empty")

        generation_budget = max(1, min(max_new_tokens, 1024))
        prompt_budget = max(1, self.model.config.max_seq_len - generation_budget)
        prompt_ids = render_prompt_for_completion(
            self.tokenizer,
            {"messages": messages},
            max_tokens=prompt_budget,
        )

        started = time.perf_counter()
        with self.lock:
            output_ids = generate_with_tools(
                self.model,
                self.tokenizer,
                prompt_ids,
                max_new_tokens=generation_budget,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )
        elapsed = max(time.perf_counter() - started, 1e-6)
        text = self.tokenizer.decode(output_ids, skip_special=True).strip()
        return {
            "message": {"role": "assistant", "content": text},
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(output_ids),
                "seconds": elapsed,
                "tokens_per_second": len(output_ids) / elapsed,
            },
        }

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ):
        if not messages:
            raise ValueError("messages cannot be empty")

        generation_budget = max(1, min(max_new_tokens, 1024))
        prompt_budget = max(1, self.model.config.max_seq_len - generation_budget)
        prompt_ids = render_prompt_for_completion(
            self.tokenizer,
            {"messages": messages},
            max_tokens=prompt_budget,
        )

        started = time.perf_counter()
        output_ids: list[int] = []
        seq = list(prompt_ids)
        forced: list[int] = []
        in_python = False
        python_expr_tokens: list[int] = []
        specials = ChatSpecialIds.from_tokenizer(self.tokenizer)
        top_k_arg = top_k if top_k > 0 else None

        with self.lock:
            for _ in range(generation_budget):
                if forced:
                    next_token = forced.pop(0)
                else:
                    if len(seq) >= self.model.config.max_seq_len:
                        break
                    x = torch.tensor([seq], dtype=torch.long, device=self.device)
                    logits = self.model(x, attention_mask=None, causal=True)
                    sampled, _ = _sample_tokens(
                        logits[:, -1, :],
                        temperature=temperature,
                        top_k=top_k_arg,
                        top_p=None,
                    )
                    next_token = int(sampled.item())

                if next_token == specials.assistant_end or next_token == self.tokenizer.bos_token_id:
                    break
                seq.append(next_token)
                output_ids.append(next_token)
                if next_token == specials.python_start:
                    in_python = True
                    python_expr_tokens = []
                elif next_token == specials.python_end and in_python:
                    in_python = False
                    expr = self.tokenizer.decode(python_expr_tokens, skip_special=True)
                    result = use_calculator(expr)
                    if result is not None:
                        forced.append(specials.output_start)
                        forced.extend(self.tokenizer.encode(str(result)))
                        forced.append(specials.output_end)
                    python_expr_tokens = []
                elif in_python:
                    python_expr_tokens.append(next_token)

                text = self.tokenizer.decode(output_ids, skip_special=True).strip()
                yield {
                    "type": "token",
                    "content": text,
                    "completion_tokens": len(output_ids),
                }

        elapsed = max(time.perf_counter() - started, 1e-6)
        text = self.tokenizer.decode(output_ids, skip_special=True).strip()
        yield {
            "type": "done",
            "message": {"role": "assistant", "content": text},
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(output_ids),
                "seconds": elapsed,
                "tokens_per_second": len(output_ids) / elapsed,
            },
        }


class ClientDisconnected(Exception):
    """Raised when the browser closes a streaming connection mid-generation."""


class ChatHandler(SimpleHTTPRequestHandler):
    service: TinyGrootChat

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[local-chat] {self.address_string()} {format % args}", flush=True)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.write_json({"ok": True, "model": self.service.info})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat_stream":
            self.handle_chat_stream()
            return
        if path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json()
            messages = sanitize_messages(payload.get("messages"))
            max_new_tokens = int(payload.get("max_new_tokens", 256))
            temperature = float(payload.get("temperature", 0.0))
            top_k = int(payload.get("top_k", 50))
            result = self.service.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            self.write_json({"ok": True, **result})
        except Exception as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def handle_chat_stream(self) -> None:
        try:
            payload = self.read_json()
            messages = sanitize_messages(payload.get("messages"))
            max_new_tokens = int(payload.get("max_new_tokens", 256))
            temperature = float(payload.get("temperature", 0.0))
            top_k = int(payload.get("top_k", 50))
        except Exception as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        # The streamed body has no Content-Length, so the client only learns the
        # response is finished when the connection closes. Force-close it (rather
        # than keep-alive) or the browser's reader blocks forever after the last
        # token and the UI stays locked.
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        # Closing the generator on exit throws GeneratorExit into stream_generate,
        # which unwinds its `with self.lock` block so a disconnect (or Stop button)
        # promptly stops decoding and frees the model for the next request.
        events = self.service.stream_generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        try:
            for event in events:
                self.write_sse(event)
        except ClientDisconnected:
            self.log_message("client disconnected; stopping generation")
        except Exception as exc:  # noqa: BLE001 - surface any model error to the client
            try:
                self.write_sse({"type": "error", "error": str(exc)})
            except ClientDisconnected:
                pass
        finally:
            events.close()

    def read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        return json.loads(raw.decode("utf-8"))

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_sse(self, payload: dict[str, Any]) -> None:
        data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClientDisconnected from exc


def sanitize_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("messages must be a list")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("last message must be a user message")
    return messages[-24:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local tinyGroot chat UI.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local checkpoint directory.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = pick_device(require_mps=not args.allow_cpu_fallback)
    print(f"loading {args.repo_id} on {device}...", flush=True)
    ChatHandler.service = TinyGrootChat(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        device=device,
        checkpoint=args.checkpoint,
    )
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"serving tinyGroot chat at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
