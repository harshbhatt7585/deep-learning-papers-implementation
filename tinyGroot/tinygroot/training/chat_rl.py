from __future__ import annotations

import argparse
import gc
import itertools
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tinygroot.chat_core_eval import extract_gsm_answer
from tinygroot.engine import Engine
from tinygroot.hf_upload import push_checkpoint_to_hub
from tinygroot.model import TinyGrootConfig, TinyGrootModel
from tinygroot.nanochat_optim import DistMuonAdamW, MuonAdamW
from tinygroot.sft_chat import ChatSpecialIds, render_prompt_for_completion
from tinygroot.sft_data import GSM8K
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.training.chat_sft import clean_state_dict
from tinygroot.training.train import apply_fp8_training
from tinygroot.utils import (
    Runtime,
    autocast_context,
    cleanup_distributed,
    create_runtime,
    init_wandb,
    is_dist,
    is_main_process,
    log,
    rank,
    save_checkpoint,
    unwrap_model,
    world_size,
)


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tinyGroot GSM8K RL with nanochat-style GRPO/REINFORCE.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SFT checkpoint.pt to initialize from.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for RL checkpoint.pt.")
    parser.add_argument("--run-name", "--run", "--wandb-name", dest="wandb_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tinyGroot-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/wandb"))

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epoch-derived number of optimizer steps.")
    parser.add_argument("--device-batch-size", "--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--examples-per-step", type=int, default=16, help="Total GSM8K questions per optimizer step across ranks.")
    parser.add_argument("--num-samples", type=int, default=16, help="Samples per question.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)

    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--eval-num-samples", type=int, default=None, help="Defaults to device batch size.")
    parser.add_argument("--log-rollouts-every", type=int, default=1, help="Log decoded rollout samples on rank 0 every N steps; 0 disables.")
    parser.add_argument("--log-rollout-samples", type=int, default=2, help="Number of rollout completions to print when logging is enabled.")
    parser.add_argument("--log-rollout-chars", type=int, default=1200, help="Maximum decoded characters per logged rollout completion.")

    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.05)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--push-to-hf", action="store_true", help="Upload the final RL checkpoint directory to Hugging Face Hub after training.")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-commit-message", type=str, default=None)
    return parser.parse_args()


def _assistant_ref_text(conversation: dict[str, Any]) -> str:
    content = conversation["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return ""


def gsm_reward(conversation: dict[str, Any], completion: str) -> float:
    ref = extract_gsm_answer(_assistant_ref_text(conversation))
    pred = extract_gsm_answer(completion)
    return float(pred is not None and pred == ref)


def load_model_and_tokenizer(args: argparse.Namespace, runtime: Runtime) -> tuple[torch.nn.Module, NanochatTokenizer, dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location=runtime.device, weights_only=False)
    config = TinyGrootConfig(**checkpoint["config"])
    tokenizer = NanochatTokenizer.load(args.checkpoint.parent / "tokenizer_hf")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config).to(runtime.device)
    model.load_state_dict(clean_state_dict(checkpoint["model_state"]), strict=True)
    if len(model.mtp_heads) > 0:
        for param in model.mtp_heads.parameters():
            param.requires_grad = False
        log(f"MTP heads frozen and unused during RL: heads={len(model.mtp_heads)}")
    model = apply_fp8_training(model, args, runtime)
    if args.compile:
        log("compiling RL model with torch.compile(dynamic=False)")
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    if is_dist() and args.optimizer == "muon":
        log("distributed RL model: replicated parameters with DistMuonAdamW gradient sync")
        return model, tokenizer, checkpoint
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    return model, tokenizer, checkpoint


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, runtime: Runtime) -> torch.optim.Optimizer:
    source = unwrap_model(model)
    trainable = lambda params: [p for p in params if p.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable(source.parameters()),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=runtime.device.type == "cuda",
        )

    model_dim = source.config.d_model
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    embedding_params = trainable(source.token_emb.parameters())
    lm_head_params = trainable(source.lm_head.parameters())
    matrix_params = trainable(source.blocks.parameters())
    seen = {id(p) for p in embedding_params + lm_head_params + matrix_params}
    scalar_params = [p for p in source.parameters() if p.requires_grad and id(p) not in seen]
    groups: list[dict[str, Any]] = [
        {
            "kind": "adamw",
            "params": lm_head_params,
            "lr": args.unembedding_lr * dmodel_lr_scale,
            "lr_multiplier": args.unembedding_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.96),
            "eps": 1e-10,
            "weight_decay": args.weight_decay,
        },
        {
            "kind": "adamw",
            "params": embedding_params,
            "lr": args.embedding_lr * dmodel_lr_scale,
            "lr_multiplier": args.embedding_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.96),
            "eps": 1e-10,
            "weight_decay": args.weight_decay,
        },
        {
            "kind": "adamw",
            "params": scalar_params,
            "lr": args.scalar_lr * dmodel_lr_scale,
            "lr_multiplier": args.scalar_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.96),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
    ]
    for shape in sorted({param.shape for param in matrix_params}):
        shape_params = [param for param in matrix_params if param.shape == shape]
        groups.append({
            "kind": "muon",
            "params": shape_params,
            "lr": args.matrix_lr * dmodel_lr_scale,
            "lr_multiplier": args.matrix_lr * dmodel_lr_scale / args.lr,
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
            "beta2": 0.9,
            "weight_decay": 0.0,
        })
    groups = [group for group in groups if group["params"]]
    opt_cls = DistMuonAdamW if is_dist() else MuonAdamW
    return opt_cls(groups)


@torch.no_grad()
def generate_batch_with_masks(
    model: torch.nn.Module,
    tokenizer: NanochatTokenizer,
    prompt_ids: list[int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    source = unwrap_model(model)
    source.eval()
    max_tokens = min(max_new_tokens, max(0, source.config.max_seq_len - len(prompt_ids)))
    engine = Engine(source, tokenizer)
    return engine.generate_batch(
        prompt_ids,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


def make_rollout_batch(
    tokenizer: NanochatTokenizer,
    prompt_len: int,
    sequences: list[list[int]],
    masks: list[list[int]],
    rewards: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = ChatSpecialIds.from_tokenizer(tokenizer).assistant_end
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    padded_masks = [mask + [0] * (max_len - len(mask)) for mask in masks]
    ids = torch.tensor(padded, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.bool, device=device)
    inputs = ids[:, :-1].contiguous()
    targets = ids[:, 1:].clone().contiguous()
    targets[~mask_ids[:, 1:]] = IGNORE_INDEX
    # Guard against malformed masks from overlong prompts.
    targets[:, : max(0, prompt_len - 1)] = IGNORE_INDEX
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = rewards_tensor - rewards_tensor.mean()
    return inputs, targets, rewards_tensor, advantages


def policy_gradient_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """DAPO-style token-level REINFORCE: per-pass mean of (logp * advantage) over
    valid positions. Returns ``-pg_obj / num_valid_in_this_pass``. Callers should
    further divide by ``num_passes * examples_per_rank`` so that summing the
    .backward() calls across one optimizer step yields the per-example mean of
    per-pass-token-means (matches nanochat/scripts/chat_rl.py)."""
    logits = model(inputs, attention_mask=None, causal=True)
    valid = targets != IGNORE_INDEX
    safe_targets = targets.masked_fill(~valid, 0)
    logp = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    pg_obj = (logp * valid.to(logp.dtype) * advantages[:, None]).sum()
    num_valid = valid.sum().clamp(min=1)
    return -pg_obj / num_valid


@torch.no_grad()
def run_gsm8k_eval(model: torch.nn.Module, tokenizer: NanochatTokenizer, runtime: Runtime, args: argparse.Namespace, task: GSM8K) -> dict[str, float]:
    source = unwrap_model(model)
    source.eval()
    num_samples = args.eval_num_samples or args.batch_size
    pass_counts = torch.zeros(num_samples, dtype=torch.float32, device=runtime.device)
    total = 0
    max_examples = min(args.eval_examples, len(task))
    for idx in range(rank(), max_examples, world_size()):
        conversation = task[idx]
        prompt_ids = render_prompt_for_completion(
            tokenizer,
            conversation,
            max_tokens=max(1, source.config.max_seq_len - args.max_new_tokens),
        )
        sequences, _masks = generate_batch_with_masks(
            source,
            tokenizer,
            prompt_ids,
            num_samples=num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=args.top_k if args.top_k > 0 else None,
            seed=(args.seed + idx) & 0x7FFFFFFF,
        )
        outcomes = []
        for seq in sequences:
            completion = tokenizer.decode(seq[len(prompt_ids):], skip_special=True)
            outcomes.append(gsm_reward(conversation, completion) > 0.0)
        for k in range(1, num_samples + 1):
            pass_counts[k - 1] += float(any(outcomes[:k]))
        total += 1
    total_tensor = torch.tensor(total, dtype=torch.float32, device=runtime.device)
    if is_dist():
        dist.all_reduce(pass_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    pass_rates = pass_counts / total_tensor.clamp(min=1)
    return {f"pass@{k}": float(pass_rates[k - 1].item()) for k in range(1, num_samples + 1)}


def log_wandb(wandb_run: Any, metrics: dict[str, float], step: int) -> None:
    if wandb_run is not None and is_main_process():
        wandb_run.log(metrics, step=step)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return str(content)


def conversation_prompt_text(conversation: dict[str, Any]) -> str:
    for message in reversed(conversation.get("messages", [])[:-1]):
        if message.get("role") == "user":
            return _message_text(message)
    return _message_text(conversation.get("messages", [{}])[0])


def compact_log_text(text: str, max_chars: int) -> str:
    text = " ".join(text.strip().split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def log_rollout_responses(
    *,
    step: int,
    example_step: int,
    example_idx: int,
    conversation: dict[str, Any],
    tokenizer: NanochatTokenizer,
    prompt_len: int,
    sequences: list[list[int]],
    rewards: list[float],
    args: argparse.Namespace,
) -> None:
    if not is_main_process() or args.log_rollouts_every <= 0:
        return
    if step % args.log_rollouts_every != 0 or example_step != 0:
        return
    prompt = compact_log_text(conversation_prompt_text(conversation), args.log_rollout_chars)
    log(f"[rollout] step={step} example_idx={example_idx} prompt={prompt!r}")
    for sample_idx, (seq, reward) in enumerate(zip(sequences, rewards)):
        if sample_idx >= args.log_rollout_samples:
            break
        completion = tokenizer.decode(seq[prompt_len:], skip_special=True)
        completion = compact_log_text(completion, args.log_rollout_chars)
        log(
            f"[rollout] step={step} example_idx={example_idx} sample={sample_idx} "
            f"reward={reward:.1f} tokens={len(seq) - prompt_len} response={completion!r}"
        )


def microbatch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    torch.manual_seed(args.seed + rank())
    model, tokenizer, checkpoint = load_model_and_tokenizer(args, runtime)
    source = unwrap_model(model)
    tokenizer_dir = args.checkpoint.parent / "tokenizer_hf"
    optimizer = build_optimizer(args, model, runtime)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"] * args.init_lr_frac
        group["lr"] = group["initial_lr"]

    train_task = GSM8K("train")
    val_task = GSM8K("test")
    epoch_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
    num_steps = args.max_steps if args.max_steps > 0 else max(1, epoch_steps)
    if args.batch_size <= 0:
        raise ValueError("--device-batch-size must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.examples_per_step % world_size() != 0:
        raise ValueError("--examples-per-step must be divisible by world size")
    examples_per_rank = args.examples_per_step // world_size()
    if (args.eval_num_samples or args.batch_size) > args.batch_size:
        raise ValueError("--eval-num-samples must be <= --device-batch-size")

    args.wandb_name = args.wandb_name or args.out_dir.name
    wandb_run = init_wandb(args, source.config, runtime)
    scaler = None
    log(f"loaded RL checkpoint: {args.checkpoint} step={checkpoint.get('step')}")
    log(f"RL steps={num_steps} examples_per_rank={examples_per_rank} num_samples={args.num_samples}")
    log(
        "rollout logging: "
        f"every={args.log_rollouts_every} samples={args.log_rollout_samples} chars={args.log_rollout_chars}"
    )
    batch_iter = itertools.cycle(range(rank(), len(train_task), world_size()))
    start_time = time.time()

    for step in range(num_steps):
        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_start = time.time()
            if is_main_process():
                log(
                    f"starting eval step={step} examples={min(args.eval_examples, len(val_task))} "
                    f"samples={args.eval_num_samples or args.batch_size} max_new_tokens={args.max_new_tokens}"
                )
            metrics = run_gsm8k_eval(model, tokenizer, runtime, args, val_task)
            if is_main_process():
                elapsed = time.time() - eval_start
                log(" ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f" eval_time={elapsed:.1f}s")
            log_wandb(wandb_run, {f"eval/{k}": v for k, v in metrics.items()}, step)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        rewards_by_example = []
        sequence_lengths = []
        for example_step in range(examples_per_rank):
            example_idx = next(batch_iter)
            conversation = train_task[example_idx]
            prompt_ids = render_prompt_for_completion(
                tokenizer,
                conversation,
                max_tokens=max(1, source.config.max_seq_len - args.max_new_tokens),
            )
            all_sequences: list[list[int]] = []
            all_masks: list[list[int]] = []
            for sampling_step, (sample_start, sample_end) in enumerate(
                microbatch_ranges(args.num_samples, args.batch_size)
            ):
                seed = hash((args.seed, step, example_idx, sampling_step, rank())) & 0x7FFFFFFF
                sequences, masks = generate_batch_with_masks(
                    model,
                    tokenizer,
                    prompt_ids,
                    num_samples=sample_end - sample_start,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k > 0 else None,
                    seed=seed,
                )
                all_sequences.extend(sequences)
                all_masks.extend(masks)
            rewards = [
                gsm_reward(conversation, tokenizer.decode(seq[len(prompt_ids):], skip_special=True))
                for seq in all_sequences
            ]
            log_rollout_responses(
                step=step,
                example_step=example_step,
                example_idx=example_idx,
                conversation=conversation,
                tokenizer=tokenizer,
                prompt_len=len(prompt_ids),
                sequences=all_sequences,
                rewards=rewards,
                args=args,
            )
            rewards_by_example.append(sum(rewards) / max(1, len(rewards)))
            sequence_lengths.extend(len(seq) for seq in all_sequences)
            inputs_all, targets_all, rewards_all, advantages_all = make_rollout_batch(
                tokenizer, len(prompt_ids), all_sequences, all_masks, rewards, runtime.device
            )
            model.train()
            batch_ranges = microbatch_ranges(inputs_all.size(0), args.batch_size)
            num_passes = len(batch_ranges)
            for pass_idx, (b0, b1) in enumerate(batch_ranges):
                with autocast_context(runtime.device, args.amp_dtype):
                    loss = policy_gradient_loss(
                        model,
                        inputs_all[b0:b1],
                        targets_all[b0:b1],
                        advantages_all[b0:b1],
                    )
                    loss = loss / (num_passes * examples_per_rank)
                loss.backward()
                log(
                    f"step {step}/{num_steps} example {example_step} pass {pass_idx} "
                    f"loss {loss.item():.6f} reward {rewards_all[b0:b1].mean().item():.4f}"
                )

        lrm = max(0.0, 1.0 - step / max(1, num_steps))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mean_reward = sum(rewards_by_example) / max(1, len(rewards_by_example))
        mean_seq_len = sum(sequence_lengths) / max(1, len(sequence_lengths))
        metrics_tensor = torch.tensor([mean_reward, mean_seq_len], dtype=torch.float32, device=runtime.device)
        if is_dist():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        log(f"step {step}/{num_steps} reward {metrics_tensor[0].item():.4f} seq_len {metrics_tensor[1].item():.2f} lrm {lrm:.4f}")
        log_wandb(
            wandb_run,
            {
                "train/reward": float(metrics_tensor[0].item()),
                "train/sequence_length": float(metrics_tensor[1].item()),
                "train/lrm": lrm,
                "train/lr": optimizer.param_groups[0]["lr"],
            },
            step,
        )

        if is_main_process() and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=step, args=args)
            log(f"saved RL checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

    if is_main_process():
        save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=num_steps, args=args)
        total_time = time.time() - start_time
        log(f"saved final RL checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if args.push_to_hf:
            if not args.hf_repo_id:
                raise ValueError("--push-to-hf requires --hf-repo-id")
            commit_url = push_checkpoint_to_hub(
                checkpoint_dir=args.out_dir,
                repo_id=args.hf_repo_id,
                private=args.hf_private,
                revision=args.hf_revision,
                commit_message=args.hf_commit_message,
            )
            log(f"uploaded final RL checkpoint to Hugging Face: {commit_url}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = num_steps
            wandb_run.summary["total_training_time"] = total_time
            wandb_run.finish()


@record
def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        train(args, runtime)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
