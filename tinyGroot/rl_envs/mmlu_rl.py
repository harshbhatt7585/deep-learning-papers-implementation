from __future__ import annotations

import argparse
import gc
import itertools
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from tinygroot.hf_upload import download_checkpoint_from_hub, push_checkpoint_to_hub
from tinygroot.sft_chat import render_prompt_for_completion
from tinygroot.sft_data import MMLU
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.training.chat_rl import (
    build_optimizer,
    generate_rollouts_batched,
    load_model_and_tokenizer,
    make_flat_rollout_batch,
    microbatch_ranges,
    weighted_pg_loss,
)
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
    resolve_checkpoint_dir,
    save_checkpoint,
    unwrap_model,
    world_size,
)


CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tinyGroot MMLU RL with flat batched rollouts.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SFT checkpoint.pt or run directory.")
    parser.add_argument("--arch", choices=["auto", "causal_mtp", "hrm"], default="auto")
    parser.add_argument("--hf-checkpoint-repo-id", type=str, default=None)
    parser.add_argument("--hf-checkpoint-revision", type=str, default=None)
    parser.add_argument("--hf-checkpoint-cache-dir", type=Path, default=Path("runs/hf_checkpoints"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", "--run", "--wandb-name", dest="wandb_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tinyGroot-mmlu-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/wandb"))

    parser.add_argument("--train-split", choices=["auxiliary_train", "validation", "test"], default="auxiliary_train")
    parser.add_argument("--eval-split", choices=["validation", "test"], default="test")
    parser.add_argument("--train-stop", type=int, default=-1, help="Limit train examples; <=0 uses all.")
    parser.add_argument("--eval-examples", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--device-batch-size", "--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--examples-per-step", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)

    parser.add_argument("--log-rollouts-every", type=int, default=5)
    parser.add_argument("--log-rollout-samples", type=int, default=4)
    parser.add_argument("--log-rollout-chars", type=int, default=240)
    parser.add_argument("--stream-rollouts", action="store_true")

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

    parser.add_argument("--push-to-hf", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-commit-message", type=str, default=None)
    return parser.parse_args()


def resolve_input_checkpoint(args: argparse.Namespace) -> None:
    if args.hf_checkpoint_repo_id:
        checkpoint_dir = download_checkpoint_from_hub(
            repo_id=args.hf_checkpoint_repo_id,
            revision=args.hf_checkpoint_revision,
            cache_dir=args.hf_checkpoint_cache_dir,
        )
        args.checkpoint = checkpoint_dir
        log(f"downloaded HF checkpoint {args.hf_checkpoint_repo_id} to {checkpoint_dir}")
    if args.checkpoint is None:
        raise SystemExit("pass --checkpoint /path/to/run or --hf-checkpoint-repo-id username/model")


def gold_choice(conversation: dict[str, Any]) -> str:
    content = conversation["messages"][-1]["content"]
    if not isinstance(content, str):
        raise ValueError("MMLU gold answer must be a string")
    return content.strip().upper()


def extract_choice(completion: str, letters: tuple[str, ...] = ("A", "B", "C", "D")) -> str | None:
    text = completion.strip().upper()
    if not text:
        return None
    first = text[0]
    if first in letters:
        return first
    match = CHOICE_RE.search(text)
    if match and match.group(1) in letters:
        return match.group(1)
    return None


def mmlu_reward(conversation: dict[str, Any], completion: str) -> float:
    letters = tuple(conversation.get("letters", ("A", "B", "C", "D")))
    pred = extract_choice(completion, letters)
    return float(pred == gold_choice(conversation))


def compact(text: str, max_chars: int) -> str:
    text = " ".join(text.strip().split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def log_mmlu_rollouts(
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
    prompt = conversation["messages"][0]["content"]
    log(f"[mmlu-rollout] step={step} example_idx={example_idx} gold={gold_choice(conversation)} prompt={compact(prompt, args.log_rollout_chars)!r}")
    for sample_idx, (seq, reward) in enumerate(zip(sequences, rewards)):
        if sample_idx >= args.log_rollout_samples:
            break
        completion = tokenizer.decode(seq[prompt_len:], skip_special=True)
        pred = extract_choice(completion, tuple(conversation.get("letters", ("A", "B", "C", "D"))))
        log(
            f"[mmlu-rollout] step={step} sample={sample_idx} reward={reward:.1f} "
            f"pred={pred} response={compact(completion, args.log_rollout_chars)!r}"
        )


@torch.no_grad()
def evaluate_mmlu_next_token(
    model: torch.nn.Module,
    tokenizer: NanochatTokenizer,
    runtime: Runtime,
    task: MMLU,
    *,
    max_examples: int,
    batch_size: int,
) -> float:
    source = unwrap_model(model)
    source.eval()
    num_examples = len(task) if max_examples <= 0 else min(len(task), max_examples)
    num_batches = -(-num_examples // batch_size)
    letter_to_id: dict[str, int] = {}
    passed = 0
    total = 0
    for batch_idx in range(rank(), num_batches, world_size()):
        i0 = batch_idx * batch_size
        i1 = min(i0 + batch_size, num_examples)
        conversations = [task[i] for i in range(i0, i1)]
        prompts = [
            render_prompt_for_completion(
                tokenizer,
                conversation,
                max_tokens=max(1, source.config.max_seq_len - 1),
            )
            for conversation in conversations
        ]
        max_len = max(len(ids) for ids in prompts)
        answer_positions = [len(ids) - 1 for ids in prompts]
        padded = [ids + [tokenizer.bos_token_id] * (max_len - len(ids)) for ids in prompts]
        x = torch.tensor(padded, dtype=torch.long, device=runtime.device)
        logits = model(x, attention_mask=None, causal=True)
        for row_idx, conversation in enumerate(conversations):
            letters = tuple(conversation.get("letters", ("A", "B", "C", "D")))
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id:
                    encoded = tokenizer.encode(letter)
                    if len(encoded) != 1:
                        raise ValueError(f"letter {letter!r} must encode to one token, got {encoded}")
                    letter_to_id[letter] = encoded[0]
                letter_ids.append(letter_to_id[letter])
            focus = logits[row_idx, answer_positions[row_idx], letter_ids]
            pred = letters[int(focus.argmax(dim=-1).item())]
            passed += int(pred == gold_choice(conversation))
            total += 1
    counts = torch.tensor([passed, total], dtype=torch.float32, device=runtime.device)
    if is_dist():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return float((counts[0] / counts[1].clamp(min=1)).item())


def make_task(split: str, stop: int = -1) -> MMLU:
    return MMLU(split, stop=None if stop <= 0 else stop)


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    torch.manual_seed(args.seed + rank())
    model, tokenizer, source_meta = load_model_and_tokenizer(args, runtime)
    source = unwrap_model(model)
    optimizer = build_optimizer(args, model, runtime)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"] * args.init_lr_frac
        group["lr"] = group["initial_lr"]

    train_task = make_task(args.train_split, args.train_stop)
    val_task = make_task(args.eval_split)
    if args.batch_size <= 0:
        raise ValueError("--device-batch-size must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.examples_per_step % world_size() != 0:
        raise ValueError("--examples-per-step must be divisible by world size")
    examples_per_rank = args.examples_per_step // world_size()
    epoch_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
    num_steps = args.max_steps if args.max_steps > 0 else max(1, epoch_steps)

    args.wandb_name = args.wandb_name or args.out_dir.name
    wandb_run = init_wandb(args, source.config, runtime)
    scaler = None
    log(f"loaded MMLU RL checkpoint: {args.checkpoint} step={source_meta.get('step')}")
    log(
        f"MMLU RL steps={num_steps} train_split={args.train_split} "
        f"examples_per_rank={examples_per_rank} num_samples={args.num_samples}"
    )
    log("flat rollout mode: all examples_per_rank * num_samples rollouts decode in one KV-cached batch")

    batch_iter = itertools.cycle(range(rank(), len(train_task), world_size()))
    start_time = time.time()

    for step in range(num_steps):
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            eval_start = time.time()
            acc = evaluate_mmlu_next_token(
                model,
                tokenizer,
                runtime,
                val_task,
                max_examples=args.eval_examples,
                batch_size=args.batch_size,
            )
            if is_main_process():
                log(f"eval step={step} mmlu_acc={acc:.4f} eval_time={time.time() - eval_start:.1f}s")
                if wandb_run is not None:
                    wandb_run.log({"eval/mmlu_acc": acc}, step=step)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_examples = []
        for example_step in range(examples_per_rank):
            example_idx = next(batch_iter)
            conversation = train_task[example_idx]
            prompt_ids = render_prompt_for_completion(
                tokenizer,
                conversation,
                max_tokens=max(1, source.config.max_seq_len - args.max_new_tokens),
            )
            step_examples.append((example_step, example_idx, conversation, prompt_ids))

        stream_prefix = None
        if args.stream_rollouts and args.log_rollouts_every > 0 and step % args.log_rollouts_every == 0 and is_main_process():
            stream_prefix = f"[mmlu-rollout:live] step={step} example_idx={step_examples[0][1]} sample=0 response: "

        generate_start = time.time()
        per_example = generate_rollouts_batched(
            model,
            tokenizer,
            [example[3] for example in step_examples],
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            seed=hash((args.seed, step, rank())) & 0x7FFFFFFF,
            stream_prefix=stream_prefix,
        )
        rollout_generate_time = time.time() - generate_start

        flat_seqs: list[list[int]] = []
        flat_masks: list[list[int]] = []
        flat_adv: list[float] = []
        flat_rewards: list[float] = []
        flat_example: list[int] = []
        flat_prompt_len: list[int] = []
        rewards_by_example: list[float] = []
        sequence_lengths: list[int] = []
        rollout_tokens = 0
        rollout_count = 0

        for ex_i, ((example_step, example_idx, conversation, prompt_ids), (sequences, masks)) in enumerate(
            zip(step_examples, per_example)
        ):
            rewards = [
                mmlu_reward(conversation, tokenizer.decode(seq[len(prompt_ids):], skip_special=True))
                for seq in sequences
            ]
            log_mmlu_rollouts(
                step=step,
                example_step=example_step,
                example_idx=example_idx,
                conversation=conversation,
                tokenizer=tokenizer,
                prompt_len=len(prompt_ids),
                sequences=sequences,
                rewards=rewards,
                args=args,
            )
            mean_reward_ex = sum(rewards) / max(1, len(rewards))
            rewards_by_example.append(mean_reward_ex)
            rollout_tokens += sum(max(0, len(seq) - len(prompt_ids)) for seq in sequences)
            rollout_count += len(sequences)
            sequence_lengths.extend(len(seq) for seq in sequences)
            for seq, mask, reward in zip(sequences, masks, rewards):
                flat_seqs.append(seq)
                flat_masks.append(mask)
                flat_rewards.append(reward)
                flat_adv.append(reward - mean_reward_ex)
                flat_example.append(ex_i)
                flat_prompt_len.append(len(prompt_ids))

        inputs_all, targets_all, scale_all = make_flat_rollout_batch(
            tokenizer,
            flat_seqs,
            flat_masks,
            flat_adv,
            flat_example,
            flat_prompt_len,
            examples_per_rank,
            runtime.device,
        )
        rewards_flat = torch.tensor(flat_rewards, dtype=torch.float32, device=runtime.device)
        loss_sum = 0.0
        loss_count = 0
        for pass_idx, (b0, b1) in enumerate(microbatch_ranges(inputs_all.size(0), args.batch_size)):
            with autocast_context(runtime.device, args.amp_dtype):
                loss = weighted_pg_loss(model, inputs_all[b0:b1], targets_all[b0:b1], scale_all[b0:b1])
            loss.backward()
            loss_sum += float(loss.detach().item())
            loss_count += 1
            log(
                f"step {step}/{num_steps} pass {pass_idx} "
                f"loss {loss.item():.6f} reward {rewards_flat[b0:b1].mean().item():.4f}"
            )

        lrm = max(0.0, 1.0 - step / max(1, num_steps))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mean_reward = sum(rewards_by_example) / max(1, len(rewards_by_example))
        mean_seq_len = sum(sequence_lengths) / max(1, len(sequence_lengths))
        metrics_tensor = torch.tensor(
            [
                mean_reward,
                mean_seq_len,
                float(rollout_tokens),
                float(rollout_count),
                rollout_generate_time,
                loss_sum,
                float(loss_count),
            ],
            dtype=torch.float32,
            device=runtime.device,
        )
        if is_dist():
            dist.all_reduce(metrics_tensor[:2], op=dist.ReduceOp.AVG)
            dist.all_reduce(metrics_tensor[2:], op=dist.ReduceOp.SUM)
        global_rollout_tokens = float(metrics_tensor[2].item())
        global_rollout_count = float(metrics_tensor[3].item())
        avg_rank_rollout_time = float(metrics_tensor[4].item()) / world_size()
        rollout_tokens_per_sec = global_rollout_tokens / max(1e-9, avg_rank_rollout_time)
        mean_loss = float(metrics_tensor[5].item()) / max(1.0, float(metrics_tensor[6].item()))
        log(
            f"step {step}/{num_steps} reward {metrics_tensor[0].item():.4f} "
            f"loss {mean_loss:.6f} seq_len {metrics_tensor[1].item():.2f} lrm {lrm:.4f} "
            f"rollouts {global_rollout_count:.0f} rollout_tokens {global_rollout_tokens:.0f} "
            f"rollout_tok/s {rollout_tokens_per_sec:.1f} rollout_time/rank {avg_rank_rollout_time:.1f}s"
        )
        if wandb_run is not None and is_main_process():
            wandb_run.log(
                {
                    "train/reward": float(metrics_tensor[0].item()),
                    "train/loss": mean_loss,
                    "train/sequence_length": float(metrics_tensor[1].item()),
                    "train/rollouts": global_rollout_count,
                    "train/rollout_tokens": global_rollout_tokens,
                    "train/rollout_tokens_per_sec": rollout_tokens_per_sec,
                    "train/rollout_time_per_rank": avg_rank_rollout_time,
                    "train/lrm": lrm,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )

        if is_main_process() and args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
            )
            log(f"saved MMLU RL checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

    if is_main_process():
        save_checkpoint(
            out_dir=args.out_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            step=num_steps,
            args=args,
        )
        log(f"saved final MMLU RL checkpoint: {args.out_dir / 'checkpoint.pt'}")
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
            log(f"uploaded final MMLU RL checkpoint to Hugging Face: {commit_url}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = num_steps
            wandb_run.summary["total_training_time"] = time.time() - start_time
            wandb_run.finish()


@record
def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        resolve_input_checkpoint(args)
        train(args, runtime)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

