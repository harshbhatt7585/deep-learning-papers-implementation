from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tinygroot.fp8 import disable_fp8
from tinygroot.hf_upload import push_checkpoint_to_hub
from tinygroot.model import TinyGrootConfig, TinyGrootModel, norm
from tinygroot.nanochat_optim import DistMuonAdamW, MuonAdamW
from tinygroot.sft_chat import evaluate_chatcore, render_conversation, sample_text
from tinygroot.sft_data import Task, build_datasets
from tinygroot.tokenizer import NanochatTokenizer
from tinygroot.training.train import apply_fp8_training, fp8_module_filter
from tinygroot.utils import (
    Runtime,
    autocast_context,
    cleanup_distributed,
    create_runtime,
    init_wandb,
    is_dist,
    is_main_process,
    load_meta,
    load_model_state,
    log,
    rank,
    resolve_checkpoint_dir,
    save_checkpoint,
    set_lr,
    unwrap_model,
    world_size,
)


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tinyGroot chat SFT from a pretrain checkpoint.pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrain checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for SFT checkpoint.pt")
    parser.add_argument("--run-name", "--run", "--wandb-name", dest="wandb_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tinyGroot-sft")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=Path("runs/wandb"))
    parser.add_argument("--push-to-hf", action="store_true", help="Upload the final SFT checkpoint directory to Hugging Face Hub after training.")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Target Hugging Face model repo, e.g. username/model-name.")
    parser.add_argument("--hf-private", action="store_true", help="Create/use a private Hugging Face model repo.")
    parser.add_argument("--hf-revision", type=str, default=None, help="Optional Hugging Face branch/revision.")
    parser.add_argument("--hf-commit-message", type=str, default=None)

    parser.add_argument("--device-batch-size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--total-batch-size", type=int, default=524_288, help="Global SFT tokens per optimizer step")
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to checkpoint max_seq_len")
    parser.add_argument("--max-steps", "--num-iterations", dest="max_steps", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-tokens", type=int, default=2_097_152)
    parser.add_argument("--chatcore-every", type=int, default=200, help="-1 disables ChatCORE eval")
    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task (-1 = all)")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task")
    parser.add_argument("--chatcore-max-new-tokens", type=int, default=512)
    parser.add_argument("--chatcore-temperature", type=float, default=0.0)
    parser.add_argument("--chatcore-top-k", type=int, default=50)
    parser.add_argument("--chatcore-batch-size", type=int, default=8, help="batch size for categorical ChatCORE eval")
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--sample-top-p", type=float, default=None)

    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.8)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise")
    parser.add_argument("--train-mtp-heads", action=argparse.BooleanOptionalAction, default=True, help="Train MTP heads with an assistant-token auxiliary loss")
    parser.add_argument("--mtp-loss-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--identity-jsonl", type=Path, default=Path("/data/identity_conversations.jsonl"))
    parser.add_argument("--identity-weight", type=int, default=2)
    parser.add_argument("--no-smoltalk", action="store_true")
    parser.add_argument("--mmlu-epochs", type=int, default=3)
    parser.add_argument("--gsm8k-epochs", type=int, default=4)
    parser.add_argument("--simple-spelling-size", type=int, default=200_000)
    parser.add_argument("--spellingbee-size", type=int, default=80_000)
    parser.add_argument("--words-path", type=Path, default=Path("/data/words_alpha.txt"))
    return parser.parse_args()


class SFTBatcher:
    def __init__(self, dataset: Task, tokenizer: NanochatTokenizer, args: argparse.Namespace, runtime: Runtime, split: str, buffer_size: int = 100) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.runtime = runtime
        self.split = split
        self.buffer_size = buffer_size
        self.conv_buffer: list[tuple[list[int], list[int]]] = []
        self.cursor = rank()
        self.consumed = rank()
        self.epoch = 1
        self.it = 0
        self.last_step = False
        self.approx_progress = 0.0

    def refill(self) -> None:
        dataset_size = len(self.dataset)
        while len(self.conv_buffer) < self.buffer_size:
            conv = self.dataset[self.cursor]
            ids, mask = render_conversation(self.tokenizer, conv, max_tokens=self.args.seq_len + 1)
            if len(ids) >= 2:
                self.conv_buffer.append((ids, mask))
            self.cursor += world_size()
            if self.cursor >= dataset_size:
                self.cursor %= dataset_size
                self.epoch += 1

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        row_capacity = self.args.seq_len + 1
        rows: list[list[int]] = []
        masks: list[list[int]] = []

        for _ in range(self.args.batch_size):
            row: list[int] = []
            mask_row: list[int] = []
            while len(row) < row_capacity:
                self.refill()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv_ids, _) in enumerate(self.conv_buffer):
                    if len(conv_ids) <= remaining and len(conv_ids) > best_len:
                        best_idx = i
                        best_len = len(conv_ids)
                if best_idx < 0:
                    row.extend([self.tokenizer.bos_token_id] * remaining)
                    mask_row.extend([0] * remaining)
                    break
                conv_ids, conv_mask = self.conv_buffer.pop(best_idx)
                row.extend(conv_ids)
                mask_row.extend(conv_mask)
                self.consumed += world_size()
            rows.append(row[:row_capacity])
            masks.append(mask_row[:row_capacity])

        self.it += 1
        if self.split == "train":
            if self.args.max_steps > 0:
                self.approx_progress = self.it / self.args.max_steps
                if self.it >= self.args.max_steps:
                    self.last_step = True
            else:
                self.approx_progress = self.consumed / max(1, len(self.dataset))
                if self.consumed >= len(self.dataset):
                    self.last_step = True

        use_cuda = self.runtime.device.type == "cuda"
        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, pin_memory=use_cuda)
        x = batch[:, :-1].to(self.runtime.device, non_blocking=use_cuda).contiguous()
        y = batch[:, 1:].to(self.runtime.device, non_blocking=use_cuda).contiguous()
        target_mask = mask_tensor[:, 1:].to(self.runtime.device, non_blocking=use_cuda)
        y = y.masked_fill(~target_mask, IGNORE_INDEX)
        return x, y


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_model_and_tokenizer(args: argparse.Namespace, runtime: Runtime) -> tuple[torch.nn.Module, NanochatTokenizer, dict[str, Any]]:
    meta = load_meta(args.checkpoint)
    config = TinyGrootConfig(**meta["config"])
    if args.seq_len is None:
        args.seq_len = config.max_seq_len
    if args.seq_len > config.max_seq_len:
        raise ValueError(f"--seq-len {args.seq_len} exceeds checkpoint max_seq_len {config.max_seq_len}")

    tokenizer_dir = resolve_checkpoint_dir(args.checkpoint) / "tokenizer_hf"
    tokenizer = NanochatTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}")

    model = TinyGrootModel(config).to(runtime.device)
    model.load_state_dict(
        clean_state_dict(load_model_state(args.checkpoint, map_location=runtime.device)),
        strict=True,
    )
    if not args.train_mtp_heads:
        for param in model.mtp_heads.parameters():
            param.requires_grad = False
        log("frozen MTP heads for chat SFT")
    elif len(model.mtp_heads) > 0:
        log(f"training MTP heads for chat SFT: heads={len(model.mtp_heads)} loss_weight={args.mtp_loss_weight:g}")
    model = apply_fp8_training(model, args, runtime)
    if args.compile:
        log("compiling model with torch.compile(dynamic=False)")
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    if is_dist() and args.optimizer == "muon":
        log("distributed model: replicated parameters with DistMuonAdamW gradient sync")
        return model, tokenizer, meta
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    return model, tokenizer, meta


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
    if args.train_mtp_heads:
        matrix_params += trainable(source.mtp_heads.parameters())
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
            "weight_decay": 0.01,
        },
        {
            "kind": "adamw",
            "params": embedding_params,
            "lr": args.embedding_lr * dmodel_lr_scale,
            "lr_multiplier": args.embedding_lr * dmodel_lr_scale / args.lr,
            "betas": (0.8, 0.995),
            "eps": 1e-10,
            "weight_decay": 0.001,
        },
    ]
    if scalar_params:
        groups.append(
            {
                "kind": "adamw",
                "params": scalar_params,
                "lr": args.scalar_lr,
                "lr_multiplier": args.scalar_lr / args.lr,
                "betas": (0.8, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            }
        )
    for shape in sorted({p.shape for p in matrix_params}):
        shape_params = [p for p in matrix_params if p.shape == shape]
        groups.append(
            {
                "kind": "muon",
                "params": shape_params,
                "lr": args.matrix_lr,
                "lr_multiplier": args.matrix_lr / args.lr,
                "momentum": args.muon_momentum,
                "weight_decay": args.weight_decay,
                "ns_steps": args.muon_ns_steps,
                "beta2": 0.9,
            }
        )
    optimizer_cls = DistMuonAdamW if is_dist() else MuonAdamW
    return optimizer_cls(groups)


def lr_multiplier(progress: float, args: argparse.Namespace) -> float:
    progress = max(0.0, min(1.0, progress))
    if args.warmup_ratio > 0 and progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if args.warmdown_ratio <= 0 or progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1.0 - decay) + decay * args.final_lr_frac


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@dataclass
class LossAccumulator:
    total: torch.Tensor
    main: torch.Tensor
    mtp: torch.Tensor
    tokens: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device, dtype: torch.dtype) -> "LossAccumulator":
        zero = torch.zeros((), device=device, dtype=dtype)
        return cls(total=zero.clone(), main=zero.clone(), mtp=zero.clone(), tokens=zero.clone())

    def add(
        self,
        total_loss: torch.Tensor,
        main_loss: torch.Tensor,
        mtp_loss: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        valid = valid_tokens.to(self.tokens.dtype)
        self.total += total_loss.detach().to(self.total.dtype) * valid
        self.main += main_loss.detach().to(self.main.dtype) * valid
        self.mtp += mtp_loss.detach().to(self.mtp.dtype) * valid
        self.tokens += valid

    def values(self, *, reduce: bool = False) -> tuple[float, float, float, int]:
        packed = torch.stack([self.total, self.main, self.mtp, self.tokens])
        if reduce:
            all_reduce_sum(packed)
        total, main, mtp, tokens = packed.tolist()
        return total, main, mtp, int(tokens)


def sft_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss_for_backward, main_loss_detached, mtp_loss_detached, valid_tokens_tensor).

    All returned tensors stay on-device; the caller can defer
    a single sync to log time by accumulating these into tensor buffers.
    """
    source = unwrap_model(model)
    use_mtp = args.train_mtp_heads and len(source.mtp_heads) > 0
    if use_mtp:
        logits, hidden = model(x, attention_mask=None, causal=True, return_hidden=True)
    else:
        logits = model(x, attention_mask=None, causal=True)
        hidden = None
    flat_y = y.reshape(-1)
    valid = (flat_y != IGNORE_INDEX).sum()
    main_loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        flat_y,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    main_loss = main_loss_sum / valid.clamp(min=1)

    mtp_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)
    if use_mtp and hidden is not None:
        mtp_hidden: torch.Tensor | None = None
        aux_terms: list[torch.Tensor] = []
        for depth, head in enumerate(source.mtp_heads, start=1):
            target_shift = depth
            if y.size(1) <= target_shift:
                break
            target = y[:, target_shift:]
            if source.config.mtp_arch == "deepseek":
                if depth == 1:
                    previous_hidden = hidden[:, :-target_shift, :]
                else:
                    if mtp_hidden is None:
                        break
                    previous_hidden = mtp_hidden[:, :-1, :]
                token_ids = x[:, depth:]
                cos_sin = (
                    source.cos[:, : previous_hidden.size(1)],
                    source.sin[:, : previous_hidden.size(1)],
                )
                mtp_hidden = head(
                    previous_hidden,
                    token_ids,
                    token_emb=source.token_emb,
                    cos_sin=cos_sin,
                )
                h_offset = mtp_hidden
            else:
                h_offset = norm(head(hidden[:, :-target_shift, :]))
            aux_logits = source.lm_head(h_offset)
            target_valid = (target != IGNORE_INDEX).sum()
            mtp_loss_sum = F.cross_entropy(
                aux_logits.reshape(-1, aux_logits.size(-1)),
                target.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            aux_terms.append(mtp_loss_sum / target_valid.clamp(min=1))
        if aux_terms:
            mtp_loss = torch.stack(aux_terms).mean()

    total = main_loss + args.mtp_loss_weight * mtp_loss
    return total, main_loss.detach(), mtp_loss.detach(), valid


@torch.no_grad()
def evaluate_sft(model: torch.nn.Module, batcher: SFTBatcher, args: argparse.Namespace, runtime: Runtime) -> dict[str, float]:
    batcher.cursor = rank()
    batcher.consumed = rank()
    batcher.conv_buffer.clear()
    model.eval()
    losses = LossAccumulator.empty(runtime.device, torch.float64)
    steps = max(1, args.eval_tokens // (args.batch_size * args.seq_len * world_size()))
    for _ in range(steps):
        x, y = batcher.next()
        with autocast_context(runtime.device, args.amp_dtype):
            with disable_fp8(unwrap_model(model)):
                total_loss, main_loss, mtp_loss, valid = sft_loss(unwrap_model(model), x, y, args)
        losses.add(total_loss, main_loss, mtp_loss, valid)
    model.train()
    _, main_sum, mtp_sum, tokens = losses.values(reduce=True)
    denom = max(1.0, float(tokens))
    return {
        "loss": main_sum / denom,
        "mtp_loss": mtp_sum / denom,
        "tokens": tokens,
    }


def log_wandb(wandb_run: Any, metrics: dict[str, float], step: int) -> None:
    if wandb_run is not None and is_main_process():
        wandb_run.log(metrics, step=step)


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    model, tokenizer, source_meta = load_model_and_tokenizer(args, runtime)
    train_dataset, val_dataset = build_datasets(args)
    tokens_per_micro = args.batch_size * args.seq_len * world_size()
    if args.total_batch_size % tokens_per_micro != 0:
        raise ValueError(f"--total-batch-size must be divisible by {tokens_per_micro:,}")
    args.grad_accum_steps = args.total_batch_size // tokens_per_micro
    args.tokens_per_step = args.total_batch_size
    args.wandb_project = args.wandb_project

    optimizer = build_optimizer(args, model, runtime)
    for group in optimizer.param_groups:
        group["lr"] *= args.init_lr_frac
        group["initial_lr"] = group["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=(runtime.device.type == "cuda" and args.amp_dtype == "float16"))
    wandb_run = init_wandb(args, unwrap_model(model).config, runtime)

    train_batcher = SFTBatcher(train_dataset, tokenizer, args, runtime, "train")
    val_batcher = SFTBatcher(val_dataset, tokenizer, args, runtime, "val")
    log(f"loaded checkpoint: {args.checkpoint} step={source_meta.get('step')}")
    log(f"train conversations: {len(train_dataset):,}; val conversations: {len(val_dataset):,}")
    log(f"tokens/micro={tokens_per_micro:,}; grad_accum={args.grad_accum_steps}; tokens/step={args.total_batch_size:,}")

    model.train()
    smooth_loss = 0.0
    total_time = 0.0
    step = 0
    while True:
        if train_batcher.last_step:
            break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        if args.eval_every > 0 and (step == 0 or step % args.eval_every == 0):
            metrics = evaluate_sft(model, val_batcher, args, runtime)
            log(
                f"step {step:05d} val_loss {metrics['loss']:.4f} "
                f"mtp_loss {metrics['mtp_loss']:.4f}"
            )
            log_wandb(
                wandb_run,
                {
                    "eval/loss": metrics["loss"],
                    "eval/mtp_loss": metrics["mtp_loss"],
                    "eval/tokens": metrics["tokens"],
                },
                step,
            )

        if args.chatcore_every > 0 and step > 0 and step % args.chatcore_every == 0:
            chatcore_metrics = evaluate_chatcore(model, tokenizer, args, runtime)
            if chatcore_metrics:
                log(
                    f"step {step:05d} chatcore {chatcore_metrics['chatcore']:.4f} "
                    f"cat {chatcore_metrics['chatcore_cat']:.4f}"
                )
                log_wandb(wandb_run, {f"eval/{k}": v for k, v in chatcore_metrics.items()}, step)

        progress = (step / args.max_steps) if args.max_steps > 0 else train_batcher.approx_progress
        lr = args.lr * lr_multiplier(progress, args)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        losses = LossAccumulator.empty(runtime.device, torch.float32)

        for micro_step in range(args.grad_accum_steps):
            x, y = train_batcher.next()
            sync_context = model.no_sync() if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1 else nullcontext()
            with sync_context:
                with autocast_context(runtime.device, args.amp_dtype):
                    loss, main_loss, mtp_loss, valid = sft_loss(model, x, y, args)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            losses.add(loss, main_loss, mtp_loss, valid)

        if is_dist():
            last_step_tensor = torch.tensor(int(train_batcher.last_step), dtype=torch.int32, device=runtime.device)
            dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
            train_batcher.last_step = bool(last_step_tensor.item())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in unwrap_model(model).parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        elapsed = time.time() - start
        total_time += elapsed
        step += 1
        total_loss_sum, main_loss_sum_v, mtp_loss_sum_v, valid_tokens = losses.values()
        denom = max(1, valid_tokens)
        train_loss = total_loss_sum / denom
        main_loss_val = main_loss_sum_v / denom
        mtp_loss_val = mtp_loss_sum_v / denom
        smooth_loss = main_loss_val if step == 1 else 0.9 * smooth_loss + 0.1 * main_loss_val
        log(
            f"step {step:05d} train_loss {train_loss:.4f} main {main_loss_val:.4f} "
            f"mtp {mtp_loss_val:.4f} smooth {smooth_loss:.4f} lr {lr:.2e} "
            f"tok/s {valid_tokens / max(elapsed, 1e-9):,.0f}"
        )
        log_wandb(
            wandb_run,
            {
                "train/loss": train_loss,
                "train/main_loss": main_loss_val,
                "train/mtp_loss": mtp_loss_val,
                "train/smooth_loss": smooth_loss,
                "train/lr": lr,
                "train/tokens": step * args.total_batch_size,
            },
            step,
        )

        if args.sample_every > 0 and step % args.sample_every == 0 and is_main_process():
            sample = sample_text(model, tokenizer, args, runtime)
            if wandb_run is not None:
                import wandb

                wandb_run.log({"sample/text": wandb.Html(f"<pre>{sample}</pre>")}, step=step)

        if is_main_process() and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=step, args=args)
            log(f"saved checkpoint: {args.out_dir / 'checkpoint.pt'}")

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

    if is_main_process():
        save_checkpoint(out_dir=args.out_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, step=step, args=args)
        log(f"saved final checkpoint: {args.out_dir / 'checkpoint.pt'}")
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
            log(f"uploaded final checkpoint to Hugging Face: {commit_url}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = step
            wandb_run.summary["total_training_time"] = total_time
            wandb_run.finish()


def main() -> None:
    args = parse_args()
    runtime = create_runtime(args)
    try:
        train(args, runtime)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
