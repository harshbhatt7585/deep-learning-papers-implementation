from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from core_eval import ensure_eval_bundle, evaluate_core
from flash_attention import describe_attention_backend
from model import (
    TextDiffusionConfig,
    TextDiffusionModel,
    generate,
    generate_causal,
    make_masked_inputs,
)
from nanochat_optim import DistMuonAdamW, MuonAdamW
from fp8 import disable_fp8
from utils import (
    Runtime,
    TokenData,
    Tokenizer,
    autocast_context,
    cleanup_distributed,
    create_runtime,
    get_batch,
    init_wandb,
    is_dist,
    is_main_process,
    learning_rate,
    log,
    save_checkpoint,
    set_lr,
    tokenize_data,
    unwrap_model,
    world_size,
)


SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


INTERNAL_DEFAULTS: dict[str, Any] = {
    "tokenizer": "nanochat",
    "tokenizer_local_files_only": False,
    "nanochat_tokenizer_vocab_size": 32_768,
    "nanochat_tokenizer_train_chars": 2_000_000_000,
    "nanochat_tokenizer_doc_cap": 10_000,
    "nanochat_tokenizer_threads": 4,
    "nanochat_tokenizer_batch_size": 128,
    "mask_prob": 0.30,
    "mtp_heads": 3,
    "mtp_loss_weight": 0.3,
    "weight_decay": 0.1,
    "aurora_weight_decay": 0.025,
    "warmup_steps": 50,
    "dropout": 0.1,
    "eval_interval": 200,
    "eval_batches": 20,
    "core_metric_every": 400,
    "core_eval_max_per_task": 500,
    "core_eval_cache_dir": Path("data/core_eval"),
    "log_interval": 1,
    "sample_interval": 200,
    "save_interval": 1000,
    "sample_prompt": "The ",
    "sample_length": 128,
    "sample_block_length": 32,
    "sample_steps": 32,
    "sample_threshold": 0.5,
    "sample_temperature": 0.6,
    "sample_top_k": 50,
    "sample_top_p": None,
    "compile_mode": "default",
    "fused_adamw": True,
    "fp8_recipe": "tensorwise",
    "matrix_lr": 0.02,
    "embedding_lr": 0.3,
    "unembedding_lr": 0.008,
    "scalar_lr": 0.5,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "aurora_pp_iterations": 2,
    "aurora_pp_beta": 0.5,
    "aurora_polar_steps": 12,
    "pin_memory": True,
    "seed": 0,
    "wandb_project": "text-diffusion",
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_group": None,
    "wandb_tags": None,
    "wandb_dir": Path("runs/wandb"),
}


def with_internal_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in INTERNAL_DEFAULTS.items():
        setattr(args, name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny text diffusion model.")

    parser.add_argument("--data", type=Path, help="Plain text dataset path.")
    parser.add_argument("--nanochat", action="store_true", help="Use nanochat's ClimbMix parquet dataset.")
    parser.add_argument("--stream-nanochat", action="store_true", help="Stream/tokenize nanochat parquet shards during training.")
    parser.add_argument("--nanochat-cache-dir", type=Path, default=Path("data/nanochat_climbmix"))
    parser.add_argument("--nanochat-train-shards", type=int, default=1)
    parser.add_argument("--max-train-chars", type=int, default=5_000_000)
    parser.add_argument("--max-val-chars", type=int, default=1_000_000)

    parser.add_argument("--nanochat-tokenizer-cache-dir", type=Path, default=Path("data/nanochat_tokenizer_32k"))
    parser.add_argument("--nanochat-tokenizer-vocab-size", type=int, default=None)
    parser.add_argument("--token-shards-dir", type=Path, default=None)

    parser.add_argument("--out-dir", type=Path, default=Path("runs/text-diffusion-nanochat"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--target-tokens", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=8.0)
    parser.add_argument("--objective", choices=["diffusion", "causal_mtp"], default="diffusion")
    parser.add_argument("--mtp-heads", type=int, default=None)
    parser.add_argument("--mtp-loss-weight", type=float, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "muon", "aurora"], default="adamw")
    parser.add_argument("--aurora-weight-decay", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")

    parsed = parser.parse_args()
    eval_interval = parsed.eval_interval
    nanochat_tokenizer_vocab_size = parsed.nanochat_tokenizer_vocab_size
    mtp_heads = parsed.mtp_heads
    mtp_loss_weight = parsed.mtp_loss_weight
    aurora_weight_decay = parsed.aurora_weight_decay
    args = with_internal_defaults(parsed)
    if eval_interval is not None:
        args.eval_interval = eval_interval
    if nanochat_tokenizer_vocab_size is not None:
        args.nanochat_tokenizer_vocab_size = nanochat_tokenizer_vocab_size
    if mtp_heads is not None:
        args.mtp_heads = mtp_heads
    if mtp_loss_weight is not None:
        args.mtp_loss_weight = mtp_loss_weight
    if aurora_weight_decay is not None:
        args.aurora_weight_decay = aurora_weight_decay
    return args


def build_config(args: argparse.Namespace, tokenizer: Tokenizer) -> TextDiffusionConfig:
    sample_prompt_len = max(
        len(tokenizer.encode(prompt))
        for prompt in [args.sample_prompt, *SAMPLE_PROMPTS]
    )
    sample_len = sample_prompt_len + args.sample_length
    max_seq_len = max(args.seq_len, sample_len + args.sample_block_length)
    return TextDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_mtp_heads=args.mtp_heads if args.objective == "causal_mtp" else 0,
    )


def tokens_per_step(args: argparse.Namespace) -> int:
    return args.batch_size * args.seq_len * args.grad_accum_steps * world_size()


def count_scaling_params(model: torch.nn.Module) -> int:
    source_model = unwrap_model(model)
    return sum(p.numel() for p in source_model.parameters())


def resolve_training_horizon(args: argparse.Namespace, model: torch.nn.Module) -> None:
    step_tokens = tokens_per_step(args)
    scaling_params = count_scaling_params(model)

    if args.max_steps > 0:
        total_tokens = args.max_steps * step_tokens
        horizon = f"max_steps={args.max_steps:,}"
    elif args.target_tokens > 0:
        args.max_steps = max(1, args.target_tokens // step_tokens)
        total_tokens = args.max_steps * step_tokens
        horizon = f"target_tokens={args.target_tokens:,}"
    elif args.target_param_data_ratio > 0:
        target_tokens = int(args.target_param_data_ratio * scaling_params)
        args.max_steps = max(1, target_tokens // step_tokens)
        total_tokens = args.max_steps * step_tokens
        horizon = f"target_param_data_ratio={args.target_param_data_ratio:g}"
        args.target_tokens = target_tokens
    else:
        raise ValueError("Set --max-steps, --target-tokens, or --target-param-data-ratio.")

    args.scaling_params = scaling_params
    args.tokens_per_step = step_tokens
    args.total_training_tokens = total_tokens
    log(
        "training_horizon: "
        f"{horizon} "
        f"scaling_params={scaling_params:,} "
        f"tokens_per_step={step_tokens:,} "
        f"max_steps={args.max_steps:,} "
        f"total_tokens={total_tokens:,} "
        f"tokens_per_scaling_param={total_tokens / scaling_params:.2f}"
    )


def load_training_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    runtime: Runtime,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=runtime.device, weights_only=False)
    source_model = unwrap_model(model)
    source_model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler_state = checkpoint.get("scaler_state")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    step = int(checkpoint.get("step", 0))
    log(f"resumed checkpoint: {checkpoint_path} at step {step:,}")
    return step


def fp8_module_filter(module: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if module.in_features % 16 != 0 or module.out_features % 16 != 0:
        return False
    if min(module.in_features, module.out_features) < 128:
        return False
    return True


def apply_fp8_training(model: torch.nn.Module, args: argparse.Namespace, runtime: Runtime) -> torch.nn.Module:
    if not args.fp8:
        return model
    if runtime.device.type != "cuda":
        raise ValueError("--fp8 requires a CUDA GPU with FP8 support, such as H100/H200.")
    if args.amp_dtype != "bfloat16":
        raise ValueError("--fp8 expects --amp-dtype bfloat16.")

    from fp8 import Float8Linear, Float8LinearConfig, convert_to_float8_training

    candidates = [
        name
        for name, module in model.named_modules()
        if fp8_module_filter(module, name)
    ]
    config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    model = model.to(dtype=torch.bfloat16)
    model = convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=fp8_module_filter,
    )
    converted = sum(1 for module in model.modules() if isinstance(module, Float8Linear))
    log(f"fp8: recipe={args.fp8_recipe} converted_linear_layers={converted}/{len(candidates)}")
    return model


def build_model(args: argparse.Namespace, config: TextDiffusionConfig, runtime: Runtime) -> torch.nn.Module:
    model: torch.nn.Module = TextDiffusionModel(config).to(runtime.device)
    model = apply_fp8_training(model, args, runtime)
    if args.compile:
        log("compiling model with torch.compile(dynamic=False)")
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    if is_dist() and args.optimizer in {"muon", "aurora"}:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)
        log(f"distributed model: using replicated model with DistMuonAdamW {args.optimizer} gradient sync")
        return model
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    return model


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, runtime: Runtime) -> torch.optim.Optimizer:
    source_model = unwrap_model(model)
    if args.optimizer in {"muon", "aurora"}:
        model_dim = source_model.config.d_model
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        embedding_params = list(source_model.token_emb.parameters())
        lm_head_params = list(source_model.lm_head.parameters())
        if getattr(source_model, "mtp_heads", None) is not None:
            lm_head_params += list(source_model.mtp_heads.parameters())
        matrix_params = list(source_model.blocks.parameters())
        seen_ids = {id(param) for param in embedding_params + lm_head_params + matrix_params}
        scalar_params = [param for param in source_model.parameters() if id(param) not in seen_ids]

        param_groups: list[dict[str, Any]] = [
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
            param_groups.append(
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
        matrix_kind = args.optimizer
        for shape in sorted({param.shape for param in matrix_params}):
            shape_params = [param for param in matrix_params if param.shape == shape]
            matrix_weight_decay = args.aurora_weight_decay if matrix_kind == "aurora" else args.weight_decay
            group = {
                "kind": matrix_kind,
                "params": shape_params,
                "lr": args.matrix_lr,
                "lr_multiplier": args.matrix_lr / args.lr,
                "momentum": args.muon_momentum,
                "weight_decay": matrix_weight_decay,
            }
            if matrix_kind == "muon":
                group.update(
                    {
                        "ns_steps": args.muon_ns_steps,
                        "beta2": 0.9,
                    }
                )
            else:
                group.update(
                    {
                        "pp_iterations": args.aurora_pp_iterations,
                        "pp_beta": args.aurora_pp_beta,
                        "polar_steps": args.aurora_polar_steps,
                    }
                )
            param_groups.append(group)

        optimizer_cls = DistMuonAdamW if is_dist() else MuonAdamW
        log(
            f"optimizer: {optimizer_cls.__name__} matrix_kind={matrix_kind} "
            f"matrix_params={sum(param.numel() for param in matrix_params):,} "
            f"lm_head_params={sum(param.numel() for param in lm_head_params):,} "
            f"embedding_params={sum(param.numel() for param in embedding_params):,} "
            f"scalar_params={sum(param.numel() for param in scalar_params):,}"
        )
        return optimizer_cls(param_groups)

    kwargs: dict[str, Any] = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
    }
    if args.fused_adamw and runtime.device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    masked = labels != -100
    return F.cross_entropy(logits[masked], labels[masked], reduction=reduction)


def causal_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
        reduction=reduction,
    )


def causal_mtp_cross_entropy(
    model: torch.nn.Module,
    source_model: TextDiffusionModel,
    input_ids: torch.Tensor,
    *,
    mtp_loss_weight: float,
) -> torch.Tensor:
    logits, hidden = model(input_ids, attention_mask=None, causal=True, return_hidden=True)
    main_loss = causal_cross_entropy(logits, input_ids)
    loss = main_loss
    for offset, head in enumerate(source_model.mtp_heads, start=2):
        if input_ids.size(1) <= offset:
            break
        aux_logits = head(hidden[:, :-offset, :])
        aux_loss = F.cross_entropy(
            aux_logits.contiguous().view(-1, aux_logits.size(-1)),
            input_ids[:, offset:].contiguous().view(-1),
        )
        loss = loss + mtp_loss_weight * aux_loss / max(1, len(source_model.mtp_heads))
    return loss


@torch.no_grad()
def estimate_eval_metrics(
    model: torch.nn.Module,
    data: torch.Tensor,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, float]:
    model.eval()
    source_model = unwrap_model(model)
    total_loss = 0.0
    total_masked_tokens = 0
    total_bytes = 0

    for _ in range(args.eval_batches):
        batch = get_batch(
            data,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runtime=runtime,
        )
        with autocast_context(runtime.device, args.amp_dtype):
            with disable_fp8(source_model):
                if args.objective == "causal_mtp":
                    logits = source_model(batch, attention_mask=None, causal=True)
                    loss_sum = causal_cross_entropy(logits, batch, reduction="sum")
                    token_count = batch.numel() - batch.size(0)
                else:
                    noised, labels = make_masked_inputs(
                        batch,
                        mask_token_id=source_model.config.mask_token_id,
                        pad_token_id=source_model.config.pad_token_id,
                        mask_prob=args.mask_prob,
                    )
                    logits = source_model(noised, attention_mask=None)
                    loss_sum = masked_cross_entropy(logits, labels, reduction="sum")
                    token_count = int((labels != -100).sum().item())

        total_loss += float(loss_sum.item())
        total_masked_tokens += token_count
        total_bytes += sum(
            len(tokenizer.decode(row.detach().cpu().tolist()).encode("utf-8"))
            for row in batch
        )

    model.train()
    totals_device = torch.device("cpu") if runtime.device.type == "mps" else runtime.device
    totals = torch.tensor(
        [total_loss, total_masked_tokens, total_bytes],
        dtype=torch.float64,
        device=totals_device,
    )
    if is_dist():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss = float(totals[0].item())
    total_masked_tokens = max(1.0, float(totals[1].item()))
    total_bytes = max(1.0, float(totals[2].item()))
    return {
        "loss": total_loss / total_masked_tokens,
        "masked_bpb": total_loss / (math.log(2) * total_bytes),
        "masked_tokens": total_masked_tokens,
    }


@torch.no_grad()
def estimate_core_metrics(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, float]:
    if args.core_metric_every <= 0:
        return {}

    model.eval()
    source_model = unwrap_model(model)
    if is_dist():
        if is_main_process():
            ensure_eval_bundle(args.core_eval_cache_dir)
        dist.barrier()

    with disable_fp8(source_model):
        results = evaluate_core(
            source_model,
            tokenizer,
            runtime.device,
            cache_dir=args.core_eval_cache_dir,
            max_per_task=args.core_eval_max_per_task,
            objective=args.objective,
        )
    model.train()
    return {
        "core": float(results["core_metric"]),
        **{
            f"core/{label}": float(value)
            for label, value in results["centered_results"].items()
        },
    }


@torch.no_grad()
def sample_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> str:
    source_model = unwrap_model(model)
    samples = []
    for prompt in SAMPLE_PROMPTS:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_bos=True),
            dtype=torch.long,
            device=runtime.device,
        )
        with disable_fp8(source_model):
            if args.objective == "causal_mtp":
                output = generate_causal(
                    source_model,
                    prompt_ids,
                    gen_length=args.sample_length,
                    temperature=args.sample_temperature,
                    top_k=args.sample_top_k,
                    top_p=args.sample_top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                output = generate(
                    source_model,
                    prompt_ids,
                    gen_length=args.sample_length,
                    block_length=args.sample_block_length,
                    steps=args.sample_steps,
                    threshold=args.sample_threshold,
                    editing_threshold=None,
                    temperature=args.sample_temperature,
                    top_k=args.sample_top_k,
                    top_p=args.sample_top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )
        samples.append(tokenizer.decode(output.detach().cpu(), skip_special=False))

    sample = "\n".join(samples)
    log("samples:\n" + sample)
    model.train()
    return sample


def log_startup(args: argparse.Namespace, data: TokenData, config: TextDiffusionConfig, model: torch.nn.Module, runtime: Runtime) -> None:
    train_chars = data.train_chars if data.train_chars is not None else len(data.train_text)
    val_chars = data.val_chars if data.val_chars is not None else len(data.train_text) - int(0.95 * len(data.train_text))
    log(f"device: {runtime.device}")
    log(f"world_size: {world_size()}")
    data_source = (
        "nanochat/climbmix-400b-shuffle streaming"
        if args.stream_nanochat
        else args.token_shards_dir or ("nanochat/climbmix-400b-shuffle" if args.nanochat else args.data)
    )
    log(f"data_source: {data_source}")
    log(f"tokenizer: {args.tokenizer}")
    log(f"train chars: {train_chars:,}")
    log(f"val chars: {val_chars:,}")
    log(f"train tokens: {data.train_tokens.numel():,}")
    log(f"val tokens: {data.val_tokens.numel():,}")
    log(f"vocab_size: {config.vocab_size:,}")
    log(f"parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}")
    log(f"objective: {args.objective}")
    if args.objective == "causal_mtp":
        log(f"mtp: heads={config.n_mtp_heads} loss_weight={args.mtp_loss_weight}")
    log(f"scaling_params: {args.scaling_params:,}")
    log(f"tokens_per_step: {args.tokens_per_step:,}")
    log(f"max_steps: {args.max_steps:,}")
    log(f"total_training_tokens: {args.total_training_tokens:,}")
    log(f"amp_dtype: {args.amp_dtype}")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16 if args.amp_dtype == "float16" else torch.float32
    log(f"attention_backend: {describe_attention_backend(masked=False, dtype=amp_dtype)}")
    log(f"compile: {args.compile}")
    log(f"fp8: {args.fp8}")
    log(f"eval_interval: {args.eval_interval}")
    log(
        "sample: "
        f"interval={args.sample_interval} "
        f"steps={args.sample_steps} "
        f"temperature={args.sample_temperature} "
        f"top_k={args.sample_top_k} "
        f"top_p={args.sample_top_p}"
    )


def log_train_metrics(wandb_run, step: int, metrics: dict[str, float]) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def create_wandb_sample_table(wandb_run):
    if wandb_run is None:
        return None

    import wandb

    return wandb.Table(
        columns=[
            "step",
            "prompt",
            "generated_text",
            "eval_loss",
            "eval_masked_bpb",
            "eval_masked_tokens",
            "sample_length",
            "block_length",
            "steps",
            "threshold",
            "temperature",
            "top_k",
            "top_p",
        ]
    )


def log_wandb_sample_table(
    wandb_run,
    table,
    *,
    step: int,
    sample: str,
    eval_metrics: dict[str, float] | None,
    args: argparse.Namespace,
) -> None:
    if wandb_run is None or table is None:
        return

    metrics = eval_metrics or {}
    table.add_data(
        step,
        args.sample_prompt,
        sample,
        metrics.get("loss"),
        metrics.get("masked_bpb"),
        metrics.get("masked_tokens"),
        args.sample_length,
        args.sample_block_length,
        args.sample_steps,
        args.sample_threshold,
        args.sample_temperature,
        args.sample_top_k,
        args.sample_top_p,
    )
    wandb_run.log({"eval/generated_samples": table}, step=step)


def train_one_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_tokens: torch.Tensor,
    args: argparse.Namespace,
    runtime: Runtime,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    source_model = unwrap_model(model)
    config = source_model.config
    step_loss = 0.0

    for micro_step in range(args.grad_accum_steps):
        batch = get_batch(
            train_tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            runtime=runtime,
        )
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1
            else nullcontext()
        )
        with sync_context:
            with autocast_context(runtime.device, args.amp_dtype):
                if args.objective == "causal_mtp":
                    loss = causal_mtp_cross_entropy(
                        model,
                        source_model,
                        batch,
                        mtp_loss_weight=args.mtp_loss_weight,
                    )
                else:
                    noised, labels = make_masked_inputs(
                        batch,
                        mask_token_id=config.mask_token_id,
                        pad_token_id=config.pad_token_id,
                        mask_prob=args.mask_prob,
                    )
                    logits = model(noised, attention_mask=None)
                    loss = masked_cross_entropy(logits, labels)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
        step_loss += loss.detach().item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return step_loss / args.grad_accum_steps


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    data = tokenize_data(args, runtime)
    config = build_config(args, data.tokenizer)
    model = build_model(args, config, runtime)
    resolve_training_horizon(args, model)
    optimizer = build_optimizer(args, model, runtime)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(runtime.device.type == "cuda" and args.amp_dtype == "float16"),
    )
    start_step = 0
    if args.resume is not None:
        resume_path = args.resume
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint.pt"
        start_step = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            runtime=runtime,
        )
        if start_step >= args.max_steps:
            log(f"checkpoint step {start_step:,} is already >= max_steps {args.max_steps:,}; nothing to train")
            return
    wandb_run = init_wandb(args, config, runtime)

    log_startup(args, data, config, model, runtime)
    if wandb_run is not None:
        wandb_run.summary["parameters"] = sum(p.numel() for p in unwrap_model(model).parameters())
        wandb_run.summary["train_tokens"] = data.train_tokens.numel()
        wandb_run.summary["val_tokens"] = data.val_tokens.numel()

    model.train()
    running_loss = 0.0
    last_log_time = time.time()
    latest_eval_metrics: dict[str, float] | None = None
    wandb_sample_table = create_wandb_sample_table(wandb_run)
    log("starting training loop")
    if args.compile:
        log("first compiled step can take several minutes")

    for step in range(start_step, args.max_steps):
        step_id = step + 1
        lr = learning_rate(
            step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
        )
        set_lr(optimizer, lr)

        step_loss = train_one_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_tokens=data.train_tokens,
            args=args,
            runtime=runtime,
        )
        running_loss += step_loss

        if args.log_interval > 0 and step_id % args.log_interval == 0:
            elapsed = time.time() - last_log_time
            tokens = args.tokens_per_step * args.log_interval
            tokens_per_second = tokens / max(elapsed, 1e-9)
            train_loss = running_loss / args.log_interval
            log(f"step {step_id:05d} train_loss {train_loss:.4f} lr {lr:.2e} tok/s {tokens_per_second:,.0f}")
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens": step_id * args.tokens_per_step,
                },
            )
            running_loss = 0.0
            last_log_time = time.time()

        if step == 0 or (args.eval_interval > 0 and step_id % args.eval_interval == 0):
            eval_metrics = estimate_eval_metrics(
                model,
                data.val_tokens,
                data.tokenizer,
                args,
                runtime,
            )
            latest_eval_metrics = eval_metrics
            log(
                f"step {step_id:05d} "
                f"train_loss {step_loss:.4f} "
                f"val_loss {eval_metrics['loss']:.4f} "
                f"masked_bpb {eval_metrics['masked_bpb']:.4f} "
                f"lr {lr:.2e}"
            )
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "eval/loss": eval_metrics["loss"],
                    "eval/masked_bpb": eval_metrics["masked_bpb"],
                    "eval/masked_tokens": eval_metrics["masked_tokens"],
                    "eval/train_loss_at_eval": step_loss,
                    "train/lr": lr,
                },
            )

        if args.core_metric_every > 0 and step_id % args.core_metric_every == 0:
            core_metrics = estimate_core_metrics(model, data.tokenizer, args, runtime)
            latest_eval_metrics = {
                **(latest_eval_metrics or {}),
                **core_metrics,
            }
            if core_metrics:
                log(f"step {step_id:05d} core {core_metrics['core']:.4f}")
                log_train_metrics(
                    wandb_run,
                    step_id,
                    {
                        "eval/core": core_metrics["core"],
                        **{
                            f"eval/{key}": value
                            for key, value in core_metrics.items()
                            if key != "core"
                        },
                    },
                )

        if is_main_process() and args.sample_interval > 0 and step_id % args.sample_interval == 0:
            sample = sample_text(model, data.tokenizer, args, runtime)
            if wandb_run is not None:
                import wandb

                wandb_run.log({"sample/text": wandb.Html(f"<pre>{sample}</pre>")}, step=step_id)
                log_wandb_sample_table(
                    wandb_run,
                    wandb_sample_table,
                    step=step_id,
                    sample=sample,
                    eval_metrics=latest_eval_metrics,
                    args=args,
                )

        if is_main_process() and args.save_interval > 0 and step_id % args.save_interval == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                model=model,
                tokenizer=data.tokenizer,
                optimizer=optimizer,
                scaler=scaler,
                step=step_id,
                args=args,
            )
            log(f"saved checkpoint: {args.out_dir / 'checkpoint.pt'}")
            log_train_metrics(wandb_run, step_id, {"checkpoint/step": step_id})

    if is_main_process():
        save_checkpoint(
            out_dir=args.out_dir,
            model=model,
            tokenizer=data.tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            step=args.max_steps,
            args=args,
        )
        log(f"saved final checkpoint: {args.out_dir / 'checkpoint.pt'}")
        if wandb_run is not None:
            wandb_run.summary["final_step"] = args.max_steps
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
