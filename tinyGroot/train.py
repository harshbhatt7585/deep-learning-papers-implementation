from __future__ import annotations

import argparse
import gc
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
    TinyGrootConfig,
    TinyGrootModel,
    generate_causal,
    norm,
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
    "mtp_heads": 3,
    "mtp_arch": "linear",
    "mtp_loss_weight": 0.3,
    "tst_bag_size": 1,
    "tst_ratio": 0.0,
    "ff_mult": 4,
    "weight_decay": 0.1,
    "aurora_weight_decay": 0.025,
    "warmup_steps": 50,
    "dropout": 0.0,
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
    "wandb_project": "tinyGroot",
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
    parser = argparse.ArgumentParser(description="Train the text MTP model.")

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

    parser.add_argument("--out-dir", type=Path, default=Path("runs/tinygroot-nanochat"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--target-tokens", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=8.0)
    parser.add_argument("--mtp-heads", type=int, default=None)
    parser.add_argument("--mtp-arch", choices=["linear", "deepseek"], default=None)
    parser.add_argument("--mtp-loss-weight", type=float, default=None)
    parser.add_argument("--tst-bag-size", type=int, default=None)
    parser.add_argument(
        "--tst-ratio",
        type=float,
        default=None,
        help="Fraction of training steps to run Token-Superposition Training before causal/MTP recovery.",
    )

    # DFlash drafter training (Chen et al. 2026, "Block Diffusion for Flash
    # Speculative Decoding"). When --dflash is set, we freeze a target
    # checkpoint, run it in eval mode each step to extract intermediate hidden
    # states, and train a small block-diffusion drafter on top. The drafter
    # shares the target's token_emb and lm_head; only its own layers train.
    parser.add_argument("--dflash", action="store_true", help="Train a DFlash drafter instead of the main MTP model.")
    parser.add_argument("--target-checkpoint", type=Path, default=None,
                        help="dflash: path to frozen target checkpoint.pt (required with --dflash)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="dflash: drafter block size (1 anchor + bs-1 mask slots). Default matches DFlash paper.")
    parser.add_argument("--n-draft-layers", type=int, default=2,
                        help="dflash: number of drafter decoder layers")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=None, help="MLP hidden expansion factor (default 4).")
    parser.add_argument("--gated-mlp", action="store_true", help="Use SwiGLU-style gated MLP instead of ReLU^2.")

    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "muon", "aurora"], default="adamw")
    parser.add_argument("--aurora-weight-decay", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--core-metric-every", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")

    parsed = parser.parse_args()
    eval_interval = parsed.eval_interval
    core_metric_every = parsed.core_metric_every
    sample_interval = parsed.sample_interval
    nanochat_tokenizer_vocab_size = parsed.nanochat_tokenizer_vocab_size
    mtp_heads = parsed.mtp_heads
    mtp_arch = parsed.mtp_arch
    mtp_loss_weight = parsed.mtp_loss_weight
    tst_bag_size = parsed.tst_bag_size
    tst_ratio = parsed.tst_ratio
    aurora_weight_decay = parsed.aurora_weight_decay
    ff_mult = parsed.ff_mult
    args = with_internal_defaults(parsed)
    if eval_interval is not None:
        args.eval_interval = eval_interval
    if core_metric_every is not None:
        args.core_metric_every = core_metric_every
    if sample_interval is not None:
        args.sample_interval = sample_interval
    if nanochat_tokenizer_vocab_size is not None:
        args.nanochat_tokenizer_vocab_size = nanochat_tokenizer_vocab_size
    if mtp_heads is not None:
        args.mtp_heads = mtp_heads
    if mtp_arch is not None:
        args.mtp_arch = mtp_arch
    if mtp_loss_weight is not None:
        args.mtp_loss_weight = mtp_loss_weight
    if tst_bag_size is not None:
        args.tst_bag_size = tst_bag_size
    if tst_ratio is not None:
        args.tst_ratio = tst_ratio
    if aurora_weight_decay is not None:
        args.aurora_weight_decay = aurora_weight_decay
    if ff_mult is not None:
        args.ff_mult = ff_mult
    return args


def build_config(args: argparse.Namespace, tokenizer: Tokenizer) -> TinyGrootConfig:
    sample_prompt_len = max(
        len(tokenizer.encode(prompt))
        for prompt in [args.sample_prompt, *SAMPLE_PROMPTS]
    )
    sample_len = sample_prompt_len + args.sample_length
    max_seq_len = max(args.seq_len, sample_len)

    n_heads = args.n_heads
    if args.dflash and args.d_model % n_heads != 0:
        # DFlash drafter inherits the target's d_model at bind time
        # (see `_build_dflash_drafter`), so this config is just a cosmetic
        # placeholder for logging / wandb metadata. If the user-chosen
        # n_heads doesn't divide the placeholder d_model (common when
        # overriding N_HEADS to match the target's head_dim while leaving
        # speed_run.sh's draft-mode D_MODEL=256 in place), fall back to the
        # largest divisor of d_model that is <= n_heads so we don't crash.
        # The drafter's *actual* n_heads (consumed by DFlashConfig) still
        # comes straight from args.n_heads downstream — this only affects
        # the unused placeholder config.
        for candidate in range(n_heads, 0, -1):
            if args.d_model % candidate == 0:
                n_heads = candidate
                break

    return TinyGrootConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        gated_mlp=args.gated_mlp,
        mtp_arch=args.mtp_arch,
        n_mtp_heads=0 if args.dflash else args.mtp_heads,
    )


# ---------------------------------------------------------------------------
# DFlash drafter training plumbing
# ---------------------------------------------------------------------------
# The frozen target is held at module scope so train_one_step / eval / sample
# can reach it without threading it through every helper signature. It's only
# set when --dflash is selected, and only the main training entry
# point (build_model -> _build_dflash) writes to it.
_DFLASH_TARGET: TinyGrootModel | None = None


def _load_frozen_target(
    checkpoint_path: Path,
    runtime: Runtime,
) -> TinyGrootModel:
    """Load a target TinyGrootModel checkpoint into eval/frozen mode."""
    blob = torch.load(checkpoint_path, map_location=runtime.device, weights_only=False)
    cfg_blob = blob.get("config")
    if cfg_blob is None:
        raise KeyError(f"target checkpoint {checkpoint_path} has no 'config' key")
    if isinstance(cfg_blob, TinyGrootConfig):
        cfg = cfg_blob
    else:
        cfg = TinyGrootConfig(**cfg_blob)
    state = blob["model_state"]
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    mtp_weight = cleaned.get("mtp_heads.0.weight")
    if (
        mtp_weight is not None
        and tuple(mtp_weight.shape) != (cfg.d_model, cfg.d_model)
    ):
        log(
            "[dflash] target checkpoint uses legacy full-vocab MTP heads; "
            "ignoring MTP heads for DFlash target loading"
        )
        cfg.n_mtp_heads = 0

    target = TinyGrootModel(cfg).to(runtime.device)
    missing, unexpected = target.load_state_dict(cleaned, strict=False)
    if missing:
        log(f"[dflash] target missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log(f"[dflash] target unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    target.eval()
    for param in target.parameters():
        param.requires_grad = False
    log(
        f"[dflash] target loaded from {checkpoint_path}: "
        f"d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
        f"params={sum(p.numel() for p in target.parameters()):,}"
    )
    return target


def _build_dflash_config(args: argparse.Namespace, target: TinyGrootModel) -> "Any":
    from dflash_model import DFlashConfig

    # Force drafter n_heads to match the target. The drafter shares d_model
    # with the target (it's bound at runtime), so n_heads completely determines
    # head_dim -- and head_dim controls RoPE's per-channel rotation frequency.
    # If the drafter's head_dim differs from the target's, the same absolute
    # position gets a different rotation in each model, so the drafter has to
    # learn its own positional code that is subtly inconsistent with the
    # representation it cross-attends to in x_ctx. q_proj/o_proj parameter
    # counts are identical (d_model -> d_model either way), so this is free.
    # ``args.n_heads`` is ignored for the drafter (still respected by the
    # cosmetic placeholder TinyGrootConfig used for logging).
    drafter_n_heads = target.config.n_heads
    drafter_n_kv_heads = target.config.n_kv_heads or target.config.n_heads
    return DFlashConfig(
        target_d_model=target.config.d_model,
        target_vocab_size=target.config.vocab_size,
        target_n_layers=target.config.n_layers,
        target_n_heads=target.config.n_heads,
        target_pad_token_id=target.config.pad_token_id,
        mask_token_id=target.config.mask_token_id,
        block_size=args.block_size,
        n_draft_layers=args.n_draft_layers,
        n_heads=drafter_n_heads,
        n_kv_heads=drafter_n_kv_heads,
        ff_mult=args.ff_mult,
        gated_mlp=args.gated_mlp,
        dropout=args.dropout,
        max_seq_len=max(args.seq_len, target.config.max_seq_len),
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
    args.tst_steps = int(args.max_steps * args.tst_ratio) if args.tst_bag_size > 1 and args.tst_ratio > 0 else 0
    args.effective_training_tokens = total_tokens + args.tst_steps * step_tokens * max(0, args.tst_bag_size - 1)
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
    if args.tst_steps > 0:
        log(
            "tst_horizon: "
            f"bag_size={args.tst_bag_size} "
            f"ratio={args.tst_ratio:g} "
            f"steps={args.tst_steps:,} "
            f"effective_raw_tokens={args.effective_training_tokens:,} "
            f"effective_tokens_per_scaling_param={args.effective_training_tokens / scaling_params:.2f}"
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


def build_model(args: argparse.Namespace, config: TinyGrootConfig, runtime: Runtime) -> torch.nn.Module:
    if args.dflash:
        return _build_dflash_drafter(args, runtime)
    model: torch.nn.Module = TinyGrootModel(config).to(runtime.device)
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


def _build_dflash_drafter(args: argparse.Namespace, runtime: Runtime) -> torch.nn.Module:
    """Load+freeze a target, build a DFlash drafter, bind, return drafter.

    Side effect: sets the module-level ``_DFLASH_TARGET`` reference so
    ``train_one_step`` / eval can reach the frozen target without threading.
    """
    global _DFLASH_TARGET
    from dflash_model import DFlashDraftModel

    if args.target_checkpoint is None:
        raise SystemExit("--dflash requires --target-checkpoint <path>")
    if not args.target_checkpoint.exists():
        raise SystemExit(f"--target-checkpoint not found: {args.target_checkpoint}")

    target = _load_frozen_target(args.target_checkpoint, runtime)
    drafter_config = _build_dflash_config(args, target)
    drafter = DFlashDraftModel(drafter_config).to(runtime.device)
    drafter.bind(target)
    _DFLASH_TARGET = target

    drafter_head_dim = drafter_config.d_model // drafter_config.n_heads
    target_head_dim = target.config.d_model // target.config.n_heads
    log(
        f"[dflash] drafter built: d_model={drafter_config.d_model} "
        f"n_draft_layers={drafter_config.n_draft_layers} "
        f"n_heads={drafter_config.n_heads} (head_dim={drafter_head_dim}, "
        f"target head_dim={target_head_dim}) "
        f"n_kv_heads={drafter_config.n_kv_heads} "
        f"target_layer_ids={drafter_config.target_layer_ids} "
        f"block_size={drafter_config.block_size} "
        f"owned_params={drafter.num_owned_parameters():,}"
    )

    # Drafter is small — torch.compile is optional and FP8 won't help much
    # (most layers are below the 128-dim FP8 threshold). Wrap in DDP if needed.
    if args.compile:
        log("[dflash] compiling drafter with torch.compile(dynamic=False)")
        drafter = torch.compile(drafter, mode=args.compile_mode, dynamic=False)
    if is_dist():
        ddp_kwargs = {"device_ids": [runtime.local_rank]} if runtime.device.type == "cuda" else {}
        drafter = DDP(drafter, **ddp_kwargs)
    return drafter


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, runtime: Runtime) -> torch.optim.Optimizer:
    source_model = unwrap_model(model)
    if args.dflash:
        # Drafter has a different module layout (no token_emb / lm_head /
        # blocks). Sidestep the muon/aurora group construction and use plain
        # AdamW over the drafter's owned parameters.
        return torch.optim.AdamW(
            source_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
    if args.optimizer in {"muon", "aurora"}:
        model_dim = source_model.config.d_model
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        embedding_params = list(source_model.token_emb.parameters())
        lm_head_params = list(source_model.lm_head.parameters())
        matrix_params = list(source_model.blocks.parameters())
        # DeepSeek-V3 style MTP heads are d_model x d_model projections that
        # feed the shared lm_head, so they belong with the other matrix params
        # (muon group), not the unembedding-vocab group.
        if getattr(source_model, "mtp_heads", None) is not None:
            matrix_params += list(source_model.mtp_heads.parameters())
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


def causal_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().view(-1),
        reduction=reduction,
    )


def causal_mtp_cross_entropy(
    model: torch.nn.Module,
    source_model: TinyGrootModel,
    input_ids: torch.Tensor,
    *,
    mtp_loss_weight: float,
) -> torch.Tensor:
    logits, hidden = model(input_ids, attention_mask=None, causal=True, return_hidden=True)
    main_loss = causal_cross_entropy(logits, input_ids)
    loss = main_loss
    mtp_hidden: torch.Tensor | None = None
    for depth, head in enumerate(source_model.mtp_heads, start=1):
        offset = depth + 1
        if input_ids.size(1) <= offset:
            break
        if source_model.config.mtp_arch == "deepseek":
            if depth == 1:
                previous_hidden = hidden[:, :-offset, :]
            else:
                previous_hidden = mtp_hidden[:, :-1, :]
            token_ids = input_ids[:, depth:-1]
            cos_sin = (
                source_model.cos[:, : previous_hidden.size(1)],
                source_model.sin[:, : previous_hidden.size(1)],
            )
            mtp_hidden = head(
                previous_hidden,
                token_ids,
                token_emb=source_model.token_emb,
                cos_sin=cos_sin,
            )
            h_offset = mtp_hidden
        else:
            h_offset = norm(head(hidden[:, :-offset, :]))
        aux_logits = source_model.lm_head(h_offset)
        aux_loss = F.cross_entropy(
            aux_logits.contiguous().view(-1, aux_logits.size(-1)),
            input_ids[:, offset:].contiguous().view(-1),
        )
        loss = loss + mtp_loss_weight * aux_loss / max(1, len(source_model.mtp_heads))
    return loss


def token_superposition_cross_entropy(
    model: torch.nn.Module,
    source_model: TinyGrootModel,
    input_ids: torch.Tensor,
    *,
    bag_size: int,
) -> torch.Tensor:
    """Token-Superposition Training loss.

    Non-overlapping input bags of ``bag_size`` raw tokens are averaged inside
    the model embedding layer. Each latent bag predicts the next raw token bag
    with the simplified multi-hot CE objective from the TST paper.
    """
    if bag_size <= 1:
        logits = model(input_ids, attention_mask=None, causal=True)
        return causal_cross_entropy(logits, input_ids)

    usable_len = (input_ids.size(1) // bag_size) * bag_size
    if usable_len < 2 * bag_size:
        raise ValueError(
            f"tst requires at least two bags: seq_len={input_ids.size(1)} bag_size={bag_size}"
        )

    bags = input_ids[:, :usable_len].view(input_ids.size(0), usable_len // bag_size, bag_size)
    input_bags = bags[:, :-1, :]
    target_bags = bags[:, 1:, :]
    logits = model(input_bags, attention_mask=None, causal=True)

    log_den = torch.logsumexp(logits, dim=-1)
    target_logit_sum = logits.gather(-1, target_bags).sum(dim=-1)
    loss = (log_den.float() - target_logit_sum.float() / bag_size).mean()
    if len(source_model.mtp_heads) > 0:
        loss = loss + logits.new_zeros(()) * sum(
            param.sum() for param in source_model.mtp_heads.parameters()
        )
    return loss


def use_tst_phase(args: argparse.Namespace, step_id: int) -> bool:
    if args.tst_bag_size <= 1 or args.tst_ratio <= 0.0:
        return False
    return step_id <= int(args.max_steps * args.tst_ratio)


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

    if args.dflash:
        # Drafter eval: dflash_loss over a few val batches, plus mean
        # acceptance-proxy (drafter argmax == ground-truth at masked positions).
        from dflash_model import dflash_loss

        if _DFLASH_TARGET is None:
            raise RuntimeError("dflash eval requires _DFLASH_TARGET")
        total_loss = 0.0
        total_acc = 0.0
        total_valid = 0
        for _ in range(args.eval_batches):
            batch = get_batch(
                data,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                runtime=runtime,
            )
            with autocast_context(runtime.device, args.amp_dtype):
                loss, metrics = dflash_loss(source_model, _DFLASH_TARGET, batch)
            n_valid = int(metrics["dflash_valid_positions"].item())
            total_loss += float(loss.item()) * max(1, n_valid)
            total_acc += float(metrics["dflash_acceptance_proxy"].item()) * max(1, n_valid)
            total_valid += max(1, n_valid)
        model.train()
        return {
            "loss": total_loss / max(1, total_valid),
            "bpb": 0.0,  # not meaningful for dflash; keep key shape consistent
            "tokens": float(total_valid),
            "dflash_acceptance_proxy": total_acc / max(1, total_valid),
        }

    total_loss = 0.0
    total_tokens = 0
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
                logits = source_model(batch, attention_mask=None, causal=True)
                loss_sum = causal_cross_entropy(logits, batch, reduction="sum")
                token_count = batch.numel() - batch.size(0)

        total_loss += float(loss_sum.item())
        total_tokens += token_count
        total_bytes += sum(
            len(tokenizer.decode(row.detach().cpu().tolist()).encode("utf-8"))
            for row in batch
        )

    model.train()
    totals_device = torch.device("cpu") if runtime.device.type == "mps" else runtime.device
    totals = torch.tensor(
        [total_loss, total_tokens, total_bytes],
        dtype=torch.float64,
        device=totals_device,
    )
    if is_dist():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss = float(totals[0].item())
    total_tokens = max(1.0, float(totals[1].item()))
    total_bytes = max(1.0, float(totals[2].item()))
    return {
        "loss": total_loss / total_tokens,
        "bpb": total_loss / (math.log(2) * total_bytes),
        "tokens": total_tokens,
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
def sample_text_dflash(
    drafter_model: torch.nn.Module,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> tuple[str, dict[str, float]]:
    """Spec-decode sampling for the DFlash drafter.

    DFlash analogue of ``sample_text``: runs the drafter against the frozen
    target on each prompt in ``SAMPLE_PROMPTS`` via ``speculate_dflash`` and
    reports both the generated text and a small batch of acceptance metrics.

    Why this matters during training:

      1. Qualitative — confirms the drafter is actually producing target-like
         text and not collapsing into a mode (e.g. EOS-spam, repetition).
      2. Quantitative — tracks ``acceptance_rate`` and ``accepted_per_block``
         step-by-step. These are the metrics we'll be scored on at inference;
         seeing them as a training curve is the fastest way to spot
         under/over-training and bad hyperparameters.

    Temperature is pinned to 0.0 so acceptance is a clean signal (no sampling
    noise). The drafter is set to ``eval()`` for the duration and returned to
    ``train()`` before this function exits.
    """
    from spec_decode import speculate_dflash

    if _DFLASH_TARGET is None:
        raise RuntimeError(
            "dflash sampling requires _DFLASH_TARGET; was build_model not called?"
        )

    source_drafter = unwrap_model(drafter_model)
    drafter_model.eval()

    samples: list[str] = []
    total_drafts_accepted = 0
    total_drafts_proposed = 0
    total_committed = 0  # sum of (acceptance_length + 1) across blocks
    total_blocks = 0
    total_target_forwards = 0
    total_tokens_generated = 0

    block_size = source_drafter.config.block_size
    for prompt in SAMPLE_PROMPTS:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_bos=True),
            dtype=torch.long,
            device=runtime.device,
        )
        output, stats = speculate_dflash(
            _DFLASH_TARGET,
            source_drafter,
            prompt_ids,
            gen_length=args.sample_length,
            block_size=block_size,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        samples.append(tokenizer.decode(output.detach().cpu(), skip_special=False))
        total_drafts_accepted += stats.drafts_accepted
        total_drafts_proposed += stats.drafts_proposed
        total_committed += sum(stats.per_step_accepted)
        total_blocks += len(stats.per_step_accepted)
        total_target_forwards += stats.target_forwards
        total_tokens_generated += stats.tokens_generated

    metrics = {
        "sample_acceptance_rate": total_drafts_accepted / max(1, total_drafts_proposed),
        "sample_accepted_per_block": total_committed / max(1, total_blocks),
        "sample_tokens_per_forward": total_tokens_generated / max(1, total_target_forwards),
        "sample_blocks": float(total_blocks),
        "sample_block_size": float(block_size),
    }
    sample_str = "\n".join(samples)
    log("samples:\n" + sample_str)
    log(
        "sample acceptance: "
        f"rate={metrics['sample_acceptance_rate'] * 100:.1f}% "
        f"accepted/block={metrics['sample_accepted_per_block']:.2f}/{block_size} "
        f"tokens/forward={metrics['sample_tokens_per_forward']:.2f} "
        f"(over {total_blocks} blocks)"
    )
    drafter_model.train()
    return sample_str, metrics


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
            output = generate_causal(
                source_model,
                prompt_ids,
                gen_length=args.sample_length,
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


def log_startup(args: argparse.Namespace, data: TokenData, config: TinyGrootConfig, model: torch.nn.Module, runtime: Runtime) -> None:
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
    log(f"mode: {'dflash' if args.dflash else 'causal_mtp'}")
    if not args.dflash:
        log(f"mtp: arch={config.mtp_arch} heads={config.n_mtp_heads} loss_weight={args.mtp_loss_weight}")
    if args.tst_steps > 0:
        log(
            "tst: "
            f"bag_size={args.tst_bag_size} "
            f"ratio={args.tst_ratio:g} "
            f"steps={args.tst_steps:,} "
            "then causal_mtp recovery"
        )
    log(f"mlp: kind={'gated_swiglu' if config.gated_mlp else 'relu2'} ff_mult={config.ff_mult}")
    log(f"scaling_params: {args.scaling_params:,}")
    log(f"tokens_per_step: {args.tokens_per_step:,}")
    log(f"max_steps: {args.max_steps:,}")
    log(f"total_training_tokens: {args.total_training_tokens:,}")
    if args.tst_steps > 0:
        log(f"effective_training_tokens: {args.effective_training_tokens:,}")
    log(f"amp_dtype: {args.amp_dtype}")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16 if args.amp_dtype == "float16" else torch.float32
    log(f"attention_backend: {describe_attention_backend(masked=False, dtype=amp_dtype)}")
    log(f"compile: {args.compile}")
    log(f"fp8: {args.fp8}")
    log(f"eval_interval: {args.eval_interval}")
    log(
        "sample: "
        f"interval={args.sample_interval} "
        f"length={args.sample_length} "
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
            "eval_bpb",
            "eval_tokens",
            "sample_length",
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
        metrics.get("bpb"),
        metrics.get("tokens"),
        args.sample_length,
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
    step_id: int,
    args: argparse.Namespace,
    runtime: Runtime,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    source_model = unwrap_model(model)
    step_loss = 0.0

    for micro_step in range(args.grad_accum_steps):
        tst_active = use_tst_phase(args, step_id)
        batch_seq_len = args.seq_len * args.tst_bag_size if tst_active else args.seq_len
        batch = get_batch(
            train_tokens,
            batch_size=args.batch_size,
            seq_len=batch_seq_len,
            runtime=runtime,
        )
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and micro_step < args.grad_accum_steps - 1
            else nullcontext()
        )
        with sync_context:
            with autocast_context(runtime.device, args.amp_dtype):
                if tst_active:
                    loss = token_superposition_cross_entropy(
                        model,
                        source_model,
                        batch,
                        bag_size=args.tst_bag_size,
                    )
                elif not args.dflash:
                    loss = causal_mtp_cross_entropy(
                        model,
                        source_model,
                        batch,
                        mtp_loss_weight=args.mtp_loss_weight,
                    )
                elif args.dflash:
                    from dflash_model import dflash_loss

                    if _DFLASH_TARGET is None:
                        raise RuntimeError(
                            "dflash training requires _DFLASH_TARGET; was build_model not called?"
                        )
                    loss, _metrics = dflash_loss(
                        source_model,
                        _DFLASH_TARGET,
                        batch,
                    )
                else:
                    raise RuntimeError("unreachable training mode")
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
    if args.tst_bag_size < 1:
        raise ValueError("--tst-bag-size must be >= 1")
    if not 0.0 <= args.tst_ratio < 1.0:
        raise ValueError("--tst-ratio must be in [0, 1) — recovery phase requires ratio < 1")
    if args.tst_ratio > 0.0 and args.dflash:
        raise ValueError("TST pretraining is supported only for main causal MTP training")
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
            step_id=step_id,
            args=args,
            runtime=runtime,
        )
        running_loss += step_loss

        if args.log_interval > 0 and step_id % args.log_interval == 0:
            elapsed = time.time() - last_log_time
            current_tokens_per_step = args.tokens_per_step * (
                args.tst_bag_size if use_tst_phase(args, step_id) else 1
            )
            tokens = current_tokens_per_step * args.log_interval
            tokens_per_second = tokens / max(elapsed, 1e-9)
            train_loss = running_loss / args.log_interval
            phase = "tst" if use_tst_phase(args, step_id) else ("dflash" if args.dflash else "causal_mtp")
            log(f"step {step_id:05d} phase {phase} train_loss {train_loss:.4f} lr {lr:.2e} tok/s {tokens_per_second:,.0f}")
            raw_tokens_seen = step_id * args.tokens_per_step
            if args.tst_steps > 0:
                raw_tokens_seen += min(step_id, args.tst_steps) * args.tokens_per_step * (args.tst_bag_size - 1)
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens": raw_tokens_seen,
                    "train/phase_is_tst": 1.0 if phase == "tst" else 0.0,
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
                f"bpb {eval_metrics['bpb']:.4f} "
                f"lr {lr:.2e}"
            )
            log_train_metrics(
                wandb_run,
                step_id,
                {
                    "eval/loss": eval_metrics["loss"],
                    "eval/bpb": eval_metrics["bpb"],
                    "eval/tokens": eval_metrics["tokens"],
                    "eval/train_loss_at_eval": step_loss,
                    "train/lr": lr,
                },
            )

        if not args.dflash and args.core_metric_every > 0 and step_id % args.core_metric_every == 0:
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

        if (
            is_main_process()
            and args.sample_interval > 0
            and step_id % args.sample_interval == 0
        ):
            if args.dflash:
                sample, sample_metrics = sample_text_dflash(
                    model, data.tokenizer, args, runtime
                )
                latest_eval_metrics = {
                    **(latest_eval_metrics or {}),
                    **sample_metrics,
                }
                if wandb_run is not None:
                    import wandb

                    wandb_run.log(
                        {
                            "sample/text": wandb.Html(f"<pre>{sample}</pre>"),
                            **{
                                f"sample/{key}": value
                                for key, value in sample_metrics.items()
                            },
                        },
                        step=step_id,
                    )
            else:
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

        # The garbage collector spends ~500ms scanning for cycles quite frequently.
        # We manually manage it to avoid these pauses during training.
        if step == start_step:
            gc.collect()  # manually collect a lot of garbage from setup
            gc.freeze()  # freeze all currently surviving objects and exclude them from GC
            gc.disable()  # disable GC entirely except:
        elif step_id % 5000 == 0:
            gc.collect()  # manually collect, just to be safe for very long runs

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
