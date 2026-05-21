"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

If ~/.cache/nanochat/base_checkpoints is empty, train a tiny base run or pull weights:

  bash runs/quick_base_checkpoint.sh
  python -m scripts.chat_sft

  # or: python -m scripts.pull_hf_checkpoint && python -m scripts.chat_sft
"""

import gc
import argparse
import os

from torch.utils.data import dataset
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default="d12", help="model tag to load from (subfolder under base_checkpoints/)")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

print(token_bytes)

# intialised the Optimizer (combined MounAdamW: Moun for matrix params, AdamW for rest)
# Not that pretraining ramps weight_decay to zero by end of pretraining, so SFT contnies with zero
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffer etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")

    else:
        print0("WARNING: optimizer checkpoint not found,starting with fresh optimizer (slightly worse)")
    


scalar = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scalar is not None:
    print0("GradScaler enabled for fp16 training")



# Ovverides the inital learning rate as a fraction of the base learning tate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["inital_lr"] = group["lr"]


# SFT data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [
    SmolTalk(split="train"),
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    SimpleSpelling(size=200000, split="train"),
    SpellingBee(size=80000, split="train")
]

train_dataset = TaskMixture(train_tasks)
print0(f"Training mixutre: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GS8K x{args.gsm8k_epochs})")
val_dataset = TaskMixture([
    SmolTalk(split="test"),
    MMLU(subset="all", split="test", stop=5200),
    GSM8K(subset="main", split="test", stop=420)
]) # total: 24K = 5.2K + 0.42K ~= 29.6K

last_step = False
approx_progress = 0.0
current_epoch = 1


# print(train_dataset[0])

def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfiit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Converrsations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """

    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1 # +1 or target at last positon
    bos_token = tokenizer.get_bos_token_id()

    # conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank # each rank processes different conversations (for fetching)
    consumed = ddp_rank # Track actual consumption seperatly from buffering
    epoch = 1
    it = 0 # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumptio, not fetching

    
    while True:
        rows = []
        mask_rows = []
        row_length = [] # track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                
                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirly
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                
                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining) # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break # Row is now full (with padding)
            
            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_length.append(content_len)
            else:
                row_length.append(row_capacity)
            
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])
    
        # Stopping condition to resepect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True
        
        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursoe wraps)
        
            if consumed >= dataset_size:
                last_step = True
            
        
        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensors = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensors[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensors[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,)
        # mask=0 for user prompts, BOS, special tokens, tool outputs
        # with targets (shiftd by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_length):
            if content_len > row_capacity:
                targets[i, content_len-1:] = -1
            
        yield inputs, targets
    

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0  # will go from 0 to 1 over the course of the epoch
# print(next(train_loader)[0])
# print(next(train_loader)[1])

# learning rate schedule (linear warmup,  constant, linearn warmdown)
# same shape as base_train uses progess (0->1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# Training loop -------
x, y = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float('inf')
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-click time of training
step = 0

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step
    print(f"FLOPS SO FAR", flops_so_far)

    # Syncronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())


    # if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
    #     model.eval()
    #     val_loader = build_val_loader()
    #     eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
    #     val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    #     print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
    #     if val_bpb < min_val_bpb:
    #         min_val_bpb = val_bpb
        
    #     wandb_run.log({
    #         "step": step,
    #         "total_training_flops": flops_so_far,
    #         "total_training_time": total_training_time,
    #         "val/bpb": val_bpb
    #     })
    #     model.train()
    
    # Once in a while: estimate that ChatCORE metrics (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # chatcore_results = {}
    # if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
    #     model.eval()
    #     engine = Engine(orig_model, tokenizer)
    #     all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    #     categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
    #     baseline_accuracies = {
    #         'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
    #         'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
    #     }
    #     task_results = {}
    #     for task_name in all_tasks:
    #         limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
    #         max_problems = None if limit < 0 else limit # -1 means no limit
    #         acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
    #                             batch_size=args.device_batch_size, max_problems=max_problems)
    #         task_results[task_name] = acc
    #         print0(f"  {task_name}: {100*acc:.2f}%")
        
    #     # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
    #     def centered_mean(tasks):
    #         return sum((task_results[t] - baseline_accuracies[t] / (1.0 - baseline_accuracies) for t in tasks))
    #     chatcore = centered_mean(all_tasks)
    #     chatcore_cat = centered_mean(categorical_tasks)
    #     print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
    #     wandb_run.log({
    #         "step": step,
    #         "total_training_flops": flops_so_far,
    #         "chatcore_metric": chatcore,
    #         "chatcore_cat": chatcore_cat,
    #         **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
    #     })
    #     model.train()


    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirnmae)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.max_seq_len,
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_mebed,
                    "window_pattern": model.config.window_pattern
                },
            "user_config": user_config # inputs to the training script
            },
            rank=ddp_rank
        )
    if last_step:
        break

    # single training step 
    # evalaute the gradient
    synchronize()
    t0 = time.time()
    model.train()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        # print(train_loss)
        loss = loss / grad_accum_steps
        if scalar is not None:
            scalar.scale(loss).backward()
        else:
            loss.backward()
        # x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        x, y = next(train_loader)
        progress = max(progress, approx_progress) # only increase progress monotonically
    
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    print("LRM:", lrm)
    moun_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["inital_lr"] * lrm
        if group["kind"] == "moun":
            group["momentum"] = moun_momentum

    if scalar is not None:
        scalar.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scalar._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
            scalar.step(optimizer)
            scalar.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1
    
    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step+1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size) 
    
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })
    
    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except
    elif step % 500 == 0:
        gc.collect()


# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")



# Log to report
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()



    
