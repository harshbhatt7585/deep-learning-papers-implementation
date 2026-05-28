from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygroot.core_eval import evaluate_core
from tinygroot.infer.sample import load_checkpoint
from tinygroot.utils import cleanup_distributed, configure_cuda, is_main_process, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a text MTP checkpoint on nanochat CORE tasks.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-cache-dir", type=Path, default=Path("data/core_eval"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--max-per-task", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = setup_distributed()
    configure_cuda()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        model, tokenizer, checkpoint = load_checkpoint(args.checkpoint_dir, device)
        model.eval()
        results = evaluate_core(
            model,
            tokenizer,
            device,
            cache_dir=args.eval_cache_dir,
            max_per_task=args.max_per_task,
        )

        if is_main_process():
            out_csv = args.out_csv or (args.checkpoint_dir / f"core_eval_step_{checkpoint['step']:06d}.csv")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", encoding="utf-8") as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in results["results"]:
                    acc = results["results"][label]
                    centered = results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {results['core_metric']:<10.6f}\n")
            print(f"CORE metric: {results['core_metric']:.4f}", flush=True)
            print(f"Results written to: {out_csv}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
