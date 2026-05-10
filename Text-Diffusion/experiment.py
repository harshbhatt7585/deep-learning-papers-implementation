from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiment_tracker import generate_experiment_report, load_metric_series, best_value


def report(args: argparse.Namespace) -> None:
    path = generate_experiment_report(args.run_dir, final_step=args.final_step)
    print(path)


def summarize_run(run_dir: Path) -> dict:
    manifest_path = run_dir / "experiment" / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    series = load_metric_series(run_dir)
    return {
        "name": manifest.get("name", run_dir.name),
        "objective": manifest.get("objective"),
        "optimizer": manifest.get("optimizer"),
        "mtp_heads": manifest.get("model", {}).get("n_mtp_heads"),
        "mtp_loss_weight": manifest.get("training", {}).get("mtp_loss_weight"),
        "eval_loss": best_value(series.get("eval/loss", []), mode="min"),
        "bpb": best_value(series.get("eval/bpb", []) or series.get("eval/masked_bpb", []), mode="min"),
        "core": best_value(series.get("eval/core", []), mode="max"),
        "tok_s": best_value(series.get("train/tokens_per_second", []), mode="max"),
    }


def compare(args: argparse.Namespace) -> None:
    rows = [summarize_run(path) for path in args.run_dirs]
    headers = ["run", "objective", "optimizer", "mtp", "weight", "best_loss", "best_bpb", "best_core", "best_tok/s"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        def fmt_step_value(item):
            if item is None:
                return ""
            step, value = item
            return f"{value:.5g} @ {step}"

        print(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["objective"]),
                    str(row["optimizer"]),
                    str(row["mtp_heads"]),
                    str(row["mtp_loss_weight"]),
                    fmt_step_value(row["eval_loss"]),
                    fmt_step_value(row["bpb"]),
                    fmt_step_value(row["core"]),
                    fmt_step_value(row["tok_s"]),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment report and comparison utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Regenerate a run report and SVG plots.")
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("--final-step", type=int, default=None)
    report_parser.set_defaults(func=report)

    compare_parser = subparsers.add_parser("compare", help="Compare experiment summaries.")
    compare_parser.add_argument("run_dirs", type=Path, nargs="+")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
