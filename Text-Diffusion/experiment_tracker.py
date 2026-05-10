from __future__ import annotations

import json
import math
import time
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Namespace):
        return _plain(vars(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, float]) -> None:
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten(name, value, out)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[name] = float(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(row), sort_keys=True) + "\n")


def _line_chart_svg(
    series: list[tuple[int, float]],
    *,
    title: str,
    y_label: str,
    width: int = 840,
    height: int = 320,
) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 64, 20, 42, 46
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    if len(series) < 2:
        return ""

    xs = [step for step, _ in series]
    ys = [value for _, value in series]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0

    def sx(x: float) -> float:
        return margin_left + (x - min_x) / max(max_x - min_x, 1) * plot_w

    def sy(y: float) -> float:
        return margin_top + (max_y - y) / max(max_y - min_y, 1e-12) * plot_h

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in series)
    grid = []
    labels = []
    for i in range(5):
        t = i / 4
        y = margin_top + t * plot_h
        value = max_y - t * (max_y - min_y)
        grid.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        labels.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#6b7280">{value:.3g}</text>'
        )
    for i in range(5):
        t = i / 4
        x = margin_left + t * plot_w
        value = min_x + t * (max_x - min_x)
        grid.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f3f4f6"/>')
        labels.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" fill="#6b7280">{value:.0f}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="18" font-family="system-ui, sans-serif" fill="#111827">{title}</text>
{''.join(grid)}
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af"/>
{''.join(labels)}
<text x="{width / 2:.1f}" y="{height - 4}" text-anchor="middle" font-size="12" font-family="system-ui, sans-serif" fill="#374151">step</text>
<text transform="translate(16 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="12" font-family="system-ui, sans-serif" fill="#374151">{y_label}</text>
<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
</svg>
"""


class ExperimentTracker:
    def __init__(
        self,
        out_dir: Path,
        *,
        args: Namespace,
        config: Any,
        description: str | None = None,
        tags: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.experiment_dir = out_dir / "experiment"
        self.plots_dir = self.experiment_dir / "plots"
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.samples_path = self.experiment_dir / "samples.jsonl"
        self.manifest_path = self.experiment_dir / "experiment.yaml"
        self.report_path = self.experiment_dir / "report.md"
        self.description = description or ""
        self.tags = [tag.strip() for tag in (tags or "").split(",") if tag.strip()]
        self.notes = notes or ""
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest(args=args, config=config)

    def write_manifest(self, *, args: Namespace, config: Any) -> None:
        manifest = {
            "name": Path(args.out_dir).name,
            "description": self.description,
            "tags": self.tags,
            "notes": self.notes,
            "started_at": self.started_at,
            "run_dir": str(args.out_dir),
            "objective": getattr(args, "objective", None),
            "optimizer": getattr(args, "optimizer", None),
            "model": {
                "d_model": getattr(args, "d_model", None),
                "n_heads": getattr(args, "n_heads", None),
                "n_layers": getattr(args, "n_layers", None),
                "n_mtp_heads": getattr(config, "n_mtp_heads", None),
                "vocab_size": getattr(config, "vocab_size", None),
            },
            "training": {
                "max_steps": getattr(args, "max_steps", None),
                "tokens_per_step": getattr(args, "tokens_per_step", None),
                "total_training_tokens": getattr(args, "total_training_tokens", None),
                "target_param_data_ratio": getattr(args, "target_param_data_ratio", None),
                "mtp_loss_weight": getattr(args, "mtp_loss_weight", None),
                "batch_size": getattr(args, "batch_size", None),
                "grad_accum_steps": getattr(args, "grad_accum_steps", None),
                "seq_len": getattr(args, "seq_len", None),
            },
            "args": _plain(args),
            "config": _plain(config),
        }
        self.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(step), "time": time.time(), "metrics": _plain(metrics)}
        _write_jsonl(self.metrics_path, row)

    def log_sample(self, step: int, text: str) -> None:
        _write_jsonl(self.samples_path, {"step": int(step), "time": time.time(), "text": text})

    def finalize(self, *, final_step: int | None = None) -> None:
        generate_experiment_report(self.out_dir, final_step=final_step)


def load_metric_series(out_dir: Path) -> dict[str, list[tuple[int, float]]]:
    rows = _read_jsonl(out_dir / "experiment" / "metrics.jsonl")
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        step = int(row["step"])
        flat: dict[str, float] = {}
        _flatten("", row.get("metrics", {}), flat)
        for key, value in flat.items():
            series.setdefault(key, []).append((step, value))
    return series


def best_value(series: list[tuple[int, float]], *, mode: str) -> tuple[int, float] | None:
    if not series:
        return None
    fn = min if mode == "min" else max
    return fn(series, key=lambda item: item[1])


def generate_experiment_report(out_dir: Path, *, final_step: int | None = None) -> Path:
    experiment_dir = out_dir / "experiment"
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = experiment_dir / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    series = load_metric_series(out_dir)

    plot_specs = [
        ("train/loss", "Train Loss", "loss"),
        ("eval/loss", "Eval Loss", "loss"),
        ("eval/masked_bpb", "Eval BPB", "bpb"),
        ("eval/core", "CORE", "score"),
        ("train/tokens_per_second", "Throughput", "tokens/sec"),
        ("train/lr", "Learning Rate", "lr"),
    ]
    generated_plots = []
    for metric, title, y_label in plot_specs:
        values = series.get(metric, [])
        svg = _line_chart_svg(values, title=title, y_label=y_label)
        if not svg:
            continue
        path = plots_dir / f"{metric.replace('/', '_')}.svg"
        path.write_text(svg, encoding="utf-8")
        generated_plots.append((metric, path.relative_to(experiment_dir)))

    summary_items = [
        ("Best eval/loss", best_value(series.get("eval/loss", []), mode="min")),
        ("Best eval/masked_bpb", best_value(series.get("eval/masked_bpb", []), mode="min")),
        ("Best eval/core", best_value(series.get("eval/core", []), mode="max")),
        ("Best train/tokens_per_second", best_value(series.get("train/tokens_per_second", []), mode="max")),
    ]

    samples = _read_jsonl(experiment_dir / "samples.jsonl")
    last_sample = samples[-1] if samples else None

    lines = [
        f"# Experiment: {manifest.get('name', out_dir.name)}",
        "",
    ]
    if manifest.get("description"):
        lines += [manifest["description"], ""]
    if manifest.get("tags"):
        lines += [f"Tags: {', '.join(manifest['tags'])}", ""]
    lines += [
        "## Configuration",
        "",
        f"- Objective: `{manifest.get('objective')}`",
        f"- Optimizer: `{manifest.get('optimizer')}`",
        f"- Model: `{manifest.get('model', {}).get('n_layers')}L`, d_model `{manifest.get('model', {}).get('d_model')}`, heads `{manifest.get('model', {}).get('n_heads')}`",
        f"- MTP heads: `{manifest.get('model', {}).get('n_mtp_heads')}`",
        f"- MTP loss weight: `{manifest.get('training', {}).get('mtp_loss_weight')}`",
        f"- Max steps: `{manifest.get('training', {}).get('max_steps')}`",
        f"- Tokens/step: `{manifest.get('training', {}).get('tokens_per_step')}`",
        f"- Final step: `{final_step}`",
        "",
        "## Scoreboard",
        "",
        "| Metric | Step | Value |",
        "| --- | ---: | ---: |",
    ]
    for label, item in summary_items:
        if item is None:
            continue
        step, value = item
        lines.append(f"| {label} | {step} | {value:.6g} |")

    if generated_plots:
        lines += ["", "## Plots", ""]
        for metric, rel_path in generated_plots:
            lines += [f"### {metric}", "", f"![{metric}]({rel_path})", ""]

    if last_sample is not None:
        lines += [
            "## Latest Samples",
            "",
            f"Step `{last_sample['step']}`",
            "",
            "```text",
            last_sample["text"],
            "```",
            "",
        ]

    if manifest.get("notes"):
        lines += ["## Notes", "", manifest["notes"], ""]

    report_path = experiment_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
