"""Generate CORE evaluation charts for the Text-MTP experiment suite.

Outputs:
    assets/core-evals-overview.png
    assets/core-vs-steps.png
    assets/core-evals-by-experiment/*.png
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "text-diffusion-matplotlib"))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np


@dataclass(frozen=True)
class Experiment:
    label: str
    slug: str
    points: tuple[tuple[int, float], ...]
    verdict: str
    note: str
    params_m: int | None = None

    @property
    def final_step(self) -> int:
        return self.points[-1][0]

    @property
    def final_core(self) -> float:
        return self.points[-1][1]

    @property
    def peak_core(self) -> float:
        return max(core for _, core in self.points)


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        label="H100 FP8 MTP2 ReLU2",
        slug="h100-fp8-mtp2-relu2",
        points=((400, 0.0710),),
        verdict="best",
        note="400-step gate winner; full-vocab MTP heads x2.",
        params_m=185,
    ),
    Experiment(
        label="A100 MTP1 ReLU2",
        slug="a100-mtp1-relu2",
        points=((400, 0.0693), (2000, 0.1001)),
        verdict="long",
        note="Only run with multiple logged CORE evaluations.",
        params_m=160,
    ),
    Experiment(
        label="H100 bf16 SwiGLU MTP1 shared",
        slug="h100-bf16-swiglu-mtp1-shared",
        points=((400, 0.0688),),
        verdict="accepted",
        note="Parameter-efficient gate result with shared MTP projection.",
        params_m=143,
    ),
    Experiment(
        label="H100 FP8 MoE 4x top-1",
        slug="h100-fp8-moe-4x-top1",
        points=((400, 0.0688),),
        verdict="accepted",
        note="Matched the SwiGLU CORE score, but with weaker loss and throughput.",
        params_m=185,
    ),
    Experiment(
        label="H100 FP8 GQA-3 ReLU2",
        slug="h100-fp8-gqa3-relu2",
        points=((400, 0.0681),),
        verdict="accepted",
        note="Small regression from full attention at the 400-step gate.",
        params_m=178,
    ),
    Experiment(
        label="Tied embeddings",
        slug="tied-embeddings",
        points=((400, 0.0615),),
        verdict="rejected",
        note="Smaller model, but CORE regressed at the gate.",
        params_m=160,
    ),
    Experiment(
        label="MTP-shared first attempt",
        slug="mtp-shared-first-attempt",
        points=((400, 0.0585),),
        verdict="rejected",
        note="Good BPB did not transfer to CORE.",
        params_m=136,
    ),
    Experiment(
        label="SwiGLU MTP1 + GQA-2",
        slug="swiglu-mtp1-gqa2",
        points=((400, 0.0459),),
        verdict="rejected",
        note="Attention rank reduction cost too much downstream signal.",
        params_m=140,
    ),
    Experiment(
        label="SwiGLU dropout=0.1",
        slug="swiglu-dropout-01",
        points=((400, 0.0447),),
        verdict="broken",
        note="Catastrophic gate result; kept as a failed-control point.",
        params_m=143,
    ),
)

STYLE = {
    "paper": "#efe4cf",
    "paper_dark": "#e5d6ba",
    "ink": "#26231f",
    "muted": "#6e665b",
    "axis": "#3c3933",
    "grid": "#c9bda7",
    "best": "#8f3228",
    "long": "#1f1e1b",
    "accepted": "#275b4b",
    "rejected": "#b9742f",
    "broken": "#7c2f2b",
    "label_box": "#f6eddb",
}

LABEL_FONT = "Avenir"
SCORE_FONT = "DIN Alternate"


def _asset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _setup_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(STYLE["paper"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.7, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(STYLE["axis"])
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(colors=STYLE["muted"], labelsize=10, length=4)
    ax.xaxis.label.set_color(STYLE["ink"])
    ax.yaxis.label.set_color(STYLE["ink"])


def _curve(points: tuple[tuple[int, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([x for x, _ in points], dtype=float)
    ys = np.array([y for _, y in points], dtype=float)
    if len(points) < 3:
        return xs, ys
    smooth_x = np.linspace(xs.min(), xs.max(), 160)
    coeffs = np.polyfit(xs, ys, deg=2)
    smooth_y = np.polyval(coeffs, smooth_x)
    return smooth_x, smooth_y


def _line_effects() -> list[pe.AbstractPathEffect]:
    return [
        pe.SimpleLineShadow(offset=(1.2, -1.2), alpha=0.14, shadow_color="#111111"),
        pe.Normal(),
    ]


def _place_labels(items: list[tuple[Experiment, float]], min_gap: float) -> dict[str, float]:
    order = sorted(items, key=lambda item: -item[1])
    placed: dict[str, float] = {}
    previous: float | None = None
    for experiment, natural_y in order:
        y = natural_y if previous is None else min(natural_y, previous - min_gap)
        placed[experiment.slug] = y
        previous = y
    return placed


def plot_overview(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=240)
    fig.patch.set_facecolor(STYLE["paper"])
    ax.set_facecolor(STYLE["paper"])

    gate_runs = sorted(EXPERIMENTS, key=lambda experiment: -experiment.points[0][1])[:5]
    gate_runs = sorted(gate_runs, key=lambda experiment: experiment.params_m or 0)
    labels = {
        "h100-bf16-swiglu-mtp1-shared": "SwiGLU",
        "a100-mtp1-relu2": "A100 MTP1",
        "h100-fp8-gqa3-relu2": "GQA-3",
        "h100-fp8-moe-4x-top1": "MoE",
        "h100-fp8-mtp2-relu2": "MTP2",
    }
    label_offsets = {
        "h100-bf16-swiglu-mtp1-shared": (-2.0, 0.0060, "right"),
        "a100-mtp1-relu2": (0.0, 0.0053, "center"),
        "h100-fp8-gqa3-relu2": (-0.8, 0.0058, "right"),
        "h100-fp8-moe-4x-top1": (5.8, 0.0026, "left"),
        "h100-fp8-mtp2-relu2": (-0.2, 0.0074, "center"),
    }

    for index, experiment in enumerate(gate_runs):
        core = experiment.points[0][1]
        params_m = experiment.params_m or 0
        color = STYLE[experiment.verdict]
        bend = -1.2 if index % 2 == 0 else 1.2
        path = MplPath(
            [
                (params_m, 0.0),
                (params_m + bend, core * 0.32),
                (params_m - bend, core * 0.68),
                (params_m, core),
            ],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        guide = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=2.2,
            alpha=0.72,
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
        guide.set_path_effects(_line_effects())
        ax.add_patch(guide)
        ax.scatter(
            params_m,
            core,
            s=190 if experiment.verdict == "best" else 145,
            color=color,
            edgecolor=STYLE["paper"],
            linewidth=2.2,
            zorder=4,
        )
        dx, dy, ha = label_offsets[experiment.slug]
        weight = "heavy" if experiment.verdict == "best" else "medium"
        ax.text(
            params_m + dx,
            core + dy,
            labels[experiment.slug],
            ha=ha,
            va="bottom",
            fontsize=10.2,
            color=STYLE["ink"],
            fontfamily=LABEL_FONT,
            weight=weight,
            zorder=5,
        )
        ax.text(
            params_m + dx,
            core + dy - 0.0031,
            f"{core:.4f}",
            ha=ha,
            va="bottom",
            fontsize=10.8,
            color=STYLE["ink"],
            fontfamily=SCORE_FONT,
            weight="bold",
            zorder=5,
        )

    ax.set_xlim(136, 192)
    ax.set_ylim(0.0, 0.086)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0.02, y=0.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    overview_path = out_dir / "core-evals-overview.png"
    legacy_path = out_dir / "core-vs-steps.png"
    fig.savefig(overview_path, bbox_inches="tight", facecolor=STYLE["paper"], dpi=220)
    fig.savefig(legacy_path, bbox_inches="tight", facecolor=STYLE["paper"], dpi=220)
    plt.close(fig)
    print(f"saved: {overview_path}")
    print(f"saved: {legacy_path}")


def plot_experiment(experiment: Experiment, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=220)
    fig.patch.set_facecolor(STYLE["paper"])
    _setup_axis(ax)

    color = STYLE[experiment.verdict]
    xs = [step for step, _ in experiment.points]
    ys = [core for _, core in experiment.points]

    if len(experiment.points) > 1:
        curve_x, curve_y = _curve(experiment.points)
        line = ax.plot(curve_x, curve_y, color=color, linewidth=3.0, zorder=3)[0]
        line.set_sketch_params(scale=0.65, length=85, randomness=1.5)
        line.set_path_effects(_line_effects())
    else:
        ax.axhline(ys[0], color=color, linewidth=1.2, linestyle=(0, (5, 6)), alpha=0.45, zorder=1)
        ax.vlines(xs[0], 0, ys[0], color=color, linewidth=1.0, alpha=0.35, zorder=1)

    ax.scatter(
        xs,
        ys,
        s=120,
        color=color,
        edgecolor=STYLE["paper"],
        linewidth=1.8,
        zorder=4,
    )

    for step, core in experiment.points:
        ax.annotate(
            f"step {step}\nCORE {core:.4f}",
            xy=(step, core),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=10.5,
            color=STYLE["ink"],
            weight="semibold" if core == experiment.peak_core else "normal",
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.12",
                "fc": STYLE["label_box"],
                "ec": "none",
                "alpha": 0.92,
            },
        )

    if len(experiment.points) == 1:
        ax.set_xlim(0, 520)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        y_pad = 0.012
    else:
        ax.set_xlim(0, 2520)
        ax.set_xticks([0, 400, 800, 1200, 1600, 2000, 2400])
        y_pad = 0.014

    y_min = max(0.0, min(ys) - y_pad)
    y_max = max(ys) + y_pad * 1.2
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("CORE evaluation score", fontsize=11)
    ax.set_title(experiment.label, loc="left", fontsize=16, color=STYLE["ink"], pad=16, weight="semibold")
    ax.text(
        0,
        1.015,
        experiment.note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=STYLE["muted"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment.slug}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=STYLE["paper"], dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    out_dir = _asset_dir()
    per_experiment_dir = out_dir / "core-evals-by-experiment"
    plot_overview(out_dir)
    for experiment in EXPERIMENTS:
        plot_experiment(experiment, per_experiment_dir)


if __name__ == "__main__":
    main()
