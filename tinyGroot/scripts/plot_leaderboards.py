"""Generate top-5 leaderboard charts from the latest BLOG.md tables.

Outputs:
    assets/leaderboard-top5-400-step.png
    assets/leaderboard-top5-global.png
    assets/leaderboard-experiments/400-step/*.png
    assets/leaderboard-experiments/global/*.png
    assets/thumbnail-nanochat-vs-mine.png
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "tinygroot-matplotlib"))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


@dataclass(frozen=True)
class LeaderboardRun:
    rank: int
    label: str
    short_label: str
    core: float
    bpb: float | None
    val_loss: float | None
    budget: str
    hardware: str
    params: str | None = None
    highlight: bool = False


TOP5_400_STEP: tuple[LeaderboardRun, ...] = (
    LeaderboardRun(
        1,
        "DeepSeek-MTP2 + TST",
        "DeepSeek MTP2",
        0.0737,
        1.1165,
        3.5576,
        "400-step gate",
        "4x H100 bf16",
        "~160M",
        True,
    ),
    LeaderboardRun(2, "H100 FP8 MTP2 ReLU2", "MTP2 ReLU2", 0.0710, 1.1157, 3.5491, "400-step gate", "8x H100 FP8", "~185M"),
    LeaderboardRun(3, "A100 MTP1 ReLU2", "A100 MTP1", 0.0693, 1.1289, 3.5953, "400-step gate", "8x A100", "~160M"),
    LeaderboardRun(4, "SwiGLU MTP1 shared", "SwiGLU MTP1", 0.0688, 1.1246, 3.5866, "400-step gate", "4x H100 bf16", "~143M"),
    LeaderboardRun(5, "H100 FP8 MoE top-1", "MoE top-1", 0.0688, 1.1439, 3.6385, "400-step gate", "8x H100 FP8", "~185M active ~143M"),
)


TOP5_GLOBAL: tuple[LeaderboardRun, ...] = (
    LeaderboardRun(
        1,
        "DeepSeek-MTP2 + TST d12",
        "DeepSeek d12",
        0.1162,
        0.9387,
        2.9866,
        "ratio-12 long run",
        "8x H100 FP8 compile",
        None,
        True,
    ),
    LeaderboardRun(2, "MTP2 + TST d12", "MTP2 d12", 0.1133, 0.9446, 3.0028, "ratio-12 long run", "8x H100 FP8 compile"),
    LeaderboardRun(3, "nanochat d12 public ref", "nanochat d12", 0.1059, 0.9825, None, "d12 reference", "8x H100"),
    LeaderboardRun(4, "DeepSeek-MTP2 + TST", "DeepSeek gate", 0.0737, 1.1165, 3.5576, "400-step gate", "4x H100 bf16"),
    LeaderboardRun(5, "H100 FP8 MTP2 ReLU2", "MTP2 ReLU2", 0.0710, 1.1157, 3.5491, "400-step gate", "8x H100 FP8"),
)

NANOCHAT_D12_CORE = 0.1059
MTP2_TST_D12_CORE = 0.1133


STYLE = {
    "paper": "#efe4cf",
    "paper_dark": "#e5d6ba",
    "ink": "#26231f",
    "muted": "#6e665b",
    "grid": "#c9bda7",
    "winner": "#8f3228",
    "green": "#275b4b",
    "amber": "#b9742f",
    "bar": "#3c3933",
    "card": "#f7efdd",
}

LABEL_FONT = "Avenir"
SCORE_FONT = "DIN Alternate"


def _asset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _slug(text: str) -> str:
    chars = []
    previous_dash = False
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def _line_effects() -> list[pe.AbstractPathEffect]:
    return [
        pe.SimpleLineShadow(offset=(1.2, -1.2), alpha=0.12, shadow_color="#111111"),
        pe.Normal(),
    ]


def _setup_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(STYLE["paper"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.8, alpha=0.42, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(STYLE["bar"])
        ax.spines[spine].set_linewidth(1.3)
    ax.tick_params(axis="x", colors=STYLE["muted"], labelsize=10, length=0)
    ax.tick_params(axis="y", colors=STYLE["muted"], labelsize=10, length=0)


def _subtitle_for(run: LeaderboardRun) -> str:
    parts = [run.budget, run.hardware]
    if run.params:
        parts.append(run.params)
    if run.val_loss is not None and run.bpb is not None:
        parts.append(f"val {run.val_loss:.4f}, BPB {run.bpb:.4f}")
    elif run.bpb is not None:
        parts.append(f"BPB {run.bpb:.4f}")
    return "  |  ".join(parts)


def plot_leaderboard(
    runs: tuple[LeaderboardRun, ...],
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    x_min: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.1), dpi=220)
    fig.patch.set_facecolor(STYLE["paper"])
    _setup_axis(ax)

    cores = np.array([run.core for run in runs], dtype=float)
    ranks = np.array([run.rank for run in runs], dtype=float)
    min_core = float(x_min if x_min is not None else max(0.0, cores.min() - 0.012))
    max_core = float(cores.max() + 0.006)

    ax.set_xlim(0.55, len(runs) + 0.45)
    ax.set_ylim(min_core, max_core)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"#{run.rank}" for run in runs])
    ax.set_xlabel("leaderboard rank", fontsize=11, color=STYLE["ink"], labelpad=12)
    ax.set_ylabel("CORE score", fontsize=11, color=STYLE["ink"], labelpad=12)

    line = ax.plot(
        ranks,
        cores,
        color=STYLE["bar"],
        linewidth=2.6,
        alpha=0.72,
        zorder=2,
    )[0]
    line.set_path_effects(_line_effects())
    ax.fill_between(ranks, min_core, cores, color=STYLE["paper_dark"], alpha=0.42, zorder=1)

    for idx, run in enumerate(runs):
        color = STYLE["winner"] if run.highlight else (STYLE["green"] if idx < 3 else STYLE["amber"])
        ax.scatter(
            run.rank,
            run.core,
            s=190 if run.highlight else 120,
            color=color,
            edgecolor=STYLE["paper"],
            linewidth=1.8,
            zorder=4,
        )

        label_y = run.core + (max_core - min_core) * (0.045 if idx % 2 == 0 else -0.075)
        ax.text(
            run.rank,
            label_y,
            run.short_label,
            ha="center",
            va="center",
            fontsize=9.8,
            color=STYLE["ink"],
            fontfamily=LABEL_FONT,
            weight="heavy" if run.highlight else "semibold",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.08",
                "fc": STYLE["card"],
                "ec": "none",
                "alpha": 0.90,
            },
        )
        ax.text(
            run.rank,
            run.core - (max_core - min_core) * 0.055,
            f"{run.core:.4f}",
            ha="center",
            va="center",
            fontsize=12.0,
            color=STYLE["ink"],
            fontfamily=SCORE_FONT,
            weight="bold",
            zorder=5,
        )

    ax.set_title(title, loc="left", fontsize=24, color=STYLE["ink"], pad=28, weight="heavy")
    ax.text(
        0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color=STYLE["muted"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=STYLE["paper"], dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_experiment_context(
    runs: tuple[LeaderboardRun, ...],
    *,
    selected_rank: int,
    board_label: str,
    out_path: Path,
    x_min: float | None = None,
) -> None:
    selected = next(run for run in runs if run.rank == selected_rank)
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=220)
    fig.patch.set_facecolor(STYLE["paper"])
    _setup_axis(ax)

    cores = np.array([run.core for run in runs], dtype=float)
    ranks = np.array([run.rank for run in runs], dtype=float)
    min_core = float(x_min if x_min is not None else max(0.0, cores.min() - 0.012))
    max_core = float(cores.max() + 0.007)
    ax.set_xlim(0.55, len(runs) + 0.45)
    ax.set_ylim(min_core, max_core)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"#{run.rank}" for run in runs])
    ax.set_xlabel("leaderboard rank", fontsize=11, color=STYLE["ink"], labelpad=12)
    ax.set_ylabel("CORE score", fontsize=11, color=STYLE["ink"], labelpad=12)

    line = ax.plot(
        ranks,
        cores,
        color="#8d8374",
        linewidth=2.5,
        alpha=0.60,
        zorder=2,
    )[0]
    line.set_path_effects(_line_effects())
    ax.fill_between(ranks, min_core, cores, color=STYLE["paper_dark"], alpha=0.34, zorder=1)

    for idx, run in enumerate(runs):
        is_selected = run.rank == selected_rank
        color = STYLE["winner"] if is_selected else "#8d8374"
        alpha = 1.0 if is_selected else 0.36

        ax.scatter(
            run.rank,
            run.core,
            s=230 if is_selected else 88,
            color=color,
            alpha=alpha,
            edgecolor=STYLE["paper"],
            linewidth=1.8,
            zorder=4,
        )

        label_y = run.core + (max_core - min_core) * (0.06 if idx % 2 == 0 else -0.08)
        ax.text(
            run.rank,
            label_y,
            run.short_label if is_selected else f"#{run.rank}",
            ha="center",
            va="center",
            fontsize=12.0 if is_selected else 10.2,
            color=STYLE["ink"] if is_selected else STYLE["muted"],
            fontfamily=LABEL_FONT,
            weight="heavy" if is_selected else "medium",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.08",
                "fc": STYLE["card"],
                "ec": "none",
                "alpha": 0.94 if is_selected else 0.72,
            },
        )
        ax.text(
            run.rank,
            run.core - (max_core - min_core) * 0.055,
            f"{run.core:.4f}",
            ha="center",
            va="center",
            fontsize=14 if is_selected else 11.5,
            color=STYLE["ink"] if is_selected else STYLE["muted"],
            fontfamily=SCORE_FONT,
            weight="bold" if is_selected else "normal",
            zorder=5,
        )

    ax.axvline(
        selected.rank,
        color=STYLE["winner"],
        linewidth=1.2,
        linestyle=(0, (4, 5)),
        alpha=0.42,
        zorder=0,
    )
    ax.axhline(
        selected.core,
        color=STYLE["winner"],
        linewidth=1.2,
        linestyle=(0, (4, 5)),
        alpha=0.42,
        zorder=0,
    )

    metric_lines = [
        f"CORE {selected.core:.4f}",
    ]
    ax.text(
        0.99,
        1.10,
        "  |  ".join(metric_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.4,
        color=STYLE["winner"],
        fontfamily=LABEL_FONT,
        weight="heavy",
    )
    ax.set_title(
        f"{board_label}: #{selected.rank} {selected.short_label}",
        loc="left",
        fontsize=21,
        color=STYLE["ink"],
        pad=28,
        weight="heavy",
    )
    ax.text(
        0,
        1.014,
        "Rank vs CORE; highlighted point is the selected run.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=STYLE["muted"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=STYLE["paper"], dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_experiment_set(
    runs: tuple[LeaderboardRun, ...],
    *,
    board_label: str,
    out_dir: Path,
    x_min: float,
) -> None:
    for run in runs:
        filename = f"rank{run.rank:02d}-{_slug(run.label)}.png"
        plot_experiment_context(
            runs,
            selected_rank=run.rank,
            board_label=board_label,
            out_path=out_dir / filename,
            x_min=x_min,
        )


def plot_thumbnail_comparison(out_path: Path) -> None:
    """Text-free thumbnail comparing nanochat D12 against the best local run."""
    fig, ax = plt.subplots(figsize=(8.0, 3.7), dpi=240)
    fig.patch.set_facecolor(STYLE["paper"])
    ax.set_facecolor(STYLE["paper"])
    ax.set_xlim(0.0, 1.45)
    ax.set_ylim(0.094, 0.120)
    ax.axis("off")

    values = [NANOCHAT_D12_CORE, MTP2_TST_D12_CORE]
    colors = [STYLE["green"], STYLE["winner"]]
    xs = [1.0, 1.0]

    # Subtle guide lines only; no labels, numbers, title, or axes.
    for y in np.linspace(0.096, 0.120, 5):
        ax.plot(
            [0.0, 1.35],
            [y, y],
            color=STYLE["grid"],
            linewidth=0.9,
            alpha=0.26,
            zorder=0,
        )

    y0 = 0.094
    ax.plot([0, 1.35], [y0, y0], color=STYLE["bar"], linewidth=1.4, alpha=0.5, zorder=1)
    ax.plot([0, 0], [y0, 0.120], color=STYLE["bar"], linewidth=1.4, alpha=0.5, zorder=1)

    labels = [
        ("nanochat D12", NANOCHAT_D12_CORE),
        ("MTP2 + TST s4 r0.3 d12", MTP2_TST_D12_CORE),
    ]
    for x, value, color, (label, score) in zip(xs, values, colors, labels, strict=True):
        line = ax.plot(
            [0.0, x],
            [y0, value],
            color=color,
            linewidth=8.0,
            solid_capstyle="round",
            alpha=0.96,
            zorder=4,
        )[0]
        line.set_path_effects(_line_effects())
        ax.scatter(
            [x],
            [value],
            s=180,
            color=color,
            edgecolor=STYLE["paper"],
            linewidth=2.0,
            zorder=5,
        )
        ax.text(
            x,
            value + 0.0011,
            f"{label}\n{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=12.0,
            color=STYLE["ink"],
            fontfamily=LABEL_FONT,
            weight="heavy",
            linespacing=1.05,
            zorder=6,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, facecolor=STYLE["paper"], dpi=240)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    out_dir = _asset_dir()
    plot_thumbnail_comparison(out_dir / "thumbnail-nanochat-vs-mine.png")
    plot_leaderboard(
        TOP5_400_STEP,
        title="400-Step Gate: Top 5",
        subtitle="Rank vs CORE.",
        out_path=out_dir / "leaderboard-top5-400-step.png",
        x_min=0.064,
    )
    plot_experiment_set(
        TOP5_400_STEP,
        board_label="400-Step Gate",
        out_dir=out_dir / "leaderboard-experiments" / "400-step",
        x_min=0.064,
    )
    plot_leaderboard(
        TOP5_GLOBAL,
        title="Global Leaderboard: Top 5",
        subtitle="Rank vs CORE.",
        out_path=out_dir / "leaderboard-top5-global.png",
        x_min=0.066,
    )
    plot_experiment_set(
        TOP5_GLOBAL,
        board_label="Global Leaderboard",
        out_dir=out_dir / "leaderboard-experiments" / "global",
        x_min=0.066,
    )


if __name__ == "__main__":
    main()
