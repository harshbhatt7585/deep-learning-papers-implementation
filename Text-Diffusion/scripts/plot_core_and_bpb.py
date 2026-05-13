"""Plot CORE and BPB per experiment in a single side-by-side view.

Two stacked panels share the same x-axis (run names, ranked by CORE).
Top panel: CORE @ step 400. Bottom panel: masked BPB @ step 400.

Same x order makes it visually obvious where the two metrics agree
(rank-aligned bars) vs disagree (e.g. GQA-2 has near-best BPB but
worst CORE — the routing-capacity signature).

The catastrophically broken `SwiGLU dropout=0.1` run is excluded
because its BPB (2.64) compresses every other bar into a thin band;
its values are noted in the figure caption instead.

Regenerate after a new gate experiment:

    cd Text-Diffusion && python scripts/plot_core_and_bpb.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# (label, core, bpb, status, params_M)
RUNS: list[tuple[str, float, float, str, int]] = [
    ("MTP2 ReLU\u00b2",            0.0710, 1.1157, "top",      185),
    ("A100 MTP1 ReLU\u00b2",       0.0693, 1.1289, "accepted", 160),
    ("SwiGLU MTP1 shared",         0.0688, 1.1246, "accepted", 143),
    ("MoE 4\u00d7top-1",           0.0688, 1.1439, "accepted", 143),
    ("GQA-3 ReLU\u00b2",           0.0681, 1.1188, "accepted", 178),
    ("Tied embeddings",            0.0615, 1.1210, "rejected", 160),
    ("MTP-shared 1st",             0.0585, 1.1105, "rejected", 136),
    ("SwiGLU MTP1 + GQA-2",        0.0459, 1.1123, "rejected", 140),
]
# `SwiGLU dropout=0.1` excluded: CORE=0.0447 BPB=2.6436 — off-scale.

NANOCHAT_CORE_TARGET = 0.1059

COLORS = {
    "top":      "#1F8A65",
    "accepted": "#2E79B5",
    "rejected": "#C04848",
    "target":   "#5A6CC0",
    "text":     "#1f2328",
    "muted":    "#57606a",
    "grid":     "#e6e8eb",
    "axis":     "#9da7b1",
    "bg":       "#fbfbfc",
}


def main() -> None:
    # Rank by CORE descending so top performers anchor the left.
    runs = sorted(RUNS, key=lambda r: -r[1])
    labels  = [r[0] for r in runs]
    cores   = [r[1] for r in runs]
    bpbs    = [r[2] for r in runs]
    statuses = [r[3] for r in runs]
    bar_colors = [COLORS[s] for s in statuses]

    fig, (ax_core, ax_bpb) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(12, 8),
        dpi=200,
        sharex=True,
        gridspec_kw={"hspace": 0.18, "height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("white")

    # ---- TOP PANEL: CORE ----
    ax_core.set_facecolor(COLORS["bg"])
    ax_core.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax_core.set_axisbelow(True)

    bars_core = ax_core.bar(
        range(len(labels)), cores,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.7,
        zorder=3,
    )
    for bar, value in zip(bars_core, cores):
        ax_core.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["text"],
            weight="medium",
        )

    # nanochat target line on CORE panel
    ax_core.axhline(
        y=NANOCHAT_CORE_TARGET,
        color=COLORS["target"],
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        zorder=2,
    )
    ax_core.annotate(
        f"nanochat D12 target · {NANOCHAT_CORE_TARGET:.4f}",
        xy=(len(labels) - 1 + 0.4, NANOCHAT_CORE_TARGET),
        xytext=(-4, 4),
        textcoords="offset points",
        color=COLORS["target"],
        fontsize=10,
        ha="right",
        weight="semibold",
    )

    ax_core.set_ylim(0.0, max(NANOCHAT_CORE_TARGET, max(cores)) * 1.18)
    ax_core.set_ylabel("CORE @ step 400  (higher is better)", fontsize=11, color=COLORS["text"])
    ax_core.set_title(
        "CORE @ 400 steps",
        fontsize=12,
        color=COLORS["text"],
        loc="left",
        weight="semibold",
        pad=6,
    )

    # ---- BOTTOM PANEL: BPB ----
    ax_bpb.set_facecolor(COLORS["bg"])
    ax_bpb.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax_bpb.set_axisbelow(True)

    bars_bpb = ax_bpb.bar(
        range(len(labels)), bpbs,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.7,
        zorder=3,
    )
    for bar, value in zip(bars_bpb, bpbs):
        ax_bpb.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["text"],
            weight="medium",
        )

    # Tighter BPB range so differences are visible (range is ~1.11–1.14).
    bpb_min = min(bpbs)
    bpb_max = max(bpbs)
    pad = (bpb_max - bpb_min) * 0.25 + 0.005
    ax_bpb.set_ylim(bpb_min - pad, bpb_max + pad * 1.6)
    ax_bpb.set_ylabel("masked BPB @ step 400  (lower is better)", fontsize=11, color=COLORS["text"])
    ax_bpb.set_title(
        "BPB @ 400 steps · same x-order as top panel",
        fontsize=12,
        color=COLORS["text"],
        loc="left",
        weight="semibold",
        pad=6,
    )

    # ---- Shared x-axis ----
    ax_bpb.set_xticks(range(len(labels)))
    ax_bpb.set_xticklabels(labels, rotation=20, ha="right", fontsize=10, color=COLORS["text"])

    for ax in (ax_core, ax_bpb):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(COLORS["axis"])
        ax.tick_params(colors=COLORS["muted"], labelsize=10)

    # ---- Figure title + caption ----
    fig.suptitle(
        "CORE and BPB per experiment · Text-MTP 400-step gate",
        fontsize=14,
        x=0.02, y=0.985,
        ha="left",
        weight="semibold",
        color=COLORS["text"],
    )
    fig.text(
        0.02, 0.952,
        "Bars share the same x order (ranked by CORE descending). "
        "Bar color encodes the gate verdict. Catastrophically broken SwiGLU dropout=0.1 "
        "(CORE 0.0447 / BPB 2.6436) is excluded — its BPB is off-scale.",
        fontsize=10,
        color=COLORS["muted"],
        ha="left",
    )

    # ---- Legend ----
    legend_handles = [
        Patch(facecolor=COLORS["top"], edgecolor="white", label="Current best (top)"),
        Patch(facecolor=COLORS["accepted"], edgecolor="white", label="Accepted (top 5)"),
        Patch(facecolor=COLORS["rejected"], edgecolor="white", label="Rejected at gate"),
        Line2D([0], [0], color=COLORS["target"], linestyle="--", linewidth=1.5,
               label="nanochat D12 CORE reference"),
    ]
    ax_core.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=10,
        framealpha=0.96,
        edgecolor="#d0d7de",
        ncol=2,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "assets"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "core-and-bpb-per-experiment.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
