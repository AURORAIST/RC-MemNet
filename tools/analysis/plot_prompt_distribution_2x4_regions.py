#!/usr/bin/env python3
"""Plot prompt routing weights over training iterations for eight regions."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# =========================
# 1. Configuration
# =========================

RUN_DIR = Path(
    "/home/ustc1958/lxy/graph/tone/complete_package0614/"
    "output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps"
)

# Use the baseline 8-prompt setting. Change this to another setting directory if needed,
# e.g. "temperature_0p3" or "episode_length_48".
SETTING = "num_prompts_8"

OUTPUT_DIR = Path(
    "/home/ustc1958/lxy/graph/tone/complete_package0614/pc_dlcmnet_figures_hybrid"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "fig_prompt_distribution_over_iterations_2x4_regions.jpg"

# Matplotlib may try to write cache under ~/.config; keep it inside a writable folder.
MPLCONFIGDIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# Smoothing window
SMOOTH_WINDOW = 25

# Plot parameters
DPI = 300
FIGSIZE = (13.6, 7.8)
RAW_ALPHA = 0.10
RAW_LW = 0.45
SMOOTH_LW = 1.15

# Show figure interactively
SHOW_FIG = False

REGIONS = [
    "Qingyang",
    "Tongling",
    "Jingxian",
    "Nanling",
    "Ningguo",
    "Lishui",
    "Chizhou",
    "Huangshan",
]


# =========================
# 2. Helpers
# =========================

def smooth_series(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def load_curve(region: str) -> tuple[pd.DataFrame, str, list[str]]:
    input_csv = RUN_DIR / SETTING / region / "curve.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing curve file: {input_csv}")

    df = pd.read_csv(input_csv)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x_col = "step" if "step" in df.columns else "epoch"
    df = df.sort_values(x_col).reset_index(drop=True)

    prompt_cols = []
    for i in range(64):
        col = f"prompt_route_mean_{i}"
        if col in df.columns:
            prompt_cols.append(col)

    if not prompt_cols:
        raise ValueError(f"No prompt_route_mean_* columns found in {input_csv}")

    final_smoothed_values = {
        col: smooth_series(df[col]).iloc[-1]
        for col in prompt_cols
    }
    ordered_prompt_cols = sorted(
        prompt_cols,
        key=lambda c: final_smoothed_values[c],
        reverse=True,
    )
    return df, x_col, ordered_prompt_cols


def plot_one_region(ax: plt.Axes, region: str) -> None:
    df, x_col, ordered_prompt_cols = load_curve(region)

    for col in ordered_prompt_cols:
        prompt_id = col.split("_")[-1]

        raw_line, = ax.plot(
            df[x_col],
            df[col],
            linewidth=RAW_LW,
            alpha=RAW_ALPHA,
            label="_nolegend_",
        )
        color = raw_line.get_color()

        ax.plot(
            df[x_col],
            smooth_series(df[col]),
            linewidth=SMOOTH_LW,
            color=color,
            label=f"Prompt {prompt_id}",
        )

    # Region label inside the panel, not as a subplot title.
    ax.text(
        0.02,
        0.96,
        region,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    # Every subplot has its own x/y axis labels.
    ax.set_xlabel("Training iteration", fontsize=9)
    ax.set_ylabel("Routing weight", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_ylim(bottom=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="upper right",
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.25,
    )


# =========================
# 3. Plot 2x4 panels
# =========================

def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, dpi=DPI)
    axes_flat = axes.flatten()

    for ax, region in zip(axes_flat, REGIONS):
        plot_one_region(ax, region)

    # No suptitle and no subplot titles, per request.
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=DPI)
    print(f"Saved figure to: {OUTPUT_PATH.resolve()}")

    if SHOW_FIG:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
