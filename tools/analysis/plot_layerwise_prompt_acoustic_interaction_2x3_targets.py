#!/usr/bin/env python3
"""Plot layerwise prompt-acoustic interaction for six target regions."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


TARGET_REGIONS = [
    ("04_Qingyang", "Qingyang"),
    ("06_Tongling", "Tongling"),
    ("08_Jingxian", "Jingxian"),
    ("10_Nanling", "Nanling"),
    ("12_Ningguo", "Ningguo"),
    ("14_Lishui", "Lishui"),
]

COLUMN_PREFIX = "prompt_interaction_mean_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="Experiment root containing target-region curve.csv files.")
    parser.add_argument("--output-dir", default="", help="Defaults to run_root/figures.")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--fig-dpi", type=int, default=120)
    parser.add_argument("--save-dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["jpg", "pdf"])
    return parser.parse_args()


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def parse_layer_id(column_name: str) -> int:
    try:
        return int(column_name.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse layer ID from column: {column_name}") from exc


def interaction_columns(df: pd.DataFrame, path: Path) -> list[str]:
    cols = [col for col in df.columns if col.startswith(COLUMN_PREFIX)]
    if not cols:
        raise ValueError(f"No {COLUMN_PREFIX}* columns found in {path}")
    return sorted(cols, key=parse_layer_id)


def load_curve(run_root: Path, region_dir: str) -> tuple[pd.DataFrame, str, list[str]]:
    for name in ["curve.csv", "result.curve.csv"]:
        path = run_root / region_dir / name
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"Missing curve.csv for {region_dir} under {run_root}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "step" in df.columns:
        x_col = "step"
    elif "epoch" in df.columns:
        x_col = "epoch"
    else:
        raise ValueError(f"Neither step nor epoch exists in {path}")

    df = df.sort_values(x_col).reset_index(drop=True)
    return df, x_col, interaction_columns(df, path)


def get_plot_order(df: pd.DataFrame, cols: list[str], smooth_window: int) -> list[str]:
    return sorted(cols, key=lambda col: smooth_series(df[col], smooth_window).iloc[-1])


def nice_upper_bound(value: float) -> float:
    if value <= 0.20:
        step = 0.02
    elif value <= 0.50:
        step = 0.05
    else:
        step = 0.10
    return max(math.ceil(value / step) * step + step, 0.15)


def compute_ymax(df: pd.DataFrame, cols: list[str], smooth_window: int) -> float:
    smoothed_max = max(smooth_series(df[col], smooth_window).max() for col in cols)
    raw_max = max(df[col].max() for col in cols)
    return nice_upper_bound(max(smoothed_max, raw_max * 0.85))


def build_layer_colors(layer_ids: list[int], plt_module):
    cmap = plt_module.get_cmap("tab10" if len(layer_ids) <= 10 else "tab20")
    return {layer_id: cmap(index % cmap.N) for index, layer_id in enumerate(layer_ids)}


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    font_size = 20
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    curve_cache = {region_dir: load_curve(run_root, region_dir) for region_dir, _ in TARGET_REGIONS}
    layer_ids = [parse_layer_id(col) for col in next(iter(curve_cache.values()))[2]]
    layer_colors = build_layer_colors(layer_ids, plt)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.0), dpi=int(args.fig_dpi))
    axes_flat = axes.flatten()

    for idx, (ax, (region_dir, title)) in enumerate(zip(axes_flat, TARGET_REGIONS)):
        df, x_col, cols = curve_cache[region_dir]
        for col in get_plot_order(df, cols, int(args.smooth_window)):
            layer_id = parse_layer_id(col)
            color = layer_colors[layer_id]
            ax.plot(df[x_col], df[col], linewidth=0.65, alpha=0.20, color=color, zorder=1)
            ax.plot(
                df[x_col],
                smooth_series(df[col], int(args.smooth_window)),
                linewidth=2.25,
                color=color,
                zorder=3,
                solid_capstyle="round",
            )

        ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Prompt-acoustic interaction" if idx % 3 == 0 else "")
        ax.tick_params(axis="both", width=1.0, length=4.5)
        ax.grid(True, linestyle="--", linewidth=0.50, alpha=0.18)
        ax.set_ylim(0, compute_ymax(df, cols, int(args.smooth_window)))
        ax.margins(x=0.02)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=layer_colors[layer_id],
            lw=2.2,
            label=f"Layer {layer_id + 1}",
            solid_capstyle="round",
        )
        for layer_id in layer_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(layer_ids),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.90, wspace=0.28, hspace=0.42)

    stem = "fig_layerwise_prompt_acoustic_interaction_2x3_targets_raw_smooth_font20"
    for fmt in args.formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"jpg", "jpeg", "png"}:
            save_kwargs["dpi"] = int(args.save_dpi)
        fig.savefig(path, **save_kwargs)
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
