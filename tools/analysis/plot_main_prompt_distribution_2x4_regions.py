#!/usr/bin/env python3
"""Plot RC-MemNet prompt-routing dynamics for the main target regions."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


SOURCE_PROMPT_NAMES = {
    0: "Chizhou",
    1: "Dangtu",
    2: "Fanchang",
    3: "Gaochun",
    4: "Huangshan",
    5: "Suncun",
    6: "Wuhu",
    7: "Xuancheng",
}

TARGET_REGIONS = [
    ("04_Qingyang", "Qingyang"),
    ("06_Tongling", "Tongling"),
    ("08_Jingxian", "Jingxian"),
    ("10_Nanling", "Nanling"),
    ("12_Ningguo", "Ningguo"),
    ("14_Lishui", "Lishui"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="Main experiment root containing region/curve.csv files.")
    parser.add_argument("--output-dir", default="", help="Figure output directory.")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--fig-dpi", type=int, default=120)
    parser.add_argument("--save-dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["jpg", "pdf"])
    return parser.parse_args()


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def parse_prompt_id(column_name: str) -> int:
    return int(column_name.rsplit("_", 1)[-1])


def prompt_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    cols = [col for col in df.columns if col.startswith("prompt_interaction_mean_")]
    if cols:
        return sorted(cols, key=parse_prompt_id), "Prompt-acoustic interaction"
    raise ValueError(
        "No prompt_interaction_mean_* columns found. This script is for "
        "encoder-layer interaction curves; prompt_route_mean_* is routing, "
        "not layerwise interaction."
    )


def load_curve(run_root: Path, region_dir: str) -> tuple[pd.DataFrame, str, list[str], str]:
    for name in ["curve.csv", "result.curve.csv"]:
        path = run_root / region_dir / name
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"Missing curve.csv for {region_dir} under {run_root}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    x_col = "step" if "step" in df.columns else "epoch"
    df = df.sort_values(x_col).reset_index(drop=True)
    cols, ylabel = prompt_columns(df)
    ordered = sorted(cols, key=lambda col: smooth_series(df[col], 3).iloc[-1])
    return df, x_col, ordered, ylabel


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


def build_prompt_colors(prompt_ids: list[int], plt_module):
    cmap = plt_module.get_cmap("tab10" if len(prompt_ids) <= 10 else "tab20")
    return {prompt_id: cmap(index % cmap.N) for index, prompt_id in enumerate(prompt_ids)}


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

    curve_cache = {
        region_dir: load_curve(run_root, region_dir)
        for region_dir, _title in TARGET_REGIONS
    }
    prompt_ids = sorted(parse_prompt_id(col) for col in next(iter(curve_cache.values()))[2])
    prompt_colors = build_prompt_colors(prompt_ids, plt)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.0), dpi=int(args.fig_dpi))
    axes_flat = axes.flatten()
    common_ylabel = "Normalized prompt-token magnitude"

    for idx, (ax, (region_dir, title)) in enumerate(zip(axes_flat, TARGET_REGIONS)):
        df, x_col, cols, ylabel = curve_cache[region_dir]
        common_ylabel = ylabel
        for col in cols:
            prompt_id = parse_prompt_id(col)
            color = prompt_colors[prompt_id]
            ax.plot(
                df[x_col],
                df[col],
                linewidth=0.65,
                alpha=0.20,
                color=color,
                zorder=1,
            )
            ax.plot(
                df[x_col],
                smooth_series(df[col], int(args.smooth_window)),
                linewidth=2.25,
                color=color,
                zorder=3,
                solid_capstyle="round",
            )

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Training iteration")
        if idx % 4 == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
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
            color=prompt_colors[prompt_id],
            lw=2.2,
            label=SOURCE_PROMPT_NAMES.get(prompt_id, f"Prompt{prompt_id}"),
            solid_capstyle="round",
        )
        for prompt_id in prompt_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(prompt_ids),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        labelspacing=0.6,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(
        left=0.060,
        right=0.995,
        bottom=0.12,
        top=0.900,
        wspace=0.28,
        hspace=0.4,
    )

    stem = "fig_main_rcmemnet_prompt_distribution_2x4_regions_raw_smooth_font20"
    for fmt in args.formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"jpg", "jpeg", "png"}:
            save_kwargs["dpi"] = int(args.save_dpi)
        fig.savefig(path, **save_kwargs)
        print(f"saved {path}")
    plt.close(fig)
    print(f"ylabel={common_ylabel}")


if __name__ == "__main__":
    main()
