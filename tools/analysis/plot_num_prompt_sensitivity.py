#!/usr/bin/env python3
"""Plot prompt-bank size sensitivity across target regions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "output/0614/matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = PACKAGE_ROOT / "output/0614/num_prompt_sweep_8targets_20260629/num_prompt_sweep_8targets_summary.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "model/num_prompt_sensitivity"
AREA_ORDER = ["Qingyang", "Tongling", "Jingxian", "Nanling", "Ningguo", "Lishui", "Chizhou", "Huangshan"]
COLORS = {
    "Qingyang": "#2F6BBA",
    "Tongling": "#D95F02",
    "Jingxian": "#1B9E77",
    "Nanling": "#7570B3",
    "Ningguo": "#E7298A",
    "Lishui": "#66A61E",
    "Chizhou": "#E6AB02",
    "Huangshan": "#666666",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "weighted_f1"])
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(summary_csv: Path, output_dir: Path, metric: str, formats: list[str]) -> None:
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    required = {"area", "num_prompts", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{summary_csv} is missing columns: {sorted(missing)}")

    df = df[df["area"].isin(AREA_ORDER)].copy()
    df["num_prompts"] = pd.to_numeric(df["num_prompts"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce") * 100.0
    df = df.dropna(subset=["num_prompts", metric])

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for area in AREA_ORDER:
        part = df[df["area"] == area].sort_values("num_prompts")
        if part.empty:
            continue
        x = part["num_prompts"].to_numpy(dtype=float)
        y = part[metric].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.0,
            linewidth=1.9,
            color=COLORS[area],
            label=area,
        )

        best_idx = int(y.argmax())
        ax.scatter(
            [x[best_idx]],
            [y[best_idx]],
            s=62,
            color=COLORS[area],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )

    ax.set_title("Prompt-bank size sensitivity across target regions")
    ax.set_xlabel("Number of prompts R")
    ax.set_ylabel(metric.replace("_", "-").title() + " (%)")
    ax.set_xticks(sorted(df["num_prompts"].unique()))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(title="Target region", frameon=False, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"num_prompt_sensitivity_{metric}"
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)

    tidy = df[["area", "num_prompts", metric]].sort_values(["area", "num_prompts"])
    tidy.to_csv(output_dir / f"{stem}_plot_data.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    configure()
    plot(Path(args.summary_csv).resolve(), Path(args.output_dir).resolve(), args.metric, args.formats)
    print(
        {
            "summary_csv": str(Path(args.summary_csv).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "metric": args.metric,
        }
    )


if __name__ == "__main__":
    main()
