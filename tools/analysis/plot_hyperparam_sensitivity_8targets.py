#!/usr/bin/env python3
"""Plot fixed-4-shot hyperparameter sensitivity from 8-target sweep summaries."""

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


DEFAULT_OUTPUT = Path("/home/ustc1958/lxy/graph/tone/model/hyperparam_sensitivity_8targets")
PANEL_ORDER = ["num_prompts", "lambda_global", "temperature", "episode_length"]
PANEL_LABEL = {
    "num_prompts": "Prompt number R",
    "lambda_global": "Global loss weight lambda_g",
    "temperature": "Temperature tau",
    "episode_length": "Episode length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metric", choices=["accuracy", "macro_f1", "weighted_f1"], default="accuracy")
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(summary_csv: Path, output_dir: Path, metric: str) -> None:
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    df = df[df["sweep"].isin(PANEL_ORDER)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for col in [metric, f"{metric}_area_std"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for ax, sweep in zip(axes.ravel(), PANEL_ORDER):
        part = df[df["sweep"] == sweep].sort_values("value")
        if part.empty:
            ax.set_visible(False)
            continue
        x = part["value"].to_numpy(dtype=float)
        y = part[metric].to_numpy(dtype=float)
        yerr = part[f"{metric}_area_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, color="#2F6BBA")
        ax.fill_between(x, y - yerr, y + yerr, color="#2F6BBA", alpha=0.12, linewidth=0)
        ax.set_title(PANEL_LABEL[sweep])
        ax.set_xlabel(PANEL_LABEL[sweep])
        ax.set_ylabel(metric.replace("_", "-").title() + " (%)")
        ax.grid(axis="y", alpha=0.24)
        ax.set_xticks(x)
        if sweep == "lambda_global" or sweep == "temperature":
            ax.set_xticklabels([f"{v:g}" for v in x])
        else:
            ax.set_xticklabels([str(int(v)) for v in x])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_hyperparam_sensitivity_8targets_k4_{metric}"
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure()
    plot(Path(args.summary_csv), Path(args.output_dir), args.metric)
    print({"output_dir": str(args.output_dir), "metric": args.metric})


if __name__ == "__main__":
    main()
