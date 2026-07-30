#!/usr/bin/env python3
"""Paper-style SSL acoustic model comparison figures."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path("/home/ustc1958/lxy/graph/tone/complete_package0614")
INPUT_CSV = ROOT / "output/0614/local_hf_ssl_4shot_s100_20260625_121230/summary_complete_models.csv"
OUTPUT_DIR = ROOT / "pc_dlcmnet_figures_hybrid" / "ssl_model_paper_examples"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
METRIC = "accuracy"
BASELINE = "wav2vec2-base"

AREAS = ["Qingyang", "Tongling", "Jingxian", "Nanling", "Ningguo", "Lishui"]
MODEL_ORDER = [
    "wav2vec2-base",
    "hubert-base-ls960",
    "wavlm-base",
    "mHuBERT-147",
]
MODEL_LABELS = {
    "wav2vec2-base": "wav2vec2",
    "hubert-base-ls960": "HuBERT",
    "wavlm-base": "WavLM",
    "mHuBERT-147": "mHuBERT",
}
COLORS = {
    "wav2vec2-base": "#4C78A8",
    "hubert-base-ls960": "#F58518",
    "wavlm-base": "#54A24B",
    "mHuBERT-147": "#E45756",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["model"].isin(MODEL_ORDER) & df["area"].isin(AREAS)].copy()
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["area"] = pd.Categorical(df["area"], AREAS, ordered=True)
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce") * 100.0
    df[f"{METRIC}_std"] = pd.to_numeric(df[f"{METRIC}_std"], errors="coerce") * 100.0
    return df.sort_values(["area", "model"])


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.tick_params(axis="both", labelsize=8)


def plot_area_small_multiples(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), dpi=DPI, sharey=True)
    x = np.arange(len(MODEL_ORDER))
    for ax, area in zip(axes.ravel(), AREAS):
        d = df[df["area"] == area].set_index("model").reindex(MODEL_ORDER).reset_index()
        vals = d[METRIC].to_numpy()
        errs = d[f"{METRIC}_std"].to_numpy()
        ax.bar(
            x,
            vals,
            yerr=errs,
            width=0.66,
            color=[COLORS[m] for m in MODEL_ORDER],
            error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
        )
        ax.text(0.02, 0.94, area, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=35, ha="right")
        ax.set_xlabel("SSL model", fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        style_axes(ax)
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_accuracy_by_area_2x3_bars.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_average_with_error(df: pd.DataFrame) -> Path:
    summary = (
        df.groupby("model", observed=True)[METRIC]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=DPI)
    ax.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        color=[COLORS[m] for m in MODEL_ORDER],
        width=0.62,
        error_kw={"elinewidth": 0.8, "capsize": 3, "capthick": 0.8},
    )
    for i, row in summary.iterrows():
        ax.text(i, row["mean"] + row["std"] + 0.8, f"{row['mean']:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax.set_xlabel("SSL acoustic model")
    ax.set_ylabel("Mean accuracy across areas (%)")
    style_axes(ax)
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_average_accuracy_errorbar.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_over_baseline(df: pd.DataFrame) -> Path:
    pivot = df.pivot(index="area", columns="model", values=METRIC).reindex(index=AREAS, columns=MODEL_ORDER)
    delta = pivot.sub(pivot[BASELINE], axis=0).drop(columns=[BASELINE])
    models = [m for m in MODEL_ORDER if m != BASELINE]
    fig, axes = plt.subplots(1, len(models), figsize=(11.0, 3.4), dpi=DPI, sharey=True)
    y = np.arange(len(AREAS))
    for ax, model in zip(axes, models):
        vals = delta[model].to_numpy()
        colors = np.where(vals >= 0, "#009E73", "#D55E00")
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.barh(y, vals, color=colors, height=0.58)
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(AREAS)
        ax.invert_yaxis()
        ax.set_xlabel("Delta Acc. (%)", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_delta_accuracy_vs_wav2vec2.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rank_lines(df: pd.DataFrame) -> Path:
    d = df.copy()
    d["rank"] = d.groupby("area", observed=True)[METRIC].rank(ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=DPI)
    x = np.arange(len(AREAS))
    for model in MODEL_ORDER:
        g = d[d["model"] == model].set_index("area").reindex(AREAS)
        ax.plot(x, g["rank"], marker="o", linewidth=1.4, markersize=4.5, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(AREAS, rotation=25, ha="right")
    ax.set_yticks(np.arange(1, len(MODEL_ORDER) + 1))
    ax.invert_yaxis()
    ax.set_xlabel("Area")
    ax.set_ylabel("Rank by accuracy")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_accuracy_rank_lines.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    paths = [
        plot_area_small_multiples(df),
        plot_average_with_error(df),
        plot_delta_over_baseline(df),
        plot_rank_lines(df),
    ]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
