#!/usr/bin/env python3
"""Plot 8-region K-shot sensitivity and effectiveness figures."""

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
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PACKAGE_ROOT
    / "output/0614/foundation_kshot_1_10_8areas_20260625_215947/foundation_kshot_summary.csv"
)
DEFAULT_OUTPUT = Path("/home/ustc1958/lxy/graph/tone/model/8region_kshot_figures")

AREA_ORDER = [
    "Qingyang",
    "Tongling",
    "Jingxian",
    "Nanling",
    "Ningguo",
    "Lishui",
    "Chizhou",
    "Huangshan",
]
MODEL_ORDER = [
    "PC-DLCMNet-prototype",
    "mHuBERT-147",
    "MR-HuBERT",
    "MS-HuBERT",
    "SALMONN-proxy",
]
MODEL_LABEL = {
    "PC-DLCMNet-prototype": "PC-DLCMNet",
    "mHuBERT-147": "mHuBERT-147",
    "MR-HuBERT": "MR-HuBERT",
    "MS-HuBERT": "MS-HuBERT",
    "SALMONN-proxy": "SALMONN-proxy",
}
COLORS = {
    "PC-DLCMNet-prototype": "#2F6BBA",
    "mHuBERT-147": "#7A68A6",
    "MR-HuBERT": "#5B9F62",
    "MS-HuBERT": "#D19A3A",
    "SALMONN-proxy": "#9AA4B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def configure_matplotlib() -> None:
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


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pct(series: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(series, dtype=float) * 100.0


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["area"] = pd.Categorical(df["area"], categories=AREA_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "area", "k"]).reset_index(drop=True)
    expected = len(MODEL_ORDER) * len(AREA_ORDER) * 10
    if len(df) != expected:
        raise ValueError(f"Expected {expected} rows, got {len(df)}")
    return df


def write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        agg = (
            df.groupby(["model", "k"], observed=True)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_area_std"})
        )
        agg["metric"] = metric
        rows.append(agg)
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "8region_kshot_model_mean_area_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline = df[df["model"] != "PC-DLCMNet-prototype"]
    best = (
        baseline.groupby(["area", "k"], observed=True)["accuracy"]
        .max()
        .rename("best_baseline_accuracy")
        .reset_index()
    )
    pc = df[df["model"] == "PC-DLCMNet-prototype"][
        ["area", "k", "accuracy", "macro_f1", "weighted_f1"]
    ].rename(columns={"accuracy": "pc_accuracy"})
    gain = pc.merge(best, on=["area", "k"], how="left")
    gain["pc_gain_over_best_baseline"] = gain["pc_accuracy"] - gain["best_baseline_accuracy"]
    gain.to_csv(output_dir / "8region_pc_gain_over_best_baseline.csv", index=False, encoding="utf-8-sig")


def plot_model_kshot_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        agg = sub.groupby("k", observed=True)["accuracy"].agg(["mean", "std"]).reset_index()
        x = agg["k"].to_numpy(dtype=float)
        y = pct(agg["mean"])
        ystd = pct(agg["std"])
        lw = 2.8 if model == "PC-DLCMNet-prototype" else 1.8
        zorder = 4 if model == "PC-DLCMNet-prototype" else 2
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=lw,
            markersize=4.0,
            color=COLORS[model],
            label=MODEL_LABEL[model],
            zorder=zorder,
        )
        alpha = 0.14 if model == "PC-DLCMNet-prototype" else 0.07
        ax.fill_between(x, y - ystd, y + ystd, color=COLORS[model], alpha=alpha, linewidth=0)

    ax.set_xlabel("Support samples per class (K)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    save(fig, output_dir, "fig8_kshot_accuracy_models")


def plot_pc_metric_stability(df: pd.DataFrame, output_dir: Path) -> None:
    pc = df[df["model"] == "PC-DLCMNet-prototype"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    specs = [
        ("accuracy", "Accuracy", "#2F6BBA", "o"),
        ("macro_f1", "Macro-F1", "#5B9F62", "s"),
        ("weighted_f1", "Weighted-F1", "#D19A3A", "D"),
    ]
    for metric, label, color, marker in specs:
        agg = pc.groupby("k", observed=True)[metric].agg(["mean", "std"]).reset_index()
        x = agg["k"].to_numpy(dtype=float)
        y = pct(agg["mean"])
        ystd = pct(agg["std"])
        ax.plot(x, y, marker=marker, linewidth=2.2, markersize=4.0, color=color, label=label)
        ax.fill_between(x, y - ystd, y + ystd, color=color, alpha=0.11, linewidth=0)

    ax.set_xlabel("Support samples per class (K)")
    ax.set_ylabel("8-region mean score (%)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, output_dir, "fig8_pc_metric_stability")


def plot_pc_area_panels(df: pd.DataFrame, output_dir: Path) -> None:
    pc = df[df["model"] == "PC-DLCMNet-prototype"].copy()
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.8), sharex=True, sharey=True)
    for ax, area in zip(axes.ravel(), AREA_ORDER):
        sub = pc[pc["area"] == area].sort_values("k")
        x = sub["k"].to_numpy(dtype=float)
        ax.plot(x, pct(sub["accuracy"]), marker="o", linewidth=1.8, color="#2F6BBA", label="Accuracy")
        ax.plot(x, pct(sub["macro_f1"]), marker="s", linewidth=1.6, color="#5B9F62", label="Macro-F1")
        ax.fill_between(
            x,
            pct(sub["accuracy"] - sub["accuracy_std"]),
            pct(sub["accuracy"] + sub["accuracy_std"]),
            color="#2F6BBA",
            alpha=0.10,
            linewidth=0,
        )
        ax.set_title(str(area))
        ax.set_xticks([1, 4, 7, 10])
        ax.grid(axis="y", alpha=0.22)
    axes[0, 0].set_ylabel("Score (%)")
    axes[1, 0].set_ylabel("Score (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("K")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, output_dir, "fig8_pc_area_kshot_panels")


def plot_gain_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    baseline = df[df["model"] != "PC-DLCMNet-prototype"]
    best = baseline.groupby(["area", "k"], observed=True)["accuracy"].max().reset_index()
    pc = df[df["model"] == "PC-DLCMNet-prototype"][["area", "k", "accuracy"]]
    gain = pc.merge(best, on=["area", "k"], suffixes=("_pc", "_baseline"))
    gain["gain"] = (gain["accuracy_pc"] - gain["accuracy_baseline"]) * 100.0
    matrix = (
        gain.pivot(index="area", columns="k", values="gain")
        .reindex(AREA_ORDER)
        .reindex(columns=range(1, 11))
    )

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    vmax = float(np.nanmax(matrix.to_numpy()))
    im = ax.imshow(matrix.to_numpy(), cmap="Reds", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(10), [str(k) for k in range(1, 11)])
    ax.set_yticks(np.arange(len(AREA_ORDER)), AREA_ORDER)
    ax.set_xlabel("Support samples per class (K)")
    for i, area in enumerate(AREA_ORDER):
        for j, k in enumerate(range(1, 11)):
            value = float(matrix.loc[area, k])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Accuracy gain (points)")
    save(fig, output_dir, "fig8_pc_gain_heatmap_accuracy")


def plot_k4_gain_bar(df: pd.DataFrame, output_dir: Path) -> None:
    k4 = df[df["k"] == 4].copy()
    baseline = k4[k4["model"] != "PC-DLCMNet-prototype"]
    best = (
        baseline.sort_values("accuracy", ascending=False)
        .groupby("area", observed=True)
        .first()
        .reset_index()[["area", "model", "accuracy"]]
        .rename(columns={"model": "best_baseline", "accuracy": "best_baseline_accuracy"})
    )
    pc = k4[k4["model"] == "PC-DLCMNet-prototype"][["area", "accuracy", "accuracy_std"]].rename(
        columns={"accuracy": "pc_accuracy", "accuracy_std": "pc_accuracy_std"}
    )
    gain = pc.merge(best, on="area", how="left")
    gain["gain_points"] = (gain["pc_accuracy"] - gain["best_baseline_accuracy"]) * 100.0
    gain["pc_acc_pct"] = gain["pc_accuracy"] * 100.0
    gain = gain.sort_values("gain_points", ascending=True)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = plt.cm.Reds((gain["gain_points"] - gain["gain_points"].min()) / (gain["gain_points"].max() - gain["gain_points"].min() + 1e-9) * 0.55 + 0.35)
    bars = ax.barh(gain["area"].astype(str), gain["gain_points"], color=colors, edgecolor="#8A1C1C", linewidth=0.4)
    ax.set_xlabel("Accuracy gain over best foundation baseline (points)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.22)
    for bar, row in zip(bars, gain.to_dict("records")):
        ax.text(
            row["gain_points"] + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"+{row['gain_points']:.1f}",
            va="center",
            fontsize=7.0,
        )
    ax.set_xlim(0, max(gain["gain_points"]) + 4.0)
    gain.to_csv(output_dir / "8region_k4_pc_gain_over_best_baseline.csv", index=False, encoding="utf-8-sig")
    save(fig, output_dir, "fig8_k4_pc_gain_bar_accuracy")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    write_summary_tables(df, output_dir)
    plot_model_kshot_accuracy(df, output_dir)
    plot_pc_metric_stability(df, output_dir)
    plot_pc_area_panels(df, output_dir)
    plot_gain_heatmap(df, output_dir)
    plot_k4_gain_bar(df, output_dir)
    print(
        {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "figures": [
                "fig8_kshot_accuracy_models",
                "fig8_pc_metric_stability",
                "fig8_pc_area_kshot_panels",
                "fig8_pc_gain_heatmap_accuracy",
                "fig8_k4_pc_gain_bar_accuracy",
            ],
        }
    )


if __name__ == "__main__":
    main()
