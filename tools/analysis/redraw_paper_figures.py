"""Redraw paper-quality figures for PC-DLCMNet experiments.

Each figure is designed to answer one concrete paper question:
1. Does episode memory consistently improve all holdout regions?
2. Does memory help harder regions more?
3. Does the read/write memory remain effective across training stages?
4. Are memory write strategies consistently useful across regions?
5. Which hyperparameters are sensitive?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("output/0614/paper_ready_results")
OUT = ROOT / "07_redrawn_paper_figures"


COLORS = {
    "global": "#4C78A8",
    "episode": "#F58518",
    "gain": "#54A24B",
    "accent": "#E45756",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def pct(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def load_region_gain() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "03_region_memory_gain" / "region_memory_gain.csv")
    for col in ["global_macro_f1", "macro_f1", "gain_macro_f1", "global_accuracy", "accuracy", "gain_accuracy"]:
        df[col] = df[col].astype(float) * 100.0
    return df


def fig_region_ranked_gain(df: pd.DataFrame) -> None:
    """Line + filled gap: one look shows every region improves."""
    d = df.sort_values("global_macro_f1").reset_index(drop=True)
    x = np.arange(len(d))
    labels = d["plot_label"].tolist()

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.plot(x, d["global_macro_f1"], marker="o", color=COLORS["global"], label="Global-only")
    ax.plot(x, d["macro_f1"], marker="o", color=COLORS["episode"], label="Global + Episode RW")
    ax.fill_between(x, d["global_macro_f1"], d["macro_f1"], color=COLORS["episode"], alpha=0.18, label="Memory gain")
    ax.set_title("Region-wise effect of episode memory")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlabel("Holdout regions sorted by Global-only Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.text(
        0.98,
        0.08,
        "Positive gain: 16/16 regions",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["gray"],
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLORS["light"], "alpha": 0.9},
    )
    save(fig, "fig1_region_ranked_macro_f1_gain")


def fig_gain_vs_difficulty(df: pd.DataFrame) -> None:
    """Scatter/regression: harder global regions tend to benefit more."""
    x = df["global_macro_f1"].to_numpy()
    y = df["gain_macro_f1"].to_numpy()
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = coef[0] * xs + coef[1]
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.scatter(x, y, s=42, color=COLORS["gain"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot(xs, ys, color=COLORS["accent"], linestyle="--", label=f"Trend, r={corr:.2f}")
    for _, row in df.nlargest(3, "gain_macro_f1").iterrows():
        ax.annotate(row["plot_label"], (row["global_macro_f1"], row["gain_macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color=COLORS["gray"], linewidth=1)
    ax.set_title("Gain vs. target difficulty")
    ax.set_xlabel("Global-only Macro-F1 (%)")
    ax.set_ylabel("Episode memory gain (%)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig2_gain_vs_target_difficulty")


def fig_stage_memory_ablation() -> None:
    stages = np.array([1, 2, 3])
    global_mf1 = np.array([48.84, 52.78, 53.49])
    episode_mf1 = np.array([52.21, 57.83, 59.14])
    gains = episode_mf1 - global_mf1

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(stages, global_mf1, marker="o", color=COLORS["global"], label="Global-only")
    ax.plot(stages, episode_mf1, marker="o", color=COLORS["episode"], label="Global + Episode RW")
    ax.fill_between(stages, global_mf1, episode_mf1, color=COLORS["episode"], alpha=0.18)
    for x, y, g in zip(stages, episode_mf1, gains):
        ax.text(x, y + 0.7, f"+{g:.2f}", ha="center", color=COLORS["accent"], fontsize=8)
    ax.set_title("Memory ablation across stages")
    ax.set_xlabel("Training stage")
    ax.set_ylabel("Mean Macro-F1 over 16 regions (%)")
    ax.set_xticks(stages)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3_stage_memory_ablation_curve")


def fig_write_modes_by_region() -> None:
    rows = pd.read_csv(ROOT / "04_memory_write_mode_ablation" / "summary.csv")
    base = load_region_gain()[["plot_label", "region"]]
    # Some summary rows do not have plot_label; map by region name.
    rows = rows.merge(base, on="region", how="left")
    rows["gain_macro_f1"] = rows["gain_macro_f1"].astype(float) * 100.0
    order = (
        rows.groupby("plot_label")["gain_macro_f1"].mean().sort_values().index.tolist()
    )
    pivot = rows.pivot_table(index="plot_label", columns="mode", values="gain_macro_f1").reindex(order)

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    x = np.arange(len(pivot))
    for mode, color, marker in [("pseudo", COLORS["global"], "o"), ("blend", COLORS["episode"], "s"), ("label", COLORS["gain"], "^")]:
        if mode in pivot.columns:
            ax.plot(x, pivot[mode], marker=marker, color=color, label=mode)
    ax.axhline(0, color=COLORS["gray"], linewidth=1)
    ax.set_title("Region-wise gains under different write modes")
    ax.set_ylabel("Macro-F1 gain over Global-only (%)")
    ax.set_xlabel("Holdout regions sorted by average write-mode gain")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
    ax.legend(title="Write mode", frameon=False, ncol=3, loc="upper left")
    save(fig, "fig4_write_mode_region_gain_lines")


def fig_write_mode_summary() -> None:
    rows = pd.read_csv(ROOT / "04_memory_write_mode_ablation" / "summary.csv")
    summary = []
    for mode, group in rows.groupby("mode"):
        summary.append(
            {
                "mode": mode,
                "macro_f1": group["macro_f1"].astype(float).mean() * 100.0,
                "gain_macro_f1": group["gain_macro_f1"].astype(float).mean() * 100.0,
                "accuracy": group["accuracy"].astype(float).mean() * 100.0,
            }
        )
    d = pd.DataFrame(summary)
    order = ["pseudo", "blend", "label"]
    d["mode"] = pd.Categorical(d["mode"], categories=order, ordered=True)
    d = d.sort_values("mode")

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x = np.arange(len(d))
    bars = ax.bar(x, d["macro_f1"], color=[COLORS["global"], COLORS["episode"], COLORS["gain"]], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pseudo", "Blend", "Label-aware"])
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlabel("Memory write strategy")
    ax.set_title("Memory write strategy")
    ymin = max(0, d["macro_f1"].min() - 3)
    ymax = d["macro_f1"].max() + 2
    ax.set_ylim(ymin, ymax)
    for bar, gain in zip(bars, d["gain_macro_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"+{gain:.2f}",
            ha="center",
            va="bottom",
            color=COLORS["accent"],
            fontsize=8,
        )
    save(fig, "fig4_write_mode_summary")


def fig_module_ablation_summary() -> None:
    labels = [
        "Speech\nbaseline",
        "+ Class\nmemory",
        "+ Acoustic\nfusion",
        "+ Prompt\ngate",
        "+ Static\nregion mem.",
        "Global\nonly",
        "+ Episode\nRW",
    ]
    macro_f1 = np.array([46.59, 47.88, 47.64, 47.17, 44.46, 53.49, 59.14])
    groups = ["static", "static", "static", "static", "static", "episode", "episode"]
    colors = [COLORS["gray"] if g == "static" else COLORS["global"] for g in groups]
    colors[-1] = COLORS["episode"]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(labels))
    ax.plot(x[:5], macro_f1[:5], marker="o", color=COLORS["gray"], label="Static module chain")
    ax.plot(x[5:], macro_f1[5:], marker="o", color=COLORS["episode"], label="Episode memory")
    ax.scatter(x, macro_f1, s=38, c=colors, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(4.5, color=COLORS["light"], linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlabel("Ablated component setting")
    ax.set_title("Module ablation")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        "+5.65",
        xy=(6, macro_f1[-1]),
        xytext=(5.45, macro_f1[-1] + 2.0),
        arrowprops={"arrowstyle": "->", "color": COLORS["accent"], "lw": 1.0},
        color=COLORS["accent"],
        fontsize=8,
    )
    save(fig, "fig0_module_ablation_summary")


def load_sensitivity() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "02_parameter_sensitivity" / "summary.csv")
    df["value_num"] = df["value"].astype(float)
    df["macro_f1_pct"] = df["macro_f1"].astype(float) * 100.0
    df["accuracy_pct"] = df["accuracy"].astype(float) * 100.0
    return df


def fig_parameter_sensitivity_panels(df: pd.DataFrame) -> None:
    panels = [
        ("episode_length", "Episode length"),
        ("num_prompts", "Number of prompts"),
        ("temperature", "Temperature"),
        ("lambda_global", r"$\lambda_{\mathrm{global}}$"),
        ("eval_support_shots", "Eval support shots"),
        ("eval_ensemble_runs", "Eval ensemble runs"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    axes = axes.ravel()
    for ax, (param, title) in zip(axes, panels):
        d = df[df["parameter"] == param].sort_values("value_num")
        ax.plot(d["value_num"], d["macro_f1_pct"], marker="o", color=COLORS["global"])
        best = d.loc[d["macro_f1_pct"].idxmax()]
        ax.scatter([best["value_num"]], [best["macro_f1_pct"]], s=70, color=COLORS["accent"], zorder=4, edgecolor="white", linewidth=0.8)
        ax.annotate(
            f"best={best['value_num']:g}",
            (best["value_num"], best["macro_f1_pct"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["accent"],
        )
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Macro-F1 (%)")
    fig.suptitle("Parameter sensitivity of PC-DLCMNet", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_parameter_sensitivity_panels")


def fig_key_parameter_curves(df: pd.DataFrame) -> None:
    """Two larger curves for the most paper-worthy sensitivity findings."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, param, title, message in [
        (axes[0], "episode_length", "Richer episodes improve memory adaptation", "Longer local context helps"),
        (axes[1], "temperature", "Matching temperature has an optimum", "Too soft weakens separation"),
    ]:
        d = df[df["parameter"] == param].sort_values("value_num")
        ax.plot(d["value_num"], d["macro_f1_pct"], marker="o", color=COLORS["episode"])
        best = d.loc[d["macro_f1_pct"].idxmax()]
        ax.scatter([best["value_num"]], [best["macro_f1_pct"]], color=COLORS["accent"], s=72, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(message, (best["value_num"], best["macro_f1_pct"]), xytext=(8, 8), textcoords="offset points", fontsize=8, color=COLORS["accent"])
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Macro-F1 (%)")
    fig.tight_layout()
    save(fig, "fig6_key_parameter_curves")


def main() -> None:
    setup_style()
    region = load_region_gain()
    sensitivity = load_sensitivity()
    fig_region_ranked_gain(region)
    fig_gain_vs_difficulty(region)
    fig_module_ablation_summary()
    fig_stage_memory_ablation()
    fig_write_mode_summary()
    fig_write_modes_by_region()
    fig_parameter_sensitivity_panels(sensitivity)
    fig_key_parameter_curves(sensitivity)


if __name__ == "__main__":
    main()
