"""Plot no-K mechanism figures from PC-DLCMNet diagnostic CSV files.

These figures aggregate over available K settings and use mechanism-relevant
axes such as vowel class, prompt index, true class, and predicted class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pc_dlcmnet.visualization import DEFAULT_VOWEL_ORDER, save_figure, setup_paper_style


COL = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def ordered(values, preferred=DEFAULT_VOWEL_ORDER) -> list[str]:
    vals = [str(v) for v in values if pd.notna(v)]
    uniq = list(dict.fromkeys(vals))
    return [v for v in preferred if v in uniq] + [v for v in uniq if v not in preferred]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("f")]


def fig_gema_absorption_by_vowel(gate: pd.DataFrame):
    setup_paper_style()
    classes = ordered(gate["class"])
    d = gate.groupby(["class"], as_index=False)["support_injection_weight"].agg(["mean", "std"]).reset_index()
    d = d.set_index("class").reindex(classes).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    x = np.arange(len(d))
    ax.bar(x, d["mean"], yerr=d["std"].fillna(0), capsize=3, color=COL["orange"], alpha=0.82, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(d["class"])
    ax.set_ylim(0, max(0.25, float((d["mean"] + d["std"].fillna(0)).max()) * 1.2))
    ax.set_xlabel("Vowel class")
    ax.set_ylabel("Support absorption ratio")
    ax.set_title("GEMA support absorption by vowel class")
    ax.grid(axis="y", color=COL["light"], linewidth=0.8)
    return fig


def compute_rho(memory: pd.DataFrame) -> pd.DataFrame:
    fcols = feature_cols(memory)
    rows = []
    group_cols = [c for c in ["region", "episode", "k", "class"] if c in memory.columns]
    for keys, group in memory.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        vectors = {str(row["memory_type"]): row[fcols].to_numpy(dtype=float) for _, row in group.iterrows()}
        if not all(k in vectors for k in ["global_memory", "support_candidate", "episode_memory"]):
            continue
        rho = np.linalg.norm(vectors["episode_memory"] - vectors["global_memory"])
        rho /= np.linalg.norm(vectors["support_candidate"] - vectors["global_memory"]) + 1e-8
        rows.append({**meta, "adaptation_ratio": float(rho)})
    return pd.DataFrame(rows)


def fig_gema_adaptation_by_vowel(memory: pd.DataFrame):
    setup_paper_style()
    rho = compute_rho(memory)
    classes = ordered(rho["class"])
    d = rho.groupby("class", as_index=False)["adaptation_ratio"].agg(["mean", "std"]).reset_index()
    d = d.set_index("class").reindex(classes).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    x = np.arange(len(d))
    ax.bar(x, d["mean"], yerr=d["std"].fillna(0), capsize=3, color=COL["green"], alpha=0.82, width=0.65)
    ax.axhline(1.0, color=COL["gray"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["class"])
    ax.set_xlabel("Vowel class")
    ax.set_ylabel("Adaptation ratio rho")
    ax.set_title("Conservative memory adaptation by vowel class")
    ax.grid(axis="y", color=COL["light"], linewidth=0.8)
    return fig


def fig_pcmr_shift_heatmap(pcmr: pd.DataFrame):
    setup_paper_style()
    shift = pcmr[pcmr["component"] == "attention_shift"].copy()
    classes = ordered(pd.concat([shift["true_class"].astype(str), shift["class"].astype(str)]))
    pivot = shift.pivot_table(index="true_class", columns="class", values="value", aggfunc="mean").reindex(index=classes, columns=classes)
    values = pivot.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Candidate memory class")
    ax.set_ylabel("True query class")
    ax.set_title("Prompt-induced PCMR attention shift")
    fig.colorbar(im, ax=ax, label="Mean attention shift")
    return fig


def fig_pcmr_true_attention_by_vowel(pcmr: pd.DataFrame):
    setup_paper_style()
    d = pcmr[pcmr["component"].isin(["attention_without_prompt", "attention_with_prompt"])].copy()
    d = d[d["true_class"].astype(str) == d["class"].astype(str)]
    classes = ordered(d["true_class"])
    pivot = d.pivot_table(index="true_class", columns="component", values="value", aggfunc="mean").reindex(index=classes)
    x = np.arange(len(classes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.bar(x - w / 2, pivot["attention_without_prompt"], w, color=COL["gray"], alpha=0.75, label="without prompt")
    ax.bar(x + w / 2, pivot["attention_with_prompt"], w, color=COL["blue"], alpha=0.85, label="with prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("True vowel class")
    ax.set_ylabel("Attention on true class")
    ax.set_title("PCMR increases true-class memory attention")
    ax.grid(axis="y", color=COL["light"], linewidth=0.8)
    ax.legend(frameon=False)
    return fig


def fig_prompt_usage(prompt: pd.DataFrame):
    setup_paper_style()
    d = prompt.groupby("prompt", as_index=False)["alpha"].mean().sort_values("prompt")
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    x = np.arange(len(d))
    ax.bar(x, d["alpha"], color=COL["blue"], alpha=0.82, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in d["prompt"]])
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("Mean routing weight")
    ax.set_title("Prompt routing usage")
    ax.grid(axis="y", color=COL["light"], linewidth=0.8)
    return fig


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.maximum(denom, 1.0), out=np.zeros_like(cm), where=denom > 0)


def fig_confusion_delta(pred: pd.DataFrame):
    setup_paper_style()
    labels = ordered(pd.concat([pred["true_class"].astype(str), pred["baseline_pred"].astype(str), pred["ours_pred"].astype(str)]))
    delta = confusion(pred["true_class"].astype(str).tolist(), pred["ours_pred"].astype(str).tolist(), labels)
    delta -= confusion(pred["true_class"].astype(str).tolist(), pred["baseline_pred"].astype(str).tolist(), labels)
    delta *= 100.0
    lim = float(np.max(np.abs(delta))) or 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted vowel")
    ax.set_ylabel("True vowel")
    ax.set_title("Error reduction over global memory")
    fig.colorbar(im, ax=ax, label="Delta normalized confusion (%)")
    return fig


def main() -> None:
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gate = pd.read_csv(inp / "gema_gate_diagnostics.csv")
    memory = pd.read_csv(inp / "memory_trajectory.csv")
    pcmr = pd.read_csv(inp / "pcmr_decomposition.csv")
    prompt = pd.read_csv(inp / "prompt_routing.csv")
    pred = pd.read_csv(inp / "predictions.csv")

    figs = {
        "fig_nok_gema_absorption_by_vowel": fig_gema_absorption_by_vowel(gate),
        "fig_nok_gema_adaptation_by_vowel": fig_gema_adaptation_by_vowel(memory),
        "fig_nok_pcmr_attention_shift_heatmap": fig_pcmr_shift_heatmap(pcmr),
        "fig_nok_pcmr_true_attention_by_vowel": fig_pcmr_true_attention_by_vowel(pcmr),
        "fig_nok_prompt_usage": fig_prompt_usage(prompt),
        "fig_nok_confusion_delta": fig_confusion_delta(pred),
    }
    for name, fig in figs.items():
        save_figure(fig, out, name, formats=args.formats)
        print(f"[write] {name}", flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
