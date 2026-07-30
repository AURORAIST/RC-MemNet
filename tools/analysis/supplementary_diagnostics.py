#!/usr/bin/env python3
"""Generate supplementary diagnostics for PC-DLCMNet experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory"))
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/supplementary_diagnostics"))
    parser.add_argument("--stage", default="stage3")
    parser.add_argument("--bootstrap-runs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confusion-regions", nargs="*", default=[])
    parser.add_argument("--top-confusion-regions", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def region_label(region: str) -> str:
    prefix = "".join(ch for ch in str(region) if ch.isdigit())[:2] or "R"
    digest = hashlib.sha1(str(region).encode("utf-8")).hexdigest()[:4]
    return f"R{prefix}_{digest}"


def collect_stage_rows(input_root: Path, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.glob(f"*/*_{stage}.json")):
        data = load_json(path)
        split = data.get("split", {})
        config = data.get("config", {})
        region = split.get("holdout_region") or config.get("holdout_region") or path.parent.name
        metrics = data.get("metrics", {})
        global_metrics = data.get("global_metrics", {})
        diag = data.get("diagnostics", {}).get("test", {})
        feature_weights = {
            key: value
            for key, value in diag.items()
            if str(key).startswith("feature_weight_")
        }
        prediction_path = data.get("files", {}).get("predictions") or str(path.with_suffix(".predictions.csv"))
        prediction_path = Path(prediction_path)
        if not prediction_path.is_absolute():
            prediction_path = PACKAGE_ROOT / prediction_path
        rows.append(
            {
                "region": str(region),
                "plot_label": region_label(str(region)),
                "result_file": str(path),
                "prediction_file": str(prediction_path),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "global_macro_f1": float(global_metrics.get("macro_f1", 0.0)),
                "global_accuracy": float(global_metrics.get("accuracy", 0.0)),
                "gain_macro_f1": float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0)),
                "gain_accuracy": float(metrics.get("accuracy", 0.0)) - float(global_metrics.get("accuracy", 0.0)),
                "episode_prompt_route_entropy": diag.get("episode_prompt_route_entropy", ""),
                "global_prompt_route_entropy": diag.get("global_prompt_route_entropy", ""),
                **feature_weights,
            }
        )
    return rows


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_stats(values_a: np.ndarray, values_b: np.ndarray, bootstrap_runs: int, seed: int) -> dict[str, Any]:
    diff = values_a - values_b
    n = len(diff)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = std_diff / math.sqrt(max(1, n))
    t_value = mean_diff / se if se > 0 else float("inf")
    # Normal approximation keeps this dependency-light; n=16 is reported with bootstrap CI as the main interval.
    p_approx = float(2.0 * (1.0 - normal_cdf(abs(t_value)))) if math.isfinite(t_value) else 0.0
    rng = np.random.default_rng(int(seed))
    boot = []
    if n > 0 and bootstrap_runs > 0:
        for _ in range(int(bootstrap_runs)):
            sample = diff[rng.integers(0, n, size=n)]
            boot.append(float(sample.mean()))
    boot_arr = np.asarray(boot, dtype=np.float64)
    ci_low, ci_high = (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))) if len(boot_arr) else (mean_diff, mean_diff)
    return {
        "n": int(n),
        "mean_a": float(values_a.mean()) if n else 0.0,
        "mean_b": float(values_b.mean()) if n else 0.0,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_value": float(t_value),
        "p_value_normal_approx": p_approx,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "positive_count": int((diff > 0).sum()),
    }


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_paired_gain(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("gain_macro_f1", ascending=False)
    labels = frame["plot_label"].tolist()
    x = np.arange(len(frame))
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(labels, frame["gain_macro_f1"] * 100.0, color="#2e86ab")
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_ylabel("Macro-F1 gain (%)")
    ax.set_title("Paired gain of episode read/write memory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "paired_region_gain.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_prompt_feature_diagnostics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for col in ["gain_macro_f1", "episode_prompt_route_entropy", "global_prompt_route_entropy"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(frame["episode_prompt_route_entropy"], frame["gain_macro_f1"] * 100.0, s=55, color="#2e86ab")
    for _, row in frame.iterrows():
        ax.annotate(str(row["plot_label"]), (row["episode_prompt_route_entropy"], row["gain_macro_f1"] * 100.0), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Episode prompt route entropy")
    ax.set_ylabel("Macro-F1 gain (%)")
    ax.set_title("Prompt routing vs memory gain")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "prompt_entropy_vs_gain.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    feature_cols = sorted([col for col in frame.columns if col.startswith("feature_weight_")])
    if feature_cols:
        weights = frame[feature_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.bar(feature_cols, weights * 100.0, color="#d9822b")
        ax.set_ylabel("Average branch weight (%)")
        ax.set_title("Average acoustic feature branch weights")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "feature_branch_weights.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def load_prediction_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def plot_confusion_pair(region_row: dict[str, Any], output_dir: Path) -> None:
    path = Path(region_row["prediction_file"])
    if not path.exists():
        return
    frame = load_prediction_frame(path)
    if "vowel" not in frame.columns or "pred_vowel" not in frame.columns or "global_pred_vowel" not in frame.columns:
        return
    labels = [label for label in VOWEL_ORDER if label in set(frame["vowel"].astype(str))]
    if not labels:
        labels = list(VOWEL_ORDER)
    y_true = frame["vowel"].astype(str).tolist()
    episode_pred = frame["pred_vowel"].astype(str).tolist()
    global_pred = frame["global_pred_vowel"].astype(str).tolist()
    cms = [
        ("Global-only", confusion_matrix(y_true, global_pred, labels=labels)),
        ("Episode RW", confusion_matrix(y_true, episode_pred, labels=labels)),
    ]
    plot_dir = output_dir / "plots" / "confusion_matrices"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    vmax = max(int(cm.max()) for _, cm in cms) if cms else 1
    for ax, (title, cm) in zip(axes, cms):
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum
        image = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] > 0:
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="Row-normalized rate")
    fig.suptitle(f"Confusion matrices: {region_row['plot_label']}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe = str(region_row["plot_label"])
    fig.savefig(plot_dir / f"{safe}_global_vs_episode_confusion.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def choose_confusion_regions(rows: list[dict[str, Any]], requested: list[str], top_n: int) -> list[dict[str, Any]]:
    if requested:
        wanted = set(requested)
        chosen = [row for row in rows if row["region"] in wanted or row["plot_label"] in wanted]
        if chosen:
            return chosen
    sorted_rows = sorted(rows, key=lambda row: float(row["gain_macro_f1"]), reverse=True)
    low_base = sorted(rows, key=lambda row: float(row["global_macro_f1"]))
    selected = []
    seen = set()
    for row in sorted_rows[: max(1, top_n)] + low_base[: max(1, top_n)]:
        key = row["region"]
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= max(1, top_n):
            break
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_stage_rows(Path(args.input_root), str(args.stage))
    write_csv(rows, output_dir / "region_diagnostics.csv")

    episode_mf1 = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    global_mf1 = np.asarray([row["global_macro_f1"] for row in rows], dtype=np.float64)
    episode_acc = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    global_acc = np.asarray([row["global_accuracy"] for row in rows], dtype=np.float64)
    stats = {
        "macro_f1": paired_stats(episode_mf1, global_mf1, int(args.bootstrap_runs), int(args.seed)),
        "accuracy": paired_stats(episode_acc, global_acc, int(args.bootstrap_runs), int(args.seed) + 17),
    }
    (output_dir / "paired_significance.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_paired_gain(rows, output_dir)
    plot_prompt_feature_diagnostics(rows, output_dir)

    chosen = choose_confusion_regions(rows, [str(v) for v in args.confusion_regions], int(args.top_confusion_regions))
    for row in chosen:
        plot_confusion_pair(row, output_dir)

    lines = [
        "# Supplementary Diagnostics",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Paired Significance",
        "",
        "| Metric | Global Mean | Episode RW Mean | Gain | 95% Bootstrap CI | Positive Regions | p approx |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, row in stats.items():
        lines.append(
            f"| {metric} | {pct(row['mean_b'])} | {pct(row['mean_a'])} | {pct(row['mean_diff'])} | "
            f"[{pct(row['bootstrap_ci_low'])}, {pct(row['bootstrap_ci_high'])}] | "
            f"{row['positive_count']}/{row['n']} | {row['p_value_normal_approx']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `plots/paired_region_gain.png`",
            "- `plots/prompt_entropy_vs_gain.png`",
            "- `plots/feature_branch_weights.png`",
            "- `plots/confusion_matrices/*_global_vs_episode_confusion.png`",
            "",
            "## Confusion Matrix Regions",
            "",
        ]
    )
    for row in chosen:
        lines.append(f"- {row['plot_label']}: {row['region']} (gain={pct(row['gain_macro_f1'])})")
    (output_dir / "supplementary_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"regions": len(rows), "confusion_regions": len(chosen), "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
