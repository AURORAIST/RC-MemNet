#!/usr/bin/env python3
"""Visualize per-region gains from episode read/write memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory"))
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/region_memory_gain"))
    parser.add_argument("--stage", default="stage3")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(input_root: Path, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.glob(f"*/*_{stage}.json")):
        try:
            data = load_json(path)
        except Exception as exc:
            rows.append({"result_file": str(path), "status": "failed", "error": str(exc)})
            continue
        metrics = data.get("metrics", {})
        global_metrics = data.get("global_metrics", {})
        split = data.get("split", {})
        config = data.get("config", {})
        region = split.get("holdout_region") or config.get("holdout_region") or path.parent.name
        prefix = "".join(ch for ch in str(region) if ch.isdigit())[:2] or "R"
        label = f"R{prefix}_{hashlib.sha1(str(region).encode('utf-8')).hexdigest()[:4]}"
        macro_f1 = metrics.get("macro_f1")
        global_macro_f1 = global_metrics.get("macro_f1")
        accuracy = metrics.get("accuracy")
        global_accuracy = global_metrics.get("accuracy")
        row = {
            "region": region,
            "plot_label": label,
            "stage": stage,
            "status": "done",
            "result_file": str(path),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "global_accuracy": global_accuracy,
            "global_macro_f1": global_macro_f1,
            "gain_accuracy": float(accuracy) - float(global_accuracy) if accuracy is not None and global_accuracy is not None else "",
            "gain_macro_f1": float(macro_f1) - float(global_macro_f1) if macro_f1 is not None and global_macro_f1 is not None else "",
            "evaluated_rows": data.get("diagnostics", {}).get("test", {}).get("evaluated_rows", ""),
            "episode_prompt_route_entropy": data.get("diagnostics", {}).get("test", {}).get("episode_prompt_route_entropy", ""),
            "global_prompt_route_entropy": data.get("diagnostics", {}).get("test", {}).get("global_prompt_route_entropy", ""),
        }
        rows.append(row)
    return rows


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "region_memory_gain.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "region_memory_gain.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    done = [row for row in rows if row.get("status") == "done" and row.get("macro_f1") not in {"", None}]
    lines = [
        "# Region Memory Gain",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Stage: `{stage}`",
        "",
    ]
    if done:
        mean_macro = sum(float(row["macro_f1"]) for row in done) / len(done)
        mean_global_macro = sum(float(row["global_macro_f1"]) for row in done) / len(done)
        mean_acc = sum(float(row["accuracy"]) for row in done) / len(done)
        mean_global_acc = sum(float(row["global_accuracy"]) for row in done) / len(done)
        positive = sum(1 for row in done if float(row["gain_macro_f1"]) > 0)
        lines.extend(
            [
                "## Summary",
                "",
                f"- Regions: {len(done)}",
                f"- Positive Macro-F1 gains: {positive}/{len(done)}",
                f"- Mean Global-only Macro-F1: {pct(mean_global_macro)}",
                f"- Mean Global + Episode RW Macro-F1: {pct(mean_macro)}",
                f"- Mean Macro-F1 gain: {pct(mean_macro - mean_global_macro)}",
                f"- Mean Global-only Acc: {pct(mean_global_acc)}",
                f"- Mean Global + Episode RW Acc: {pct(mean_acc)}",
                f"- Mean Acc gain: {pct(mean_acc - mean_global_acc)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-Region Results",
            "",
            "| Region | Global Macro-F1 | Episode RW Macro-F1 | Gain | Global Acc | Episode RW Acc | Gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(done, key=lambda item: float(item.get("gain_macro_f1", 0.0)), reverse=True):
        lines.append(
            f"| {row.get('region')} | {pct(row.get('global_macro_f1'))} | {pct(row.get('macro_f1'))} | "
            f"{pct(row.get('gain_macro_f1'))} | {pct(row.get('global_accuracy'))} | {pct(row.get('accuracy'))} | "
            f"{pct(row.get('gain_accuracy'))} |"
        )
    lines.extend(
        [
            "",
            "Figures:",
            "- `plots/region_macro_f1_comparison.png`",
            "- `plots/region_macro_f1_gain.png`",
            "- `plots/gain_vs_global_macro_f1.png`",
        ]
    )
    (output_dir / "region_memory_gain.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_rows(done, output_dir / "plots")


def plot_rows(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    if not rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in ["macro_f1", "global_macro_f1", "gain_macro_f1", "accuracy", "global_accuracy", "gain_accuracy"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values("gain_macro_f1", ascending=False)
    regions = frame["region"].astype(str).tolist()
    labels = frame["plot_label"].astype(str).tolist() if "plot_label" in frame.columns else regions
    x = range(len(frame))

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    width = 0.38
    ax.bar([i - width / 2 for i in x], frame["global_macro_f1"] * 100.0, width=width, label="Global-only", color="#7f8c8d")
    ax.bar([i + width / 2 for i in x], frame["macro_f1"] * 100.0, width=width, label="Global + Episode RW", color="#2e86ab")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Per-region effect of episode read/write memory")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "region_macro_f1_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    colors = ["#2e86ab" if v >= 0 else "#c0392b" for v in frame["gain_macro_f1"]]
    ax.bar(labels, frame["gain_macro_f1"] * 100.0, color=colors)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Macro-F1 gain (%)")
    ax.set_title("Episode read/write memory gain by region")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "region_macro_f1_gain.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(frame["global_macro_f1"] * 100.0, frame["gain_macro_f1"] * 100.0, s=52, color="#2e86ab")
    for _, row in frame.iterrows():
        ax.annotate(str(row.get("plot_label", row["region"])), (row["global_macro_f1"] * 100.0, row["gain_macro_f1"] * 100.0), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Global-only Macro-F1 (%)")
    ax.set_ylabel("Episode RW gain (%)")
    ax.set_title("Does memory help harder regions more?")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "gain_vs_global_macro_f1.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.input_root), str(args.stage))
    write_outputs(rows, Path(args.output_dir), str(args.stage))
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
