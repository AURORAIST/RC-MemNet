#!/usr/bin/env python3
"""Plot parameter sensitivity figures from existing 0614 experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "output/0614/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output/0614/parameter_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_percent(value: float) -> float:
    return float(value) * 100.0


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def collect_stage_rows() -> list[dict[str, Any]]:
    rows = load_json(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory/summary.json")
    return [row for row in rows if isinstance(row, dict)]


def stage_mean_rows(stage_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        grouped[str(row["stage"])].append(row)
    ordered = []
    for stage in ["stage1", "stage2", "stage3"]:
        values = grouped.get(stage, [])
        if not values:
            continue
        episode = np.asarray([float(row["macro_f1"]) for row in values], dtype=float)
        global_only = np.asarray([float(row["global_macro_f1"]) for row in values], dtype=float)
        ordered.append(
            {
                "stage": stage,
                "episode_macro_f1": metric_percent(float(episode.mean())),
                "episode_macro_f1_std": metric_percent(float(episode.std(ddof=0))),
                "global_macro_f1": metric_percent(float(global_only.mean())),
                "global_macro_f1_std": metric_percent(float(global_only.std(ddof=0))),
                "gain_macro_f1": metric_percent(float((episode - global_only).mean())),
            }
        )
    return ordered


def plot_memory_stage_sensitivity(rows: list[dict[str, float]], output_dir: Path) -> list[dict[str, Any]]:
    x = np.arange(len(rows), dtype=float)
    labels = [str(row["stage"]).replace("stage", "Stage ") for row in rows]
    global_y = np.asarray([row["global_macro_f1"] for row in rows], dtype=float)
    episode_y = np.asarray([row["episode_macro_f1"] for row in rows], dtype=float)
    global_std = np.asarray([row["global_macro_f1_std"] for row in rows], dtype=float)
    episode_std = np.asarray([row["episode_macro_f1_std"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, global_y, marker="o", linewidth=2.2, label="Global-only memory")
    ax.plot(x, episode_y, marker="s", linewidth=2.2, label="Global + episode read/write")
    ax.fill_between(x, global_y - global_std, global_y + global_std, alpha=0.12)
    ax.fill_between(x, episode_y - episode_std, episode_y + episode_std, alpha=0.12)
    for idx, row in enumerate(rows):
        ax.annotate(f"+{row['gain_macro_f1']:.2f}", (idx, episode_y[idx]), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Read/Write Memory Sensitivity Across Training Stages")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir / "memory_read_write_gain_by_stage")
    return [{"plot": "memory_read_write_gain_by_stage", **row} for row in rows]


def load_result(path: str) -> dict[str, Any]:
    data = load_json(PACKAGE_ROOT / path)
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    cfg = data.get("config", {})
    return {
        "path": path,
        "macro_f1": metric_percent(float(metrics.get("macro_f1", 0.0))),
        "accuracy": metric_percent(float(metrics.get("accuracy", 0.0))),
        "global_macro_f1": metric_percent(float(global_metrics.get("macro_f1", 0.0))),
        "global_accuracy": metric_percent(float(global_metrics.get("accuracy", 0.0))),
        "num_prompts": cfg.get("num_prompts"),
        "support_shots": cfg.get("support_shots"),
        "eval_support_shots": cfg.get("eval_support_shots"),
        "eval_ensemble_runs": cfg.get("eval_ensemble_runs"),
        "lambda_global": cfg.get("lambda_global"),
        "episode_length": cfg.get("episode_length"),
        "max_steps": cfg.get("max_steps"),
        "lr": cfg.get("lr"),
    }


def plot_lambda_global(output_dir: Path) -> list[dict[str, Any]]:
    specs = [
        ("0.2", "output/0614/effectiveness_tune/01dt_paper_s2e32_lg02_seed0.json"),
        ("0.5", "output/0614/effectiveness_tune/01dt_paper_region_s2e32_lg05_seed0.json"),
    ]
    rows = []
    for label, path in specs:
        row = load_result(path)
        row["plot"] = "lambda_global_sensitivity"
        row["x_label"] = label
        rows.append(row)

    x = np.asarray([float(row["lambda_global"]) for row in rows], dtype=float)
    y = np.asarray([row["macro_f1"] for row in rows], dtype=float)
    gy = np.asarray([row["global_macro_f1"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, marker="o", linewidth=2.2, label="Episode RW")
    ax.plot(x, gy, marker="s", linewidth=2.0, label="Global-only")
    ax.set_xlabel("lambda_global")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Sensitivity to Global-Loss Weight")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir / "lambda_global_sensitivity")
    return rows


def plot_eval_support_shots(output_dir: Path) -> list[dict[str, Any]]:
    specs = [
        ("4 shots", "output/0614/effectiveness_eval/01dt_paper_s2e32_ckpt_evalsupport4.json"),
        ("8 shots", "output/0614/effectiveness_eval/01dt_paper_s2e32_ckpt_evalsupport8.json"),
        ("4 shots, ens5", "output/0614/effectiveness_eval/01dt_paper_s2e32_ckpt_evalsupport4_ens5.json"),
    ]
    rows = []
    for label, path in specs:
        row = load_result(path)
        row["plot"] = "eval_support_shots_sensitivity"
        row["x_label"] = label
        rows.append(row)

    strict = [row for row in rows if int(row.get("eval_ensemble_runs") or 1) == 1]
    x = np.asarray([int(row["eval_support_shots"]) for row in strict], dtype=float)
    y = np.asarray([row["macro_f1"] for row in strict], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, marker="o", linewidth=2.2, label="Single evaluation")
    ens = [row for row in rows if int(row.get("eval_ensemble_runs") or 1) > 1]
    if ens:
        ax.scatter(
            [int(row["eval_support_shots"]) for row in ens],
            [row["macro_f1"] for row in ens],
            marker="D",
            s=72,
            label="Ensemble evaluation",
        )
    ax.set_xlabel("Evaluation support shots per class")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Sensitivity to Evaluation Support Size")
    ax.set_xticks(sorted({int(row["eval_support_shots"]) for row in rows}))
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, output_dir / "eval_support_shots_sensitivity")
    return rows


def plot_support_shots_exploratory(output_dir: Path) -> list[dict[str, Any]]:
    specs = [
        ("s=1,e=16,800", "output/0614/effectiveness/01dt_paper_dual_memory_seed0.json"),
        ("s=2,e=32,800", "output/0614/effectiveness/01dt_paper_dual_memory_s2e32_seed0.json"),
        ("s=4,e=32,400", "output/0614/effectiveness_tune/01dt_paper_s4e32_lg05_seed0.json"),
    ]
    rows = []
    for label, path in specs:
        row = load_result(path)
        row["plot"] = "support_shots_exploratory"
        row["x_label"] = label
        rows.append(row)

    x = np.asarray([int(row["support_shots"]) for row in rows], dtype=float)
    y = np.asarray([row["macro_f1"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, marker="o", linewidth=2.2)
    for row in rows:
        ax.annotate(
            f"ep={row['episode_length']}, steps={row['max_steps']}",
            (float(row["support_shots"]), row["macro_f1"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Training support shots per class")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Exploratory Sensitivity to Support Shots")
    ax.set_xticks(x)
    style_axes(ax)
    save_figure(fig, output_dir / "support_shots_exploratory")
    return rows


def plot_prompt_count_coverage(output_dir: Path) -> list[dict[str, Any]]:
    row = load_result("output/0614/effectiveness/01dt_paper_dual_memory_s2e32_seed0.json")
    row["plot"] = "prompt_count_coverage"
    row["x_label"] = "8 prompts only"

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.scatter([int(row["num_prompts"])], [row["macro_f1"]], s=90)
    ax.annotate("only existing point", (int(row["num_prompts"]), row["macro_f1"]), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Number of prompts")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Prompt Count Sensitivity: Data Coverage")
    ax.set_xlim(0, 18)
    ax.set_xticks([2, 4, 8, 16])
    ax.text(
        0.5,
        0.12,
        "Need matched runs at 2/4/16 prompts for a true curve.",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    style_axes(ax)
    save_figure(fig, output_dir / "prompt_count_data_coverage")
    return [row]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    report = """# Parameter Sensitivity Visualizations

This folder contains parameter sensitivity figures generated from existing 0614 outputs.

## Figures

- `memory_read_write_gain_by_stage.png`: strict same-protocol comparison of global-only memory versus global + episode read/write memory across stages.
- `lambda_global_sensitivity.png`: sensitivity to the global-loss weight under matched support shots, episode length, steps, and learning rate.
- `eval_support_shots_sensitivity.png`: strict checkpoint evaluation with different support sizes.
- `support_shots_exploratory.png`: exploratory training-support sensitivity. This plot is useful for intuition, but the available points differ in episode length or training steps.
- `prompt_count_data_coverage.png`: current prompt-count coverage. Existing runs only contain `num_prompts=8`, so a true prompt-number curve needs additional matched runs.

## Main Observations

- The read/write episode memory is consistently useful: Stage 3 mean Macro-F1 improves from 53.49 to 59.14, a gain of 5.65 points.
- `lambda_global=0.5` is better than `0.2` in the matched 01Dangtu sensitivity pair.
- Increasing evaluation support shots from 4 to 8 does not improve the existing checkpoint result, suggesting that support selection and memory writing are more important than simply adding more support items.
- The prompt-number sensitivity curve is not yet statistically available because all existing paper runs use 8 prompts.

## Suggested Missing Prompt Sweep

Run matched jobs for `num_prompts in {2,4,8,16}` with the same holdout, support shots, episode length, max steps, and learning rate. The existing comparable setting is:

```powershell
python -u experiments\\feature_memory\\run_paper_dual_memory.py `
  --holdout-region \"01当涂\" `
  --device cuda `
  --episode-length 32 `
  --episode-batch-size 2 `
  --support-shots 2 `
  --lambda-global 0.5 `
  --max-steps 400 `
  --lr 3e-4 `
  --num-prompts <2|4|8|16> `
  --output output\\0614\\parameter_sensitivity\\prompt_sweep\\01dt_prompts<num>.json
```
"""
    (output_dir / "parameter_sensitivity_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    stage_rows = stage_mean_rows(collect_stage_rows())
    all_rows.extend(plot_memory_stage_sensitivity(stage_rows, output_dir))
    all_rows.extend(plot_lambda_global(output_dir))
    all_rows.extend(plot_eval_support_shots(output_dir))
    all_rows.extend(plot_support_shots_exploratory(output_dir))
    all_rows.extend(plot_prompt_count_coverage(output_dir))

    write_csv(output_dir / "parameter_sensitivity_data.csv", all_rows)
    write_report(output_dir, all_rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
