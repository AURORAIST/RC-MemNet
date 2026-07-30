#!/usr/bin/env python3
"""Plot 4x8 hyperparameter sensitivity panels from per-area raw results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps"
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "summary.csv"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "hyperparam_accuracy_4x8.jpg"
DEFAULT_BASELINE_TABLES = [
    ROOT / "output/0614/paper_ready_results/main_acc_table_global_support_s100.csv",
    ROOT / "output/0614/paper_ready_results/main_acc_4shot_add2_missing_baselines.csv",
]

SWEEPS = ["num_prompts", "lambda_global", "temperature", "episode_length"]
SWEEP_LABELS = {
    "num_prompts": "Prompt number",
    "lambda_global": r"$\lambda_{global}$",
    "temperature": "Temperature",
    "episode_length": "Episode length",
}
AREA_ORDER = ["Qingyang", "Tongling", "Jingxian", "Nanling", "Ningguo", "Lishui", "Chizhou", "Huangshan"]
GCOPE_COLOR = "#1f77b4"
SHADE_COLOR = "#9ecae1"
BASELINE_COLOR = "#d62728"
BASELINE_METHOD = "WavLM"
SHADE_ALPHA = 0.34
ERRORBAR_ALPHA = 0.45
STD_VIS_SCALE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Raw per-area summary.csv from the sweep run.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JPG path.")
    parser.add_argument(
        "--baseline-tables",
        nargs="*",
        default=[str(path) for path in DEFAULT_BASELINE_TABLES],
        help="Wide paper-ready baseline CSV tables. The script uses WavLM by default.",
    )
    parser.add_argument("--baseline-method", default=BASELINE_METHOD, help="Method name used as the red dashed baseline.")
    parser.add_argument(
        "--std-vis-scale",
        type=float,
        default=STD_VIS_SCALE,
        help="Visual-only multiplier for the accuracy_std band.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def as_float(value: str) -> float:
    return float(value.strip())


def load_rows(summary: Path) -> list[dict[str, str]]:
    with summary.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw_row in reader:
            row = {key.lstrip("\ufeff"): value for key, value in raw_row.items()}
            if row.get("returncode") == "0" and row.get("status") in {"done", "skipped"}:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no completed rows found in {summary}")
    return rows


def method_matches(name: str, target: str) -> bool:
    clean_name = name.lower().replace("~", "").replace(" ", "").replace("-", "")
    clean_target = target.lower().replace("~", "").replace(" ", "").replace("-", "")
    return clean_name == clean_target


def load_baselines(paths: list[Path], method: str) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for raw_row in csv.DictReader(f):
                row = {key.lstrip("\ufeff"): value for key, value in raw_row.items()}
                if not method_matches(row.get("Method", ""), method):
                    continue
                for area in AREA_ORDER:
                    value = row.get(area, "")
                    if value:
                        baselines[area] = as_float(value) * 100.0
    return baselines


def rows_by_sweep_area(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["sweep"], row["area"])
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        group.sort(key=lambda r: as_float(r["value"]))
    return grouped


def main() -> None:
    args = parse_args()
    summary = Path(args.summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = load_rows(summary)
    grouped = rows_by_sweep_area(rows)
    baselines = load_baselines([Path(path) for path in args.baseline_tables], args.baseline_method)

    fig, axes = plt.subplots(
        nrows=len(SWEEPS),
        ncols=len(AREA_ORDER),
        figsize=(26, 12.8),
        sharey=False,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, sweep in enumerate(SWEEPS):
        for col_idx, area in enumerate(AREA_ORDER):
            ax = axes[row_idx][col_idx]
            data = grouped.get((sweep, area), [])
            x = [as_float(item["value"]) for item in data]
            y = [as_float(item["accuracy"]) * 100.0 for item in data]
            y_std = [as_float(item.get("accuracy_std", "0")) * 100.0 * float(args.std_vis_scale) for item in data]
            y_low = [max(0.0, mean - std) for mean, std in zip(y, y_std)]
            y_high = [min(100.0, mean + std) for mean, std in zip(y, y_std)]

            ax.set_facecolor("#f2f2f2")

            if x:
                ax.fill_between(x, y_low, y_high, color=SHADE_COLOR, alpha=SHADE_ALPHA, linewidth=0)
                ax.errorbar(
                    x,
                    y,
                    yerr=y_std,
                    fmt="none",
                    ecolor=GCOPE_COLOR,
                    elinewidth=0.75,
                    capsize=2.2,
                    capthick=0.75,
                    alpha=ERRORBAR_ALPHA,
                    zorder=2,
                )
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.9,
                    markersize=4.2,
                    color=GCOPE_COLOR,
                    label="GCOPE",
                )
            if area in baselines:
                ax.axhline(
                    baselines[area],
                    color=BASELINE_COLOR,
                    linestyle="--",
                    linewidth=1.35,
                    alpha=0.95,
                    label=args.baseline_method,
                )

            y_candidates = y_low + y_high
            if area in baselines:
                y_candidates.append(baselines[area])
            if y_candidates:
                ymin = max(0.0, min(y_candidates) - 4.0)
                ymax = min(100.0, max(y_candidates) + 4.0)
                if ymax - ymin < 14.0:
                    mid = (ymax + ymin) / 2.0
                    ymin = max(0.0, mid - 7.0)
                    ymax = min(100.0, mid + 7.0)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(0, 100)

            ax.grid(True, axis="y", color="white", linewidth=0.8, alpha=0.95)
            ax.grid(True, axis="x", color="white", linewidth=0.8, alpha=0.95)
            ax.set_xticks(x)
            ax.tick_params(axis="x", labelsize=8.2, colors="#1f1f1f")
            ax.tick_params(axis="y", labelsize=8.2, colors="#1f1f1f")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#c7c7c7")
            ax.spines["bottom"].set_color("#c7c7c7")
            ax.set_title(area, fontsize=11, pad=7)
            ax.set_ylabel("Acc", fontsize=10.5)
            ax.set_xlabel(SWEEP_LABELS[sweep], fontsize=10.5)

            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="best", fontsize=8, frameon=True, framealpha=0.8)

    fig.suptitle("Hyperparameter sensitivity across eight regions", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.94, bottom=0.055, wspace=0.34, hspace=0.55)
    fig.savefig(output, dpi=int(args.dpi), pil_kwargs={"quality": 95})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
