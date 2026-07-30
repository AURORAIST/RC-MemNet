#!/usr/bin/env python3
"""Run representative 100-split ablations and build paper tables."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc_dlcmnet.utils.paths import (
    default_audio_root,
    default_dataset,
    default_matrix_cache,
    default_whisper_cache,
)

WORKSPACE = ROOT
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"
DATASET = default_dataset()
WHISPER_CACHE = default_whisper_cache()
MATRIX_CACHE = default_matrix_cache()
AUDIO_ROOT = default_audio_root()
OUT_DIR = ROOT / "output/0614/component_ablation_representative_global_support_s100"
PAPER_DIR = ROOT / "output/0614/paper_ready_results"


TARGETS = {
    "qingyang": {
        "area": "Qingyang",
        "region_column": "region",
        "holdout": "04\u9752\u9633",
        "ckpt": ROOT / "output/0614/multiregion_paper_dual_memory/04_1a770e49/04_1a770e49_stage3.pt",
        "train_output": None,
    },
    "ningguo": {
        "area": "Ningguo",
        "region_column": "site",
        "holdout": "12\u5b81\u56fd",
        "ckpt": ROOT / "output/0614/site_domain_paper_dual_memory/12_ningguo/12_ningguo_stage3.pt",
        "train_output": ROOT / "output/0614/site_domain_paper_dual_memory/12_ningguo/12_ningguo_stage3.json",
    },
    "lishui": {
        "area": "Lishui",
        "region_column": "site",
        "holdout": "14\u6ea7\u6c34",
        "ckpt": ROOT / "output/0614/site_domain_paper_dual_memory/14_lishui/14_lishui_stage3.pt",
        "train_output": ROOT / "output/0614/site_domain_paper_dual_memory/14_lishui/14_lishui_stage3.json",
    },
}


COMPONENT_VARIANTS = {
    "support_prototype": {
        "method": "Support prototype",
        "args": ["--eval-classifier", "prototype", "--no-prototype-fallback-global", "--prototype-support-weight", "1.0"],
    },
    "global_only_memory": {
        "method": "Global-only memory",
        "args": ["--eval-classifier", "global_memory"],
    },
    "global_support_average": {
        "method": "Global-support average",
        "args": ["--eval-classifier", "prototype", "--prototype-fallback-global", "--prototype-support-weight", "0.5"],
    },
    "gema_only": {
        "method": "GEMA only",
        "args": ["--eval-classifier", "gema_cosine", "--support-write-mode", "label", "--support-label-blend", "1.0"],
    },
    "pcdlcmnet": {
        "method": "PC-DLCMNet",
        "args": ["--eval-classifier", "memory", "--support-write-mode", "label", "--support-label-blend", "1.0"],
    },
}


WEIGHTING_VARIANTS = {
    "pseudo": ["--eval-classifier", "memory", "--support-write-mode", "pseudo", "--support-label-blend", "0.0"],
    "blend": ["--eval-classifier", "memory", "--support-write-mode", "blend", "--support-label-blend", "0.5"],
    "label": ["--eval-classifier", "memory", "--support-write-mode", "label", "--support-label-blend", "1.0"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--train-eval-every", type=int, default=100)
    parser.add_argument("--train-early-stop-patience", type=int, default=500)
    parser.add_argument("--train-early-stop-min-steps", type=int, default=800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shared_command(args: argparse.Namespace, target: dict[str, Any], output: Path) -> list[str]:
    return [
        args.python,
        "-u",
        str(RUNNER),
        "--holdout-region",
        str(target["holdout"]),
        "--region-column",
        str(target["region_column"]),
        "--device",
        args.device,
        "--seed",
        "0",
        "--episode-length",
        "256",
        "--episode-batch-size",
        "2",
        "--support-shots",
        "2",
        "--eval-support-shots",
        str(args.k),
        "--eval-ensemble-runs",
        "1",
        "--eval-split-runs",
        str(args.splits),
        "--eval-query-only",
        "--eval-split-mode",
        "global_support",
        "--eval-query-shots-per-class",
        "1",
        "--lambda-global",
        "0.5",
        "--hidden-dim",
        "256",
        "--score-dim",
        "128",
        "--prompt-dim",
        "128",
        "--num-prompts",
        "8",
        "--layers",
        "3",
        "--heads",
        "4",
        "--ffn-dim",
        "768",
        "--dropout",
        "0.1",
        "--temperature",
        "0.2",
        "--token-chunks",
        "32",
        "--aux-source",
        "acoustic",
        "--aux-representation",
        "sequence",
        "--aux-cache-dir",
        "output/0614/feature_memory_aux_cache",
        "--output",
        str(output),
        "--skip-val-eval",
        "--quiet",
        "--progress",
        "--csv",
        str(DATASET),
        "--cache-dir",
        str(WHISPER_CACHE),
        "--matrix-cache-dir",
        str(MATRIX_CACHE),
        "--audio-root",
        str(AUDIO_ROOT),
    ]


def cli_path(path: Path) -> str:
    try:
        return os.path.relpath(Path(path), WORKSPACE)
    except Exception:
        return str(path)


def train_command(args: argparse.Namespace, target: dict[str, Any]) -> list[str]:
    output = target.get("train_output") or OUT_DIR / f"{target['area'].lower()}_train.json"
    cmd = shared_command(args, target, Path(output))
    cmd.extend(
        [
            "--max-steps",
            str(args.train_steps),
            "--eval-classifier",
            "memory",
            "--support-write-mode",
            "label",
            "--support-label-blend",
            "1.0",
            "--lr",
            "3e-4",
            "--weight-decay",
            "1e-4",
            "--label-smoothing",
            "0.0",
            "--class-weight-power",
            "0.5",
            "--eval-every",
            str(args.train_eval_every),
            "--eval-split-runs",
            "1",
            "--eval-ensemble-runs",
            "1",
            "--early-stop-patience",
            str(args.train_early_stop_patience),
            "--early-stop-min-steps",
            str(args.train_early_stop_min_steps),
            "--save-checkpoint",
            "--checkpoint-output",
            cli_path(Path(target["ckpt"])),
            "--progress",
        ]
    )
    return cmd


def eval_command(args: argparse.Namespace, target: dict[str, Any], output: Path) -> list[str]:
    cmd = shared_command(args, target, output)
    cmd.extend(["--max-steps", "0", "--init-checkpoint", cli_path(Path(target["ckpt"]))])
    return cmd


def run_one(cmd: list[str], output: Path, force: bool, dry_run: bool) -> None:
    if output.exists() and not force:
        print(f"[skip] {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)
    print(f"[done] {output} time={time.perf_counter() - start:.1f}s", flush=True)


def nested_get(data: dict[str, Any], dotted: str) -> float:
    cur: Any = data
    for part in dotted.split("."):
        cur = cur[part]
    return float(cur)


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100.0:.2f}"


def pct_pm(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "--"
    if std_value is None:
        return pct(mean_value)
    return f"{mean_value * 100.0:.2f} $\\pm$ {std_value * 100.0:.2f}"


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean_std(data: dict[str, Any], metric_key: str = "accuracy") -> tuple[float, float | None]:
    mean_value = nested_get(data, f"metrics.{metric_key}")
    split = data.get("evaluation_splits", {}).get("test", {})
    std_value = None
    if isinstance(split, dict):
        metrics_std = split.get("metrics_std", {})
        if isinstance(metrics_std, dict) and metric_key in metrics_std:
            std_value = float(metrics_std[metric_key])
    return mean_value, std_value


def global_metric_mean_std(data: dict[str, Any], metric_key: str = "accuracy") -> tuple[float, float | None]:
    mean_value = nested_get(data, f"global_metrics.{metric_key}")
    split = data.get("evaluation_splits", {}).get("test", {})
    std_value = None
    if isinstance(split, dict):
        metrics_std = split.get("global_metrics_std", {})
        if isinstance(metrics_std, dict) and metric_key in metrics_std:
            std_value = float(metrics_std[metric_key])
    return mean_value, std_value


def aggregate_mean_std(values: list[tuple[float | None, float | None]]) -> tuple[float | None, float | None]:
    valid_means = [mean for mean, _ in values if mean is not None]
    if not valid_means:
        return None, None
    mean_value = sum(valid_means) / len(valid_means)
    valid_stds = [std for mean, std in values if mean is not None and std is not None]
    if not valid_stds:
        return mean_value, None
    std_value = (sum(std * std for std in valid_stds) ** 0.5) / len(valid_stds)
    return mean_value, std_value


def ensure_checkpoint(args: argparse.Namespace, target: dict[str, Any]) -> None:
    ckpt = Path(target["ckpt"])
    if ckpt.exists():
        print(f"[ckpt] {ckpt}", flush=True)
        return
    if not args.train_missing:
        raise FileNotFoundError(f"Missing checkpoint for {target['area']}: {ckpt}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # A previous run may have written metrics but failed before torch.save.
    # In that case the checkpoint, not the JSON file, is the source of truth.
    run_one(train_command(args, target), Path(target["train_output"]), force=True, dry_run=args.dry_run)


def build_tables() -> None:
    component_values: dict[str, dict[str, tuple[float, float | None]]] = {key: {} for key in COMPONENT_VARIANTS}
    weighting_values: dict[str, dict[str, tuple[float, float | None]]] = {key: {} for key in WEIGHTING_VARIANTS}

    for target_name, target in TARGETS.items():
        for variant_key in COMPONENT_VARIANTS:
            data = load_result(OUT_DIR / f"{target_name}_{variant_key}_s100.json")
            area = str(target["area"])
            component_values[variant_key][area] = metric_mean_std(data, "accuracy")
        for mode_key in WEIGHTING_VARIANTS:
            data = load_result(OUT_DIR / f"{target_name}_weight_{mode_key}_s100.json")
            weighting_values[mode_key][str(target["area"])] = metric_mean_std(data, "accuracy")

    areas = ["Qingyang", "Ningguo", "Lishui"]
    component_rows = []
    for key, meta in COMPONENT_VARIANTS.items():
        row = {"Method": meta["method"]}
        for area in areas:
            mean_value, std_value = component_values[key].get(area, (None, None))
            row[area] = mean_value
            row[f"{area}_std"] = std_value
        row["Avg."], row["Avg._std"] = aggregate_mean_std([(row[a], row[f"{a}_std"]) for a in areas])
        component_rows.append(row)

    component_df = pd.DataFrame(component_rows)
    weighting_df = pd.DataFrame(
        [
            {
                "Target Region": area,
                "Pseudo-label": weighting_values["pseudo"].get(area, (None, None))[0],
                "Pseudo-label_std": weighting_values["pseudo"].get(area, (None, None))[1],
                "Blended": weighting_values["blend"].get(area, (None, None))[0],
                "Blended_std": weighting_values["blend"].get(area, (None, None))[1],
                "Label-aware": weighting_values["label"].get(area, (None, None))[0],
                "Label-aware_std": weighting_values["label"].get(area, (None, None))[1],
            }
            for area in areas
        ]
    )
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    component_df.to_csv(PAPER_DIR / "component_ablation_table_representative_global_support_s100.csv", index=False, encoding="utf-8-sig")
    weighting_df.to_csv(PAPER_DIR / "support_weighting_table_representative_global_support_s100.csv", index=False, encoding="utf-8-sig")

    flags = {
        "Support prototype": ("\\checkmark", "$\\times$", "$\\times$", "$\\times$"),
        "Global-only memory": ("$\\times$", "\\checkmark", "$\\times$", "$\\times$"),
        "Global-support average": ("\\checkmark", "\\checkmark", "$\\times$", "$\\times$"),
        "GEMA only": ("\\checkmark", "\\checkmark", "\\checkmark", "$\\times$"),
        "PC-DLCMNet": ("\\checkmark", "\\checkmark", "\\checkmark", "\\checkmark"),
    }
    bs = " \\\\"
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Component ablation study of PC-DLCMNet under the cross-region $K$-shot setting. Accuracy (\\%) is averaged over 100 random target-support splits. Each split samples $K$ labeled support examples per class from the target region and evaluates on the remaining target samples.}",
        "\\label{tab:component_ablation}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc|cccc}",
        "\\toprule",
        "Method & Support & Global & GEMA & PCMR & Qingyang & Ningguo & Lishui & Avg." + bs,
        "\\midrule",
    ]
    for _, row in component_df.iterrows():
        support, global_flag, gema, pcmr = flags[row["Method"]]
        lines.append(
            f"{row['Method']} & {support} & {global_flag} & {gema} & {pcmr} & "
            f"{pct_pm(row['Qingyang'], row['Qingyang_std'])} & "
            f"{pct_pm(row['Ningguo'], row['Ningguo_std'])} & "
            f"{pct_pm(row['Lishui'], row['Lishui_std'])} & "
            f"{pct_pm(row['Avg.'], row['Avg._std'])}" + bs
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""])
    lines.extend(
        [
            "\\begin{table}[!t]",
            "\\centering",
            "\\caption{Analysis of support evidence weighting strategies in GEMA. Accuracy (\\%) is averaged over 100 random target-support splits.}",
            "\\label{tab:support_weighting}",
            "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l|ccc}",
            "\\toprule",
            "Target Region & \\multicolumn{3}{c}{Accuracy (\\%)}" + bs,
            "\\cline{2-4}",
            " & Pseudo-label & Blended & Label-aware" + bs,
            "\\midrule",
        ]
    )
    for _, row in weighting_df.iterrows():
        lines.append(
            f"{row['Target Region']} & "
            f"{pct_pm(row['Pseudo-label'], row['Pseudo-label_std'])} & "
            f"{pct_pm(row['Blended'], row['Blended_std'])} & "
            f"{pct_pm(row['Label-aware'], row['Label-aware_std'])}" + bs
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}"])
    tex = "\n".join(lines) + "\n"
    (PAPER_DIR / "component_ablation_and_support_weighting_tables_global_support_s100.tex").write_text(tex, encoding="utf-8")
    print(tex, flush=True)


def main() -> None:
    args = parse_args()
    for target_name, target in TARGETS.items():
        ensure_checkpoint(args, target)
        for variant_key, variant in COMPONENT_VARIANTS.items():
            output = OUT_DIR / f"{target_name}_{variant_key}_s100.json"
            run_one(eval_command(args, target, output) + variant["args"], output, args.force, args.dry_run)
        for mode_key, mode_args in WEIGHTING_VARIANTS.items():
            output = OUT_DIR / f"{target_name}_weight_{mode_key}_s100.json"
            run_one(eval_command(args, target, output) + mode_args, output, args.force, args.dry_run)
    if not args.dry_run:
        build_tables()


if __name__ == "__main__":
    main()
