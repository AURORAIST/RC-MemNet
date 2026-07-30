#!/usr/bin/env python3
"""Retrain/evaluate PC-DLCMNet for the 01-14 target areas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"

TARGETS = [
    ("01_Dangtu", "01当涂", "region", "01当涂"),
    ("02_Wuhu", "02芜湖", "region", "02芜湖"),
    ("03_Chizhou", "03池州", "region", "03池州"),
    ("04_Qingyang", "04青阳", "region", "04青阳"),
    ("05_Suncun", "05孙村", "region", "05孙村"),
    ("06_Tongling", "06铜陵", "region", "06铜陵"),
    ("07_Xuancheng", "07宣城", "region", "07宣城"),
    ("08_Jingxian", "08泾县", "region", "08泾县"),
    ("09_Fanchang", "09繁昌", "region", "09繁昌"),
    ("10_Nanling", "10南陵", "region", "10南陵"),
    ("11_Huangshan", "11黄山", "region", "11黄山"),
    ("12_Ningguo", "12宁国", "site", "12宁国"),
    ("13_Gaochun", "13高淳", "region", "13高淳"),
    ("14_Lishui", "14溧水", "site", "14溧水"),
]

REUSE_SIX = {
    "04_Qingyang": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Qingyang",
    "06_Tongling": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Tongling",
    "08_Jingxian": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Jingxian",
    "10_Nanling": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Nanling",
    "12_Ningguo": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Ningguo",
    "14_Lishui": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Lishui",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--csv", default="data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-shot", type=int, default=4)
    parser.add_argument("--train-support-shots", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--early-stop-patience", type=int, default=200)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--reuse-six", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-stems", nargs="*", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def usable_result(path: Path, holdout: str, column: str, args: argparse.Namespace) -> bool:
    result = path / "result.json"
    checkpoint = path / "result.pt"
    if not result.exists() or not checkpoint.exists():
        return False
    try:
        data = load_json(result)
    except Exception:
        return False
    config = data.get("config", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    return (
        data.get("metrics", {}).get("macro_f1") is not None
        and str(config.get("holdout_region")) == holdout
        and str(config.get("region_column")) == column
        and int(config.get("eval_support_shots", -1)) == int(args.k_shot)
        and int(split.get("runs", config.get("eval_split_runs", -1))) == int(args.num_tasks)
    )


def copy_reuse(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["result.json", "result.pt", "audit.json", "curve.csv", "debug.json", "predictions.csv"]:
        source = src / name
        if source.exists() and not (dst / name).exists():
            shutil.copy2(source, dst / name)


def command(args: argparse.Namespace, column: str, holdout: str, out_dir: Path) -> list[str]:
    return [
        str(args.python),
        "-u",
        str(RUNNER),
        "--csv",
        str(args.csv),
        "--region-column",
        column,
        "--holdout-region",
        holdout,
        "--device",
        str(args.device),
        "--eval-support-shots",
        str(args.k_shot),
        "--eval-split-runs",
        str(args.num_tasks),
        "--eval-split-mode",
        "global_support",
        "--support-shots",
        str(args.train_support_shots),
        "--max-steps",
        str(args.max_steps),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--early-stop-min-steps",
        str(args.early_stop_min_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
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
        "--episode-length",
        str(args.episode_length),
        "--episode-batch-size",
        str(args.episode_batch_size),
        "--eval-ensemble-runs",
        "1",
        "--log-every",
        "50",
        "--eval-every",
        "50",
        "--save-checkpoint",
        "--output",
        str(out_dir / "result.json"),
        "--curve-output",
        str(out_dir / "curve.csv"),
        "--debug-output",
        str(out_dir / "debug.json"),
        "--audit-output",
        str(out_dir / "audit.json"),
        "--predictions-output",
        str(out_dir / "predictions.csv"),
    ]


def run_target(args: argparse.Namespace, out_root: Path, stem: str, label: str, column: str, holdout: str) -> tuple[str, int]:
    out_dir = out_root / stem
    if args.reuse_six and stem in REUSE_SIX and not usable_result(out_dir, holdout, column, args):
        copy_reuse(REUSE_SIX[stem], out_dir)
    if args.skip_existing and usable_result(out_dir, holdout, column, args):
        print(f"[skip] {label} -> {out_dir}", flush=True)
        return "skipped", 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(args, column, holdout, out_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
    with (out_dir / "train.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).returncode
    status = "done" if rc == 0 and usable_result(out_dir, holdout, column, args) else "failed"
    print(f"[{status}] {label} rc={rc}", flush=True)
    return status, int(rc)


def fmt_pm(mean: Any, std: Any) -> str:
    try:
        return f"{float(mean) * 100.0:.2f} ± {float(std) * 100.0:.2f}"
    except Exception:
        return ""


def write_summary(out_root: Path) -> None:
    rows = []
    for stem, label, _column, _holdout in TARGETS:
        result_path = out_root / stem / "result.json"
        row: dict[str, Any] = {"Target": label, "stem": stem, "result_file": str(result_path)}
        if result_path.exists():
            try:
                data = load_json(result_path)
                split = data.get("evaluation_splits", {}).get("test", {})
                std = split.get("metrics_std", {}) if isinstance(split, dict) else {}
                metrics = data.get("metrics", {})
                row.update(
                    {
                        "macro_f1": metrics.get("macro_f1", ""),
                        "macro_f1_std": std.get("macro_f1", ""),
                        "accuracy": metrics.get("accuracy", ""),
                        "accuracy_std": std.get("accuracy", ""),
                        "status": "done" if metrics.get("macro_f1") is not None else "pending",
                    }
                )
            except Exception as exc:
                row["status"] = f"error: {exc}"
        else:
            row["status"] = "pending"
        rows.append(row)
    with (out_root / "main_table_14_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_root / "main_table_14_summary_with_std.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "Test Macro-F1", "Test Acc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Target": row["Target"],
                    "Test Macro-F1": fmt_pm(row.get("macro_f1"), row.get("macro_f1_std")),
                    "Test Acc": fmt_pm(row.get("accuracy"), row.get("accuracy_std")),
                }
            )


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir) if args.output_dir else ROOT / f"output/0614/main_table_retrain_pcdlcmnet_14targets_4shot_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    selected_targets = TARGETS
    if args.target_stems:
        wanted = set(args.target_stems)
        selected_targets = [target for target in TARGETS if target[0] in wanted]
        missing = sorted(wanted.difference({target[0] for target in selected_targets}))
        if missing:
            raise ValueError(f"unknown target stems: {missing}")
    print(json.dumps({"output_dir": str(out_root), "targets": len(selected_targets)}, ensure_ascii=False), flush=True)
    failures = []
    for stem, label, column, holdout in selected_targets:
        status, rc = run_target(args, out_root, stem, label, column, holdout)
        write_summary(out_root)
        if status == "failed":
            failures.append((label, rc))
            break
    write_summary(out_root)
    if failures:
        raise SystemExit(f"failures: {failures}")
    print(json.dumps({"summary": str(out_root / "main_table_14_summary_with_std.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
