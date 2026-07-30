#!/usr/bin/env python3
"""Run the PCMR x episode-memory factor ablation on three target regions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"

TARGETS = {
    "Qingyang": {
        "region_column": "region",
        "holdout": "04青阳",
        "ckpt": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Qingyang/result.pt",
    },
    "Ningguo": {
        "region_column": "site",
        "holdout": "12宁国",
        "ckpt": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Ningguo/result.pt",
    },
    "Lishui": {
        "region_column": "site",
        "holdout": "14溧水",
        "ckpt": ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342/Lishui/result.pt",
    },
}

RUN_VARIANTS = {
    "global_standard": {
        "label": "Global memory + standard reader",
        "eval_classifier": "global_memory",
    },
    "episode_standard": {
        "label": "Episode memory + standard reader",
        "eval_classifier": "gema_cosine",
    },
    "episode_pcmr": {
        "label": "Episode memory + PCMR",
        "eval_classifier": "memory",
    },
}

TABLE_ROWS = [
    ("Global memory + standard reader", "global_standard", "metrics"),
    ("Global memory + PCMR", "episode_pcmr", "global_metrics"),
    ("Episode memory + standard reader", "episode_standard", "metrics"),
    ("Episode memory + PCMR", "episode_pcmr", "metrics"),
    ("Full PC-DLCMNet", "episode_pcmr", "metrics"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / f"output/0614/reader_memory_factor_ablation_3targets_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--csv", default="data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv")
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--eval-query-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def cli_path(path: Path) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except Exception:
        return str(path)


def build_command(args: argparse.Namespace, target: dict[str, Any], variant: dict[str, str], output: Path) -> list[str]:
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--csv",
        str(args.csv),
        "--holdout-region",
        str(target["holdout"]),
        "--region-column",
        str(target["region_column"]),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--episode-length",
        str(args.episode_length),
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
        "--eval-split-mode",
        "global_support",
        "--eval-query-shots-per-class",
        "1",
        "--eval-classifier",
        str(variant["eval_classifier"]),
        "--support-write-mode",
        "label",
        "--support-label-blend",
        "1.0",
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
        "--max-steps",
        "0",
        "--init-checkpoint",
        cli_path(Path(target["ckpt"])),
    ]
    if bool(args.eval_query_only):
        command.append("--eval-query-only")
    return command


def result_is_usable(path: Path, target: dict[str, Any], variant: dict[str, str], args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    cfg = data.get("config", {})
    return (
        data.get("metrics", {}).get("accuracy") is not None
        and str(cfg.get("holdout_region")) == str(target["holdout"])
        and str(cfg.get("region_column")) == str(target["region_column"])
        and str(cfg.get("eval_classifier")) == str(variant["eval_classifier"])
        and int(cfg.get("eval_support_shots", -1)) == int(args.k)
    )


def run_one(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        f.flush()
        proc = subprocess.run(command, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    return int(proc.returncode)


def load_metric(output_dir: Path, area: str, run_key: str, section: str, metric: str) -> float | None:
    path = output_dir / area / f"{run_key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(section, {}).get(metric)
    return None if value is None else float(value)


def write_metric_table(output_dir: Path, metric: str, areas: list[str]) -> None:
    table_path = output_dir / f"reader_memory_factor_ablation_{metric}.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Variant", *areas])
        writer.writeheader()
        for label, run_key, section in TABLE_ROWS:
            row: dict[str, Any] = {"Variant": label}
            for area in areas:
                value = load_metric(output_dir, area, run_key, section, metric)
                row[area] = "" if value is None else round(value * 100.0, 2)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    wanted = [area for area in args.targets if area in TARGETS]
    missing = sorted(set(args.targets).difference(TARGETS))
    if missing:
        raise ValueError(f"unknown targets: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    print(json.dumps({"output_dir": str(output_dir), "targets": wanted, "jobs": len(wanted) * len(RUN_VARIANTS)}, ensure_ascii=False), flush=True)

    for area in wanted:
        target = TARGETS[area]
        if not Path(target["ckpt"]).exists():
            raise FileNotFoundError(target["ckpt"])
        for run_key, variant in RUN_VARIANTS.items():
            out = output_dir / area / f"{run_key}.json"
            log = output_dir / area / f"{run_key}.log"
            command = build_command(args, target, variant, out)
            if args.skip_existing and result_is_usable(out, target, variant, args):
                status, rc = "skipped", 0
            else:
                rc = run_one(command, log, args.dry_run)
                status = "dry_run" if args.dry_run else ("done" if rc == 0 else "failed")
            rows.append(
                {
                    "area": area,
                    "variant": run_key,
                    "label": variant["label"],
                    "eval_classifier": variant["eval_classifier"],
                    "status": status,
                    "returncode": rc,
                    "result_file": str(out),
                    "log_file": str(log),
                    "command": subprocess.list2cmdline(command),
                }
            )
            print(f"[{status}] {area} {run_key}", flush=True)

    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.dry_run:
        write_metric_table(output_dir, "accuracy", wanted)
        write_metric_table(output_dir, "macro_f1", wanted)


if __name__ == "__main__":
    main()
