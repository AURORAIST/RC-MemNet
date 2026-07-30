#!/usr/bin/env python3
"""Evaluate trained PC-DLCMNet checkpoints with repeated random K-shot splits.

This script fixes the trained model parameters and repeats target-region support
sampling multiple times. It reports mean and standard deviation over random
few-shot splits, which is the appropriate way to describe K-shot evaluation when
the support examples are randomly selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import pandas as pd

RUNNER = PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", default="output/0614/multiregion_paper_dual_memory")
    parser.add_argument("--output-dir", default="output/0614/random_fewshot_repeats")
    parser.add_argument("--csv", default="")
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--use-all-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-support-shots", type=int, default=4)
    parser.add_argument("--eval-split-runs", type=int, default=10)
    parser.add_argument("--eval-ensemble-runs", type=int, default=1)
    parser.add_argument("--support-write-mode", choices=["label", "pseudo", "blend"], default="label")
    parser.add_argument("--support-label-blend", type=float, default=1.0)
    parser.add_argument("--eval-split-mode", choices=["global_support", "rolling", "fixed_query"], default="global_support")
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet-child", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-child", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cfg(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get("config", {}).get(key, default)


def region_stem(region: str) -> str:
    prefix_match = re.match(r"^(\d+)", str(region).strip())
    prefix = prefix_match.group(1) if prefix_match else "region"
    digest = __import__("hashlib").sha1(str(region).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def discover_stage3(stage_dir: Path) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    found: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for json_path in sorted(stage_dir.glob("*/*_stage3.json")):
        try:
            data = load_json(json_path)
        except Exception:
            continue
        region = str(data.get("config", {}).get("holdout_region") or data.get("split", {}).get("holdout_region") or "")
        if not region:
            continue
        pt_path = json_path.with_suffix(".pt")
        if not pt_path.exists():
            continue
        found[region] = (json_path, pt_path, data)
    return found


def discover_regions_from_csv(csv_path: str) -> list[str]:
    if not csv_path:
        return []
    df = pd.read_csv(csv_path)
    return sorted(df["region"].astype(str).unique().tolist())


def result_is_usable(path: Path, args: argparse.Namespace, region: str) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    config = data.get("config", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    return (
        "macro_f1" in data.get("metrics", {})
        and str(config.get("holdout_region")) == str(region)
        and int(config.get("eval_support_shots", -1)) == int(args.eval_support_shots)
        and int(config.get("eval_ensemble_runs", -1)) == int(args.eval_ensemble_runs)
        and int(split.get("runs", config.get("eval_split_runs", -1))) == int(args.eval_split_runs)
        and str(config.get("eval_split_mode")) == str(args.eval_split_mode)
        and str(config.get("eval_classifier")) == "memory"
        and bool(config.get("eval_query_only", False))
    )


def build_command(args: argparse.Namespace, region: str, stage_json: Path, stage_pt: Path, output_json: Path) -> list[str]:
    data = load_json(stage_json)
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--max-steps",
        "0",
        "--init-checkpoint",
        str(stage_pt),
        "--holdout-region",
        str(region),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--episode-length",
        str(cfg(data, "episode_length", 32)),
        "--episode-batch-size",
        str(cfg(data, "episode_batch_size", 2)),
        "--support-shots",
        str(cfg(data, "support_shots", 2)),
        "--eval-support-shots",
        str(args.eval_support_shots),
        "--eval-ensemble-runs",
        str(args.eval_ensemble_runs),
        "--eval-split-runs",
        str(args.eval_split_runs),
        "--support-write-mode",
        str(args.support_write_mode),
        "--support-label-blend",
        str(args.support_label_blend),
        "--eval-query-only",
        "--eval-split-mode",
        str(args.eval_split_mode),
        "--eval-query-shots-per-class",
        str(args.eval_query_shots_per_class),
        "--eval-classifier",
        "memory",
        "--lambda-global",
        str(cfg(data, "lambda_global", 0.5)),
        "--hidden-dim",
        str(cfg(data, "hidden_dim", 256)),
        "--score-dim",
        str(cfg(data, "score_dim", 128)),
        "--prompt-dim",
        str(cfg(data, "prompt_dim", 128)),
        "--num-prompts",
        str(cfg(data, "num_prompts", 8)),
        "--layers",
        str(cfg(data, "layers", 3)),
        "--heads",
        str(cfg(data, "heads", 4)),
        "--ffn-dim",
        str(cfg(data, "ffn_dim", 768)),
        "--dropout",
        str(cfg(data, "dropout", 0.1)),
        "--temperature",
        str(cfg(data, "temperature", 0.2)),
        "--token-chunks",
        str(cfg(data, "token_chunks", 32)),
        "--aux-source",
        str(cfg(data, "aux_source", "acoustic")),
        "--aux-representation",
        str(cfg(data, "aux_representation", "sequence")),
        "--aux-cache-dir",
        str(cfg(data, "aux_cache_dir", "output/0614/feature_memory_aux_cache")),
        "--output",
        str(output_json),
        "--skip-val-eval",
    ]
    if args.quiet_child:
        command.append("--quiet")
    command.append("--progress" if args.progress_child else "--no-progress")
    for optional in ["csv", "cache_dir", "matrix_cache_dir", "audio_root"]:
        value = cfg(data, optional, "")
        if optional == "csv" and args.csv:
            value = args.csv
        if value:
            command.extend([f"--{optional.replace('_', '-')}", str(value)])
    return command


def run_job(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        f.flush()
        if dry_run:
            return 0
        process = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return int(process.returncode)


def std_of_gains(split_summary: dict[str, Any], key: str) -> float | str:
    metrics = split_summary.get("metrics", [])
    global_metrics = split_summary.get("global_metrics", [])
    gains = []
    for row, global_row in zip(metrics, global_metrics):
        try:
            gains.append(float(row[key]) - float(global_row[key]))
        except Exception:
            pass
    if len(gains) > 1:
        return float(statistics.stdev(gains))
    if len(gains) == 1:
        return 0.0
    return ""


def collect_result(region: str, output_json: Path, log_path: Path, status: str, returncode: int, command: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": region,
        "status": status,
        "returncode": int(returncode),
        "result_file": str(output_json),
        "log_file": str(log_path),
        "command": subprocess.list2cmdline(command),
    }
    if not output_json.exists():
        return row
    try:
        data = load_json(output_json)
    except Exception as exc:
        row["error"] = str(exc)
        return row
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    metrics_std = split.get("metrics_std", {}) if isinstance(split, dict) else {}
    global_std = split.get("global_metrics_std", {}) if isinstance(split, dict) else {}
    diag = data.get("diagnostics", {}).get("test", {})
    row.update(
        {
            "eval_split_runs": split.get("runs", "") if isinstance(split, dict) else "",
            "eval_support_shots": data.get("config", {}).get("eval_support_shots", ""),
            "eval_ensemble_runs": data.get("config", {}).get("eval_ensemble_runs", ""),
            "accuracy": metrics.get("accuracy", ""),
            "accuracy_std": metrics_std.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "macro_f1_std": metrics_std.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "weighted_f1_std": metrics_std.get("weighted_f1", ""),
            "global_accuracy": global_metrics.get("accuracy", ""),
            "global_accuracy_std": global_std.get("accuracy", ""),
            "global_macro_f1": global_metrics.get("macro_f1", ""),
            "global_macro_f1_std": global_std.get("macro_f1", ""),
            "gain_accuracy": (
                float(metrics.get("accuracy", 0.0)) - float(global_metrics.get("accuracy", 0.0))
                if metrics and global_metrics
                else ""
            ),
            "gain_accuracy_std": std_of_gains(split, "accuracy") if isinstance(split, dict) else "",
            "gain_macro_f1": (
                float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0))
                if metrics and global_metrics
                else ""
            ),
            "gain_macro_f1_std": std_of_gains(split, "macro_f1") if isinstance(split, dict) else "",
            "evaluated_rows": diag.get("evaluated_rows", ""),
            "mean_support_size": diag.get("mean_support_size", ""),
            "mean_query_size": diag.get("mean_query_size", ""),
        }
    )
    return row


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    done = [row for row in rows if row.get("status") in {"done", "skipped"} and row.get("macro_f1") not in {"", None}]
    aggregate: dict[str, Any] = {"regions": len(done)}
    for key in ["accuracy", "macro_f1", "weighted_f1", "global_accuracy", "global_macro_f1", "gain_accuracy", "gain_macro_f1"]:
        values = [float(row[key]) for row in done if row.get(key) not in {"", None}]
        if values:
            aggregate[key] = float(statistics.mean(values))
            aggregate[f"{key}_region_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    for key in ["accuracy_std", "macro_f1_std", "global_accuracy_std", "global_macro_f1_std", "gain_accuracy_std", "gain_macro_f1_std"]:
        values = [float(row[key]) for row in done if row.get(key) not in {"", None}]
        if values:
            aggregate[f"mean_{key}"] = float(statistics.mean(values))
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "aggregate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(aggregate.keys()))
        writer.writeheader()
        writer.writerow(aggregate)


def main() -> None:
    args = parse_args()
    stage_dir = Path(args.stage_dir)
    output_dir = Path(args.output_dir)
    found = discover_stage3(stage_dir)
    if args.regions:
        regions = args.regions
    elif args.use_all_regions:
        regions = sorted(found.keys())
    else:
        regions = discover_regions_from_csv(args.csv)
    if not regions:
        raise ValueError("No regions selected.")

    rows: list[dict[str, Any]] = []
    for region in regions:
        if region not in found:
            print(f"[skip] no stage3 checkpoint for region={region}", flush=True)
            continue
        stage_json, stage_pt, _data = found[region]
        stem = region_stem(region)
        output_json = output_dir / "results" / f"{stem}_k{args.eval_support_shots}_splits{args.eval_split_runs}.json"
        log_path = output_dir / "logs" / f"{stem}_k{args.eval_support_shots}_splits{args.eval_split_runs}.log"
        command = build_command(args, region, stage_json, stage_pt, output_json)
        if args.skip_completed and result_is_usable(output_json, args, region):
            status = "skipped"
            returncode = 0
        else:
            print(f"[run] {region} k={args.eval_support_shots} splits={args.eval_split_runs}", flush=True)
            returncode = run_job(command, log_path, args.dry_run)
            status = "done" if returncode == 0 else "failed"
        rows.append(collect_result(region, output_json, log_path, status, returncode, command))
        write_tables(rows, output_dir)
    write_tables(rows, output_dir)
    aggregate_path = output_dir / "aggregate.json"
    print(aggregate_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
