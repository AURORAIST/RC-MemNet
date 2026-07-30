#!/usr/bin/env python3
"""Pure k-shot sensitivity evaluation for the six main-table checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"
DEFAULT_SOURCE = PACKAGE_ROOT / "output/0614/main_table_retrain_pcdlcmnet_4shot_20260624_212342"
AREAS = ["Qingyang", "Tongling", "Jingxian", "Nanling", "Ningguo", "Lishui"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-values", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cfg(data: dict[str, Any], key: str, default: Any = "") -> Any:
    return data.get("config", {}).get(key, default)


def usable(path: Path, k: int, runs: int) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    split = data.get("evaluation_splits", {}).get("test", {})
    return (
        data.get("metrics", {}).get("macro_f1") is not None
        and int(data.get("config", {}).get("eval_support_shots", -1)) == int(k)
        and int(split.get("runs", -1)) == int(runs)
        and bool(data.get("config", {}).get("eval_query_only", False))
        and str(data.get("config", {}).get("eval_split_mode", "")) == "fixed_query"
        and str(data.get("config", {}).get("eval_classifier", "")) == "prototype"
    )


def build_command(args: argparse.Namespace, source_dir: Path, area: str, k: int, out_dir: Path) -> list[str]:
    data = load_json(source_dir / area / "result.json")
    checkpoint = source_dir / area / "result.pt"
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--csv",
        str(cfg(data, "csv")),
        "--region-column",
        str(cfg(data, "region_column")),
        "--holdout-region",
        str(cfg(data, "holdout_region")),
        "--device",
        str(args.device),
        "--seed",
        str(cfg(data, "seed", 0)),
        "--max-steps",
        "0",
        "--init-checkpoint",
        str(checkpoint),
        "--support-shots",
        str(cfg(data, "support_shots", 2)),
        "--eval-support-shots",
        str(k),
        "--eval-split-runs",
        str(args.eval_split_runs),
        "--eval-split-mode",
        "fixed_query",
        "--eval-query-only",
        "--eval-query-shots-per-class",
        str(args.eval_query_shots_per_class),
        "--eval-ensemble-runs",
        "1",
        "--eval-classifier",
        "prototype",
        "--prototype-fallback-global",
        "--prototype-support-weight",
        "1.0",
        "--support-write-mode",
        "label",
        "--support-label-blend",
        "1.0",
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
        "--episode-length",
        str(cfg(data, "episode_length", 32)),
        "--episode-batch-size",
        str(cfg(data, "episode_batch_size", 2)),
        "--token-chunks",
        str(cfg(data, "token_chunks", 32)),
        "--aux-source",
        str(cfg(data, "aux_source", "acoustic")),
        "--aux-representation",
        str(cfg(data, "aux_representation", "sequence")),
        "--aux-cache-dir",
        str(cfg(data, "aux_cache_dir", "output/0614/feature_memory_aux_cache")),
        "--cache-dir",
        str(cfg(data, "cache_dir")),
        "--matrix-cache-dir",
        str(cfg(data, "matrix_cache_dir")),
        "--skip-val-eval",
        "--quiet",
        "--no-progress",
        "--output",
        str(out_dir / "result.json"),
        "--debug-output",
        str(out_dir / "debug.json"),
        "--audit-output",
        str(out_dir / "audit.json"),
    ]
    return command


def run_job(args: argparse.Namespace, source_dir: Path, output_dir: Path, area: str, k: int) -> tuple[str, int, str, int]:
    out_dir = output_dir / area / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if args.skip_existing and usable(result_path, k, args.eval_split_runs):
        return area, k, "skipped", 0
    command = build_command(args, source_dir, area, k, out_dir)
    env = os.environ.copy()
    env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
    with (out_dir / "eval.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        returncode = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).returncode
    status = "done" if returncode == 0 and usable(result_path, k, args.eval_split_runs) else "failed"
    return area, k, status, int(returncode)


def collect_row(output_dir: Path, area: str, k: int, runs: int) -> dict[str, Any]:
    out_dir = output_dir / area / f"k{k}"
    result_path = out_dir / "result.json"
    row: dict[str, Any] = {
        "Target": area,
        "K_SHOT": int(k),
        "result_file": str(result_path),
        "log_file": str(out_dir / "eval.log"),
    }
    if not usable(result_path, k, runs):
        row["status"] = "pending"
        return row
    data = load_json(result_path)
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    metrics_std = split.get("metrics_std", {}) if isinstance(split, dict) else {}
    global_std = split.get("global_metrics_std", {}) if isinstance(split, dict) else {}
    row.update(
        {
            "status": "done",
            "Test Macro-F1": metrics.get("macro_f1", ""),
            "Test Macro-F1 std": metrics_std.get("macro_f1", ""),
            "Test Acc": metrics.get("accuracy", ""),
            "Test Acc std": metrics_std.get("accuracy", ""),
            "Global Macro-F1": global_metrics.get("macro_f1", ""),
            "Global Macro-F1 std": global_std.get("macro_f1", ""),
            "Global Acc": global_metrics.get("accuracy", ""),
            "Global Acc std": global_std.get("accuracy", ""),
            "eval_split_runs": split.get("runs", ""),
        }
    )
    return row


def fmt_pct(value: Any, std: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f} ± {float(std) * 100.0:.2f}"
    except Exception:
        return ""


def write_tables(output_dir: Path, k_values: list[int], runs: int) -> int:
    rows = [collect_row(output_dir, area, k, runs) for area in AREAS for k in k_values]
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "pure_kshot_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "pure_kshot_summary_with_std.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "K_SHOT", "Test Macro-F1", "Test Acc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Target": row["Target"],
                    "K_SHOT": row["K_SHOT"],
                    "Test Macro-F1": fmt_pct(row.get("Test Macro-F1"), row.get("Test Macro-F1 std")),
                    "Test Acc": fmt_pct(row.get("Test Acc"), row.get("Test Acc std")),
                }
            )
    aggregates = []
    done = [row for row in rows if row.get("status") == "done" and row.get("Test Macro-F1") not in {"", None}]
    for k in k_values:
        part = [row for row in done if int(row["K_SHOT"]) == int(k)]
        if not part:
            continue
        mf1 = [float(row["Test Macro-F1"]) for row in part]
        acc = [float(row["Test Acc"]) for row in part]
        aggregates.append(
            {
                "K_SHOT": int(k),
                "regions": len(part),
                "Mean Macro-F1": statistics.mean(mf1),
                "Across-region Macro-F1 std": statistics.stdev(mf1) if len(mf1) > 1 else 0.0,
                "Mean Acc": statistics.mean(acc),
                "Across-region Acc std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            }
        )
    with (output_dir / "pure_kshot_across_region_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["K_SHOT", "regions", "Mean Macro-F1", "Across-region Macro-F1 std", "Mean Acc", "Across-region Acc std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(done)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else PACKAGE_ROOT / f"output/0614/main_table_pure_kshot_prototype_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(area, int(k)) for area in AREAS for k in args.k_values]
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "k_values": [int(k) for k in args.k_values],
                "eval_split_runs": int(args.eval_split_runs),
                "eval_query_only": True,
                "eval_split_mode": "fixed_query",
                "eval_classifier": "prototype",
                "eval_query_shots_per_class": int(args.eval_query_shots_per_class),
                "workers": int(args.workers),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(jobs), "workers": int(args.workers)}, ensure_ascii=False), flush=True)
    write_tables(output_dir, [int(k) for k in args.k_values], int(args.eval_split_runs))
    failures: list[tuple[str, int, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_job, args, source_dir, output_dir, area, k): (area, k) for area, k in jobs}
        for idx, future in enumerate(as_completed(futures), start=1):
            area, k, status, returncode = future.result()
            done = write_tables(output_dir, [int(v) for v in args.k_values], int(args.eval_split_runs))
            print(f"[{idx}/{len(jobs)}] {status} {area} k={k} rc={returncode}; completed={done}/{len(jobs)}", flush=True)
            if status == "failed":
                failures.append((area, k, returncode))
    if failures:
        raise SystemExit(f"failures: {failures}")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_with_std": str(output_dir / "pure_kshot_summary_with_std.csv"),
                "summary": str(output_dir / "pure_kshot_summary.csv"),
                "aggregate": str(output_dir / "pure_kshot_across_region_summary.csv"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
