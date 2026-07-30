#!/usr/bin/env python3
"""Pure prototype k-shot sensitivity for the 01-14 target checkpoints."""

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


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"
TARGETS = [
    ("01_Dangtu", "01当涂"),
    ("02_Wuhu", "02芜湖"),
    ("03_Chizhou", "03池州"),
    ("04_Qingyang", "04青阳"),
    ("05_Suncun", "05孙村"),
    ("06_Tongling", "06铜陵"),
    ("07_Xuancheng", "07宣城"),
    ("08_Jingxian", "08泾县"),
    ("09_Fanchang", "09繁昌"),
    ("10_Nanling", "10南陵"),
    ("11_Huangshan", "11黄山"),
    ("12_Ningguo", "12宁国"),
    ("13_Gaochun", "13高淳"),
    ("14_Lishui", "14溧水"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-values", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-stems", nargs="*", default=None)
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


def build_command(args: argparse.Namespace, source_dir: Path, stem: str, k: int, out_dir: Path) -> list[str]:
    data = load_json(source_dir / stem / "result.json")
    checkpoint = source_dir / stem / "result.pt"
    return [
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


def run_job(args: argparse.Namespace, source_dir: Path, output_dir: Path, stem: str, label: str, k: int) -> tuple[str, int, str, int]:
    out_dir = output_dir / stem / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / "result.json"
    if args.skip_existing and usable(result, k, args.eval_split_runs):
        return label, k, "skipped", 0
    cmd = build_command(args, source_dir, stem, k, out_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
    with (out_dir / "eval.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace").returncode
    status = "done" if rc == 0 and usable(result, k, args.eval_split_runs) else "failed"
    return label, k, status, int(rc)


def collect_row(output_dir: Path, stem: str, label: str, k: int, runs: int) -> dict[str, Any]:
    result = output_dir / stem / f"k{k}" / "result.json"
    row: dict[str, Any] = {"Target": label, "K_SHOT": int(k), "result_file": str(result)}
    if not usable(result, k, runs):
        row["status"] = "pending"
        return row
    data = load_json(result)
    split = data.get("evaluation_splits", {}).get("test", {})
    metrics = data.get("metrics", {})
    std = split.get("metrics_std", {}) if isinstance(split, dict) else {}
    row.update(
        {
            "status": "done",
            "macro_f1": metrics.get("macro_f1", ""),
            "macro_f1_std": std.get("macro_f1", ""),
            "accuracy": metrics.get("accuracy", ""),
            "accuracy_std": std.get("accuracy", ""),
            "eval_split_runs": split.get("runs", ""),
        }
    )
    return row


def fmt_pm(mean: Any, std: Any) -> str:
    try:
        return f"{float(mean) * 100.0:.2f} ± {float(std) * 100.0:.2f}"
    except Exception:
        return ""


def write_tables(output_dir: Path, k_values: list[int], runs: int, targets: list[tuple[str, str]] | None = None) -> int:
    selected = TARGETS if targets is None else targets
    rows = [collect_row(output_dir, stem, label, k, runs) for stem, label in selected for k in k_values]
    with (output_dir / "pure_kshot_14_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "pure_kshot_14_summary_with_std.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "K_SHOT", "Test Macro-F1", "Test Acc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Target": row["Target"],
                    "K_SHOT": row["K_SHOT"],
                    "Test Macro-F1": fmt_pm(row.get("macro_f1"), row.get("macro_f1_std")),
                    "Test Acc": fmt_pm(row.get("accuracy"), row.get("accuracy_std")),
                }
            )
    aggregates = []
    done = [row for row in rows if row.get("status") == "done" and row.get("macro_f1") not in {"", None}]
    for k in k_values:
        part = [row for row in done if int(row["K_SHOT"]) == int(k)]
        if not part:
            continue
        mf1 = [float(row["macro_f1"]) for row in part]
        acc = [float(row["accuracy"]) for row in part]
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
    with (output_dir / "pure_kshot_14_across_region_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["K_SHOT", "regions", "Mean Macro-F1", "Across-region Macro-F1 std", "Mean Acc", "Across-region Acc std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    return len(done)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / f"output/0614/main_table_pure_kshot_14targets_prototype_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_targets = TARGETS
    if args.target_stems:
        wanted = set(args.target_stems)
        selected_targets = [target for target in TARGETS if target[0] in wanted]
        missing = sorted(wanted.difference({target[0] for target in selected_targets}))
        if missing:
            raise ValueError(f"unknown target stems: {missing}")
    jobs = [(stem, label, int(k)) for stem, label in selected_targets for k in args.k_values]
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(jobs)}, ensure_ascii=False), flush=True)
    write_tables(output_dir, [int(k) for k in args.k_values], int(args.eval_split_runs), selected_targets)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_job, args, source_dir, output_dir, stem, label, k): (stem, k) for stem, label, k in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            label, k, status, rc = future.result()
            done = write_tables(output_dir, [int(v) for v in args.k_values], int(args.eval_split_runs), selected_targets)
            print(f"[{idx}/{len(jobs)}] {status} {label} k={k} rc={rc}; completed={done}/{len(jobs)}", flush=True)
            if status == "failed":
                failures.append((label, k, rc))
    if failures:
        raise SystemExit(f"failures: {failures}")
    print(json.dumps({"summary": str(output_dir / "pure_kshot_14_summary_with_std.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
