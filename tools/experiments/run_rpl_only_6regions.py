#!/usr/bin/env python3
"""Train six-region RPL-only checkpoints for downstream representation plots.

This is the no-RGMA ablation used as the RPL Only panel: context prompts remain
enabled, while memory read/write adaptation is disabled via no_memory_adapter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = sys.executable
SCRIPT_PATH = ROOT / "tools/experiments/run_rc_dmnet.py"
CSV_PATH = ROOT / "output/0722/data/wu_vowel_segments.paper_regions.csv"
CACHE_DIR = Path("/home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base")
AUX_CACHE_DIR = ROOT / "output/0722/feature_memory_aux_cache"
MATRIX_CACHE_DIR = Path("/home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache")

TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("06_Tongling", "06铜陵", "Tongling"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("10_Nanling", "10南陵", "Nanling"),
    ("12_Ningguo", "12宁国", "Ningguo"),
    ("14_Lishui", "14溧水", "Lishui"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-variant", default="no_memory_adapter")
    parser.add_argument(
        "--output-base",
        default=str(ROOT / "output/0722/rpl_only_6regions_20260727"),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-split-runs", type=int, default=1)
    parser.add_argument("--target-adapt-steps", type=int, default=30)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, region_folder: str, region_name: str, clean_name: str) -> dict[str, float] | None:
    out_dir = Path(args.output_base) / region_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json = out_dir / "result.json"
    checkpoint = out_dir / "result.pt"

    if result_json.exists() and checkpoint.exists() and not args.force:
        data = json.loads(result_json.read_text(encoding="utf-8"))
        return {
            "accuracy": float(data["metrics"]["accuracy"]) * 100.0,
            "macro_f1": float(data["metrics"]["macro_f1"]) * 100.0,
            "query_size": float(data["metrics"]["query_size"]),
        }

    cmd = [
        PYTHON_BIN,
        "-u",
        str(SCRIPT_PATH),
        "--csv",
        str(CSV_PATH),
        "--output",
        str(result_json),
        "--checkpoint-output",
        str(checkpoint),
        "--save-checkpoint",
        "--holdout-region",
        region_name,
        "--region-column",
        "paper_region",
        "--seed",
        "0",
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--eval-split-runs",
        str(args.eval_split_runs),
        "--target-adapt-steps",
        str(args.target_adapt_steps),
        "--target-support-shots",
        str(args.target_support_shots),
        "--batch-size",
        "64",
        "--eval-batch-size",
        "256",
        "--hidden-dim",
        "256",
        "--score-dim",
        "128",
        "--prompt-dim",
        "128",
        "--num-slots",
        "4",
        "--write-top-k",
        "1",
        "--num-prompts",
        "8",
        "--device",
        str(args.device),
        "--cache-dir",
        str(CACHE_DIR),
        "--aux-cache-dir",
        str(AUX_CACHE_DIR),
        "--matrix-cache-dir",
        str(MATRIX_CACHE_DIR),
        "--ablation-variant",
        str(args.ablation_variant),
    ]

    started = time.time()
    print(f"[run] {clean_name}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "run.log").write_text(res.stdout, encoding="utf-8")
    if res.returncode != 0:
        print(f"[error] {clean_name}: return code {res.returncode}", flush=True)
        print(res.stdout[-3000:], flush=True)
        return None

    data = json.loads(result_json.read_text(encoding="utf-8"))
    print(f"[done] {clean_name}: {(time.time() - started) / 60.0:.1f} min", flush=True)
    return {
        "accuracy": float(data["metrics"]["accuracy"]) * 100.0,
        "macro_f1": float(data["metrics"]["macro_f1"]) * 100.0,
        "query_size": float(data["metrics"]["query_size"]),
    }


def main() -> None:
    args = parse_args()
    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    started = time.time()

    print(f"RPL-only six-region run -> {out_base}", flush=True)
    print(
        f"Protocol: seed=0, max_steps={args.max_steps}, target_support_shots={args.target_support_shots}, "
        f"target_adapt_steps={args.target_adapt_steps}, eval_split_runs={args.eval_split_runs}, device={args.device}",
        flush=True,
    )
    for idx, (region_folder, region_name, clean_name) in enumerate(TARGET_REGIONS, start=1):
        print(f"\n[{idx}/{len(TARGET_REGIONS)}] {clean_name}", flush=True)
        metrics = run_one(args, region_folder, region_name, clean_name)
        if metrics is not None:
            summary[clean_name] = metrics
            print(
                f"  acc={metrics['accuracy']:.2f}, macro_f1={metrics['macro_f1']:.2f}, "
                f"query={metrics['query_size']:.0f}",
                flush=True,
            )

    if summary:
        summary["Average"] = {
            "accuracy": sum(item["accuracy"] for item in summary.values()) / len(summary),
            "macro_f1": sum(item["macro_f1"] for item in summary.values()) / len(summary),
            "query_size": sum(item["query_size"] for item in summary.values()),
        }
    summary_path = out_base / "rpl_only_6regions_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Elapsed: {(time.time() - started) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
