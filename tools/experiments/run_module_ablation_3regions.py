#!/usr/bin/env python3
"""Retrain the three-region module ablation table for RC-MemNet.

Rows:
- Baseline: no regional prompts, no memory adapter.
- Baseline + RPL: regional prompts only.
- Baseline + RGMA: memory adapter only, with acoustic memory queries.
- RC-MemNet: full model.
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
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("03_Chizhou", "03池州", "Chizhou"),
]

EXPERIMENTS = {
    "baseline": {
        "label": "Baseline",
        "args": ["--ablation-variant", "no_prompt_no_memory"],
    },
    "baseline_rpl": {
        "label": "Baseline + RPL",
        "args": ["--ablation-variant", "no_memory_adapter"],
    },
    "baseline_rgma": {
        "label": "Baseline + RGMA",
        "args": ["--ablation-variant", "no_context_prompt"],
    },
    "rc_memnet": {
        "label": "RC-MemNet",
        "args": ["--ablation-variant", "rc_memnet"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        default=str(ROOT / "output/0722/module_ablation_3regions_20260726"),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--target-adapt-steps", type=int, default=30)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, exp_key: str, region_folder: str, region_name: str) -> dict[str, float] | None:
    out_dir = Path(args.output_base) / exp_key / region_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json = out_dir / "result.json"

    if result_json.exists() and not args.force:
        data = json.loads(result_json.read_text(encoding="utf-8"))
        return {
            "accuracy": float(data["metrics"]["accuracy"]) * 100.0,
            "accuracy_std": float(data.get("metrics_std", {}).get("accuracy", 0.0)) * 100.0,
            "macro_f1": float(data["metrics"]["macro_f1"]) * 100.0,
            "macro_f1_std": float(data.get("metrics_std", {}).get("macro_f1", 0.0)) * 100.0,
        }

    cmd = [
        PYTHON_BIN,
        "-u",
        str(SCRIPT_PATH),
        "--csv",
        str(CSV_PATH),
        "--output",
        str(result_json),
        "--holdout-region",
        region_name,
        "--region-column",
        "paper_region",
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
    ] + EXPERIMENTS[exp_key]["args"]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "run.log").write_text(res.stdout, encoding="utf-8")
    if res.returncode != 0:
        print(f"ERROR {exp_key}/{region_folder}: return code {res.returncode}", flush=True)
        print(res.stdout[-2000:], flush=True)
        return None

    data = json.loads(result_json.read_text(encoding="utf-8"))
    return {
        "accuracy": float(data["metrics"]["accuracy"]) * 100.0,
        "accuracy_std": float(data.get("metrics_std", {}).get("accuracy", 0.0)) * 100.0,
        "macro_f1": float(data["metrics"]["macro_f1"]) * 100.0,
        "macro_f1_std": float(data.get("metrics_std", {}).get("macro_f1", 0.0)) * 100.0,
    }


def main() -> None:
    args = parse_args()
    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    total = len(EXPERIMENTS) * len(TARGET_REGIONS)
    current = 0
    started = time.time()
    summary: dict[str, dict[str, object]] = {}

    print(f"Running module ablation: {total} runs -> {out_base}", flush=True)
    print(
        f"Protocol: max_steps={args.max_steps}, target_adapt_steps={args.target_adapt_steps}, "
        f"eval_split_runs={args.eval_split_runs}, shots={args.target_support_shots}",
        flush=True,
    )

    for exp_key, exp in EXPERIMENTS.items():
        rows: dict[str, dict[str, float]] = {}
        print(f"\n>>> {exp['label']} ({exp_key})", flush=True)
        for region_folder, region_name, clean_name in TARGET_REGIONS:
            current += 1
            t0 = time.time()
            print(f"[{current}/{total}] {clean_name}", flush=True)
            metrics = run_one(args, exp_key, region_folder, region_name)
            if metrics is None:
                continue
            rows[clean_name] = metrics
            print(
                f"  acc={metrics['accuracy']:.2f}±{metrics['accuracy_std']:.2f}, "
                f"macro_f1={metrics['macro_f1']:.2f}±{metrics['macro_f1_std']:.2f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        if rows:
            avg_acc = sum(item["accuracy"] for item in rows.values()) / len(rows)
            avg_f1 = sum(item["macro_f1"] for item in rows.values()) / len(rows)
            summary[exp_key] = {
                "label": exp["label"],
                "regions": rows,
                "avg_accuracy": avg_acc,
                "avg_macro_f1": avg_f1,
            }
            print(f"  AVG acc={avg_acc:.2f}, macro_f1={avg_f1:.2f}", flush=True)

    summary_path = out_base / "module_ablation_3regions_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# RC-MemNet module ablation, three representative regions",
        "",
        "| Variant | Qing. | Jing. | Chiz. | Avg. |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp_key in EXPERIMENTS:
        if exp_key not in summary:
            continue
        item = summary[exp_key]
        regions = item["regions"]
        md_lines.append(
            "| {label} | {qing:.2f} | {jing:.2f} | {chiz:.2f} | {avg:.2f} |".format(
                label=item["label"],
                qing=regions["Qingyang"]["accuracy"],
                jing=regions["Jingxian"]["accuracy"],
                chiz=regions["Chizhou"]["accuracy"],
                avg=item["avg_accuracy"],
            )
        )
    md_path = out_base / "module_ablation_3regions_accuracy.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Elapsed: {(time.time() - started) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
