#!/usr/bin/env python3
"""Run all paper ablation and hyperparameter sweep experiments under the formal submission protocol.

Formal Protocol:
- max_steps: 3000 (full training convergence)
- target_adapt_steps: 30 (matching main model)
- eval_split_runs: 100 (matching 100 Monte Carlo evaluation splits)
- base parameters: L=8, M=4, K=1
- seed: 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON_BIN = sys.executable
SCRIPT_PATH = ROOT / "tools/experiments/run_rc_dmnet.py"
CSV_PATH = ROOT / "output/0722/data/wu_vowel_segments.paper_regions.csv"
OUT_BASE = ROOT / "output/0722/formal_ablations_20260724"
CACHE_DIR = Path("/home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base")
AUX_CACHE_DIR = ROOT / "output/0722/feature_memory_aux_cache"
MATRIX_CACHE_DIR = Path("/home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache")

TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("03_Chizhou", "03池州", "Chizhou"),
]

# Formal Experiment Definitions (Controlled Variable Protocol)
EXPERIMENTS = {
    # Prompt Ablations (Table 4)
    "single_prompt_L1": {
        "desc": "Single Prompt (L=1)",
        "args": ["--num-prompts", "1", "--ablation-variant", "rc_memnet"],
    },
    "uniform_query_routing": {
        "desc": "Uniform Query Routing",
        "args": ["--ablation-variant", "uniform_query_routing"],
    },
    "uniform_source_composition": {
        "desc": "Uniform Source Composition",
        "args": ["--ablation-variant", "uniform_source_composition"],
    },
    "single_source_prompt": {
        "desc": "Single Source Prompt",
        "args": ["--ablation-variant", "single_source_prompt"],
    },
    "unconstrained_composition": {
        "desc": "Unconstrained Composition",
        "args": ["--ablation-variant", "unconstrained_composition"],
    },
    # Hyperparameter Sweeps (Table 6)
    "slots_M1": {
        "desc": "Memory Slots M=1",
        "args": ["--num-slots", "1", "--ablation-variant", "rc_memnet"],
    },
    "slots_M2": {
        "desc": "Memory Slots M=2",
        "args": ["--num-slots", "2", "--ablation-variant", "rc_memnet"],
    },
    "slots_M8": {
        "desc": "Memory Slots M=8",
        "args": ["--num-slots", "8", "--ablation-variant", "rc_memnet"],
    },
    "topk_K2": {
        "desc": "Write Top-K K=2",
        "args": ["--write-top-k", "2", "--ablation-variant", "rc_memnet"],
    },
    "topk_K4": {
        "desc": "Write Top-K K=4",
        "args": ["--write-top-k", "4", "--ablation-variant", "rc_memnet"],
    },
}

def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_results: dict[str, dict[str, float]] = {}

    total_runs = len(EXPERIMENTS) * len(TARGET_REGIONS)
    current_run = 0
    start_time = time.time()

    print(f"==========================================================================", flush=True)
    print(f" Starting Formal Paper Retraining Protocol ({total_runs} runs on GPU)...", flush=True)
    print(f" Protocol: max_steps=3000, target_adapt_steps=30, eval_split_runs=100", flush=True)
    print(f"==========================================================================", flush=True)

    for exp_key, exp_info in EXPERIMENTS.items():
        print(f"\n>>> Running Experiment: {exp_key} ({exp_info['desc']})", flush=True)
        region_accs = {}

        for folder_name, region_name, clean_name in TARGET_REGIONS:
            current_run += 1
            out_dir = OUT_BASE / exp_key / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)
            result_json = out_dir / "result.json"

            if result_json.exists():
                print(f"[{current_run}/{total_runs}] Skipping existing: {exp_key} / {clean_name}", flush=True)
                try:
                    data = json.loads(result_json.read_text(encoding="utf-8"))
                    acc = float(data["metrics"]["accuracy"])
                    region_accs[clean_name] = acc * 100.0
                    continue
                except Exception:
                    pass

            cmd = [
                PYTHON_BIN, "-u", str(SCRIPT_PATH),
                "--csv", str(CSV_PATH),
                "--output", str(result_json),
                "--holdout-region", region_name,
                "--region-column", "paper_region",
                "--max-steps", "3000",
                "--eval-every", "300",
                "--eval-split-runs", "100",
                "--target-adapt-steps", "30",
                "--target-support-shots", "4",
                "--batch-size", "64",
                "--eval-batch-size", "256",
                "--hidden-dim", "256",
                "--score-dim", "128",
                "--prompt-dim", "128",
                "--num-slots", "4",
                "--write-top-k", "1",
                "--num-prompts", "8",
                "--device", "cuda",
                "--cache-dir", str(CACHE_DIR),
                "--aux-cache-dir", str(AUX_CACHE_DIR),
                "--matrix-cache-dir", str(MATRIX_CACHE_DIR),
            ] + exp_info["args"]

            print(f"[{current_run}/{total_runs}] Executing {exp_key} on {clean_name} (3000 steps)...", flush=True)
            proc_start = time.time()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            proc_time = time.time() - proc_start

            if res.returncode != 0:
                print(f"ERROR running {exp_key} on {clean_name} (code {res.returncode}):\n{res.stdout[-1000:]}", flush=True)
                continue

            if result_json.exists():
                data = json.loads(result_json.read_text(encoding="utf-8"))
                acc = float(data["metrics"]["accuracy"]) * 100.0
                acc_std = float(data.get("metrics_std", {}).get("accuracy", 0.0)) * 100.0
                region_accs[clean_name] = acc
                print(f" -> Finished in {proc_time:.1f}s | {clean_name}: {acc:.2f}% ± {acc_std:.2f}%", flush=True)
            else:
                print(f" -> Failed to write result.json", flush=True)

        if region_accs:
            avg_acc = sum(region_accs.values()) / len(region_accs)
            region_accs["Avg"] = avg_acc
            summary_results[exp_key] = region_accs
            print(f"\n[Summary] {exp_key}: Qing={region_accs.get('Qingyang', 0):.2f}%, Jing={region_accs.get('Jingxian', 0):.2f}%, Chiz={region_accs.get('Chizhou', 0):.2f}% | Avg={avg_acc:.2f}%", flush=True)

    elapsed = time.time() - start_time
    print(f"\n==========================================================================", flush=True)
    print(f" All 30 formal submission experiments completed in {elapsed/60.0:.2f} minutes!", flush=True)
    print(f"==========================================================================", flush=True)

    summary_file = OUT_BASE / "formal_ablations_summary.json"
    summary_file.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"Wrote formal summary to {summary_file}", flush=True)

if __name__ == "__main__":
    main()
