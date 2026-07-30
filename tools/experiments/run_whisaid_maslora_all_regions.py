#!/usr/bin/env python3
"""
Run WhisAID (2026) and MAS-LoRA (Interspeech 2025) across all 8 paper target regions.
Strict 4-shot / 100-run split evaluation protocol.
Saves checkpoints to /home/ustc1958/lxy/graph/tone/model/
Saves results to output/0722/baselines_whisaid_maslora_20260725/
"""

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
SCRIPT_PATH = ROOT / "tools/experiments/ssl_finetune_whisaid_maslora.py"
CSV_PATH = ROOT / "output/0722/data/wu_vowel_segments.paper_regions.csv"
OUT_BASE = ROOT / "output/0722/baselines_whisaid_maslora_20260725"
MODEL_DIR = Path("/home/ustc1958/lxy/graph/tone/model")

TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("06_Tongling", "06铜陵", "Tongling"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("10_Nanling", "10南陵", "Nanling"),
    ("11_Ningguo", "11宁国", "Ningguo"),
    ("12_Lishui", "12溧水", "Lishui"),
    ("03_Chizhou", "03池州", "Chizhou"),
    ("14_Huangshan", "14黄山", "Huangshan"),
]

MODELS = ["whisaid", "mas_lora"]


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary_results: dict[str, dict[str, Any]] = {}

    total_runs = len(MODELS) * len(TARGET_REGIONS)
    current_run = 0
    start_time = time.time()

    print(f"==========================================================================", flush=True)
    print(f" Starting WhisAID (2026) & MAS-LoRA (Interspeech 2025) Benchmark Protocol", flush=True)
    print(f" Protocol: max_steps=3000, target_adapt_steps=30, eval_split_runs=100", flush=True)
    print(f"==========================================================================", flush=True)

    for model_type in MODELS:
        summary_results[model_type] = {}
        print(f"\n>>> Running Model Baseline: {model_type.upper()}", flush=True)

        for folder_name, region_name, label_name in TARGET_REGIONS:
            current_run += 1
            out_dir = OUT_BASE / model_type / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)
            result_file = out_dir / "result.json"

            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text())
                    acc = data["metrics"]["accuracy"] * 100
                    std = data.get("metrics_std", {}).get("accuracy", 0) * 100
                    summary_results[model_type][label_name] = (acc, std)
                    print(f"[{current_run}/{total_runs}] Skipping existing: {model_type} / {label_name} -> {acc:.2f}% ± {std:.2f}%", flush=True)
                    continue
                except Exception:
                    pass

            cmd = [
                PYTHON_BIN,
                str(SCRIPT_PATH),
                "--csv", str(CSV_PATH),
                "--output", str(result_file),
                "--holdout-region", region_name,
                "--region-column", "paper_region",
                "--speaker-column", "speaker_id",
                "--label-column", "vowel",
                "--model-type", model_type,
                "--max-steps", "3000",
                "--batch-size", "64",
                "--target-adapt-steps", "30",
                "--target-support-shots", "4",
                "--eval-split-runs", "100",
                "--device", "cuda",
                "--cache-dir", "/home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base",
                "--aux-cache-dir", str(ROOT / "output/0722/feature_memory_aux_cache"),
                "--matrix-cache-dir", "/home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache",
            ]

            print(f"[{current_run}/{total_runs}] Executing {model_type} on {label_name} (1500 steps)...", flush=True)
            t0 = time.time()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            t_elapsed = time.time() - t0

            if res.returncode != 0:
                print(f"ERROR running {model_type} on {label_name} (code {res.returncode}):\n{res.stdout[-1500:]}", flush=True)
            else:
                if result_file.exists():
                    data = json.loads(result_file.read_text())
                    acc = data["metrics"]["accuracy"] * 100
                    std = data.get("metrics_std", {}).get("accuracy", 0) * 100
                    summary_results[model_type][label_name] = (acc, std)
                    print(f" -> Finished in {t_elapsed:.1f}s | {label_name}: {acc:.2f}% ± {std:.2f}%", flush=True)
                else:
                    print(f" -> Finished in {t_elapsed:.1f}s | Missing result.json", flush=True)

    # Output aggregated summary
    summary_file = OUT_BASE / "summary.json"
    summary_file.write_text(json.dumps(summary_results, indent=2, ensure_ascii=False))
    print(f"\n==========================================================================", flush=True)
    print(f" Finished All WhisAID & MAS-LoRA Benchmarks in {time.time() - start_time:.1f}s", flush=True)
    print(f" Summary written to {summary_file}", flush=True)
    print(f"==========================================================================", flush=True)


if __name__ == "__main__":
    main()
