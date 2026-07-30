#!/usr/bin/env python3
"""Audit script to inspect and print raw result.json files for all ablation variants and sweeps."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "output/0722/missing_ablations_20260724"
MAIN_DIR = ROOT / "output/0722/main_table_rc_memnet_14targets_20260722_231203"

VARIANTS = [
    ("Single Prompt (L=1)", BASE_DIR / "single_prompt_L1"),
    ("Uniform Routing", BASE_DIR / "uniform_routing"),
    ("Uniform Composition", BASE_DIR / "uniform_composition"),
    ("Single Source Prompt", BASE_DIR / "single_source_prompt"),
    ("Unconstrained Composition", BASE_DIR / "unconstrained_composition"),
    ("M = 1", BASE_DIR / "slots_M1"),
    ("M = 2", BASE_DIR / "slots_M2"),
    ("M = 4 (Default)", MAIN_DIR),
    ("M = 8", BASE_DIR / "slots_M8"),
    ("K = 1 (Default)", MAIN_DIR),
    ("K = 2", BASE_DIR / "topk_K2"),
    ("K = 4", BASE_DIR / "topk_K4"),
]

REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("03_Chizhou", "03池州", "Chizhou"),
]

def main():
    rows = []

    print("==========================================================================")
    print(" RAW FILE DISK AUDIT FOR ALL EXPERIMENT VARIANTS & TARGET REGIONS")
    print("==========================================================================")

    for var_name, var_dir in VARIANTS:
        print(f"\n>>> VARIANT: {var_name}")
        for reg_folder, reg_code, reg_clean in REGIONS:
            if var_dir == MAIN_DIR:
                # Main table directory layout
                target_json = var_dir / reg_folder / "result.json"
                target_ckpt = var_dir / reg_folder / "result.pt"
            else:
                target_json = var_dir / reg_folder / "result.json"
                target_ckpt = var_dir / reg_folder / "result.pt"

            if not target_json.exists():
                print(f"  [{reg_clean}] MISSING: {target_json}")
                rows.append({
                    "variant": var_name,
                    "region": reg_clean,
                    "status": "MISSING",
                    "json_path": str(target_json),
                    "ckpt_path": str(target_ckpt) if target_ckpt.exists() else "MISSING",
                    "mtime": "N/A",
                    "holdout_region": reg_code,
                    "num_prompts": "N/A",
                    "num_slots": "N/A",
                    "write_top_k": "N/A",
                    "ablation_variant": "N/A",
                    "target_adapt_steps": "N/A",
                    "eval_split_runs": "N/A",
                    "accuracy": "N/A",
                    "accuracy_std": "N/A",
                })
                continue

            # File stats
            mtime_ts = target_json.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime_ts).strftime('%Y-%m-%d %H:%M:%S')

            data = json.loads(target_json.read_text(encoding="utf-8"))

            metrics = data.get("metrics", {})
            metrics_std = data.get("metrics_std", {})
            eval_splits = data.get("evaluation_splits", {}).get("test", {})

            accuracy = metrics.get("accuracy", 0.0)
            accuracy_std = metrics_std.get("accuracy", 0.0)
            runs = eval_splits.get("runs", 0)

            config = data.get("config", {})
            num_prompts = config.get("num_prompts", data.get("num_prompts", "N/A"))
            num_slots = config.get("num_slots", data.get("num_slots", "N/A"))
            write_top_k = config.get("write_top_k", data.get("write_top_k", "N/A"))
            ablation_variant = config.get("ablation_variant", data.get("ablation_variant", "N/A"))
            target_adapt_steps = config.get("target_adapt_steps", data.get("target_adapt_steps", "N/A"))
            holdout_region = config.get("holdout_region", data.get("holdout_region", reg_code))
            seed = config.get("seed", data.get("seed", 0))

            ckpt_str = str(target_ckpt) if target_ckpt.exists() else "N/A (Memory cache mode)"

            print(f"  [{reg_clean}] EXISTS:")
            print(f"    JSON Path: {target_json}")
            print(f"    Ckpt Path: {ckpt_str}")
            print(f"    Mod Time : {mtime_str}")
            print(f"    Holdout  : {holdout_region}")
            print(f"    Params   : prompts={num_prompts}, slots={num_slots}, top_k={write_top_k}, variant={ablation_variant}, adapt_steps={target_adapt_steps}, seed={seed}")
            print(f"    Metrics  : Accuracy = {accuracy*100.0:.4f}% ± {accuracy_std*100.0:.4f}% ({runs} eval split runs)")

            rows.append({
                "variant": var_name,
                "region": reg_clean,
                "status": "EXISTS",
                "json_path": str(target_json),
                "ckpt_path": ckpt_str,
                "mtime": mtime_str,
                "holdout_region": holdout_region,
                "num_prompts": num_prompts,
                "num_slots": num_slots,
                "write_top_k": write_top_k,
                "ablation_variant": ablation_variant,
                "target_adapt_steps": target_adapt_steps,
                "eval_split_runs": runs,
                "accuracy_pct": round(accuracy * 100.0, 4),
                "accuracy_std_pct": round(accuracy_std * 100.0, 4),
            })

    df = pd.DataFrame(rows)
    csv_out = ROOT / "output/0722/missing_ablations_disk_audit.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved disk audit CSV to {csv_out}")

if __name__ == "__main__":
    main()
