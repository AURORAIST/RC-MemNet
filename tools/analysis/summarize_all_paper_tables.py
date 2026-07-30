#!/usr/bin/env python3
"""Summarize paper-aligned ablation tables and text parameters from existing runs."""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    ("04_Qingyang", "04青阳", "Qing."),
    ("08_Jingxian", "08泾县", "Jing."),
    ("03_Chizhou", "03池州", "Chiz."),
]

def load_res(root_path: Path, folder: str) -> float:
    p = root_path / folder / "result.json"
    if p.exists():
        data = json.loads(p.read_text())
        acc = data.get("metrics", {}).get("accuracy")
        if acc is not None:
            return float(acc) * 100.0
    return 0.0

def main():
    rc_memnet_root = PACKAGE_ROOT / "output/0722/main_table_rc_memnet_14targets_20260722_231203"
    prompt_ablation_root = PACKAGE_ROOT / "output/0722/ablation_prompt_20260723_001604"
    memory_ablation_root = PACKAGE_ROOT / "output/0722/ablation_memory_count_based_v2_20260723_012548"

    # RC-MemNet
    rc_vals = [load_res(rc_memnet_root, folder) for folder, _, _ in TARGETS]
    rc_avg = sum(rc_vals) / len(rc_vals)

    # w/o Prompt Routing (Baseline + RGMA)
    no_prompt_vals = [load_res(prompt_ablation_root, folder) for folder, _, _ in TARGETS]
    no_prompt_avg = sum(no_prompt_vals) / len(no_prompt_vals)

    # w/o RGMA Adapter (Baseline + RPL)
    no_mem_vals = [load_res(memory_ablation_root, folder) for folder, _, _ in TARGETS]
    no_mem_avg = sum(no_mem_vals) / len(no_mem_vals)

    print("=== Collected Existing Run Ablations ===")
    print(f"RC-MemNet: Qing={rc_vals[0]:.2f}, Jing={rc_vals[1]:.2f}, Chiz={rc_vals[2]:.2f}, Avg={rc_avg:.2f}")
    print(f"Baseline + RGMA (w/o Prompt Routing): Qing={no_prompt_vals[0]:.2f}, Jing={no_prompt_vals[1]:.2f}, Chiz={no_prompt_vals[2]:.2f}, Avg={no_prompt_avg:.2f}")
    print(f"Baseline + RPL (w/o RGMA Write Adapter): Qing={no_mem_vals[0]:.2f}, Jing={no_mem_vals[1]:.2f}, Chiz={no_mem_vals[2]:.2f}, Avg={no_mem_avg:.2f}")

if __name__ == "__main__":
    main()
