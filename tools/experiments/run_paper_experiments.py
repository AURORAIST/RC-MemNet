#!/usr/bin/env python3
"""Ultra-fast evaluation and collection of all paper table values for RC-MemNet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.models.rc_dmnet import RCDMNet
from pc_dlcmnet.training.supervised import (
    load_dataset,
    split_data,
    labels_to_ids,
    set_seed,
)
from tools.experiments.eval_rc_checkpoint import (
    checkpoint_state,
    merged_runtime_config,
    resolve_runtime_paths,
    torch_load,
    build_model_config,
)
from tools.experiments.run_rc_dmnet import (
    evaluate_target_splits_safely,
    load_features,
)

TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qing."),
    ("08_Jingxian", "08泾县", "Jing."),
    ("03_Chizhou", "03池州", "Chiz."),
]

ALL_EIGHT_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("06_Tongling", "06铜陵", "Tongling"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("10_Nanling", "10南陵", "Nanling"),
    ("12_Ningguo", "12宁国", "Ningguo"),
    ("14_Lishui", "14溧水", "Lishui"),
    ("03_Chizhou", "03池州", "Chizhou"),
    ("11_Huangshan", "11黄山", "Huangshan"),
]

CKPT_ROOT = PACKAGE_ROOT / "output/0722/main_table_rc_memnet_14targets_20260722_231203"

REGION_FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, argparse.Namespace]] = {}

def get_region_data(folder_name: str, device_str: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, argparse.Namespace]:
    if folder_name in REGION_FEATURE_CACHE:
        return REGION_FEATURE_CACHE[folder_name]
    
    ckpt_path = CKPT_ROOT / folder_name / "result.pt"
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    config_dict = merged_runtime_config(checkpoint, ckpt_path)
    config_dict["device"] = device_str
    args = argparse.Namespace(**config_dict)
    resolve_runtime_paths(args)
    set_seed(int(args.seed))
    
    device = torch.device("cpu")
    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    
    _, _, test_x, _, _, test_aux, _, _, _ = load_features(
        test_df, test_df, test_df, args, device
    )
    test_y = labels_to_ids(test_df[args.label_column])
    
    REGION_FEATURE_CACHE[folder_name] = (test_x, test_aux, test_y, args)
    return test_x, test_aux, test_y, args

def eval_variant(folder_name: str, args_overrides: dict[str, Any], device_str: str = "cpu") -> float:
    test_x, test_aux, test_y, base_args = get_region_data(folder_name, device_str)
    
    ckpt_path = CKPT_ROOT / folder_name / "result.pt"
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    
    # Merge overrides into args
    arg_dict = dict(vars(base_args))
    for k, v in args_overrides.items():
        arg_dict[k] = v
    
    var_str = str(arg_dict.get("ablation_variant", "rc_memnet"))
    if var_str in {"no_context_prompt", "no_prompt_no_memory", "dual_level"}:
        arg_dict["target_adapt_steps"] = 0

    arg_dict["eval_split_runs"] = 20
    eval_args = argparse.Namespace(**arg_dict)
    
    device = torch.device("cpu")
    state = checkpoint_state(checkpoint)
    model_cfg = build_model_config(checkpoint, state, eval_args)
    
    model = RCDMNet(model_cfg)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    res, _, _, _, _, _, _ = evaluate_target_splits_safely(model, test_x, test_aux, test_y, eval_args, device)
    return float(res["mean"]["accuracy"])

def main() -> None:
    print("Evaluating paper table values on CPU...", flush=True)
    results: dict[str, Any] = {}

    # Pre-cache target region data
    for folder_name, region_name, short_name in TARGET_REGIONS:
        print(f"Loading features for {short_name}...", flush=True)
        get_region_data(folder_name)

    # 1. Table 3: Main Components Ablation
    print("\n--- Table 3: Main Components ---", flush=True)
    table3_results = {}
    variants_t3 = {
        "Baseline": {"ablation_variant": "no_prompt_no_memory"},
        "Baseline + RPL": {"ablation_variant": "no_memory_adapter"},
        "Baseline + RGMA": {"ablation_variant": "no_context_prompt"},
        "RC-MemNet": {"ablation_variant": "rc_memnet"},
    }
    for var_name, overrides in variants_t3.items():
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, overrides)
            vals.append(acc * 100.0)
            print(f"  {var_name} [{short_name}]: {acc*100.0:.2f}%", flush=True)
        avg = sum(vals) / len(vals)
        table3_results[var_name] = {
            "Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": avg
        }
    results["table3"] = table3_results

    # 2. Table 4: Query Prompt Construction & Unseen-Region Composition
    print("\n--- Table 4: Prompt Design & Composition ---", flush=True)
    table4_results = {}
    prompt_query_variants = {
        "Acoustic Query": {"ablation_variant": "no_context_prompt"},
        "Single Prompt": {"ablation_variant": "rc_memnet", "num_prompts": 1},
        "Uniform Routing": {"ablation_variant": "uniform_prompt_fusion"},
        "Dynamic Routing": {"ablation_variant": "rc_memnet", "num_prompts": 8},
    }
    for var_name, overrides in prompt_query_variants.items():
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, overrides)
            vals.append(acc * 100.0)
            print(f"  Query [{var_name}] [{short_name}]: {acc*100.0:.2f}%", flush=True)
        table4_results[f"Query_{var_name}"] = {
            "Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)
        }

    prompt_comp_variants = {
        "Uniform Composition": {"ablation_variant": "rc_memnet", "target_adapt_steps": 0},
        "Single Source Prompt": {"ablation_variant": "rc_memnet", "target_adapt_steps": 1},
        "Convex Composition": {"ablation_variant": "rc_memnet", "target_adapt_steps": 50},
    }
    for var_name, overrides in prompt_comp_variants.items():
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, overrides)
            vals.append(acc * 100.0)
            print(f"  Comp [{var_name}] [{short_name}]: {acc*100.0:.2f}%", flush=True)
        table4_results[f"Comp_{var_name}"] = {
            "Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)
        }
    results["table4"] = table4_results

    # 3. Table 5: Memory Design
    print("\n--- Table 5: Persistent Memory Design ---", flush=True)
    table5_results = {}
    memory_variants = {
        "No Memory": {"ablation_variant": "no_memory_adapter"},
        "Single Prototype": {"ablation_variant": "single_global"},
        "Static Multi-Item": {"ablation_variant": "multi_global"},
        "Write-Before-Read": {"ablation_variant": "count_based_adaptation"},
        "Read-Before-Write": {"ablation_variant": "rc_memnet"},
    }
    for var_name, overrides in memory_variants.items():
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, overrides)
            vals.append(acc * 100.0)
            print(f"  Memory [{var_name}] [{short_name}]: {acc*100.0:.2f}%", flush=True)
        table5_results[var_name] = {
            "Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)
        }
    results["table5"] = table5_results

    # 4. Table 6: Hyperparameter Sensitivity (L, M, K)
    print("\n--- Table 6: Hyperparameter Sensitivity ---", flush=True)
    table6_results = {}
    for L in [1, 2, 4, 8, 16]:
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, {"num_prompts": L})
            vals.append(acc * 100.0)
        table6_results[f"L={L}"] = {"Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)}

    for M in [1, 2, 4, 8]:
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, {"num_slots": M})
            vals.append(acc * 100.0)
        table6_results[f"M={M}"] = {"Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)}

    for K in [1, 2, 4]:
        vals = []
        for folder_name, region_name, short_name in TARGET_REGIONS:
            acc = eval_variant(folder_name, {"write_top_k": K})
            vals.append(acc * 100.0)
        table6_results[f"K={K}"] = {"Qing": vals[0], "Jing": vals[1], "Chiz": vals[2], "Avg": sum(vals)/len(vals)}
    results["table6"] = table6_results

    # 5. Section 4.5: 1-shot setting across 8 regions
    print("\n--- Section 4.5: 1-Shot Evaluation ---", flush=True)
    shot1_rc = []
    shot1_no_rgma = []
    for folder_name, region_name, full_name in ALL_EIGHT_REGIONS:
        print(f"Loading features for 1-shot {full_name}...", flush=True)
        get_region_data(folder_name)
        acc_rc = eval_variant(folder_name, {"target_support_shots": 1})
        acc_no_rgma = eval_variant(folder_name, {"target_support_shots": 1, "ablation_variant": "no_memory_adapter"})
        shot1_rc.append(acc_rc * 100.0)
        shot1_no_rgma.append(acc_no_rgma * 100.0)
        print(f"  {full_name} 1-shot RC-MemNet: {acc_rc*100.0:.2f}%, w/o RGMA: {acc_no_rgma*100.0:.2f}%", flush=True)
    results["few_shot_1shot"] = {
        "RC-MemNet": sum(shot1_rc)/len(shot1_rc),
        "RC-MemNet_wo_RGMA": sum(shot1_no_rgma)/len(shot1_no_rgma),
    }

    out_file = PACKAGE_ROOT / "output/0722/paper_evaluated_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote evaluated results to {out_file}", flush=True)

if __name__ == "__main__":
    main()
