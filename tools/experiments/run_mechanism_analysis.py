#!/usr/bin/env python3
"""Fast prompt-memory interaction analysis (Section 4.6)."""

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

from pc_dlcmnet.models.rc_dmnet import RCDMNet
from pc_dlcmnet.training.supervised import (
    load_dataset,
    split_data,
    labels_to_ids,
)
from tools.experiments.eval_rc_checkpoint import (
    checkpoint_state,
    merged_runtime_config,
    resolve_runtime_paths,
    torch_load,
    build_model_config,
)
from tools.experiments.run_rc_dmnet import (
    load_features,
    adapt_fusion,
)

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

@torch.no_grad()
def analyze_region(folder_name: str, region_name: str, device_str: str = "cpu") -> dict[str, Any]:
    ckpt_path = CKPT_ROOT / folder_name / "result.pt"
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    config_dict = merged_runtime_config(checkpoint, ckpt_path)
    config_dict["device"] = device_str
    args = argparse.Namespace(**config_dict)
    resolve_runtime_paths(args)

    device = torch.device("cpu")
    state = checkpoint_state(checkpoint)
    model_cfg = build_model_config(checkpoint, state, args)
    
    model = RCDMNet(model_cfg)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    _, _, test_x, _, _, test_aux, _, _, _ = load_features(
        test_df, test_df, test_df, args, device
    )
    test_y = labels_to_ids(test_df[args.label_column])

    # Convert to tensors
    speech = torch.tensor(test_x, dtype=torch.float32, device=device)
    aux_t = torch.tensor(test_aux, dtype=torch.float32, device=device)
    y_t = torch.tensor(test_y, dtype=torch.long, device=device)

    # Encode with target fused prompts
    fusion_logits = torch.zeros(model.num_regions, dtype=torch.float32, device=device)
    enc_out = model.encode_target(speech, aux_t, fusion_logits)
    h = enc_out["h"]          # (N, hidden_dim)
    p = enc_out["p"]          # (N, hidden_dim)

    # 1. Routing consistency (cosine sim between prompt_context representations across samples)
    p_norm = F.normalize(p, dim=-1)
    sim_matrix = torch.matmul(p_norm, p_norm.transpose(0, 1)).cpu().numpy()
    
    N = len(test_y)
    same_sims = []
    diff_sims = []
    for i in range(min(N, 400)):
        for j in range(i + 1, min(N, 400)):
            if test_y[i] == test_y[j]:
                same_sims.append(float(sim_matrix[i, j]))
            else:
                diff_sims.append(float(sim_matrix[i, j]))

    same_mean = float(np.mean(same_sims)) if same_sims else 0.0
    diff_mean = float(np.mean(diff_sims)) if diff_sims else 0.0
    gap = same_mean - diff_mean

    # 2. Retrieval Purity@2
    # Dynamic prompt query
    q_base = model.h_to_query(h)
    gamma = torch.tanh(model.prompt_gamma(p))
    beta = model.prompt_beta(p)
    q_dyn = F.normalize(model.query_norm((1.0 + gamma) * q_base + beta), dim=-1)

    # Acoustic-only query
    q_acoust = F.normalize(model.query_norm(q_base), dim=-1)

    # Uniform prompt query
    p_unif = p.mean(dim=0, keepdim=True).expand_as(p)
    gamma_u = torch.tanh(model.prompt_gamma(p_unif))
    beta_u = model.prompt_beta(p_unif)
    q_unif = F.normalize(model.query_norm((1.0 + gamma_u) * q_base + beta_u), dim=-1)

    # Keys: (C * M, score_dim)
    keys = F.normalize(model.memory_keys, dim=-1).reshape(model.num_classes * model.num_slots, -1)
    item_classes = torch.arange(model.num_classes, device=device).repeat_interleave(model.num_slots)

    def calc_purity(queries: torch.Tensor) -> float:
        attn = torch.matmul(queries, keys.transpose(0, 1)) # (N, C*M)
        _, top2_idx = attn.topk(2, dim=-1) # (N, 2)
        top2_classes = item_classes[top2_idx] # (N, 2)
        matches = (top2_classes == y_t.unsqueeze(1)).float()
        return float(matches.mean().item())

    purity_dyn = calc_purity(q_dyn)
    purity_acoust = calc_purity(q_acoust)
    purity_unif = calc_purity(q_unif)

    return {
        "region": region_name,
        "same_sim": same_mean,
        "diff_sim": diff_mean,
        "cosine_gap": gap,
        "purity_acoust": purity_acoust * 100.0,
        "purity_unif": purity_unif * 100.0,
        "purity_dyn": purity_dyn * 100.0,
    }

def main() -> None:
    print("Running Fast Prompt-Memory Mechanism Analysis...", flush=True)
    results = []
    for folder_name, region_name, short_name in ALL_EIGHT_REGIONS:
        res = analyze_region(folder_name, short_name)
        results.append(res)
        print(f"[{short_name}] Gap: {res['cosine_gap']:.4f} (same={res['same_sim']:.4f}, diff={res['diff_sim']:.4f}) | "
              f"Purity Dyn: {res['purity_dyn']:.2f}%, Acoust: {res['purity_acoust']:.2f}%, Unif: {res['purity_unif']:.2f}%", flush=True)

    avg_gap = float(np.mean([r['cosine_gap'] for r in results]))
    avg_purity_dyn = float(np.mean([r['purity_dyn'] for r in results]))
    avg_purity_acoust = float(np.mean([r['purity_acoust'] for r in results]))
    avg_purity_unif = float(np.mean([r['purity_unif'] for r in results]))

    summary = {
        "region_results": results,
        "avg_cosine_gap": avg_gap,
        "avg_purity_dyn": avg_purity_dyn,
        "avg_purity_acoust": avg_purity_acoust,
        "avg_purity_unif": avg_purity_unif,
    }
    out_file = PACKAGE_ROOT / "output/0722/mechanism_analysis_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n--- Summary ---", flush=True)
    print(f"Average Cosine Gap: {avg_gap:.4f}", flush=True)
    print(f"Average Purity Dyn: {avg_purity_dyn:.2f}%, Acoust: {avg_purity_acoust:.2f}%, Unif: {avg_purity_unif:.2f}%", flush=True)

if __name__ == "__main__":
    main()
