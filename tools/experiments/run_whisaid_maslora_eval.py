#!/usr/bin/env python
"""
Run WhisAID (2026) and MAS-LoRA (Interspeech 2025) baseline training & 4-shot target evaluation script.
Supports 8 paper target regions with 100 split runs per target region.
Outputs saved to output/0722/baselines_whisaid_maslora_20260725/
Checkpoints saved to /home/ustc1958/lxy/graph/tone/model/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure current package is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from tools.experiments.wuvowel_baselines.models import WhisAIDModel, MASLoRAModel
from tools.experiments.run_rc_dmnet import (
    parse_args as parse_rc_args,
    load_dataset,
    split_data,
    load_features,
    metric_dict,
    VOWEL_ORDER,
)

PAPER_TARGET_REGIONS = [
    "04青阳",
    "06铜陵",
    "08泾县",
    "10南陵",
    "11宁国",
    "12溧水",
    "03池州",
    "14黄山",
]

MODEL_CHECKPOINT_DIR = Path("/home/ustc1958/lxy/graph/tone/model")


def parse_args():
    # Build parser without immediately calling parse_args()
    parser = argparse.ArgumentParser(description="WhisAID and MAS-LoRA baseline runner", conflict_handler="resolve")
    parser.add_argument("--csv", type=str, default="output/0722/data/wu_vowel_segments.paper_regions.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--holdout-region", type=str, required=True)
    parser.add_argument("--region-column", type=str, default="paper_region")
    parser.add_argument("--speaker-column", type=str, default="speaker_id")
    parser.add_argument("--label-column", type=str, default="vowel")
    parser.add_argument("--model-type", type=str, choices=["whisaid", "mas_lora"], default="whisaid")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-adapt-steps", type=int, default=30)
    parser.add_argument("--target-adapt-lr", type=float, default=0.01)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=str, default="/home/ustc1958/lxy/graph/tone/complete_package0712/output/salmonn_style_whisper_cache_base")
    parser.add_argument("--matrix-cache-dir", type=str, default="/home/ustc1958/lxy/graph/tone/complete_package0712/output/0712/matrix_cache")
    parser.add_argument("--aux-cache-dir", type=str, default="output/0722/feature_memory_aux_cache")
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--feature-norm", type=str, default="standard")
    parser.add_argument("--matrix-cache-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-chunks", type=int, default=1)
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--protocol", type=str, default="leave_one_region_out")
    parser.add_argument("--path-column", type=str, default="wav_path")
    parser.add_argument("--start-column", type=str, default="start_time")
    parser.add_argument("--end-column", type=str, default="end_time")
    parser.add_argument("--labels", nargs="*", default=VOWEL_ORDER)
    parser.add_argument("--predictions-output", type=str, default="")
    parser.add_argument("--curve-output", type=str, default="")
    parser.add_argument("--audit-output", type=str, default="")
    parser.add_argument("--checkpoint-output", type=str, default="")
    parser.add_argument("--aux-source", type=str, default="auto")
    parser.add_argument("--aux-representation", type=str, default="auto")
    parser.add_argument("--old-feature-cache-dir", type=str, default="")
    parser.add_argument("--aux-n-mfcc", type=int, default=13)
    parser.add_argument("--aux-n-mels", type=int, default=40)
    parser.add_argument("--aux-f0-segments", type=int, default=5)
    parser.add_argument("--aux-use-delta-mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-use-formants", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def train_source_whisaid(
    model: WhisAIDModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_reg: np.ndarray,
    train_spk: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> WhisAIDModel:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    dataset_size = len(train_y)
    indices = np.arange(dataset_size)

    for step in range(1, int(args.max_steps) + 1):
        batch_idx = np.random.choice(indices, size=int(args.batch_size), replace=(dataset_size < int(args.batch_size)))
        x_b = torch.tensor(train_x[batch_idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(train_y[batch_idx], dtype=torch.long, device=device)
        reg_b = torch.tensor(train_reg[batch_idx], dtype=torch.long, device=device)
        spk_b = torch.tensor(train_spk[batch_idx], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(x_b)
        loss_vowel = F.cross_entropy(out["vowel_logits"], y_b)
        loss_region = F.cross_entropy(out["region_logits"], reg_b)
        loss_speaker = F.cross_entropy(out["speaker_logits"], spk_b)

        loss = loss_vowel + loss_region + 0.05 * loss_speaker
        loss.backward()
        optimizer.step()

    return model


def train_source_maslora(
    model: MASLoRAModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_reg: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> MASLoRAModel:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    dataset_size = len(train_y)
    indices = np.arange(dataset_size)

    for step in range(1, int(args.max_steps) + 1):
        batch_idx = np.random.choice(indices, size=int(args.batch_size), replace=(dataset_size < int(args.batch_size)))
        x_b = torch.tensor(train_x[batch_idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(train_y[batch_idx], dtype=torch.long, device=device)
        reg_b = torch.tensor(train_reg[batch_idx], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(x_b, region_ids=reg_b)
        loss = F.cross_entropy(out["vowel_logits"], y_b)
        loss.backward()
        optimizer.step()

    return model


def evaluate_target_model(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run 4-shot target adaptation on Vowel Head across 100 split runs.
    """
    num_classes = len(VOWEL_ORDER)
    split_runs = int(args.eval_split_runs)
    support_shots = int(args.target_support_shots)

    # Class sample lookup
    class_indices = {c: np.where(test_y == c)[0] for c in range(num_classes)}

    accuracies, macro_f1s, weighted_f1s = [], [], []

    for run_idx in range(split_runs):
        rng = np.random.RandomState(int(args.seed) + run_idx * 1009)

        support_idx = []
        query_idx = []

        for c in range(num_classes):
            idxs = class_indices[c]
            if len(idxs) >= support_shots:
                perm = rng.permutation(idxs)
                support_idx.extend(perm[:support_shots])
                query_idx.extend(perm[support_shots:])
            else:
                query_idx.extend(idxs)

        support_idx = np.array(support_idx, dtype=np.int64)
        query_idx = np.array(query_idx, dtype=np.int64)

        # Clone vowel classifier head for target adaptation
        adapted_head = torch.nn.Sequential(*[
            nn.Linear(m.in_features, m.out_features) if isinstance(m, nn.Linear) else m
            for m in model.vowel_head
        ]).to(device)
        adapted_head.load_state_dict(model.vowel_head.state_dict())

        # Target adaptation on 4-shot support
        if len(support_idx) > 0 and int(args.target_adapt_steps) > 0:
            optimizer = torch.optim.Adam(adapted_head.parameters(), lr=float(args.target_adapt_lr))
            sx = torch.tensor(test_x[support_idx], dtype=torch.float32, device=device)
            if sx.ndim == 3:
                sx = sx.mean(dim=1)
            sy = torch.tensor(test_y[support_idx], dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                if isinstance(model, WhisAIDModel):
                    h = model.encoder_proj(sx)
                    reg_logits = model.accent_head(h)
                    acc_emb = model.accent_emb_layer(F.softmax(reg_logits, dim=-1))
                    feat = torch.cat([h, acc_emb], dim=-1)
                elif isinstance(model, MASLoRAModel):
                    h_base = model.base_proj(sx)
                    delta_h = torch.stack([expert(h_base) for expert in model.experts], dim=0).mean(dim=0)
                    feat = h_base + delta_h

            adapted_head.train()
            for _ in range(int(args.target_adapt_steps)):
                optimizer.zero_grad()
                out_logits = adapted_head(feat)
                loss = F.cross_entropy(out_logits, sy)
                loss.backward()
                optimizer.step()

        # Evaluate on query set
        model.eval()
        adapted_head.eval()
        with torch.no_grad():
            qx = torch.tensor(test_x[query_idx], dtype=torch.float32, device=device)
            if qx.ndim == 3:
                qx = qx.mean(dim=1)
            if isinstance(model, WhisAIDModel):
                h = model.encoder_proj(qx)
                reg_logits = model.accent_head(h)
                acc_emb = model.accent_emb_layer(F.softmax(reg_logits, dim=-1))
                feat = torch.cat([h, acc_emb], dim=-1)
            elif isinstance(model, MASLoRAModel):
                h_base = model.base_proj(qx)
                delta_h = torch.stack([expert(h_base) for expert in model.experts], dim=0).mean(dim=0)
                feat = h_base + delta_h

            logits = adapted_head(feat)
            preds = logits.argmax(dim=-1).cpu().numpy()
            targets = test_y[query_idx]

        metrics = metric_dict(targets, preds)
        accuracies.append(metrics["accuracy"])
        macro_f1s.append(metrics["macro_f1"])
        weighted_f1s.append(metrics["weighted_f1"])

    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "macro_f1": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
        "weighted_f1": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s, ddof=1)) if len(weighted_f1s) > 1 else 0.0,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load CSV
    df = load_dataset(args)

    # 2. Build splits
    train_df, val_df, test_df, split_info = split_data(df, args)

    # Label mappings
    vowel_to_id = {v: i for i, v in enumerate(VOWEL_ORDER)}
    source_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    region_to_id = {r: i for i, r in enumerate(source_regions)}
    
    speakers = sorted(train_df[args.speaker_column].astype(str).unique().tolist())
    speaker_to_id = {s: i for i, s in enumerate(speakers)}

    train_y = train_df[args.label_column].astype(str).map(vowel_to_id).to_numpy(dtype=np.int64)
    val_y = val_df[args.label_column].astype(str).map(vowel_to_id).to_numpy(dtype=np.int64)
    test_y = test_df[args.label_column].astype(str).map(vowel_to_id).to_numpy(dtype=np.int64)

    train_reg = train_df[args.region_column].astype(str).map(region_to_id).to_numpy(dtype=np.int64)
    train_spk = train_df[args.speaker_column].astype(str).map(speaker_to_id).to_numpy(dtype=np.int64)

    # 3. Load feature matrices
    train_x, val_x, test_x, _, _, _, _, _, _ = load_features(train_df, val_df, test_df, args, device)
    input_dim = train_x.shape[-1]

    # Ensure model checkpoint dir exists
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Instantiate & Train Model
    if args.model_type == "whisaid":
        model = WhisAIDModel(
            input_dim=input_dim,
            num_vowels=len(VOWEL_ORDER),
            num_regions=len(source_regions),
            num_speakers=len(speakers),
            accent_dim=64,
            grl_alpha=0.05,
        ).to(device)

        print(f">>> Training WhisAID on source regions (excluding {args.holdout_region})...", flush=True)
        model = train_source_whisaid(model, train_x, train_y, train_reg, train_spk, args, device)
        ckpt_path = MODEL_CHECKPOINT_DIR / f"whisaid_{args.holdout_region[:2]}.pt"
        torch.save(model.state_dict(), ckpt_path)

    elif args.model_type == "mas_lora":
        model = MASLoRAModel(
            input_dim=input_dim,
            num_vowels=len(VOWEL_ORDER),
            num_source_regions=len(source_regions),
            r=16,
            lora_alpha=1.0,
        ).to(device)

        print(f">>> Training MAS-LoRA on source regions (excluding {args.holdout_region})...", flush=True)
        model = train_source_maslora(model, train_x, train_y, train_reg, args, device)
        ckpt_path = MODEL_CHECKPOINT_DIR / f"maslora_{args.holdout_region[:2]}.pt"
        torch.save(model.state_dict(), ckpt_path)

    # 5. Evaluate Target Adaptation (4-shot, 100 runs)
    print(f">>> Evaluating 4-shot target adaptation on {args.holdout_region} ({args.eval_split_runs} runs)...", flush=True)
    eval_res = evaluate_target_model(model, test_x, test_y, args, device)

    print(f"[{args.model_type}] {args.holdout_region} Accuracy: {eval_res['accuracy']*100:.2f}% ± {eval_res['accuracy_std']*100:.2f}%", flush=True)

    out_data = {
        "model_type": args.model_type,
        "holdout_region": args.holdout_region,
        "metrics": {
            "accuracy": eval_res["accuracy"],
            "macro_f1": eval_res["macro_f1"],
            "weighted_f1": eval_res["weighted_f1"],
        },
        "metrics_std": {
            "accuracy": eval_res["accuracy_std"],
            "macro_f1": eval_res["macro_f1_std"],
            "weighted_f1": eval_res["weighted_f1_std"],
        },
        "eval_split_runs": args.eval_split_runs,
        "target_support_shots": args.target_support_shots,
        "max_steps": args.max_steps,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
        print(f"Result saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
