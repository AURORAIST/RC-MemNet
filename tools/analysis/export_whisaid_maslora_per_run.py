#!/usr/bin/env python3
"""Export per-repetition metrics for saved WhisAID and MAS-LoRA runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import WhisperFeatureExtractor

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.training.supervised import split_data
from tools.experiments.ssl_finetune_whisaid_maslora import (
    WHISPER_PATH,
    EndToEndMASLoRA,
    EndToEndWhisAID,
    extract_test_features,
)


TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang", "04"),
    ("06_Tongling", "06铜陵", "Tongling", "06"),
    ("08_Jingxian", "08泾县", "Jingxian", "08"),
    ("10_Nanling", "10南陵", "Nanling", "10"),
    ("11_Ningguo", "11宁国", "Ningguo", "11"),
    ("12_Lishui", "12溧水", "Lishui", "12"),
    ("03_Chizhou", "03池州", "Chizhou", "03"),
    ("14_Huangshan", "14黄山", "Huangshan", "14"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/0722/data/wu_vowel_segments.paper_regions.csv")
    parser.add_argument("--output-dir", default="output/0722/baselines_whisaid_maslora_20260725_per_run")
    parser.add_argument("--checkpoint-dir", default="/home/ustc1958/lxy/graph/tone/model")
    parser.add_argument("--model-type", choices=["whisaid", "mas_lora", "both"], default="both")
    parser.add_argument("--regions", nargs="*", default=[r[0] for r in TARGET_REGIONS])
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--region-column", default="paper_region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--protocol", default="leave_one_region_out")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(
    model_type: str,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    source_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    speakers = sorted(train_df[args.speaker_column].astype(str).unique().tolist())
    if model_type == "whisaid":
        return EndToEndWhisAID(
            whisper_path=WHISPER_PATH,
            num_vowels=len(VOWEL_ORDER),
            num_regions=len(source_regions),
            num_speakers=len(speakers),
            accent_dim=64,
        ).to(device)
    return EndToEndMASLoRA(
        whisper_path=WHISPER_PATH,
        num_vowels=len(VOWEL_ORDER),
        num_source_regions=len(source_regions),
        r=16,
        lora_alpha=1.0,
    ).to(device)


def per_run_metrics(test_x: np.ndarray, test_y: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_classes = len(VOWEL_ORDER)
    for run_idx in range(int(args.eval_split_runs)):
        split_seed = int(args.seed) + run_idx * 1009
        support_idx, query_idx = split_global_support_query(
            test_y,
            int(args.target_support_shots),
            split_seed,
        )
        support_x = torch.tensor(test_x[support_idx], dtype=torch.float32)
        query_x = torch.tensor(test_x[query_idx], dtype=torch.float32)
        support_y = torch.tensor(test_y[support_idx], dtype=torch.long)

        prototypes = []
        for class_id in range(num_classes):
            mask = support_y == class_id
            if bool(mask.any()):
                prototypes.append(F.normalize(support_x[mask].mean(dim=0), dim=0))
            else:
                prototypes.append(F.normalize(support_x.mean(dim=0), dim=0))
        proto = torch.stack(prototypes, dim=0)
        logits = torch.matmul(F.normalize(query_x, dim=-1), F.normalize(proto, dim=-1).transpose(0, 1))
        preds = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        true = test_y[query_idx]
        rows.append(
            {
                "run_idx": run_idx,
                "split_seed": split_seed,
                "support_size": int(len(support_idx)),
                "query_size": int(len(query_idx)),
                "accuracy": float(accuracy_score(true, preds)),
                "macro_f1": float(f1_score(true, preds, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(true, preds, average="weighted", zero_division=0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    df = pd.read_csv(args.csv)
    df = df[df[args.region_column].astype(str) != "nan"].reset_index(drop=True)
    extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_PATH, local_files_only=True)

    model_types = ["whisaid", "mas_lora"] if args.model_type == "both" else [args.model_type]
    selected = {r for r in args.regions}
    summary_rows: list[dict[str, Any]] = []

    for folder_name, region_name, label_name, code in TARGET_REGIONS:
        if folder_name not in selected and region_name not in selected and label_name not in selected:
            continue
        region_args = argparse.Namespace(**vars(args))
        region_args.holdout_region = region_name
        train_df, _, test_df, _ = split_data(df, region_args)
        for model_type in model_types:
            ckpt_name = f"{'whisaid' if model_type == 'whisaid' else 'maslora'}_e2e_{code}.pt"
            ckpt_path = checkpoint_dir / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            model = build_model(model_type, train_df, region_args, device)
            model.load_state_dict(load_state(ckpt_path))
            test_x, test_y = extract_test_features(model, extractor, test_df, device)
            rows = per_run_metrics(test_x, test_y, region_args)
            run_df = pd.DataFrame(rows)
            out_subdir = output_dir / model_type / folder_name
            out_subdir.mkdir(parents=True, exist_ok=True)
            run_csv = out_subdir / "per_run_metrics.csv"
            run_df.to_csv(run_csv, index=False)
            summary = {
                "model_type": model_type,
                "region": label_name,
                "holdout_region": region_name,
                "runs": int(len(run_df)),
                "accuracy": float(run_df["accuracy"].mean()),
                "accuracy_std": float(run_df["accuracy"].std(ddof=1)),
                "macro_f1": float(run_df["macro_f1"].mean()),
                "macro_f1_std": float(run_df["macro_f1"].std(ddof=1)),
                "weighted_f1": float(run_df["weighted_f1"].mean()),
                "weighted_f1_std": float(run_df["weighted_f1"].std(ddof=1)),
                "per_run_csv": str(run_csv),
                "checkpoint": str(ckpt_path),
            }
            (out_subdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.append(summary)
            print(
                f"{model_type} {label_name}: "
                f"acc={summary['accuracy'] * 100:.2f}±{summary['accuracy_std'] * 100:.2f}, "
                f"macro_f1={summary['macro_f1'] * 100:.2f}±{summary['macro_f1_std'] * 100:.2f}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
