#!/usr/bin/env python3
"""
End-to-End Fine-Tuning and Evaluation for WhisAID (2026) and MAS-LoRA (Interspeech 2025).
Directly fine-tunes Whisper acoustic encoder on WuVowelSet source regions and evaluates 4-shot target adaptation.
Saves checkpoints to /home/ustc1958/lxy/graph/tone/model/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor, WhisperModel

# Ensure package root on sys.path
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.training.supervised import labels_to_ids, load_dataset, split_data
from pc_dlcmnet.utils.paths import default_audio_root
from tools.experiments.local_hf_ssl_4shot_eval import load_segment

MODEL_CHECKPOINT_DIR = Path("/home/ustc1958/lxy/graph/tone/model")
WHISPER_PATH = Path("/home/ustc1958/lxy/graph/tone/model/whisper-base")
if not WHISPER_PATH.exists():
    WHISPER_PATH = Path("/home/ustc1958/lxy/graph/tone/model/whisper-small")


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.05):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=0.05):
    return GradientReversalFunction.apply(x, alpha)


class AudioSegmentDataset(Dataset[Any]):
    def __init__(self, df: pd.DataFrame, region_map: dict[str, int], speaker_map: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.vowel_map = {v: i for i, v in enumerate(VOWEL_ORDER)}
        self.region_map = region_map
        self.speaker_map = speaker_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dargs = argparse.Namespace(
            path_column="wav_path",
            start_column="start_time",
            end_column="end_time",
            target_sr=16000,
            audio_root=str(default_audio_root()),
        )
        audio = load_segment(row, dargs)
        vowel_id = self.vowel_map[str(row["vowel"])]
        region_id = self.region_map.get(str(row["paper_region"]), 0)
        speaker_id = self.speaker_map.get(str(row["speaker_id"]), 0)
        return {
            "audio": audio,
            "vowel": vowel_id,
            "region": region_id,
            "speaker": speaker_id,
        }


def collate_fn(batch):
    audio_list = [item["audio"] for item in batch]
    vowel_tensor = torch.tensor([item["vowel"] for item in batch], dtype=torch.long)
    region_tensor = torch.tensor([item["region"] for item in batch], dtype=torch.long)
    speaker_tensor = torch.tensor([item["speaker"] for item in batch], dtype=torch.long)
    return {
        "audio": audio_list,
        "vowel": vowel_tensor,
        "region": region_tensor,
        "speaker": speaker_tensor,
    }


class EndToEndWhisAID(nn.Module):
    """
    WhisAID (2026) End-to-End Speech Model:
      - Whisper Encoder
      - Accent Head (Region Classifier)
      - Speaker Head (GRL, lambda=0.05)
      - Vowel Head
    """
    def __init__(self, whisper_path: Path, num_vowels: int, num_regions: int, num_speakers: int, accent_dim: int = 64):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(whisper_path, local_files_only=True)
        self.encoder = self.whisper.encoder
        hidden_dim = self.whisper.config.d_model

        # Freeze feature extractor
        if hasattr(self.encoder, "conv1"):
            for p in self.encoder.conv1.parameters():
                p.requires_grad = False
        if hasattr(self.encoder, "conv2"):
            for p in self.encoder.conv2.parameters():
                p.requires_grad = False

        self.accent_head = nn.Sequential(
            nn.Linear(hidden_dim, accent_dim),
            nn.ReLU(),
            nn.Linear(accent_dim, num_regions),
        )
        self.accent_emb = nn.Sequential(
            nn.Linear(num_regions, accent_dim),
            nn.ReLU(),
        )
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_speakers),
        )
        self.vowel_head = nn.Sequential(
            nn.Linear(hidden_dim + accent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_vowels),
        )

    def extract_features(self, input_features: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_features=input_features, return_dict=True)
        hidden = output.last_hidden_state  # (B, L, D)
        return hidden.mean(dim=1)  # (B, D)

    def forward(self, input_features: torch.Tensor):
        h = self.extract_features(input_features)

        reg_logits = self.accent_head(h)
        acc_embedding = self.accent_emb(F.softmax(reg_logits, dim=-1))

        h_grl = grad_reverse(h, alpha=0.05)
        spk_logits = self.speaker_head(h_grl)

        combined = torch.cat([h, acc_embedding], dim=-1)
        vowel_logits = self.vowel_head(combined)

        return {
            "vowel_logits": vowel_logits,
            "region_logits": reg_logits,
            "speaker_logits": spk_logits,
            "h": h,
            "acc_emb": acc_embedding,
        }


class EndToEndMASLoRA(nn.Module):
    """
    MAS-LoRA (Interspeech 2025) End-to-End Speech Model:
      - Fine-tuned Whisper Encoder top layers
      - S Regional LoRA Experts (r=16, alpha=1.0)
      - Accent-agnostic expert blending for unseen target evaluation
      - Vowel Head
    """
    def __init__(self, whisper_path: Path, num_vowels: int, num_source_regions: int, r: int = 16, lora_alpha: float = 1.0):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(whisper_path, local_files_only=True)
        self.encoder = self.whisper.encoder
        hidden_dim = self.whisper.config.d_model
        self.num_source_regions = num_source_regions

        # Freeze encoder base parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Freeze feature extractor conv layers
        if hasattr(self.encoder, "conv1"):
            for p in self.encoder.conv1.parameters():
                p.requires_grad = False
        if hasattr(self.encoder, "conv2"):
            for p in self.encoder.conv2.parameters():
                p.requires_grad = False

        # Unfreeze top encoder layers for speech fine-tuning
        for layer in self.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for name in ["layer_norm", "final_layer_norm"]:
            module = getattr(self.encoder, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

        # Regional LoRA Experts (Bottleneck Adapter per source region)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, r, bias=False),
                nn.ReLU(),
                nn.Linear(r, hidden_dim, bias=False),
            ) for _ in range(num_source_regions)
        ])
        # Initialize zero output for LoRA experts
        for exp in self.experts:
            nn.init.kaiming_uniform_(exp[0].weight, a=math.sqrt(5))
            nn.init.zeros_(exp[2].weight)

        self.vowel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_vowels),
        )

    def extract_features(self, input_features: torch.Tensor, region_id: int = None) -> torch.Tensor:
        output = self.encoder(input_features=input_features, return_dict=True)
        h = output.last_hidden_state.mean(dim=1)

        if region_id is not None and 0 <= region_id < self.num_source_regions:
            delta = self.experts[region_id](h)
            pooled = h + delta
        else:
            # Accent-Agnostic Blend across all source experts for unseen target evaluation
            deltas = torch.stack([exp(h) for exp in self.experts], dim=0)
            pooled = h + deltas.mean(dim=0)

        return pooled

    def forward(self, input_features: torch.Tensor, region_id: int = None):
        h = self.extract_features(input_features, region_id=region_id)
        vowel_logits = self.vowel_head(h)
        return {"vowel_logits": vowel_logits, "h": h}


def train_whisaid_e2e(
    model: EndToEndWhisAID,
    extractor: WhisperFeatureExtractor,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> EndToEndWhisAID:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    model.train()

    step = 0
    max_steps = int(args.max_steps)

    while step < max_steps:
        for batch in train_loader:
            step += 1
            audio = batch["audio"]
            inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
            feat = inputs["input_features"].to(device)
            y_vowel = batch["vowel"].to(device)
            y_region = batch["region"].to(device)
            y_speaker = batch["speaker"].to(device)

            optimizer.zero_grad()
            out = model(feat)
            loss_vowel = F.cross_entropy(out["vowel_logits"], y_vowel)
            loss_region = F.cross_entropy(out["region_logits"], y_region)
            loss_speaker = F.cross_entropy(out["speaker_logits"], y_speaker)

            loss = loss_vowel + loss_region + 0.05 * loss_speaker
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step >= max_steps:
                break
    return model


def train_maslora_e2e(
    model: EndToEndMASLoRA,
    extractor: WhisperFeatureExtractor,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> EndToEndMASLoRA:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    model.train()

    step = 0
    max_steps = int(args.max_steps)

    while step < max_steps:
        for batch in train_loader:
            step += 1
            audio = batch["audio"]
            inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
            feat = inputs["input_features"].to(device)
            y_vowel = batch["vowel"].to(device)
            y_region = batch["region"][0].item()  # Primary region for batch

            optimizer.zero_grad()
            out = model(feat, region_id=y_region)
            loss = F.cross_entropy(out["vowel_logits"], y_vowel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step >= max_steps:
                break
    return model


@torch.no_grad()
def extract_test_features(
    model: nn.Module,
    extractor: WhisperFeatureExtractor,
    test_df: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dargs = argparse.Namespace(
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        target_sr=16000,
        audio_root=str(default_audio_root()),
    )
    vowel_map = {v: i for i, v in enumerate(VOWEL_ORDER)}

    model.eval()
    feats_list = []
    labels_list = []

    batch_size = 32
    for i in range(0, len(test_df), batch_size):
        sub_df = test_df.iloc[i : i + batch_size]
        audio_list = [load_segment(row, dargs) for _, row in sub_df.iterrows()]
        inputs = extractor(audio_list, sampling_rate=16000, return_tensors="pt", padding="max_length")
        feat_inp = inputs["input_features"].to(device)

        if isinstance(model, EndToEndWhisAID):
            out = model(feat_inp)
            h = out["h"]
            acc = out["acc_emb"]
            feat = torch.cat([h, acc], dim=-1)
        elif isinstance(model, EndToEndMASLoRA):
            feat = model.extract_features(feat_inp, region_id=None)

        feats_list.append(feat.cpu().numpy())
        labels_list.extend([vowel_map[str(r["vowel"])] for _, r in sub_df.iterrows()])

    return np.concatenate(feats_list, axis=0), np.array(labels_list, dtype=np.int64)


def evaluate_target_4shot(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    num_classes = len(VOWEL_ORDER)
    split_runs = int(args.eval_split_runs)
    support_shots = int(args.target_support_shots)

    run_metrics = []
    for run_idx in range(split_runs):
        support_idx, query_idx = split_global_support_query(test_y, support_shots, int(args.seed) + run_idx * 1009)
        support_x = torch.tensor(test_x[support_idx], dtype=torch.float32)
        query_x = torch.tensor(test_x[query_idx], dtype=torch.float32)
        support_y = torch.tensor(test_y[support_idx], dtype=torch.long)
        
        prototypes = []
        for c in range(num_classes):
            mask = support_y == c
            if bool(mask.any()):
                prototypes.append(F.normalize(support_x[mask].mean(dim=0), dim=0))
            else:
                prototypes.append(F.normalize(support_x.mean(dim=0), dim=0))
        proto = torch.stack(prototypes, dim=0)
        logits = torch.matmul(F.normalize(query_x, dim=-1), F.normalize(proto, dim=-1).transpose(0, 1))
        preds = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        run_metrics.append(float(accuracy_score(test_y[query_idx], preds)))

    return {
        "accuracy": float(np.mean(run_metrics)),
        "accuracy_std": float(np.std(run_metrics, ddof=1)) if len(run_metrics) > 1 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End WhisAID & MAS-LoRA Runner")
    parser.add_argument("--csv", default="output/0722/data/wu_vowel_segments.paper_regions.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-region", required=True)
    parser.add_argument("--model-type", choices=["whisaid", "mas_lora"], required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-adapt-steps", type=int, default=30)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--region-column", default="paper_region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--protocol", default="leave_one_region_out")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--aux-cache-dir", default="")
    parser.add_argument("--matrix-cache-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load Data
    df = pd.read_csv(args.csv)
    if args.holdout_region:
        df = df[df[args.region_column].astype(str) != "nan"].reset_index(drop=True)

    train_df, val_df, test_df, split_info = split_data(df, args)

    source_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    region_map = {r: i for i, r in enumerate(source_regions)}

    speakers = sorted(train_df[args.speaker_column].astype(str).unique().tolist())
    speaker_map = {s: i for i, s in enumerate(speakers)}

    extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_PATH, local_files_only=True)
    train_ds = AudioSegmentDataset(train_df, region_map, speaker_map)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. Build & Train Model End-to-End
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model_type == "whisaid":
        model = EndToEndWhisAID(
            whisper_path=WHISPER_PATH,
            num_vowels=len(VOWEL_ORDER),
            num_regions=len(source_regions),
            num_speakers=len(speakers),
            accent_dim=64,
        ).to(device)

        print(f">>> E2E Training WhisAID on source regions (excluding {args.holdout_region})...", flush=True)
        model = train_whisaid_e2e(model, extractor, train_loader, args, device)
        torch.save(model.state_dict(), MODEL_CHECKPOINT_DIR / f"whisaid_e2e_{args.holdout_region[:2]}.pt")

    elif args.model_type == "mas_lora":
        model = EndToEndMASLoRA(
            whisper_path=WHISPER_PATH,
            num_vowels=len(VOWEL_ORDER),
            num_source_regions=len(source_regions),
            r=16,
            lora_alpha=1.0,
        ).to(device)

        print(f">>> E2E Training MAS-LoRA on source regions (excluding {args.holdout_region})...", flush=True)
        model = train_maslora_e2e(model, extractor, train_loader, args, device)
        torch.save(model.state_dict(), MODEL_CHECKPOINT_DIR / f"maslora_e2e_{args.holdout_region[:2]}.pt")

    # 3. Extract Target Features & Evaluate 4-Shot Adaptation
    print(f">>> Extracting tuned target features and evaluating 4-shot adaptation on {args.holdout_region}...", flush=True)
    test_x, test_y = extract_test_features(model, extractor, test_df, device)

    eval_res = evaluate_target_4shot(model, test_x, test_y, args, device)
    print(f"[{args.model_type.upper()}] {args.holdout_region} Accuracy: {eval_res['accuracy']*100:.2f}% ± {eval_res['accuracy_std']*100:.2f}%", flush=True)

    out_data = {
        "model_type": args.model_type,
        "holdout_region": args.holdout_region,
        "metrics": {"accuracy": eval_res["accuracy"]},
        "metrics_std": {"accuracy": eval_res["accuracy_std"]},
        "eval_split_runs": args.eval_split_runs,
        "target_support_shots": args.target_support_shots,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
