#!/usr/bin/env python3
"""Run one leave-one-region-out Feature-Memory Transformer experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from pc_dlcmnet.data.acoustic_features import (
    apply_vector_norm,
    audio_path_candidates,
    aux_rows_hash,
    extract_aux_matrix_cached,
    fit_vector_norm,
    serializable_norm,
)
from pc_dlcmnet.utils.paths import default_project_root
from pc_dlcmnet.models.feature_memory import (
    VOWEL_ORDER,
    VOWEL_TO_ID,
    anchor_alignment_loss,
    build_model_from_variant,
    memory_diagnostics,
    region_compactness_loss,
    residual_norm_loss,
)


def default_old_root() -> Path:
    return default_project_root()


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--curve-output", default="")
    parser.add_argument("--debug-output", default="")
    parser.add_argument("--audit-output", default="")
    parser.add_argument("--checkpoint-output", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--protocol", choices=["loro", "random"], default="loro")
    parser.add_argument("--holdout-region", default="")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(old_root / "output/salmonn_style_whisper_cache_base"))
    parser.add_argument("--matrix-cache-dir", default=str(old_root / "output/ppm_supervised_matrix_cache"))
    parser.add_argument("--feature-norm", choices=["none", "global"], default="global")
    parser.add_argument("--matrix-cache-scan", action="store_true", default=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--token-chunks", type=int, default=32)
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--aux-source", choices=["auto", "acoustic", "whisper_stats", "old_fusion_cache"], default="auto")
    parser.add_argument("--aux-representation", choices=["auto", "sequence", "stats"], default="auto")
    parser.add_argument("--aux-cache-dir", default="output/0614/feature_memory_aux_cache")
    parser.add_argument("--old-feature-cache-dir", default=str(old_root / "output/feature_cache_vowel"))
    parser.add_argument("--aux-n-mfcc", type=int, default=39)
    parser.add_argument("--aux-n-mels", type=int, default=128)
    parser.add_argument("--aux-f0-segments", type=int, default=5)
    parser.add_argument("--aux-use-delta-mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-use-formants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--variant",
        choices=[
            "speech_baseline",
            "memory_only",
            "feature_memory_no_prompt",
            "feature_memory_prompt",
            "region_memory_prompt",
            "recurrent_memory_prompt",
            "recurrent_region_memory_prompt",
        ],
        default="feature_memory_prompt",
    )
    parser.add_argument("--recurrent-episode-column", default="")
    parser.add_argument("--recurrent-episode-length", type=int, default=16)
    parser.add_argument("--recurrent-episode-batch-size", type=int, default=4)
    parser.add_argument("--recurrent-eval-mode", choices=["frozen", "online", "supervised"], default="frozen")
    parser.add_argument("--recurrent-online-threshold", type=float, default=0.85)
    parser.add_argument("--anchor-kind", choices=["none", "random", "onehot", "ipa"], default="onehot")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--sampler", choices=["random", "domain_balanced", "domain_class_balanced"], default="random")
    parser.add_argument("--sampler-domain-power", type=float, default=1.0)
    parser.add_argument("--sampler-class-power", type=float, default=0.5)
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-residual", type=float, default=0.0)
    parser.add_argument("--lambda-recurrent-static", type=float, default=0.0)
    parser.add_argument("--lambda-region-cls", type=float, default=0.0)
    parser.add_argument("--lambda-region-compact", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--score-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--audit-every", type=int, default=50)
    parser.add_argument("--require-all-source-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(requested: str) -> torch.device:
    def cuda_works(device: torch.device) -> tuple[bool, str]:
        try:
            torch.empty(1, device=device)
            return True, ""
        except RuntimeError as exc:
            return False, str(exc)

    if requested == "auto":
        cuda_device = torch.device("cuda")
        ok, _message = cuda_works(cuda_device)
        return cuda_device if ok else torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        ok, message = cuda_works(device)
        if not ok:
            if os.environ.get("PC_DLCMNET_REQUIRE_CUDA") == "1":
                raise RuntimeError(f"requested {requested}, but CUDA initialization failed: {message}")
            print(f"[warn] requested {requested}, but CUDA initialization failed; using cpu: {message}", flush=True)
            return torch.device("cpu")
    return device


def dataframe_hash(df: pd.DataFrame, columns: list[str]) -> str:
    present = [col for col in columns if col in df.columns]
    payload = df[present].sort_values(present).to_csv(index=False) if present else str(len(df))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_rows_hash(df: pd.DataFrame, args: argparse.Namespace) -> str:
    return dataframe_hash(df, ["sample_id", args.path_column, args.start_column, args.end_column])


def matrix_cache_key(rows: pd.DataFrame, args: argparse.Namespace, cache_dir: Path) -> str:
    payload = {
        "rows": feature_rows_hash(rows, args),
        "whisper_model": args.whisper_model,
        "token_chunks": int(args.token_chunks),
        "cache_dir": str(cache_dir.resolve()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def find_matrix_cache_by_metadata(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    split_name: str,
) -> Path | None:
    cache_dir = Path(args.matrix_cache_dir)
    if not cache_dir.exists() or not bool(args.matrix_cache_scan):
        return None
    row_hash = feature_rows_hash(rows, args)
    expected_rows = int(len(rows))
    candidates = sorted(cache_dir.glob(f"{split_name}_*.json"))
    for meta_path in candidates:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("row_hash") != row_hash:
            continue
        if int(meta.get("rows", -1)) != expected_rows:
            continue
        if str(meta.get("whisper_model", "")) != str(args.whisper_model):
            continue
        if int(meta.get("token_chunks", -1)) != int(args.token_chunks):
            continue
        matrix_path = meta_path.with_suffix(".npy")
        if matrix_path.exists():
            return matrix_path
    return None


def row_key(row: pd.Series, args: argparse.Namespace, path_value: str | None = None) -> str:
    wav_path = str(row[args.path_column]) if path_value is None else str(path_value)
    key = "|".join(
        [
            wav_path,
            f"{float(row[args.start_column]):.4f}",
            f"{float(row[args.end_column]):.4f}",
            str(args.whisper_model),
            str(args.token_chunks),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def legacy_row_key(row: pd.Series, args: argparse.Namespace, label_column: str, path_value: str | None = None) -> str:
    wav_path = str(row[args.path_column]) if path_value is None else str(path_value)
    key = "|".join(
        [
            wav_path,
            f"{float(row[args.start_column]):.4f}",
            f"{float(row[args.end_column]):.4f}",
            str(row[label_column]),
            str(args.whisper_model),
            str(args.token_chunks),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def resolve_audio_path(row: pd.Series, args: argparse.Namespace) -> str | None:
    for candidate in audio_path_candidates(row[args.path_column], args):
        if Path(candidate).exists():
            return candidate
    return None


def load_segment(row: pd.Series, args: argparse.Namespace, path_value: str) -> np.ndarray:
    audio, sr = sf.read(path_value)
    if audio.ndim == 2:
        audio = audio[:, 0]
    start = max(0, int(round(float(row[args.start_column]) * sr)))
    end = max(start + 1, int(round(float(row[args.end_column]) * sr)))
    audio = audio[start:end].astype(np.float32)
    if sr != args.target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.target_sr)
    if audio.size < args.target_sr:
        audio = np.pad(audio, (0, args.target_sr - audio.size))
    return audio.astype(np.float32)


def chunk_tokens(encoded: torch.Tensor, token_chunks: int) -> np.ndarray:
    seq = encoded.squeeze(0)
    chunks = torch.chunk(seq, int(token_chunks), dim=0)
    pooled = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
    return pooled.detach().cpu().numpy().astype(np.float16)


def extract_one(
    row: pd.Series,
    args: argparse.Namespace,
    whisper_model: Any,
    device: torch.device,
    cache_dir: Path,
) -> np.ndarray:
    path_candidates = audio_path_candidates(row[args.path_column], args)
    primary_cache_path = cache_dir / f"{row_key(row, args, path_candidates[0])}.npy"
    for path_value in path_candidates:
        cache_path = cache_dir / f"{row_key(row, args, path_value)}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)
        for label_column in (args.label_column, "vowel", "tone"):
            if label_column in row:
                legacy_cache_path = cache_dir / f"{legacy_row_key(row, args, label_column, path_value)}.npy"
                if legacy_cache_path.exists():
                    tokens = np.load(legacy_cache_path).astype(np.float32)
                    if path_value == path_candidates[0] and not primary_cache_path.exists():
                        np.save(primary_cache_path, tokens.astype(np.float16))
                    return tokens

    if whisper_model is None:
        raise AttributeError("Whisper model is required because token cache missed.")
    resolved_path = resolve_audio_path(row, args)
    if resolved_path is None:
        raise FileNotFoundError(f"audio file not found for {row[args.path_column]}")
    import whisper

    audio = load_segment(row, args, resolved_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(device)
    with torch.no_grad():
        encoded = whisper_model.encoder(mel.unsqueeze(0))
    tokens = chunk_tokens(encoded, args.token_chunks)
    cache_path = cache_dir / f"{row_key(row, args, resolved_path)}.npy"
    np.save(cache_path, tokens)
    return tokens.astype(np.float32)


def extract_matrix_uncached(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    whisper_model: Any,
    device: torch.device,
    cache_dir: Path,
) -> np.ndarray:
    feats = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="whisper"):
        feats.append(extract_one(row, args, whisper_model, device, cache_dir))
    return np.stack(feats, axis=0).astype(np.float32)


def extract_matrix_cached(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    whisper_model: Any,
    device: torch.device,
    cache_dir: Path,
    split_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix_cache_dir = Path(args.matrix_cache_dir)
    matrix_cache_dir.mkdir(parents=True, exist_ok=True)
    key = matrix_cache_key(rows, args, cache_dir)
    exact_path = matrix_cache_dir / f"{split_name}_{key}.npy"
    if exact_path.exists():
        print(f"[matrix-cache] loaded exact {split_name}: {exact_path}", flush=True)
        return np.load(exact_path).astype(np.float32), {"source": "exact", "path": str(exact_path)}
    scanned = find_matrix_cache_by_metadata(rows, args, split_name)
    if scanned is not None:
        print(f"[matrix-cache] loaded metadata {split_name}: {scanned}", flush=True)
        return np.load(scanned).astype(np.float32), {"source": "metadata_scan", "path": str(scanned)}

    x = extract_matrix_uncached(rows, args, whisper_model, device, cache_dir)
    np.save(exact_path, x.astype(np.float16))
    meta_path = exact_path.with_suffix(".json")
    meta = {
        "split_name": split_name,
        "rows": int(len(rows)),
        "shape": list(x.shape),
        "dtype": "float16",
        "row_hash": feature_rows_hash(rows, args),
        "whisper_model": args.whisper_model,
        "token_chunks": int(args.token_chunks),
        "cache_dir": str(cache_dir.resolve()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[matrix-cache] saved {split_name}: {exact_path}", flush=True)
    return x.astype(np.float32), {"source": "computed", "path": str(exact_path)}


def fit_token_norm(x: np.ndarray, mode: str) -> dict[str, np.ndarray | str]:
    if mode == "none":
        return {"mode": "none"}
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    return {"mode": "global", "mean": mean, "std": np.maximum(std, 1e-6)}


def apply_token_norm(x: np.ndarray, norm: dict[str, np.ndarray | str]) -> np.ndarray:
    if norm["mode"] == "none":
        return x.astype(np.float32)
    return ((x - norm["mean"]) / norm["std"]).astype(np.float32)


def stratify_or_none(labels: pd.Series):
    counts = labels.value_counts()
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def normalize_region_value(value: str, available: pd.Series) -> str:
    regions = set(available.dropna().astype(str).unique().tolist())
    if value in regions:
        return value
    for encoding in ("gbk", "cp936", "latin1"):
        try:
            candidate = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if candidate in regions:
            return candidate
    return value


def stratified_train_subset(df: pd.DataFrame, fraction: float, seed: int, label_col: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(int(seed))
    for _label, group in df.groupby(label_col, sort=True):
        n = max(1, int(round(len(group) * float(fraction))))
        chosen = rng.choice(group.index.to_numpy(), size=min(n, len(group)), replace=False)
        rows.extend(chosen.tolist())
    return df.loc[rows].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    if args.labels is None:
        args.labels = list(VOWEL_ORDER)
    required = [args.path_column, args.start_column, args.end_column, args.label_column, args.region_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.dropna(subset=required).copy()
    df[args.label_column] = df[args.label_column].astype(str)
    df = df[df[args.label_column].isin(args.labels)].copy()
    df[args.start_column] = pd.to_numeric(df[args.start_column], errors="coerce")
    df[args.end_column] = pd.to_numeric(df[args.end_column], errors="coerce")
    df = df.dropna(subset=[args.start_column, args.end_column])
    df = df[df[args.end_column] > df[args.start_column]].copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"row_{idx:08d}" for idx in range(len(df))]
    if df.empty:
        raise ValueError("No usable rows after dataset filtering.")
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    label_col = args.label_column
    if args.protocol == "loro":
        if not args.holdout_region:
            raise ValueError("--holdout-region is required for --protocol loro")
        holdout = normalize_region_value(str(args.holdout_region), df[args.region_column])
        args.holdout_region = holdout
        is_test = df[args.region_column].astype(str) == holdout
        train_val = df[~is_test].copy()
        test = df[is_test].copy()
        if test.empty:
            raise ValueError(f"Holdout region has no rows: {holdout}")
        train, val = train_test_split(
            train_val,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=stratify_or_none(train_val[label_col]),
        )
        split_info: dict[str, Any] = {
            "protocol": "loro",
            "holdout_region": holdout,
            "target_labels_used_for_training": False,
        }
    else:
        train_val, test = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify_or_none(df[label_col]),
        )
        train, val = train_test_split(
            train_val,
            test_size=args.val_size / max(1e-9, 1.0 - args.test_size),
            random_state=args.seed,
            stratify=stratify_or_none(train_val[label_col]),
        )
        split_info = {"protocol": "random", "target_labels_used_for_training": False}
    if args.train_fraction < 1.0:
        train = stratified_train_subset(train, args.train_fraction, args.seed, label_col)
        split_info["train_fraction"] = float(args.train_fraction)
    train_regions = sorted(train[args.region_column].astype(str).unique().tolist())
    split_info["train_regions"] = train_regions
    split_info["num_train_regions"] = int(len(train_regions))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), split_info


def labels_to_ids(labels: pd.Series) -> np.ndarray:
    return labels.astype(str).map(VOWEL_TO_ID).to_numpy(dtype=np.int64)


def class_loss_weight(y: np.ndarray, power: float, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(VOWEL_ORDER)).astype(np.float32)
    weights = np.ones(len(VOWEL_ORDER), dtype=np.float32)
    valid = counts > 0
    weights[valid] = (float(len(y)) / (len(VOWEL_ORDER) * counts[valid])) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def label_counts(labels: pd.Series, class_names: Sequence[str]) -> dict[str, int]:
    counts = labels.astype(str).value_counts()
    return {str(label): int(counts.get(str(label), 0)) for label in class_names}


def region_counts(rows: pd.DataFrame, region_column: str) -> dict[str, int]:
    counts = rows[region_column].astype(str).value_counts().sort_index()
    return {str(region): int(count) for region, count in counts.items()}


def region_label_counts(rows: pd.DataFrame, args: argparse.Namespace) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for region, group in rows.groupby(args.region_column, sort=True):
        nested[str(region)] = label_counts(group[args.label_column], VOWEL_ORDER)
    return nested


def build_split_audit(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    split_info: dict[str, Any],
) -> dict[str, Any]:
    all_regions = sorted(full_df[args.region_column].astype(str).unique().tolist())
    holdout = str(args.holdout_region)
    source_pool = full_df[full_df[args.region_column].astype(str) != holdout]
    source_regions = sorted(source_pool[args.region_column].astype(str).unique().tolist())
    train_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    val_regions = sorted(val_df[args.region_column].astype(str).unique().tolist())
    test_regions = sorted(test_df[args.region_column].astype(str).unique().tolist())
    train_region_set = set(train_regions)
    source_region_set = set(source_regions)
    target_in_train = int((train_df[args.region_column].astype(str) == holdout).sum())
    target_in_val = int((val_df[args.region_column].astype(str) == holdout).sum())
    non_target_in_test = int((test_df[args.region_column].astype(str) != holdout).sum())
    missing_source_regions = sorted(source_region_set - train_region_set)
    audit = {
        "protocol": str(args.protocol),
        "holdout_region": holdout,
        "all_regions": all_regions,
        "num_all_regions": int(len(all_regions)),
        "source_pool_regions": source_regions,
        "num_source_pool_regions": int(len(source_regions)),
        "train_regions": train_regions,
        "val_regions": val_regions,
        "test_regions": test_regions,
        "num_train_regions": int(len(train_regions)),
        "source_pool_rows": int(len(source_pool)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_plus_val_rows": int(len(train_df) + len(val_df)),
        "target_rows_in_train": target_in_train,
        "target_rows_in_val": target_in_val,
        "non_target_rows_in_test": non_target_in_test,
        "target_labels_used_for_training": bool(split_info.get("target_labels_used_for_training", True)),
        "train_regions_cover_all_non_target_regions": train_region_set == source_region_set,
        "missing_source_regions_in_train": missing_source_regions,
        "val_regions_subset_of_source_pool": set(val_regions).issubset(source_region_set),
        "test_is_holdout_only": test_regions == [holdout],
        "train_fraction": float(args.train_fraction),
        "val_size": float(args.val_size),
        "region_counts": {
            "full": region_counts(full_df, args.region_column),
            "source_pool": region_counts(source_pool, args.region_column),
            "train": region_counts(train_df, args.region_column),
            "val": region_counts(val_df, args.region_column),
            "test": region_counts(test_df, args.region_column),
        },
        "label_counts_by_region": {
            "train": region_label_counts(train_df, args),
            "test": region_label_counts(test_df, args),
        },
    }
    if args.protocol == "loro":
        if target_in_train or target_in_val or non_target_in_test:
            raise ValueError(f"LORO split audit failed: {json.dumps(audit, ensure_ascii=False)}")
        if bool(args.require_all_source_regions) and missing_source_regions:
            raise ValueError(f"Training split is missing source regions: {missing_source_regions}")
    return audit


def grad_norm_named(named_params: list[tuple[str, nn.Parameter]], predicate) -> tuple[float, int]:
    total = 0.0
    count = 0
    for name, param in named_params:
        if not predicate(name) or param.grad is None:
            continue
        total += float(param.grad.detach().norm(2).cpu()) ** 2
        count += 1
    return total ** 0.5, count


def module_grad_audit(model: nn.Module, region_head: nn.Module | None) -> dict[str, float]:
    named_params = [(name, param) for name, param in model.named_parameters()]
    groups = {
        "class_memory": lambda name: "class_memory" in name,
        "region_memory": lambda name: name.startswith("region_") or "region_memory" in name,
        "feature_fusion": lambda name: name.startswith("feature_fusion."),
        "prompt": lambda name: ".prompt" in name or "prompt_to_gates" in name,
        "memory_write": lambda name: name.startswith("memory_write_gate."),
        "transformer": lambda name: name.startswith("layers."),
        "speech": lambda name: name.startswith("speech_") or name in {"audio_pos", "audio_type"},
        "scorer": lambda name: name.startswith("memory_mlp.") or name.startswith("shared_scorer.") or name == "class_score_bias",
    }
    out: dict[str, float] = {}
    for group_name, predicate in groups.items():
        norm, count = grad_norm_named(named_params, predicate)
        out[f"grad_{group_name}"] = float(norm)
        out[f"grad_{group_name}_params"] = float(count)
    if region_head is not None:
        total = 0.0
        count = 0
        for param in region_head.parameters():
            if param.grad is None:
                continue
            total += float(param.grad.detach().norm(2).cpu()) ** 2
            count += 1
        out["grad_region_head"] = float(total ** 0.5)
        out["grad_region_head_params"] = float(count)
    else:
        out["grad_region_head"] = 0.0
        out["grad_region_head_params"] = 0.0
    return out


def forward_activation_audit(out: dict[str, torch.Tensor]) -> dict[str, float]:
    audit: dict[str, float] = {}
    feature_weights = out.get("feature_weights")
    if feature_weights is not None and feature_weights.numel() > 0:
        weights = feature_weights.detach()
        means = weights.mean(dim=0).cpu().tolist()
        for idx, value in enumerate(means):
            audit[f"feature_weight_{idx}_mean"] = float(value)
        audit["feature_weight_entropy"] = float((-(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)).mean().cpu())
    feature_bias = out.get("feature_bias")
    if feature_bias is not None and feature_bias.numel() > 0:
        audit["feature_bias_norm"] = float(feature_bias.detach().norm(dim=-1).mean().cpu())
    feature_gate = out.get("feature_gate")
    if feature_gate is not None and feature_gate.numel() > 0:
        gate = feature_gate.detach()
        audit["feature_gate_mean"] = float(gate.mean().cpu())
        audit["feature_gate_max"] = float(gate.max().cpu())
    region_attention = out.get("region_attention")
    if region_attention is not None and region_attention.numel() > 0:
        attn = region_attention.detach().clamp_min(1e-8)
        audit["region_attention_entropy"] = float((-(attn.log() * attn).sum(dim=-1)).mean().cpu())
        audit["region_attention_max"] = float(attn.max(dim=-1).values.mean().cpu())
    region_context = out.get("region_context")
    if region_context is not None and region_context.numel() > 0:
        audit["region_context_norm"] = float(region_context.detach().norm(dim=-1).mean().cpu())
    conditioned_memory = out.get("conditioned_memory")
    if conditioned_memory is not None and conditioned_memory.numel() > 0:
        audit["conditioned_memory_norm"] = float(conditioned_memory.detach().norm(dim=-1).mean().cpu())
    base_memory = out.get("base_memory")
    if base_memory is not None and base_memory.numel() > 0:
        audit["base_memory_norm"] = float(base_memory.detach().norm(dim=-1).mean().cpu())
    candidate_memory = out.get("candidate_memory")
    if candidate_memory is not None and candidate_memory.numel() > 0:
        audit["candidate_memory_norm"] = float(candidate_memory.detach().norm(dim=-1).mean().cpu())
    write_gate = out.get("write_gate")
    if write_gate is not None and write_gate.numel() > 0:
        gate = write_gate.detach()
        audit["write_gate_mean"] = float(gate.mean().cpu())
        audit["write_gate_min"] = float(gate.min().cpu())
        audit["write_gate_max"] = float(gate.max().cpu())
    write_update = out.get("write_update")
    if write_update is not None and write_update.numel() > 0:
        audit["write_update_norm"] = float(write_update.detach().norm(dim=-1).mean().cpu())
    updated_memory = out.get("updated_memory")
    if updated_memory is not None and updated_memory.numel() > 0:
        audit["updated_memory_norm"] = float(updated_memory.detach().norm(dim=-1).mean().cpu())
    write_weights = out.get("write_weights")
    if write_weights is not None and write_weights.numel() > 0:
        audit["write_weight_active"] = float((write_weights.detach() > 0).float().sum(dim=-1).mean().cpu())
    prompt_stats = out.get("prompt_stats")
    if prompt_stats is not None and prompt_stats.numel() > 0:
        prompt = prompt_stats.detach().mean(dim=0).cpu().tolist()
        names = [
            "prompt_q_gate_mean",
            "prompt_q_gate_std",
            "prompt_k_gate_mean",
            "prompt_k_gate_std",
            "prompt_v_gate_mean",
            "prompt_v_gate_std",
        ]
        for name, value in zip(names, prompt):
            audit[name] = float(value)
    return audit


def train_trace_summary(curve: list[dict[str, float]]) -> dict[str, Any]:
    numeric_keys = [
        "grad_class_memory",
        "grad_feature_fusion",
        "grad_prompt",
        "grad_memory_write",
        "grad_region_memory",
        "grad_transformer",
        "grad_speech",
        "grad_scorer",
        "feature_bias_norm",
        "feature_gate_mean",
        "region_attention_entropy",
        "region_attention_max",
        "region_context_norm",
        "conditioned_memory_norm",
        "candidate_memory_norm",
        "write_gate_mean",
        "write_update_norm",
        "write_weight_active",
        "prompt_q_gate_mean",
        "prompt_v_gate_mean",
    ]
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in curve if key in row]
        if values:
            summary[f"{key}_last"] = values[-1]
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_mean"] = float(np.mean(values))
    return summary


def build_loader(
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    domains: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    mask = np.ones((x.shape[0], x.shape[1]), dtype=bool)
    tensors = [
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(aux, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(y, dtype=torch.long),
    ]
    if domains is not None:
        tensors.append(torch.tensor(domains, dtype=torch.long))
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
        shuffle = False
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)


def is_recurrent_model(model: nn.Module) -> bool:
    return bool(getattr(model, "use_recurrent_memory", False))


def resolve_episode_column(rows: pd.DataFrame, args: argparse.Namespace) -> str:
    requested = str(getattr(args, "recurrent_episode_column", "") or "").strip()
    if requested and requested in rows.columns:
        return requested
    if args.speaker_column in rows.columns:
        return args.speaker_column
    return args.region_column


def build_episode_batches(rows: pd.DataFrame, args: argparse.Namespace) -> tuple[list[np.ndarray], dict[str, Any]]:
    max_len = int(getattr(args, "recurrent_episode_length", 16) or 16)
    max_len = max(1, max_len)
    column = resolve_episode_column(rows, args)
    frame = rows.reset_index(drop=True).copy()
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["_episode_key"] = frame[column].fillna("__missing__").astype(str)
    sort_cols = ["_episode_key"]
    for candidate in [args.path_column, args.start_column, "sample_id"]:
        if candidate in frame.columns:
            sort_cols.append(candidate)
    frame = frame.sort_values(sort_cols, kind="mergesort")
    batches: list[np.ndarray] = []
    episode_lengths = []
    for _key, group in frame.groupby("_episode_key", sort=False):
        indices = group["_row_index"].to_numpy(dtype=np.int64)
        episode_lengths.append(int(len(indices)))
        for start in range(0, len(indices), max_len):
            batches.append(indices[start : start + max_len])
    meta = {
        "episode_column": column,
        "episode_length": max_len,
        "num_episodes": int(frame["_episode_key"].nunique()),
        "num_episode_batches": int(len(batches)),
        "episode_size_min": int(min(episode_lengths)) if episode_lengths else 0,
        "episode_size_max": int(max(episode_lengths)) if episode_lengths else 0,
        "episode_size_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }
    return batches, meta


def merge_forward_outputs(outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not outputs:
        return {}
    merged: dict[str, torch.Tensor] = {}
    concatenate_keys = [
        "logits",
        "previous_memory",
        "candidate_memory",
        "updated_memory",
        "conditioned_memory",
        "memory_residual",
        "write_gate",
        "write_weights",
        "write_update",
        "audio_summary",
        "acoustic_condition",
        "feature_bias",
        "feature_gate",
        "feature_weights",
        "prompt_stats",
        "region_logits",
        "region_attention",
        "region_context",
        "conditioned_region",
    ]
    for key in concatenate_keys:
        tensors = [out[key] for out in outputs if key in out and out[key].numel() > 0]
        if tensors:
            merged[key] = torch.cat(tensors, dim=0)
    for key in ["base_memory", "anchors"]:
        if key in outputs[-1]:
            merged[key] = outputs[-1][key]
    return merged


def build_sample_weights(
    labels: np.ndarray,
    domains: np.ndarray,
    sampler: str,
    domain_power: float,
    class_power: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    sampler = str(sampler)
    if sampler == "random":
        return None, {"sampler": "random"}
    weights = np.ones(len(labels), dtype=np.float64)
    meta: dict[str, Any] = {
        "sampler": sampler,
        "domain_power": float(domain_power),
        "class_power": float(class_power),
    }
    if sampler in {"domain_balanced", "domain_class_balanced"}:
        domain_counts = np.bincount(domains.astype(np.int64))
        domain_weights = np.ones_like(domain_counts, dtype=np.float64)
        valid = domain_counts > 0
        domain_weights[valid] = (float(len(labels)) / (float(valid.sum()) * domain_counts[valid])) ** float(domain_power)
        weights *= domain_weights[domains]
        meta["domain_counts"] = {str(idx): int(count) for idx, count in enumerate(domain_counts.tolist()) if count > 0}
    if sampler == "domain_class_balanced":
        class_counts = np.bincount(labels.astype(np.int64), minlength=len(VOWEL_ORDER))
        class_weights = np.ones_like(class_counts, dtype=np.float64)
        valid = class_counts > 0
        class_weights[valid] = (float(len(labels)) / (float(valid.sum()) * class_counts[valid])) ** float(class_power)
        weights *= class_weights[labels]
        meta["class_counts"] = {VOWEL_ORDER[idx]: int(count) for idx, count in enumerate(class_counts.tolist()) if count > 0}
    weights = weights / max(float(weights.mean()), 1e-12)
    meta["weight_min"] = float(weights.min())
    meta["weight_max"] = float(weights.max())
    meta["weight_mean"] = float(weights.mean())
    return weights.astype(np.float64), meta


class RegionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_recurrent_online(
    model: nn.Module,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    episode_batches: list[np.ndarray] | None,
    threshold: float,
    supervised: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    class_ids = torch.arange(len(VOWEL_ORDER), dtype=torch.long, device=device)
    prob_np = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    pred_np = np.zeros(len(y), dtype=np.int64)
    batches = episode_batches if episode_batches else [np.arange(len(y), dtype=np.int64)]
    residual_norms = []
    feature_bias_norms = []
    feature_gate_means = []
    region_attention_entropies = []
    region_attention_maxes = []
    region_context_norms = []
    prompt_rows = []
    memory_norms = []
    candidate_memory_norms = []
    write_gate_means = []
    write_update_norms = []
    updated_memory_norms = []
    write_weight_active = []
    with torch.no_grad():
        for episode in batches:
            memory_state = None
            for raw_idx in episode:
                idx = int(raw_idx)
                batch_x = torch.tensor(x[idx : idx + 1], dtype=torch.float32, device=device)
                batch_aux = torch.tensor(aux[idx : idx + 1], dtype=torch.float32, device=device)
                batch_mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
                out = model(batch_x, batch_aux, batch_mask, class_ids, memory_state=memory_state)
                prob = torch.softmax(out["logits"], dim=-1)
                pred = int(prob.argmax(dim=-1).item())
                conf = float(prob.max(dim=-1).values.item())
                prob_np[idx] = prob.squeeze(0).cpu().numpy()
                pred_np[idx] = pred

                weights = prob.new_zeros(1, len(VOWEL_ORDER))
                if supervised:
                    weights[0, int(y[idx])] = 1.0
                elif conf >= float(threshold):
                    weights[0, pred] = 1.0
                updated, write_gate, applied_weights = model.apply_memory_write(
                    previous_memory=out["previous_memory"],
                    candidate_memory=out["candidate_memory"],
                    audio_summary=out["audio_summary"],
                    class_ids=class_ids,
                    write_weights=weights,
                )
                out = dict(out)
                out["updated_memory"] = updated
                out["write_gate"] = write_gate
                out["write_weights"] = applied_weights
                out["write_update"] = updated - out["previous_memory"]
                memory_state = updated.squeeze(0)

                if out.get("memory_residual") is not None and out["memory_residual"].numel() > 0:
                    residual_norms.append(out["memory_residual"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("feature_bias") is not None and out["feature_bias"].numel() > 0:
                    feature_bias_norms.append(out["feature_bias"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("feature_gate") is not None and out["feature_gate"].numel() > 0:
                    feature_gate_means.append(out["feature_gate"].mean().detach().cpu().item())
                if out.get("region_attention") is not None and out["region_attention"].numel() > 0:
                    attn = out["region_attention"].detach().clamp_min(1e-8)
                    region_attention_entropies.append((-(attn.log() * attn).sum(dim=-1)).mean().cpu().item())
                    region_attention_maxes.append(attn.max(dim=-1).values.mean().cpu().item())
                if out.get("region_context") is not None and out["region_context"].numel() > 0:
                    region_context_norms.append(out["region_context"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("prompt_stats") is not None and out["prompt_stats"].numel() > 0:
                    prompt_rows.append(out["prompt_stats"].detach().cpu().numpy())
                if out.get("conditioned_memory") is not None and out["conditioned_memory"].numel() > 0:
                    memory_norms.append(out["conditioned_memory"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("candidate_memory") is not None and out["candidate_memory"].numel() > 0:
                    candidate_memory_norms.append(out["candidate_memory"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("write_gate") is not None and out["write_gate"].numel() > 0:
                    write_gate_means.append(out["write_gate"].mean().detach().cpu().item())
                if out.get("write_update") is not None and out["write_update"].numel() > 0:
                    write_update_norms.append(out["write_update"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("updated_memory") is not None and out["updated_memory"].numel() > 0:
                    updated_memory_norms.append(out["updated_memory"].norm(dim=-1).mean().detach().cpu().item())
                if out.get("write_weights") is not None and out["write_weights"].numel() > 0:
                    write_weight_active.append((out["write_weights"] > 0).float().sum(dim=-1).mean().detach().cpu().item())

    metrics = {
        "accuracy": float(accuracy_score(y, pred_np)),
        "macro_f1": float(f1_score(y, pred_np, labels=list(range(len(VOWEL_ORDER))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred_np, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y, pred_np, average="micro", zero_division=0)),
    }
    prompt = np.concatenate(prompt_rows, axis=0) if prompt_rows else np.zeros((1, 6), dtype=np.float32)
    diag = {
        "memory_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "feature_bias_norm": float(np.mean(feature_bias_norms)) if feature_bias_norms else 0.0,
        "feature_gate_mean": float(np.mean(feature_gate_means)) if feature_gate_means else 0.0,
        "region_attention_entropy": float(np.mean(region_attention_entropies)) if region_attention_entropies else 0.0,
        "region_attention_max": float(np.mean(region_attention_maxes)) if region_attention_maxes else 0.0,
        "region_context_norm": float(np.mean(region_context_norms)) if region_context_norms else 0.0,
        "conditioned_memory_norm": float(np.mean(memory_norms)) if memory_norms else 0.0,
        "candidate_memory_norm": float(np.mean(candidate_memory_norms)) if candidate_memory_norms else 0.0,
        "write_gate_mean": float(np.mean(write_gate_means)) if write_gate_means else 0.0,
        "write_update_norm": float(np.mean(write_update_norms)) if write_update_norms else 0.0,
        "updated_memory_norm": float(np.mean(updated_memory_norms)) if updated_memory_norms else 0.0,
        "write_weight_active": float(np.mean(write_weight_active)) if write_weight_active else 0.0,
        "prompt_q_gate_mean": float(prompt[:, 0].mean()),
        "prompt_q_gate_std": float(prompt[:, 1].mean()),
        "prompt_k_gate_mean": float(prompt[:, 2].mean()),
        "prompt_k_gate_std": float(prompt[:, 3].mean()),
        "prompt_v_gate_mean": float(prompt[:, 4].mean()),
        "prompt_v_gate_std": float(prompt[:, 5].mean()),
    }
    return metrics, pred_np, prob_np, diag


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    episode_batches: list[np.ndarray] | None = None,
    recurrent_eval_mode: str = "frozen",
    recurrent_online_threshold: float = 0.85,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float]]:
    if is_recurrent_model(model) and str(recurrent_eval_mode) in {"online", "supervised"}:
        return evaluate_recurrent_online(
            model=model,
            x=x,
            aux=aux,
            y=y,
            device=device,
            episode_batches=episode_batches,
            threshold=float(recurrent_online_threshold),
            supervised=str(recurrent_eval_mode) == "supervised",
        )
    model.eval()
    class_ids = torch.arange(len(VOWEL_ORDER), dtype=torch.long, device=device)
    probs = []
    preds = []
    residual_norms = []
    feature_bias_norms = []
    feature_gate_means = []
    region_attention_entropies = []
    region_attention_maxes = []
    region_context_norms = []
    prompt_rows = []
    memory_norms = []
    candidate_memory_norms = []
    write_gate_means = []
    write_update_norms = []
    updated_memory_norms = []
    write_weight_active = []
    loader = build_loader(x, aux, y, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_aux, batch_mask, _batch_y in loader:
            out = model(batch_x.to(device), batch_aux.to(device), batch_mask.to(device), class_ids)
            prob = torch.softmax(out["logits"], dim=-1)
            probs.append(prob.cpu().numpy())
            preds.append(prob.argmax(dim=-1).cpu().numpy())
            if out.get("memory_residual") is not None and out["memory_residual"].numel() > 0:
                residual_norms.append(out["memory_residual"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("feature_bias") is not None and out["feature_bias"].numel() > 0:
                feature_bias_norms.append(out["feature_bias"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("feature_gate") is not None and out["feature_gate"].numel() > 0:
                feature_gate_means.append(out["feature_gate"].mean().detach().cpu().item())
            if out.get("region_attention") is not None and out["region_attention"].numel() > 0:
                attn = out["region_attention"].detach().clamp_min(1e-8)
                region_attention_entropies.append((-(attn.log() * attn).sum(dim=-1)).mean().cpu().item())
                region_attention_maxes.append(attn.max(dim=-1).values.mean().cpu().item())
            if out.get("region_context") is not None and out["region_context"].numel() > 0:
                region_context_norms.append(out["region_context"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("prompt_stats") is not None and out["prompt_stats"].numel() > 0:
                prompt_rows.append(out["prompt_stats"].detach().cpu().numpy())
            if out.get("conditioned_memory") is not None and out["conditioned_memory"].numel() > 0:
                memory_norms.append(out["conditioned_memory"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("candidate_memory") is not None and out["candidate_memory"].numel() > 0:
                candidate_memory_norms.append(out["candidate_memory"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("write_gate") is not None and out["write_gate"].numel() > 0:
                write_gate_means.append(out["write_gate"].mean().detach().cpu().item())
            if out.get("write_update") is not None and out["write_update"].numel() > 0:
                write_update_norms.append(out["write_update"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("updated_memory") is not None and out["updated_memory"].numel() > 0:
                updated_memory_norms.append(out["updated_memory"].norm(dim=-1).mean().detach().cpu().item())
            if out.get("write_weights") is not None and out["write_weights"].numel() > 0:
                write_weight_active.append((out["write_weights"] > 0).float().sum(dim=-1).mean().detach().cpu().item())
    prob_np = np.concatenate(probs, axis=0)
    pred_np = np.concatenate(preds, axis=0).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, pred_np)),
        "macro_f1": float(f1_score(y, pred_np, labels=list(range(len(VOWEL_ORDER))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred_np, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y, pred_np, average="micro", zero_division=0)),
    }
    prompt = np.concatenate(prompt_rows, axis=0) if prompt_rows else np.zeros((1, 6), dtype=np.float32)
    diag = {
        "memory_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "feature_bias_norm": float(np.mean(feature_bias_norms)) if feature_bias_norms else 0.0,
        "feature_gate_mean": float(np.mean(feature_gate_means)) if feature_gate_means else 0.0,
        "region_attention_entropy": float(np.mean(region_attention_entropies)) if region_attention_entropies else 0.0,
        "region_attention_max": float(np.mean(region_attention_maxes)) if region_attention_maxes else 0.0,
        "region_context_norm": float(np.mean(region_context_norms)) if region_context_norms else 0.0,
        "conditioned_memory_norm": float(np.mean(memory_norms)) if memory_norms else 0.0,
        "candidate_memory_norm": float(np.mean(candidate_memory_norms)) if candidate_memory_norms else 0.0,
        "write_gate_mean": float(np.mean(write_gate_means)) if write_gate_means else 0.0,
        "write_update_norm": float(np.mean(write_update_norms)) if write_update_norms else 0.0,
        "updated_memory_norm": float(np.mean(updated_memory_norms)) if updated_memory_norms else 0.0,
        "write_weight_active": float(np.mean(write_weight_active)) if write_weight_active else 0.0,
        "prompt_q_gate_mean": float(prompt[:, 0].mean()),
        "prompt_q_gate_std": float(prompt[:, 1].mean()),
        "prompt_k_gate_mean": float(prompt[:, 2].mean()),
        "prompt_k_gate_std": float(prompt[:, 3].mean()),
        "prompt_v_gate_mean": float(prompt[:, 4].mean()),
        "prompt_v_gate_std": float(prompt[:, 5].mean()),
    }
    return metrics, pred_np, prob_np, diag


def run_recurrent_training_batch(
    model: nn.Module,
    region_head: nn.Module | None,
    episodes: Sequence[np.ndarray] | np.ndarray,
    train_x: np.ndarray,
    train_aux: np.ndarray,
    train_y: np.ndarray,
    train_domains: np.ndarray,
    class_ids: torch.Tensor,
    class_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if isinstance(episodes, np.ndarray):
        episode_list = [episodes.astype(np.int64)]
    else:
        episode_list = [np.asarray(ep, dtype=np.int64) for ep in episodes if len(ep) > 0]
    if not episode_list:
        raise ValueError("empty recurrent episode batch")

    flat_indices = np.concatenate(episode_list).astype(np.int64)
    static_x = torch.tensor(train_x[flat_indices], dtype=torch.float32, device=device)
    static_aux = torch.tensor(train_aux[flat_indices], dtype=torch.float32, device=device)
    static_mask = torch.ones((len(flat_indices), train_x.shape[1]), dtype=torch.bool, device=device)
    batch_y = torch.tensor(train_y[flat_indices], dtype=torch.long, device=device)
    batch_domain = torch.tensor(train_domains[flat_indices], dtype=torch.long, device=device)

    outputs: list[dict[str, torch.Tensor]] = []
    recurrent_labels = []
    recurrent_domains = []
    states: list[torch.Tensor | None] = [None for _ in episode_list]
    max_len = max(len(ep) for ep in episode_list)
    for pos in range(max_len):
        active_slots = [slot for slot, ep in enumerate(episode_list) if pos < len(ep)]
        step_indices = np.asarray([episode_list[slot][pos] for slot in active_slots], dtype=np.int64)
        batch_x = torch.tensor(train_x[step_indices], dtype=torch.float32, device=device)
        batch_aux = torch.tensor(train_aux[step_indices], dtype=torch.float32, device=device)
        batch_mask = torch.ones((len(step_indices), train_x.shape[1]), dtype=torch.bool, device=device)
        step_y = torch.tensor(train_y[step_indices], dtype=torch.long, device=device)
        step_domain = torch.tensor(train_domains[step_indices], dtype=torch.long, device=device)
        active_states = [states[slot] for slot in active_slots]
        if all(state is None for state in active_states):
            memory_state = None
        else:
            fallback_state = model.class_memory[class_ids].detach()
            memory_state = torch.stack(
                [state if state is not None else fallback_state for state in active_states],
                dim=0,
            )
        out = model(
            batch_x,
            batch_aux,
            batch_mask,
            class_ids,
            step_domain,
            memory_state=memory_state,
            write_labels=step_y,
        )
        outputs.append(out)
        recurrent_labels.append(step_y)
        recurrent_domains.append(step_domain)
        for out_pos, slot in enumerate(active_slots):
            states[slot] = out["updated_memory"][out_pos]

    out = merge_forward_outputs(outputs)
    recurrent_y = torch.cat(recurrent_labels, dim=0)
    recurrent_domain = torch.cat(recurrent_domains, dim=0)
    loss_cls = F.cross_entropy(
        out["logits"],
        recurrent_y,
        weight=class_weight,
        label_smoothing=float(args.label_smoothing),
    )
    static_out = model(static_x, static_aux, static_mask, class_ids, batch_domain)
    loss_static = F.cross_entropy(
        static_out["logits"],
        batch_y,
        weight=class_weight,
        label_smoothing=float(args.label_smoothing),
    )
    loss_anchor = anchor_alignment_loss(out["base_memory"], out["anchors"])
    loss_residual = residual_norm_loss(out["memory_residual"])
    loss_region_cls = out["logits"].new_tensor(0.0)
    if out.get("region_logits") is not None and out["region_logits"].numel() > 0:
        loss_region_cls = F.cross_entropy(out["region_logits"], recurrent_domain)
    elif region_head is not None:
        loss_region_cls = F.cross_entropy(region_head(out["acoustic_condition"]), recurrent_domain)
    loss_region_compact = region_compactness_loss(out["acoustic_condition"], recurrent_domain)
    return loss_cls, loss_static, loss_anchor, loss_residual, loss_region_cls, loss_region_compact, out, batch_y, batch_domain, static_out["logits"]


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_aux: np.ndarray,
    train_y: np.ndarray,
    train_domains: np.ndarray,
    val_x: np.ndarray,
    val_aux: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    num_domains: int,
    train_episode_batches: list[np.ndarray] | None = None,
    val_episode_batches: list[np.ndarray] | None = None,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    model.to(device)
    region_head = None
    if float(args.lambda_region_cls) > 0 and num_domains > 1 and not bool(getattr(model, "use_region_memory", False)):
        region_head = RegionHead(args.hidden_dim, num_domains, args.dropout).to(device)
    params = list(model.parameters()) + ([] if region_head is None else list(region_head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sample_weights, sampler_meta = build_sample_weights(
        train_y,
        train_domains,
        args.sampler,
        args.sampler_domain_power,
        args.sampler_class_power,
    )
    loader = build_loader(
        train_x,
        train_aux,
        train_y,
        args.batch_size,
        shuffle=True,
        domains=train_domains,
        sample_weights=sample_weights,
    )
    use_recurrent = is_recurrent_model(model)
    recurrent_batches = train_episode_batches if train_episode_batches else [np.arange(len(train_y), dtype=np.int64)]
    if use_recurrent:
        sampler_meta["effective_sampler"] = "episode_order"
        sampler_meta["episode_batch_size"] = int(max(1, getattr(args, "recurrent_episode_batch_size", 4)))
        sampler_meta["sample_weight_sampler_ignored"] = bool(sample_weights is not None)
    class_ids = torch.arange(len(VOWEL_ORDER), dtype=torch.long, device=device)
    weight = class_loss_weight(train_y, args.class_weight_power, device)
    best_state = deepcopy(model.state_dict())
    best_region_state = deepcopy(region_head.state_dict()) if region_head is not None else None
    best_macro = -1.0
    curve: list[dict[str, float]] = []
    initial_memory = None
    if hasattr(model, "class_memory"):
        initial_memory = model.class_memory.detach().cpu().clone()
    step = 0
    epoch = 0
    start_time = time.perf_counter()
    while step < int(args.max_steps):
        epoch += 1
        model.train()
        if region_head is not None:
            region_head.train()
        if use_recurrent:
            order = list(range(len(recurrent_batches)))
            random.Random(int(args.seed) + epoch).shuffle(order)
            ordered_batches = [recurrent_batches[idx] for idx in order]
            episode_batch_size = int(max(1, getattr(args, "recurrent_episode_batch_size", 4)))
            batch_iter = [
                ordered_batches[start : start + episode_batch_size]
                for start in range(0, len(ordered_batches), episode_batch_size)
            ]
        else:
            batch_iter = loader
        for batch in batch_iter:
            step += 1
            if use_recurrent:
                (
                    loss_cls,
                    loss_static,
                    loss_anchor,
                    loss_residual,
                    loss_region_cls,
                    loss_region_compact,
                    out,
                    batch_y,
                    batch_domain,
                    static_logits,
                ) = run_recurrent_training_batch(
                    model=model,
                    region_head=region_head,
                    episodes=batch,
                    train_x=train_x,
                    train_aux=train_aux,
                    train_y=train_y,
                    train_domains=train_domains,
                    class_ids=class_ids,
                    class_weight=weight,
                    args=args,
                    device=device,
                )
            else:
                batch_x, batch_aux, batch_mask, batch_y, batch_domain = batch
                batch_x = batch_x.to(device)
                batch_aux = batch_aux.to(device)
                batch_mask = batch_mask.to(device)
                batch_y = batch_y.to(device)
                batch_domain = batch_domain.to(device)
                out = model(batch_x, batch_aux, batch_mask, class_ids, batch_domain)
                loss_cls = F.cross_entropy(
                    out["logits"],
                    batch_y,
                    weight=weight,
                    label_smoothing=float(args.label_smoothing),
                )
                loss_anchor = anchor_alignment_loss(out["base_memory"], out["anchors"])
                loss_residual = residual_norm_loss(out["memory_residual"])
                loss_region_cls = out["logits"].new_tensor(0.0)
                if out.get("region_logits") is not None and out["region_logits"].numel() > 0:
                    loss_region_cls = F.cross_entropy(out["region_logits"], batch_domain)
                elif region_head is not None:
                    loss_region_cls = F.cross_entropy(region_head(out["acoustic_condition"]), batch_domain)
                loss_region_compact = region_compactness_loss(out["acoustic_condition"], batch_domain)
                loss_static = out["logits"].new_tensor(0.0)
                static_logits = out["logits"]
            loss = (
                loss_cls
                + float(args.lambda_recurrent_static) * loss_static
                + float(args.lambda_anchor) * loss_anchor
                + float(args.lambda_residual) * loss_residual
                + float(args.lambda_region_cls) * loss_region_cls
                + float(args.lambda_region_compact) * loss_region_compact
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = 0.0
            for param in params:
                if param.grad is not None:
                    grad_norm += float(param.grad.detach().norm(2).cpu()) ** 2
            grad_norm = grad_norm ** 0.5
            grad_audit = module_grad_audit(model, region_head)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                train_acc = float((out["logits"].argmax(dim=1) == batch_y).float().mean().detach().cpu())
                batch_domain_counts = torch.bincount(batch_domain, minlength=num_domains).float()
                batch_label_counts = torch.bincount(batch_y, minlength=len(VOWEL_ORDER)).float()
                active_domains = batch_domain_counts[batch_domain_counts > 0]
                active_labels = batch_label_counts[batch_label_counts > 0]
            activation_audit = forward_activation_audit(out)
            row = {
                "step": float(step),
                "epoch": float(epoch),
                "loss_total": float(loss.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_recurrent_static": float(loss_static.detach().cpu()),
                "loss_anchor": float(loss_anchor.detach().cpu()),
                "loss_residual": float(loss_residual.detach().cpu()),
                "loss_region_cls": float(loss_region_cls.detach().cpu()),
                "loss_region_compact": float(loss_region_compact.detach().cpu()),
                "train_batch_acc": train_acc,
                "train_batch_static_acc": float((static_logits.argmax(dim=1) == batch_y).float().mean().detach().cpu()),
                "grad_norm": float(grad_norm),
                "domain_batch_unique": float(batch_domain.detach().unique().numel()),
                "domain_batch_min_count": float(active_domains.min().detach().cpu()) if active_domains.numel() else 0.0,
                "domain_batch_max_count": float(active_domains.max().detach().cpu()) if active_domains.numel() else 0.0,
                "label_batch_unique": float(batch_y.detach().unique().numel()),
                "label_batch_min_count": float(active_labels.min().detach().cpu()) if active_labels.numel() else 0.0,
                "label_batch_max_count": float(active_labels.max().detach().cpu()) if active_labels.numel() else 0.0,
                "sampler_is_balanced": float(args.sampler != "random"),
                "lambda_anchor": float(args.lambda_anchor),
                "lambda_residual": float(args.lambda_residual),
                "lambda_recurrent_static": float(args.lambda_recurrent_static),
                "lambda_region_cls": float(args.lambda_region_cls),
                "lambda_region_compact": float(args.lambda_region_compact),
            }
            row.update(grad_audit)
            row.update(activation_audit)
            if step == 1 or step % args.eval_every == 0 or step >= args.max_steps:
                val_metrics, _pred, _prob, val_diag = evaluate(
                    model,
                    val_x,
                    val_aux,
                    val_y,
                    args.eval_batch_size,
                    device,
                    episode_batches=val_episode_batches,
                    recurrent_eval_mode=args.recurrent_eval_mode,
                    recurrent_online_threshold=args.recurrent_online_threshold,
                )
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                row.update({f"val_diag_{key}": value for key, value in val_diag.items()})
                if val_metrics["macro_f1"] > best_macro:
                    best_macro = float(val_metrics["macro_f1"])
                    best_state = deepcopy(model.state_dict())
                    best_region_state = deepcopy(region_head.state_dict()) if region_head is not None else None
                print(
                    f"step={step} loss={row['loss_total']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                    f"g_mem={row.get('grad_class_memory', 0.0):.3e} "
                    f"g_feat={row.get('grad_feature_fusion', 0.0):.3e} "
                    f"g_prompt={row.get('grad_prompt', 0.0):.3e} "
                    f"g_reg={row.get('grad_region_memory', 0.0):.3e} "
                    f"feat_norm={row.get('feature_bias_norm', 0.0):.3f}",
                    flush=True,
                )
            elif step % args.log_every == 0 or step % max(1, int(args.audit_every)) == 0:
                print(
                    f"step={step} loss={row['loss_total']:.4f} train_acc={row['train_batch_acc']:.4f} "
                    f"g_mem={row.get('grad_class_memory', 0.0):.3e} "
                    f"g_feat={row.get('grad_feature_fusion', 0.0):.3e} "
                    f"g_prompt={row.get('grad_prompt', 0.0):.3e} "
                    f"g_reg={row.get('grad_region_memory', 0.0):.3e} "
                    f"domains={row.get('domain_batch_unique', 0.0):.0f}",
                    flush=True,
                )
            curve.append(row)
            if step >= args.max_steps:
                break
    model.load_state_dict(best_state)
    if region_head is not None and best_region_state is not None:
        region_head.load_state_dict(best_region_state)
    memory_update_norm = 0.0
    if initial_memory is not None and hasattr(model, "class_memory"):
        memory_update_norm = float((model.class_memory.detach().cpu() - initial_memory).norm(dim=-1).mean())
    summary = {
        "best_val_macro_f1": float(best_macro),
        "training_time_sec": float(time.perf_counter() - start_time),
        "region_head_enabled": bool(region_head is not None),
        "class_memory_update_norm": float(memory_update_norm),
        "sampler": sampler_meta,
        "trace_summary": train_trace_summary(curve),
    }
    curve.append({"step": float(step), "epoch": float(epoch), **summary})
    return model, curve, summary


def write_curve(curve: list[dict[str, float]], path: Path) -> None:
    keys = sorted({key for row in curve for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(curve)


def summarize_confusions(cm: np.ndarray, labels: Sequence[str], top_n: int = 10) -> list[dict[str, Any]]:
    pairs = []
    for true_idx, true_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                pairs.append({"true_label": str(true_label), "pred_label": str(pred_label), "count": count})
    pairs.sort(key=lambda item: int(item["count"]), reverse=True)
    return pairs[:top_n]


def prediction_frame(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base_cols = [col for col in ["sample_id", args.label_column, args.region_column, "site", args.speaker_column] if col in rows.columns]
    out = rows[base_cols].reset_index(drop=True).copy()
    out["true_label_id"] = y_true
    out["pred_label_id"] = y_pred
    out["pred_vowel"] = [VOWEL_ORDER[idx] for idx in y_pred]
    out["pred_confidence"] = prob.max(axis=1)
    for idx, label in enumerate(VOWEL_ORDER):
        out[f"prob_{label}"] = prob[:, idx]
    return out


def serializable_token_norm(norm: dict[str, np.ndarray | str]) -> dict[str, Any]:
    out: dict[str, Any] = {"mode": str(norm["mode"])}
    if "mean" in norm:
        out["mean_shape"] = list(np.asarray(norm["mean"]).shape)
        out["std_shape"] = list(np.asarray(norm["std"]).shape)
    return out


def infer_aux_branch_dims(aux_meta: dict[str, Any], aux_array: np.ndarray) -> list[int]:
    branch_dims = aux_meta.get("branch_dims") or []
    if branch_dims and sum(int(v) for v in branch_dims) == int(aux_array.shape[-1]):
        return [int(v) for v in branch_dims]
    return [int(aux_array.shape[-1])]


def load_model_init_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path:
        return {"enabled": False}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state dict: {path}")
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "enabled": True,
        "path": str(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def main() -> None:
    configure_stdout()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output.with_suffix(".predictions.csv")
    curve_output = Path(args.curve_output) if args.curve_output else output.with_suffix(".curve.csv")
    debug_output = Path(args.debug_output) if args.debug_output else output.with_suffix(".debug.json")
    audit_output = Path(args.audit_output) if args.audit_output else output.with_suffix(".audit.json")
    checkpoint_output = Path(args.checkpoint_output) if args.checkpoint_output else output.with_suffix(".pt")

    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    split_audit = build_split_audit(df, train_df, val_df, test_df, args, split_info)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "split": split_info,
                "split_audit": {
                    "holdout_region": split_audit["holdout_region"],
                    "num_all_regions": split_audit["num_all_regions"],
                    "num_source_pool_regions": split_audit["num_source_pool_regions"],
                    "num_train_regions": split_audit["num_train_regions"],
                    "target_rows_in_train": split_audit["target_rows_in_train"],
                    "target_rows_in_val": split_audit["target_rows_in_val"],
                    "test_is_holdout_only": split_audit["test_is_holdout_only"],
                    "train_regions_cover_all_non_target_regions": split_audit["train_regions_cover_all_non_target_regions"],
                },
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "variant": args.variant,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = None
    matrix_meta: dict[str, Any] = {}
    try:
        train_x_raw, matrix_meta["train"] = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, matrix_meta["val"] = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, matrix_meta["test"] = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")
    except (AttributeError, FileNotFoundError) as exc:
        print(f"[matrix-cache] miss needs Whisper extraction: {exc}", flush=True)
        import whisper

        whisper_model = whisper.load_model(args.whisper_model, device=str(device))
        whisper_model.eval()
        for param in whisper_model.parameters():
            param.requires_grad = False
        train_x_raw, matrix_meta["train"] = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, matrix_meta["val"] = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, matrix_meta["test"] = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")

    token_norm = fit_token_norm(train_x_raw, args.feature_norm)
    train_x = apply_token_norm(train_x_raw, token_norm)
    val_x = apply_token_norm(val_x_raw, token_norm)
    test_x = apply_token_norm(test_x_raw, token_norm)

    train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, args, train_x_raw, "train")
    val_aux_raw, val_aux_meta = extract_aux_matrix_cached(val_df, args, val_x_raw, "val")
    test_aux_raw, test_aux_meta = extract_aux_matrix_cached(test_df, args, test_x_raw, "test")
    aux_norm = fit_vector_norm(train_aux_raw, "global")
    train_aux = apply_vector_norm(train_aux_raw, aux_norm)
    val_aux = apply_vector_norm(val_aux_raw, aux_norm)
    test_aux = apply_vector_norm(test_aux_raw, aux_norm)

    train_y = labels_to_ids(train_df[args.label_column])
    val_y = labels_to_ids(val_df[args.label_column])
    test_y = labels_to_ids(test_df[args.label_column])
    domain_values = sorted(train_df[args.region_column].astype(str).unique().tolist())
    domain_to_id = {region: idx for idx, region in enumerate(domain_values)}
    train_domains = train_df[args.region_column].astype(str).map(domain_to_id).to_numpy(dtype=np.int64)
    train_episode_batches, train_episode_meta = build_episode_batches(train_df, args)
    val_episode_batches, val_episode_meta = build_episode_batches(val_df, args)
    test_episode_batches, test_episode_meta = build_episode_batches(test_df, args)

    model = build_model_from_variant(
        variant=args.variant,
        input_dim=train_x.shape[-1],
        aux_dim=train_aux.shape[-1],
        aux_branch_dims=infer_aux_branch_dims(train_aux_meta, train_aux),
        num_domains=len(domain_values),
        hidden_dim=args.hidden_dim,
        score_dim=args.score_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_audio_tokens=train_x.shape[1],
        anchor_kind=args.anchor_kind,
        projection_seed=args.projection_seed,
    )
    init_checkpoint_info = load_model_init_checkpoint(model, args.init_checkpoint, device)
    model, curve, train_summary = train_model(
        model,
        train_x,
        train_aux,
        train_y,
        train_domains,
        val_x,
        val_aux,
        val_y,
        args,
        device,
        num_domains=len(domain_values),
        train_episode_batches=train_episode_batches,
        val_episode_batches=val_episode_batches,
    )
    write_curve(curve, curve_output)

    val_metrics, val_pred, _val_prob, val_diag = evaluate(
        model,
        val_x,
        val_aux,
        val_y,
        args.eval_batch_size,
        device,
        episode_batches=val_episode_batches,
        recurrent_eval_mode=args.recurrent_eval_mode,
        recurrent_online_threshold=args.recurrent_online_threshold,
    )
    test_metrics, test_pred, test_prob, test_diag = evaluate(
        model,
        test_x,
        test_aux,
        test_y,
        args.eval_batch_size,
        device,
        episode_batches=test_episode_batches,
        recurrent_eval_mode=args.recurrent_eval_mode,
        recurrent_online_threshold=args.recurrent_online_threshold,
    )
    labels = list(range(len(VOWEL_ORDER)))
    report = classification_report(test_y, test_pred, labels=labels, target_names=VOWEL_ORDER, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_y, test_pred, labels=labels)
    pred_df = prediction_frame(test_df, test_y, test_pred, test_prob, args)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    class_ids = torch.arange(len(VOWEL_ORDER), dtype=torch.long, device=device)
    model_diag = {}
    if hasattr(model, "base_memory"):
        with torch.no_grad():
            base_memory, anchors = model.base_memory(class_ids)
            model_diag.update(memory_diagnostics(base_memory, anchors))
    recurrent_enabled = is_recurrent_model(model)
    feature_variants = {"feature_memory_no_prompt", "feature_memory_prompt", "region_memory_prompt", "recurrent_memory_prompt", "recurrent_region_memory_prompt"}
    prompt_variants = {"feature_memory_prompt", "region_memory_prompt", "recurrent_memory_prompt", "recurrent_region_memory_prompt"}
    region_variants = {"region_memory_prompt", "recurrent_region_memory_prompt"}
    recurrent_variants = {"recurrent_memory_prompt", "recurrent_region_memory_prompt"}
    algorithm_flow = [
        "freeze/reuse Whisper encoder tokens as speech stream",
        "extract and align Log-Mel, MFCC, and MFCC delta/delta2 acoustic sequences when wav files are available",
        "adaptively fuse acoustic feature branches into one token-aligned feature sequence",
        "form H_i^0 by adding projected speech tokens and fused acoustic feature tokens",
        "for region-memory variants, infer a source-region acoustic memory token from auxiliary acoustic cues",
        "prepend class-memory tokens to H_i^0",
        "prepend the inferred region-memory token before class tokens without modifying M^0 before the Transformer",
        "apply one Transformer where learned prompts modulate Q/K/V attention gates and are not input tokens",
        "for region-memory variants, condition prompt gates on the inferred region-memory context",
        "score each final class-memory token with a shared scorer",
    ]
    if recurrent_enabled:
        algorithm_flow.extend(
            [
                "initialize each episode with the learned global class memory M_global",
                "read M_{t-1} together with current speech tokens before prediction",
                "after prediction, perform gated class-selective write to produce M_t",
                "carry M_t to the next speech segment and reset only at episode boundaries",
            ]
        )
    result = {
        "task": "leave_one_region_out_vowel_classification",
        "method": "PromptGuidedRecurrentClassMemoryTransformer" if recurrent_enabled else "FeatureConditionedPromptMemoryTransformer",
        "algorithm_flow": algorithm_flow,
        "training_objective": {
            "main": "source-domain vowel cross entropy over all non-target regions in LORO",
            "sampler": str(args.sampler),
            "shared_memory_update": "all source-region samples backpropagate to the same learnable M^0",
            "recurrent_memory_update": "source labels are used only after prediction for class-selective write within source episodes" if recurrent_enabled else "",
            "recurrent_eval_mode": str(args.recurrent_eval_mode) if recurrent_enabled else "",
            "recurrent_static_regularization": float(args.lambda_recurrent_static) if recurrent_enabled else 0.0,
            "region_memory_update": "source-region IDs supervise acoustic-to-region-memory matching when lambda_region_cls > 0",
            "optional_anchor_regularization": float(args.lambda_anchor),
            "optional_region_auxiliary": float(args.lambda_region_cls),
            "optional_region_compactness": float(args.lambda_region_compact),
            "target_labels_used_for_training": False,
        },
        "method_audit": {
            "class_memory_inside_transformer": args.variant != "speech_baseline",
            "class_memory_pre_transformer_is_shared": args.variant != "speech_baseline",
            "feature_stream_enabled": args.variant in feature_variants,
            "prompt_attention_gate_enabled": args.variant in prompt_variants,
            "region_memory_enabled": args.variant in region_variants,
            "region_memory_token_inside_transformer": args.variant in region_variants,
            "region_conditioned_prompt_enabled": args.variant in region_variants,
            "region_gated_feature_residual_enabled": args.variant in region_variants,
            "recurrent_memory_enabled": args.variant in recurrent_variants,
            "memory_state_persists_within_episode": recurrent_enabled,
            "gated_memory_write_enabled": recurrent_enabled,
            "class_selective_write_enabled": recurrent_enabled,
            "predict_before_write": recurrent_enabled,
            "online_test_time_memory_enabled": recurrent_enabled and args.recurrent_eval_mode == "online",
            "region_supervision_enabled": float(args.lambda_region_cls) > 0.0,
            "region_compactness_enabled": float(args.lambda_region_compact) > 0.0,
            "domain_balanced_sampling_enabled": args.sampler in {"domain_balanced", "domain_class_balanced"},
            "class_balanced_sampling_enabled": args.sampler == "domain_class_balanced",
            "curve_contains_module_gradients": True,
            "audit_file": str(audit_output),
        },
        "variant": args.variant,
        "anchor_kind": args.anchor_kind,
        "protocol": args.protocol,
        "holdout_region": args.holdout_region,
        "seed": int(args.seed),
        "train_fraction": float(args.train_fraction),
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_confusions": summarize_confusions(cm, VOWEL_ORDER, top_n=10),
        "diagnostics": {
            "val": val_diag,
            "test": test_diag,
            "memory": model_diag,
            "recurrent_episode": {
                "train": train_episode_meta,
                "val": val_episode_meta,
                "test": test_episode_meta,
            },
        },
        "split": split_info,
        "split_audit": split_audit,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "label_counts": {
            "train": label_counts(train_df[args.label_column], VOWEL_ORDER),
            "val": label_counts(val_df[args.label_column], VOWEL_ORDER),
            "test": label_counts(test_df[args.label_column], VOWEL_ORDER),
        },
        "cache": {
            "matrix": matrix_meta,
            "aux": {"train": train_aux_meta, "val": val_aux_meta, "test": test_aux_meta},
            "token_norm": serializable_token_norm(token_norm),
            "aux_norm": serializable_norm(aux_norm),
            "aux_rows_hash": {
                "train": aux_rows_hash(train_df, args),
                "val": aux_rows_hash(val_df, args),
                "test": aux_rows_hash(test_df, args),
            },
        },
        "config": vars(args),
        "training": train_summary,
        "init_checkpoint": init_checkpoint_info,
        "files": {
            "output": str(output),
            "curve": str(curve_output),
            "predictions": str(predictions_output),
            "debug": str(debug_output),
            "audit": str(audit_output),
            "checkpoint": str(checkpoint_output) if args.save_checkpoint else "",
        },
        "parameters": {
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total": int(sum(p.numel() for p in model.parameters())),
        },
        "split_hash": {
            "train": dataframe_hash(train_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "val": dataframe_hash(val_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "test": dataframe_hash(test_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_payload = {
        "result_file": str(output),
        "variant": args.variant,
        "metrics": test_metrics,
        "diagnostics": result["diagnostics"],
        "cache": result["cache"],
        "split": result["split"],
        "split_audit": result["split_audit"],
        "method_audit": result["method_audit"],
        "training": result["training"],
        "command": " ".join(sys.argv),
        "init_checkpoint": init_checkpoint_info,
    }
    debug_output.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "token_norm": token_norm,
                "aux_norm": aux_norm,
                "labels": VOWEL_ORDER,
            },
            checkpoint_output,
        )
    print(json.dumps({"output": str(output), "metrics": test_metrics, "val_metrics": val_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
