#!/usr/bin/env python3
"""Run external SSL speech encoder baselines under LORO evaluation.

The script extracts frozen representations from HuggingFace speech encoders
(wav2vec 2.0, HuBERT, WavLM, data2vec-audio, etc.), caches the segment-level
features, and evaluates simple downstream heads using the same splits as
PC-DLCMNet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from pc_dlcmnet.data.acoustic_features import audio_path_candidates
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.data.episodes import split_global_support_query
from tools.experiments.baseline_comparisons import (
    class_centroid_predict,
    metric_dict,
    prototype_split_metrics,
    safe_name,
)
from pc_dlcmnet.training.supervised import (
    build_split_audit,
    configure_stdout,
    default_old_root,
    label_counts,
    labels_to_ids,
    load_dataset,
    normalize_region_value,
    resolve_device,
    set_seed,
    split_data,
)
from pc_dlcmnet.utils.paths import default_audio_root, default_project_root


MODEL_PRESETS: dict[str, str] = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-large": "facebook/wav2vec2-large",
    "hubert-base": "facebook/hubert-base-ls960",
    "hubert-large": "facebook/hubert-large-ls960-ft",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    "data2vec-audio-base": "facebook/data2vec-audio-base-960h",
    "data2vec-audio-large": "facebook/data2vec-audio-large-960h",
    "mhubert-147": "utter-project/mHuBERT-147",
    "mhubert-147-base-3rd-iter": "utter-project/mHuBERT-147-base-3rd-iter",
}


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--output-dir", default="output/0614/ssl_pretrained_baselines")
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--protocol", choices=["loro"], default="loro")
    parser.add_argument("--holdout-regions", nargs="*", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--models", nargs="*", default=["wav2vec2-base", "hubert-base", "wavlm-base-plus", "data2vec-audio-base"])
    parser.add_argument("--methods", nargs="*", default=["ridge", "svm", "centroid", "target-proto"])
    parser.add_argument("--cache-dir", default="output/0614/ssl_feature_cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--prototype-splits", type=int, default=20)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-all-source-regions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_audio_path(row: pd.Series, args: argparse.Namespace) -> str:
    for candidate in audio_path_candidates(row[args.path_column], args):
        if Path(candidate).exists():
            return str(candidate)
    raw = str(row[args.path_column])
    marker = "/home/ustc1958/lxy/graph/tone/tone/"
    if raw.startswith(marker):
        relative = raw[len(marker) :].replace("/", "\\")
        for root in [default_audio_root(), default_project_root()]:
            candidate = root / relative
            if candidate.exists():
                return str(candidate)
    raise FileNotFoundError(f"Audio not found: {row[args.path_column]}")


def load_segment(row: pd.Series, args: argparse.Namespace) -> np.ndarray:
    path = resolve_audio_path(row, args)
    raw_file = Path(path).read_bytes()
    sr, channels, sampwidth, data_offset, data_size = parse_pcm_wav_header(raw_file, path)
    start = max(0, int(round(float(row[args.start_column]) * sr)))
    end = max(start + 1, int(round(float(row[args.end_column]) * sr)))
    frame_size = channels * sampwidth
    total_frames = data_size // frame_size
    start = min(start, total_frames)
    end = min(max(start + 1, end), total_frames)
    byte_start = data_offset + start * frame_size
    byte_end = data_offset + end * frame_size
    raw = raw_file[byte_start:byte_end]
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sampwidth}: {path}")
    if channels > 1 and audio.size:
        audio = audio.reshape(-1, channels).mean(axis=1)
    segment = audio.astype(np.float32)
    if sr != int(args.target_sr):
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(sr, int(args.target_sr))
        segment = resample_poly(segment, int(args.target_sr) // divisor, sr // divisor).astype(np.float32)
    if segment.size == 0:
        segment = np.zeros(int(args.target_sr * 0.05), dtype=np.float32)
    return segment


def parse_pcm_wav_header(raw: bytes, path: str) -> tuple[int, int, int, int, int]:
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")
    pos = 12
    sample_rate = channels = sample_width = 0
    data_offset = data_size = 0
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little", signed=False)
        payload = pos + 8
        if chunk_id == b"fmt ":
            audio_format = int.from_bytes(raw[payload : payload + 2], "little", signed=False)
            channels = int.from_bytes(raw[payload + 2 : payload + 4], "little", signed=False)
            sample_rate = int.from_bytes(raw[payload + 4 : payload + 8], "little", signed=False)
            bits_per_sample = int.from_bytes(raw[payload + 14 : payload + 16], "little", signed=False)
            if audio_format != 1:
                raise ValueError(f"Only PCM WAV is supported, got format {audio_format}: {path}")
            sample_width = bits_per_sample // 8
        elif chunk_id == b"data":
            data_offset = payload
            data_size = chunk_size
            break
        pos = payload + chunk_size + (chunk_size % 2)
    if not (sample_rate and channels and sample_width and data_offset and data_size):
        raise ValueError(f"Incomplete WAV header: {path}")
    return sample_rate, channels, sample_width, data_offset, data_size


def rows_hash(rows: pd.DataFrame, args: argparse.Namespace) -> str:
    cols = [c for c in ["sample_id", args.path_column, args.start_column, args.end_column, args.label_column] if c in rows.columns]
    payload = rows[cols].to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_id(name_or_id: str) -> str:
    return MODEL_PRESETS.get(name_or_id, name_or_id)


def cache_path(rows: pd.DataFrame, args: argparse.Namespace, model_name: str, split_name: str) -> Path:
    payload = {
        "rows": rows_hash(rows, args),
        "model": model_id(model_name),
        "target_sr": int(args.target_sr),
        "pooling": "mean_std_last_hidden",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(args.cache_dir) / safe_name(model_name) / f"{split_name}_{key}.npy"


def load_hf_model(model_name: str, args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    patch_huggingface_hub_download()
    from transformers import AutoFeatureExtractor, AutoModel

    resolved = model_id(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(
        resolved,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModel.from_pretrained(
        resolved,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return extractor, model


def patch_huggingface_hub_download() -> None:
    """Make older Transformers calls compatible with newer huggingface_hub.

    Transformers 4.28 still passes ``use_auth_token`` to ``hf_hub_download``.
    Recent huggingface_hub versions removed that keyword. Dropping it is safe for
    public model downloads and local-file loading.
    """
    try:
        import huggingface_hub
        import huggingface_hub.file_download as file_download
    except Exception:
        return
    if getattr(huggingface_hub.hf_hub_download, "_pcdlcmnet_patched", False):
        return
    original = huggingface_hub.hf_hub_download

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    wrapped._pcdlcmnet_patched = True  # type: ignore[attr-defined]
    huggingface_hub.hf_hub_download = wrapped  # type: ignore[assignment]
    file_download.hf_hub_download = wrapped  # type: ignore[assignment]


def extract_ssl_features(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    model_name: str,
    split_name: str,
    extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = cache_path(rows, args, model_name, split_name)
    meta_path = path.with_suffix(".json")
    if path.exists():
        return np.load(path).astype(np.float32), {"source": "cache", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    iterator = range(0, len(rows), int(args.batch_size))
    if not args.quiet:
        iterator = tqdm(iterator, desc=f"{model_name} {split_name}", unit="batch")
    frame = rows.reset_index(drop=True)
    with torch.no_grad():
        for start in iterator:
            batch_rows = frame.iloc[start : start + int(args.batch_size)]
            audio = [load_segment(row, args) for _, row in batch_rows.iterrows()]
            inputs = extractor(audio, sampling_rate=int(args.target_sr), return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if getattr(model.config, "model_type", "") == "whisper" and "input_features" in inputs:
                out = model.encoder(input_features=inputs["input_features"])
                hidden = getattr(out, "last_hidden_state", None)
            else:
                out = model(**inputs)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = getattr(out, "encoder_last_hidden_state", None)
            if hidden is None:
                raise AttributeError("Model output has neither last_hidden_state nor encoder_last_hidden_state")
            if "attention_mask" in inputs:
                # Convert sample-level mask to frame-level approximation when possible.
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
            else:
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean = (hidden * mask_f).sum(dim=1) / denom
            var = (((hidden - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / denom
            pooled = torch.cat([mean, var.clamp_min(0).sqrt()], dim=-1)
            features.append(pooled.detach().cpu().numpy().astype(np.float32))
    x = np.concatenate(features, axis=0).astype(np.float32)
    np.save(path, x)
    meta = {
        "rows": int(len(rows)),
        "shape": list(x.shape),
        "model": model_id(model_name),
        "model_alias": model_name,
        "row_hash": rows_hash(rows, args),
        "target_sr": int(args.target_sr),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return x, {"source": "computed", "path": str(path), "meta": str(meta_path)}


def fit_head(method: str, train_x: np.ndarray, train_y: np.ndarray, seed: int) -> Any:
    if method == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0, class_weight="balanced")).fit(train_x, train_y)
    if method == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=0.5, class_weight="balanced", max_iter=8000, random_state=seed)).fit(train_x, train_y)
    raise ValueError(f"Unsupported fit method: {method}")


def evaluate_one_method(method: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "centroid":
        pred = class_centroid_predict(train_x, train_y, test_x)
        return metric_dict(test_y, pred), test_y, pred, {}
    if method == "target-proto":
        metrics, true, pred, extra = prototype_split_metrics(
            test_x,
            test_y,
            k=int(args.prototype_k),
            splits=int(args.prototype_splits),
            seed=int(args.seed),
        )
        return metrics, true, pred, extra
    clf = fit_head(method, train_x, train_y, int(args.seed))
    pred = clf.predict(test_x).astype(np.int64)
    return metric_dict(test_y, pred), test_y, pred, {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_model(args: argparse.Namespace, full_df: pd.DataFrame, regions: list[str], model_name: str, device: torch.device) -> list[dict[str, Any]]:
    extractor, model = load_hf_model(model_name, args, device)
    rows_out = []
    for region in regions:
        region_args = argparse.Namespace(**vars(args))
        region_args.holdout_region = normalize_region_value(str(region), full_df[args.region_column])
        train_df, _val_df, test_df, split_info = split_data(full_df, region_args)
        audit = build_split_audit(full_df, train_df, _val_df, test_df, region_args, split_info)
        train_x, train_meta = extract_ssl_features(train_df, args, model_name, "train", extractor, model, device)
        test_x, test_meta = extract_ssl_features(test_df, args, model_name, "test", extractor, model, device)
        train_y = labels_to_ids(train_df[args.label_column])
        test_y = labels_to_ids(test_df[args.label_column])
        for method in args.methods:
            started = time.perf_counter()
            status = "done"
            error = ""
            try:
                metrics, y_true, pred, extra = evaluate_one_method(method, train_x, train_y, test_x, test_y, args)
            except Exception as exc:
                status = "failed"
                error = repr(exc)
                metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan")}
                y_true = np.zeros(0, dtype=np.int64)
                pred = np.zeros(0, dtype=np.int64)
                extra = {}
            row = {
                "status": status,
                "model": model_name,
                "model_id": model_id(model_name),
                "method": method,
                "region": region,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "accuracy_std": metrics.get("accuracy_std", 0.0),
                "macro_f1_std": metrics.get("macro_f1_std", 0.0),
                "weighted_f1_std": metrics.get("weighted_f1_std", 0.0),
                "evaluated_rows": int(len(y_true)),
                "train_rows": int(len(train_y)),
                "test_rows": int(len(test_y)),
                "time_sec": float(time.perf_counter() - started),
                "error": error,
                **extra,
            }
            rows_out.append(row)
            result_path = Path(args.output_dir) / "results" / safe_name(model_name) / f"{safe_name(region)}_{method}.json"
            payload = {
                "row": row,
                "metrics": metrics,
                "split": split_info,
                "split_audit": audit,
                "train_label_counts": label_counts(train_df[args.label_column], VOWEL_ORDER),
                "test_label_counts": label_counts(test_df[args.label_column], VOWEL_ORDER),
                "feature_cache": {"train": train_meta, "test": test_meta},
            }
            if len(y_true) and len(pred):
                payload["report"] = classification_report(
                    y_true,
                    pred,
                    labels=list(range(len(VOWEL_ORDER))),
                    target_names=list(VOWEL_ORDER),
                    output_dict=True,
                    zero_division=0,
                )
                payload["confusion_matrix"] = confusion_matrix(y_true, pred, labels=list(range(len(VOWEL_ORDER)))).tolist()
            write_json(result_path, payload)
            if not args.quiet:
                print(
                    f"{model_name} {method} {region}: mf1={100.0 * float(row['macro_f1']):.2f} "
                    f"acc={100.0 * float(row['accuracy']):.2f} status={status}",
                    flush=True,
                )
    return rows_out


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    done = pd.DataFrame([row for row in rows if row.get("status") == "done"])
    aggregate = []
    if not done.empty:
        for (model_name, method), group in done.groupby(["model", "method"], sort=True):
            aggregate.append(
                {
                    "model": model_name,
                    "model_id": str(group["model_id"].iloc[0]),
                    "method": method,
                    "regions": int(group["region"].nunique()),
                    "accuracy": float(group["accuracy"].mean()),
                    "accuracy_std_region": float(group["accuracy"].std(ddof=1)),
                    "macro_f1": float(group["macro_f1"].mean()),
                    "macro_f1_std_region": float(group["macro_f1"].std(ddof=1)),
                    "weighted_f1": float(group["weighted_f1"].mean()),
                    "weighted_f1_std_region": float(group["weighted_f1"].std(ddof=1)),
                    "mean_time_sec": float(group["time_sec"].mean()),
                }
            )
    with (output_dir / "aggregate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "model",
            "model_id",
            "method",
            "regions",
            "accuracy",
            "accuracy_std_region",
            "macro_f1",
            "macro_f1_std_region",
            "weighted_f1",
            "weighted_f1_std_region",
            "mean_time_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)
    write_json(output_dir / "summary.json", {"rows": rows, "aggregate": aggregate})


def main() -> None:
    configure_stdout()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = load_dataset(args)
    regions = args.holdout_regions or sorted(full_df[args.region_column].astype(str).unique().tolist())
    if int(args.max_regions) > 0:
        regions = regions[: int(args.max_regions)]
    all_rows: list[dict[str, Any]] = []
    for name in args.models:
        all_rows.extend(run_model(args, full_df, regions, name, device))
        write_summary(output_dir, all_rows)
    write_summary(output_dir, all_rows)


if __name__ == "__main__":
    main()
