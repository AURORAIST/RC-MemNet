#!/usr/bin/env python3
"""Run LORO baseline comparisons for Wu vowel recognition.

The script reuses the same dataset split and cached Whisper/acoustic features as
PC-DLCMNet. It intentionally keeps the baselines simple and reproducible:

- source-supervised heads: train on source regions and test on the held-out region;
- target-prototype heads: sample K target support examples per class and classify
  the remaining target query examples by nearest prototype.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from pc_dlcmnet.data.acoustic_features import (
    apply_vector_norm,
    extract_aux_matrix_cached,
    fit_vector_norm,
)
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    build_split_audit,
    configure_stdout,
    default_old_root,
    extract_matrix_cached,
    fit_token_norm,
    label_counts,
    labels_to_ids,
    load_dataset,
    resolve_device,
    set_seed,
    split_data,
)


def load_whisper_encoder(model_spec: str, device: torch.device):
    import whisper

    try:
        return whisper.load_model(model_spec, device=str(device))
    except Exception:
        checkpoint = torch.load(model_spec, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise
        state_dict = checkpoint["model_state_dict"]
        encoder_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("whisper_model.encoder."):
                encoder_state[key.removeprefix("whisper_model.")] = value
            elif key.startswith("encoder."):
                encoder_state[key] = value
        if not encoder_state:
            raise RuntimeError(f"Unable to locate Whisper encoder weights in checkpoint: {model_spec}")
        model_size = "base"
        args = checkpoint.get("args")
        if isinstance(args, dict):
            model_size = str(args.get("model_size", model_size))
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_size = str(config.get("model_size", model_size))
        if "dims" in checkpoint and isinstance(checkpoint["dims"], dict):
            try:
                whisper_model = whisper.Whisper(whisper.ModelDimensions(**checkpoint["dims"]))
            except Exception:
                whisper_model = whisper.load_model(model_size, device=str(device))
        else:
            whisper_model = whisper.load_model(model_size, device=str(device))
        whisper_model.encoder.load_state_dict(encoder_state, strict=False)
        return whisper_model.to(device)


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--output-dir", default="output/0614/baseline_comparisons")
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
    parser.add_argument("--cache-dir", default=str(old_root / "output/salmonn_style_whisper_cache_base"))
    parser.add_argument("--matrix-cache-dir", default=str(old_root / "output/ppm_supervised_matrix_cache"))
    parser.add_argument("--feature-norm", choices=["none", "global"], default="global")
    parser.add_argument("--matrix-cache-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--token-chunks", type=int, default=32)
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--aux-source", choices=["auto", "acoustic", "whisper_stats", "old_fusion_cache"], default="acoustic")
    parser.add_argument("--aux-representation", choices=["auto", "sequence", "stats"], default="sequence")
    parser.add_argument("--aux-cache-dir", default="output/0614/feature_memory_aux_cache")
    parser.add_argument("--old-feature-cache-dir", default=str(old_root / "output/feature_cache_vowel"))
    parser.add_argument("--aux-n-mfcc", type=int, default=39)
    parser.add_argument("--aux-n-mels", type=int, default=128)
    parser.add_argument("--aux-f0-segments", type=int, default=5)
    parser.add_argument("--aux-use-delta-mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-use-formants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "majority",
            "whisper-centroid",
            "whisper-ridge",
            "whisper-logreg",
            "whisper-svm",
            "acoustic-centroid",
            "acoustic-ridge",
            "acoustic-svm",
            "fused-ridge",
            "target-proto-whisper",
            "target-proto-acoustic",
            "target-proto-fused",
        ],
    )
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--prototype-splits", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-all-source-regions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def pool_mean_std(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return np.concatenate([mean, std], axis=1).astype(np.float32)


def pool_mean(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, 1e-8)).astype(np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def class_centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x = l2_normalize(train_x)
    test_x = l2_normalize(test_x)
    centroids = np.zeros((len(VOWEL_ORDER), train_x.shape[1]), dtype=np.float32)
    for class_id in range(len(VOWEL_ORDER)):
        mask = train_y == class_id
        if mask.any():
            centroids[class_id] = train_x[mask].mean(axis=0)
    centroids = l2_normalize(centroids)
    return np.matmul(test_x, centroids.T).argmax(axis=1).astype(np.int64)


def majority_predict(train_y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(train_y, minlength=len(VOWEL_ORDER))
    return np.full(n, int(counts.argmax()), dtype=np.int64)


def train_source_head(method: str, train_x: np.ndarray, train_y: np.ndarray, seed: int, rf_trees: int) -> Any:
    if method.endswith("logreg"):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1200,
                C=1.0,
                class_weight="balanced",
                solver="saga",
                n_jobs=1,
                random_state=seed,
            ),
        ).fit(train_x, train_y)
    if method.endswith("ridge"):
        return make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
        ).fit(train_x, train_y)
    if method.endswith("svm"):
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight="balanced", max_iter=8000, random_state=seed),
        ).fit(train_x, train_y)
    if method.endswith("rbf-svm"):
        return make_pipeline(
            StandardScaler(),
            SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=seed),
        ).fit(train_x, train_y)
    if method.endswith("rf"):
        return RandomForestClassifier(
            n_estimators=int(rf_trees),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
            min_samples_leaf=2,
        ).fit(train_x, train_y)
    raise ValueError(f"Unsupported source head: {method}")


def prototype_split_metrics(x: np.ndarray, y: np.ndarray, k: int, splits: int, seed: int) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float]]:
    metrics = []
    all_true = []
    all_pred = []
    support_sizes = []
    query_sizes = []
    for split_idx in range(max(1, int(splits))):
        support_idx, query_idx = split_global_support_query(y, k, seed + split_idx * 1009)
        pred = class_centroid_predict(x[support_idx], y[support_idx], x[query_idx])
        metrics.append(metric_dict(y[query_idx], pred))
        all_true.append(y[query_idx])
        all_pred.append(pred)
        support_sizes.append(int(len(support_idx)))
        query_sizes.append(int(len(query_idx)))
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    mean = {key: float(np.mean([row[key] for row in metrics])) for key in keys}
    for key in keys:
        mean[f"{key}_std"] = float(np.std([row[key] for row in metrics], ddof=1)) if len(metrics) > 1 else 0.0
    mean["mean_support_size"] = float(np.mean(support_sizes))
    mean["mean_query_size"] = float(np.mean(query_sizes))
    return mean, np.concatenate(all_true), np.concatenate(all_pred), {
        "splits": int(max(1, int(splits))),
        "support_k": int(k),
    }


def maybe_subsample(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        n = max(1, int(round(max_rows * len(idx) / len(y))))
        chosen.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    if len(chosen) > max_rows:
        chosen = rng.choice(chosen, size=max_rows, replace=False)
    return x[chosen], y[chosen]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_region(args: argparse.Namespace, full_df: pd.DataFrame, region: str, device: torch.device) -> list[dict[str, Any]]:
    region_args = argparse.Namespace(**vars(args))
    region_args.holdout_region = region
    train_df, val_df, test_df, split_info = split_data(full_df, region_args)
    audit = build_split_audit(full_df, train_df, val_df, test_df, region_args, split_info)

    cache_dir = Path(args.cache_dir)
    whisper_model = load_whisper_encoder(args.whisper_model, device)
    whisper_model.eval()
    for param in whisper_model.parameters():
        param.requires_grad = False
    try:
        train_tokens_raw, train_matrix_meta = extract_matrix_cached(train_df, region_args, whisper_model, device, cache_dir, "train")
        test_tokens_raw, test_matrix_meta = extract_matrix_cached(test_df, region_args, whisper_model, device, cache_dir, "test")
    except (AttributeError, FileNotFoundError) as exc:
        print(f"[matrix-cache] miss needs Whisper extraction: {exc}", flush=True)
        train_tokens_raw, train_matrix_meta = extract_matrix_cached(train_df, region_args, whisper_model, device, cache_dir, "train")
        test_tokens_raw, test_matrix_meta = extract_matrix_cached(test_df, region_args, whisper_model, device, cache_dir, "test")
    token_norm = fit_token_norm(train_tokens_raw, args.feature_norm)
    train_tokens = apply_token_norm(train_tokens_raw, token_norm)
    test_tokens = apply_token_norm(test_tokens_raw, token_norm)

    need_acoustic = any(
        method.startswith("acoustic-")
        or method.startswith("fused-")
        or method == "target-proto-acoustic"
        or method == "target-proto-fused"
        for method in args.methods
    )
    if need_acoustic:
        train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, region_args, train_tokens_raw, "train")
        test_aux_raw, test_aux_meta = extract_aux_matrix_cached(test_df, region_args, test_tokens_raw, "test")
        aux_norm = fit_vector_norm(train_aux_raw, "global")
        train_aux = apply_vector_norm(train_aux_raw, aux_norm)
        test_aux = apply_vector_norm(test_aux_raw, aux_norm)
    else:
        train_aux_meta = None
        test_aux_meta = None
        train_aux = None
        test_aux = None

    train_y = labels_to_ids(train_df[args.label_column])
    test_y = labels_to_ids(test_df[args.label_column])

    feature_sets = {
        "whisper": (pool_mean(train_tokens), pool_mean(test_tokens)),
        "whisper-ms": (pool_mean_std(train_tokens), pool_mean_std(test_tokens)),
    }
    if need_acoustic and train_aux is not None and test_aux is not None:
        feature_sets["acoustic"] = (pool_mean_std(train_aux), pool_mean_std(test_aux))
        feature_sets["fused"] = (
            np.concatenate([pool_mean(train_tokens), pool_mean_std(train_aux)], axis=1).astype(np.float32),
            np.concatenate([pool_mean(test_tokens), pool_mean_std(test_aux)], axis=1).astype(np.float32),
        )

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        started = time.perf_counter()
        status = "done"
        error = ""
        feature_name = "none"
        pred = None
        metrics: dict[str, float] = {}
        extra: dict[str, Any] = {}
        try:
            if method == "majority":
                feature_name = "none"
                pred = majority_predict(train_y, len(test_y))
                metrics = metric_dict(test_y, pred)
                true_for_report = test_y
            elif method.startswith("target-proto-"):
                feature_name = method.replace("target-proto-", "")
                if feature_name == "whisper":
                    x_test = feature_sets["whisper"][1]
                elif feature_name == "acoustic":
                    x_test = feature_sets["acoustic"][1]
                elif feature_name == "fused":
                    x_test = feature_sets["fused"][1]
                else:
                    raise ValueError(f"Unknown prototype feature: {feature_name}")
                metrics, true_for_report, pred, extra = prototype_split_metrics(
                    x_test,
                    test_y,
                    k=int(args.prototype_k),
                    splits=int(args.prototype_splits),
                    seed=int(args.seed),
                )
            else:
                if method.startswith("whisper-"):
                    feature_name = "whisper"
                elif method.startswith("acoustic-"):
                    feature_name = "acoustic"
                elif method.startswith("fused-"):
                    feature_name = "fused"
                else:
                    raise ValueError(f"Unknown method prefix: {method}")
                x_train, x_test = feature_sets[feature_name]
                if method.endswith("centroid"):
                    pred = class_centroid_predict(x_train, train_y, x_test)
                else:
                    fit_x, fit_y = maybe_subsample(x_train, train_y, int(args.max_train_rows), int(args.seed))
                    clf = train_source_head(method, fit_x, fit_y, int(args.seed), int(args.rf_trees))
                    pred = clf.predict(x_test).astype(np.int64)
                    extra["train_rows_used"] = int(len(fit_y))
                metrics = metric_dict(test_y, pred)
                true_for_report = test_y
        except Exception as exc:  # Keep long multiregion sweeps alive.
            status = "failed"
            error = repr(exc)
            pred = np.zeros(0, dtype=np.int64)
            true_for_report = np.zeros(0, dtype=np.int64)
            metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan")}

        elapsed = time.perf_counter() - started
        row = {
            "status": status,
            "method": method,
            "feature": feature_name,
            "region": region,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "accuracy_std": metrics.get("accuracy_std", 0.0),
            "macro_f1_std": metrics.get("macro_f1_std", 0.0),
            "weighted_f1_std": metrics.get("weighted_f1_std", 0.0),
            "evaluated_rows": int(len(true_for_report)),
            "test_rows": int(len(test_y)),
            "train_rows": int(len(train_y)),
            "time_sec": float(elapsed),
            "error": error,
            **extra,
        }
        rows.append(row)

        result_path = Path(args.output_dir) / "results" / f"{safe_name(region)}_{method}.json"
        payload = {
            "row": row,
            "metrics": metrics,
            "method": method,
            "feature": feature_name,
            "region": region,
            "split": split_info,
            "split_audit": audit,
            "train_label_counts": label_counts(train_df[args.label_column], VOWEL_ORDER),
            "test_label_counts": label_counts(test_df[args.label_column], VOWEL_ORDER),
            "matrix_cache": {"train": train_matrix_meta, "test": test_matrix_meta},
            "aux_cache": {"train": train_aux_meta, "test": test_aux_meta},
        }
        if len(true_for_report) and len(pred):
            payload["report"] = classification_report(
                true_for_report,
                pred,
                labels=list(range(len(VOWEL_ORDER))),
                target_names=list(VOWEL_ORDER),
                output_dict=True,
                zero_division=0,
            )
            payload["confusion_matrix"] = confusion_matrix(
                true_for_report,
                pred,
                labels=list(range(len(VOWEL_ORDER))),
            ).tolist()
        write_json(result_path, payload)
        if not args.quiet:
            print(
                f"{region} {method}: mf1={100.0 * float(row['macro_f1']):.2f} "
                f"acc={100.0 * float(row['accuracy']):.2f} status={status} time={elapsed:.1f}s",
                flush=True,
            )
    return rows


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    done = pd.DataFrame([row for row in rows if row.get("status") == "done"])
    aggregate_rows = []
    if not done.empty:
        for method, group in done.groupby("method", sort=True):
            aggregate_rows.append(
                {
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
    aggregate_csv = output_dir / "aggregate.csv"
    with aggregate_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
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
        writer.writerows(aggregate_rows)
    write_json(output_dir / "summary.json", {"rows": rows, "aggregate": aggregate_rows})


def main() -> None:
    configure_stdout()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = load_dataset(args)
    regions = args.holdout_regions or sorted(full_df[args.region_column].astype(str).unique().tolist())
    all_rows: list[dict[str, Any]] = []
    for region in regions:
        all_rows.extend(run_region(args, full_df, str(region), device))
        write_summary(output_dir, all_rows)
    write_summary(output_dir, all_rows)


if __name__ == "__main__":
    main()
