#!/usr/bin/env python3
"""Run 4-shot target-domain downstream baselines for traditional classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pc_dlcmnet.data.acoustic_features import apply_vector_norm, extract_aux_matrix_cached, fit_vector_norm
from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    default_old_root,
    extract_matrix_cached,
    fit_token_norm,
    labels_to_ids,
    load_dataset,
    resolve_device,
    split_data,
)
from pc_dlcmnet.utils.paths import default_audio_root, default_matrix_cache, default_whisper_cache
from tools.experiments.baseline_comparisons import pool_mean, pool_mean_std

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


TARGETS = {
    "Chizhou": ("region", "03池州"),
    "Qingyang": ("region", "04青阳"),
    "Tongling": ("region", "06铜陵"),
    "Jingxian": ("region", "08泾县"),
    "Nanling": ("region", "10南陵"),
    "Ningguo": ("site", "12宁国"),
    "Lishui": ("site", "14溧水"),
    "Huangshan": ("region", "11黄山"),
}

DEFAULT_CSV = PACKAGE_ROOT / "data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", default="output/0614/traditional_downstream_baselines_4shot_s100")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature", choices=["whisper", "acoustic", "fused"], default="fused")
    parser.add_argument("--methods", nargs="*", default=["logreg", "random_forest", "xgboost", "lightgbm", "svm"])
    parser.add_argument("--targets", nargs="*", default=list(TARGETS.keys()), choices=list(TARGETS.keys()))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_estimator(method: str, seed: int) -> Any:
    if method == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
        )
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )
    if method == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=seed),
        )
    if method == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=1,
        )
    if method == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200,
            num_leaves=7,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            min_child_samples=1,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"unsupported method: {method}")


def fit_predict(estimator: Any, train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    classes = np.asarray(sorted(np.unique(train_y).tolist()), dtype=np.int64)
    remap = {int(label): idx for idx, label in enumerate(classes.tolist())}
    inv = {idx: int(label) for label, idx in remap.items()}
    mapped_y = np.asarray([remap[int(label)] for label in train_y], dtype=np.int64)
    clf = clone(estimator)
    clf.fit(train_x, mapped_y)
    mapped_pred = np.asarray(clf.predict(query_x), dtype=np.int64)
    return np.asarray([inv[int(label)] for label in mapped_pred], dtype=np.int64)


def mean_std(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = ["accuracy", "macro_f1", "micro_f1", "weighted_f1"]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    std = {key: float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0 for key in keys}
    return mean, std


def dataset_args(args: argparse.Namespace, region_column: str, holdout: str) -> SimpleNamespace:
    old_root = default_old_root()
    return SimpleNamespace(
        csv=str(args.csv),
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        label_column="vowel",
        region_column=region_column,
        speaker_column="speaker_id",
        labels=None,
        protocol="loro",
        holdout_region=holdout,
        holdout_regions=None,
        val_size=0.1,
        test_size=0.1,
        train_fraction=1.0,
        seed=int(args.seed),
        cache_dir=str(default_whisper_cache()),
        matrix_cache_dir=str(default_matrix_cache()),
        feature_norm="global",
        matrix_cache_scan=True,
        whisper_model=str(args.whisper_model),
        device=str(args.device),
        target_sr=16000,
        token_chunks=32,
        audio_root=str(default_audio_root()),
        aux_source="acoustic",
        aux_representation="sequence",
        aux_cache_dir="output/0614/feature_memory_aux_cache",
        old_feature_cache_dir=str(old_root / "output/feature_cache_vowel"),
        aux_n_mfcc=39,
        aux_n_mels=128,
        aux_f0_segments=5,
        aux_use_delta_mfcc=True,
        aux_use_formants=True,
        require_all_source_regions=True,
    )


def extract_target_features(full_df: pd.DataFrame, args: argparse.Namespace, area: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    region_column, holdout = TARGETS[area]
    dargs = dataset_args(args, region_column, holdout)
    Path(dargs.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(dargs.matrix_cache_dir).mkdir(parents=True, exist_ok=True)
    train_df, _val_df, test_df, split_info = split_data(full_df, dargs)
    test_y = labels_to_ids(test_df[dargs.label_column])

    try:
        tokens_raw, matrix_meta = extract_matrix_cached(test_df, dargs, None, device, Path(dargs.cache_dir), f"{area}_test")
    except (AttributeError, FileNotFoundError):
        import whisper

        whisper_model = whisper.load_model(str(args.whisper_model), device=str(device))
        whisper_model.eval()
        for param in whisper_model.parameters():
            param.requires_grad = False
        tokens_raw, matrix_meta = extract_matrix_cached(test_df, dargs, whisper_model, device, Path(dargs.cache_dir), f"{area}_test")
    token_norm = fit_token_norm(tokens_raw, "global")
    tokens = apply_token_norm(tokens_raw, token_norm)

    aux_raw, aux_meta = extract_aux_matrix_cached(test_df, dargs, tokens_raw, f"{area}_test")
    aux_norm = fit_vector_norm(aux_raw, "global")
    aux = apply_vector_norm(aux_raw, aux_norm)

    whisper_x = pool_mean(tokens)
    acoustic_x = pool_mean_std(aux)
    if args.feature == "whisper":
        x = whisper_x
    elif args.feature == "acoustic":
        x = acoustic_x
    else:
        x = np.concatenate([whisper_x, acoustic_x], axis=1).astype(np.float32)

    meta = {
        "area": area,
        "region_column": region_column,
        "holdout": holdout,
        "split": split_info,
        "test_rows": int(len(test_y)),
        "feature": args.feature,
        "feature_dim": int(x.shape[1]),
        "matrix_cache": matrix_meta,
        "aux_cache": aux_meta,
        "source_train_rows": int(len(train_df)),
    }
    return x.astype(np.float32), test_y.astype(np.int64), meta


def run_method(area: str, method: str, x: np.ndarray, y: np.ndarray, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    estimator = make_estimator(method, int(args.seed))
    split_rows: list[dict[str, float]] = []
    support_sizes: list[int] = []
    query_sizes: list[int] = []
    for split_idx in range(int(args.splits)):
        split_seed = int(args.seed) + split_idx * 1009
        support_idx, query_idx = split_global_support_query(y, int(args.k), split_seed)
        pred = fit_predict(estimator, x[support_idx], y[support_idx], x[query_idx])
        split_rows.append(metric_dict(y[query_idx], pred))
        support_sizes.append(int(len(support_idx)))
        query_sizes.append(int(len(query_idx)))
    metrics, metrics_std = mean_std(split_rows)
    return {
        "area": area,
        "method": method,
        "config": {
            "k": int(args.k),
            "splits": int(args.splits),
            "seed": int(args.seed),
            "feature": args.feature,
            "protocol": "target_global_support",
        },
        "metrics": metrics,
        "metrics_std": metrics_std,
        "runs": split_rows,
        "support_size_mean": float(np.mean(support_sizes)),
        "query_size_mean": float(np.mean(query_sizes)),
        "time_sec": float(time.perf_counter() - started),
        **meta,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = load_dataset(
        SimpleNamespace(
            csv=str(args.csv),
            path_column="wav_path",
            start_column="start_time",
            end_column="end_time",
            label_column="vowel",
            region_column="region",
            labels=None,
        )
    )
    device = resolve_device(str(args.device))
    rows: list[dict[str, Any]] = []
    for area in args.targets:
        x, y, meta = extract_target_features(full_df, args, area, device)
        for method in args.methods:
            result_path = output_dir / "results" / f"{area}_{method}.json"
            if result_path.exists() and not args.force:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"[run] {area} {method} feature={args.feature} splits={args.splits}", flush=True)
                result = run_method(area, method, x, y, args, meta)
                write_json(result_path, result)
            rows.append(
                {
                    "area": area,
                    "method": method,
                    "feature": result["feature"],
                    "macro_f1": result["metrics"]["macro_f1"],
                    "macro_f1_std": result["metrics_std"]["macro_f1"],
                    "accuracy": result["metrics"]["accuracy"],
                    "accuracy_std": result["metrics_std"]["accuracy"],
                    "weighted_f1": result["metrics"]["weighted_f1"],
                    "weighted_f1_std": result["metrics_std"]["weighted_f1"],
                    "time_sec": result["time_sec"],
                    "test_rows": result["test_rows"],
                    "support_size_mean": result["support_size_mean"],
                    "query_size_mean": result["query_size_mean"],
                }
            )
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for method in args.methods:
        group = [row for row in rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "regions": len(group),
                "macro_f1": float(np.mean([row["macro_f1"] for row in group])),
                "accuracy": float(np.mean([row["accuracy"] for row in group])),
                "weighted_f1": float(np.mean([row["weighted_f1"] for row in group])),
            }
        )
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)


if __name__ == "__main__":
    main()
