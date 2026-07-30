#!/usr/bin/env python3
"""K-shot sweep for local foundation speech encoders on target regions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import torch

from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.training.supervised import labels_to_ids, load_dataset, resolve_device, split_data
from tools.experiments.baseline_comparisons import class_centroid_predict
from tools.experiments.local_hf_ssl_4shot_eval import (
    DEFAULT_MODELS,
    dataset_args,
    extract_features,
    load_model,
    metric_dict,
    write_json,
)


TARGETS = {
    "Qingyang": ("region", "04青阳"),
    "Tongling": ("region", "06铜陵"),
    "Jingxian": ("region", "08泾县"),
    "Nanling": ("region", "10南陵"),
    "Ningguo": ("site", "12宁国"),
    "Lishui": ("site", "14溧水"),
    "Chizhou": ("region", "03池州"),
    "Huangshan": ("region", "11黄山"),
}

DEFAULT_RUN_MODELS = ["mHuBERT-147", "MR-HuBERT", "MS-HuBERT", "SALMONN-proxy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(PACKAGE_ROOT / "data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv"))
    parser.add_argument("--output-dir", default="output/0614/foundation_kshot_1_10_8areas")
    parser.add_argument("--cache-dir", default="output/0614/local_hf_ssl_feature_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--k-values", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=DEFAULT_RUN_MODELS)
    parser.add_argument("--areas", nargs="*", default=list(TARGETS.keys()))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def evaluate_kshot(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    splits: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    runs = []
    for split_idx in range(int(splits)):
        support_idx, query_idx = split_global_support_query(y, int(k), int(seed) + split_idx * 1009)
        pred = class_centroid_predict(x[support_idx], y[support_idx], x[query_idx])
        runs.append(metric_dict(y[query_idx], pred))
    keys = ["accuracy", "macro_f1", "micro_f1", "weighted_f1"]
    mean = {key: float(np.mean([row[key] for row in runs])) for key in keys}
    std = {key: float(np.std([row[key] for row in runs], ddof=1)) if len(runs) > 1 else 0.0 for key in keys}
    return mean, std, runs


def result_usable(path: Path, k: int, splits: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    return (
        int(config.get("k", -1)) == int(k)
        and int(config.get("splits", -1)) == int(splits)
        and int(config.get("seed", -1)) == int(seed)
        and metrics.get("macro_f1") is not None
        and metrics.get("accuracy") is not None
    )


def pct(mean: Any, std: Any) -> str:
    try:
        return f"{float(mean) * 100.0:.2f} ± {float(std) * 100.0:.2f}"
    except Exception:
        return ""


def write_summary_tables(output_dir: Path, rows: list[dict[str, Any]], k_values: list[int]) -> None:
    fieldnames = [
        "model",
        "area",
        "holdout",
        "region_column",
        "k",
        "macro_f1",
        "macro_f1_std",
        "accuracy",
        "accuracy_std",
        "weighted_f1",
        "weighted_f1_std",
        "test_rows",
        "feature_source",
        "feature_path",
        "time_sec",
        "result_file",
    ]
    with (output_dir / "foundation_kshot_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "foundation_kshot_summary_with_std.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "area", "k", "Macro-F1", "Accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "area": row["area"],
                    "k": row["k"],
                    "Macro-F1": pct(row["macro_f1"], row["macro_f1_std"]),
                    "Accuracy": pct(row["accuracy"], row["accuracy_std"]),
                }
            )

    for metric, label in [("macro_f1", "macro_f1"), ("accuracy", "accuracy")]:
        columns = ["model", "area"] + [f"k{k}" for k in k_values]
        with (output_dir / f"foundation_kshot_{label}_wide_mean_std.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for model in sorted({row["model"] for row in rows}):
                for area in TARGETS:
                    group = {(row["area"], int(row["k"])): row for row in rows if row["model"] == model and row["area"] == area}
                    if not group:
                        continue
                    out = {"model": model, "area": area}
                    for k in k_values:
                        row = group.get((area, int(k)))
                        out[f"k{k}"] = pct(row.get(metric), row.get(f"{metric}_std")) if row else ""
                    writer.writerow(out)

        with (output_dir / f"foundation_kshot_{label}_across_area.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "k", f"mean_{label}", f"std_area_{label}", f"{label}_mean_std"])
            writer.writeheader()
            for model in sorted({row["model"] for row in rows}):
                for k in k_values:
                    vals = [float(row[metric]) for row in rows if row["model"] == model and int(row["k"]) == int(k)]
                    if not vals:
                        continue
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    writer.writerow(
                        {
                            "model": model,
                            "k": int(k),
                            f"mean_{label}": mean,
                            f"std_area_{label}": std,
                            f"{label}_mean_std": pct(mean, std),
                        }
                    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_areas = {area: TARGETS[area] for area in args.areas}
    k_values = [int(k) for k in args.k_values]

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

    run_config = {
        "models": list(args.models),
        "areas": list(selected_areas.keys()),
        "k_values": k_values,
        "splits": int(args.splits),
        "seed": int(args.seed),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "cache_dir": str(args.cache_dir),
        "note": "SALMONN-proxy uses local whisper-base plus SALMONN Q-former/projection weights, not full official SALMONN generation.",
    }
    write_json(output_dir / "run_config.json", run_config)

    for model_name in args.models:
        extractor, model, model_path = load_model(model_name, device)
        for area, (region_column, holdout) in selected_areas.items():
            dargs = dataset_args(args, region_column, holdout)
            _train_df, _val_df, test_df, split_info = split_data(full_df, dargs)
            y = labels_to_ids(test_df["vowel"])
            started = time.perf_counter()
            x, cache_meta = extract_features(test_df, args, model_name, area, extractor, model, device)
            feature_time = float(time.perf_counter() - started)
            for k in k_values:
                result_path = output_dir / "results" / model_name / area / f"k{k}.json"
                if result_usable(result_path, k, int(args.splits), int(args.seed)) and not args.force:
                    data = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    eval_started = time.perf_counter()
                    metrics, metrics_std, runs = evaluate_kshot(x, y, k, int(args.splits), int(args.seed))
                    data = {
                        "model": model_name,
                        "model_path": str(model_path),
                        "area": area,
                        "holdout": holdout,
                        "region_column": region_column,
                        "config": {"k": int(k), "splits": int(args.splits), "seed": int(args.seed), "protocol": "target_global_support"},
                        "metrics": metrics,
                        "metrics_std": metrics_std,
                        "runs": runs,
                        "feature_cache": cache_meta,
                        "split": split_info,
                        "test_rows": int(len(y)),
                        "feature_time_sec": feature_time,
                        "time_sec": float(time.perf_counter() - eval_started),
                    }
                    write_json(result_path, data)
                row = {
                    "model": model_name,
                    "area": area,
                    "holdout": holdout,
                    "region_column": region_column,
                    "k": int(k),
                    "macro_f1": data["metrics"]["macro_f1"],
                    "macro_f1_std": data["metrics_std"]["macro_f1"],
                    "accuracy": data["metrics"]["accuracy"],
                    "accuracy_std": data["metrics_std"]["accuracy"],
                    "weighted_f1": data["metrics"]["weighted_f1"],
                    "weighted_f1_std": data["metrics_std"]["weighted_f1"],
                    "test_rows": data["test_rows"],
                    "feature_source": data.get("feature_cache", {}).get("source", ""),
                    "feature_path": data.get("feature_cache", {}).get("path", ""),
                    "time_sec": data.get("time_sec", ""),
                    "result_file": str(result_path),
                }
                rows.append(row)
                print(
                    f"{model_name} {area} k={k}: mf1={row['macro_f1']*100:.2f}±{row['macro_f1_std']*100:.2f} "
                    f"acc={row['accuracy']*100:.2f}±{row['accuracy_std']*100:.2f}",
                    flush=True,
                )
            write_summary_tables(output_dir, rows, k_values)
        del model
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary_tables(output_dir, rows, k_values)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "foundation_kshot_summary.csv"),
                "summary_with_std": str(output_dir / "foundation_kshot_summary_with_std.csv"),
                "macro_f1_wide": str(output_dir / "foundation_kshot_macro_f1_wide_mean_std.csv"),
                "accuracy_wide": str(output_dir / "foundation_kshot_accuracy_wide_mean_std.csv"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
