#!/usr/bin/env python3
"""Build the six-area main Accuracy table.

The first four columns are target regions from the standard LORO protocol.
Ningguo and Lishui are target sites inside the SpeechSignal holdout region, so
their values are computed from site-level query predictions in that holdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch

from pc_dlcmnet.data.acoustic_features import apply_vector_norm, extract_aux_matrix_cached, fit_vector_norm
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.data.episodes import split_global_support_query
from tools.experiments.baseline_comparisons import (
    class_centroid_predict,
    pool_mean,
    pool_mean_std,
    train_source_head,
)
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


AREA_COLUMNS = [
    ("Qingyang", "region", "04"),
    ("Tongling", "region", "06"),
    ("Jingxian", "region", "08"),
    ("Nanling", "region", "10"),
    ("Ningguo", "site", "12"),
    ("Lishui", "site", "14"),
]


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--baseline-summary", default="output/0614/baseline_comparisons_full/summary.csv")
    parser.add_argument("--memory-summary", default="output/0614/random_fewshot_repeats_rolling_s100_e1/summary.csv")
    parser.add_argument(
        "--speechsignal-predictions",
        default="output/0614/random_fewshot_repeats_rolling_s100_e1/results/01_a9538d5b_k4_splits100.predictions.csv",
    )
    parser.add_argument("--output-tex", default="output/0614/paper_ready_results/main_acc_table_six_areas_100split.tex")
    parser.add_argument("--output-csv", default="output/0614/paper_ready_results/main_acc_table_six_areas_100split.csv")
    parser.add_argument("--cache-dir", default=str(old_root / "output/salmonn_style_whisper_cache_base"))
    parser.add_argument("--matrix-cache-dir", default=str(old_root / "output/ppm_supervised_matrix_cache"))
    parser.add_argument("--aux-cache-dir", default="output/0614/feature_memory_aux_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--prototype-splits", type=int, default=20)
    return parser.parse_args()


def region_by_prefix(df: pd.DataFrame, prefix: str) -> str:
    matches = sorted(df.loc[df["region"].astype(str).str.startswith(prefix), "region"].astype(str).unique())
    if not matches:
        raise KeyError(f"No region starts with {prefix}")
    return matches[0]


def speechsignal_region(df: pd.DataFrame) -> str:
    matches = sorted(df.loc[df["site"].astype(str).str.startswith("12"), "region"].astype(str).unique())
    if not matches:
        raise KeyError("Could not locate the SpeechSignal region containing site prefix 12")
    return matches[0]


def fmt(value: float | None) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{100.0 * value:.2f}"


def site_source_baselines(args: argparse.Namespace, full_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    holdout = speechsignal_region(full_df)
    region_args = SimpleNamespace(
        csv=args.csv,
        output_dir="",
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        label_column="vowel",
        region_column="region",
        speaker_column="speaker_id",
        labels=None,
        protocol="loro",
        holdout_region=holdout,
        holdout_regions=None,
        val_size=0.1,
        test_size=0.1,
        train_fraction=1.0,
        seed=args.seed,
        cache_dir=args.cache_dir,
        matrix_cache_dir=args.matrix_cache_dir,
        feature_norm="global",
        matrix_cache_scan=True,
        whisper_model="base",
        device=args.device,
        target_sr=16000,
        token_chunks=32,
        audio_root="",
        aux_source="acoustic",
        aux_representation="sequence",
        aux_cache_dir=args.aux_cache_dir,
        old_feature_cache_dir=str(default_old_root() / "output/feature_cache_vowel"),
        aux_n_mfcc=39,
        aux_n_mels=128,
        aux_f0_segments=5,
        aux_use_delta_mfcc=True,
        aux_use_formants=True,
        rf_trees=300,
        max_train_rows=0,
        quiet=True,
        require_all_source_regions=True,
    )
    train_df, _val_df, test_df, _split_info = split_data(full_df, region_args)
    device = resolve_device(args.device)
    train_tokens_raw, _ = extract_matrix_cached(train_df, region_args, None, device, Path(args.cache_dir), "train")
    test_tokens_raw, _ = extract_matrix_cached(test_df, region_args, None, device, Path(args.cache_dir), "test")
    token_norm = fit_token_norm(train_tokens_raw, "global")
    train_tokens = apply_token_norm(train_tokens_raw, token_norm)
    test_tokens = apply_token_norm(test_tokens_raw, token_norm)
    train_aux_raw, _ = extract_aux_matrix_cached(train_df, region_args, train_tokens_raw, "train")
    test_aux_raw, _ = extract_aux_matrix_cached(test_df, region_args, test_tokens_raw, "test")
    aux_norm = fit_vector_norm(train_aux_raw, "global")
    train_aux = apply_vector_norm(train_aux_raw, aux_norm)
    test_aux = apply_vector_norm(test_aux_raw, aux_norm)

    train_y = labels_to_ids(train_df["vowel"])
    test_y = labels_to_ids(test_df["vowel"])
    feature_sets = {
        "whisper": (pool_mean(train_tokens), pool_mean(test_tokens)),
        "acoustic": (pool_mean_std(train_aux), pool_mean_std(test_aux)),
        "fused": (
            np.concatenate([pool_mean(train_tokens), pool_mean_std(train_aux)], axis=1).astype(np.float32),
            np.concatenate([pool_mean(test_tokens), pool_mean_std(test_aux)], axis=1).astype(np.float32),
        ),
    }
    site_masks = {name: test_df["site"].astype(str).str.startswith(prefix).to_numpy() for name, _, prefix in AREA_COLUMNS if name in {"Ningguo", "Lishui"}}
    out: dict[str, dict[str, float]] = {name: {} for name in site_masks}
    methods = {
        "Acoustic centroid": ("acoustic", "centroid"),
        "Acoustic ridge classifier": ("acoustic", "acoustic-ridge"),
        "Acoustic linear SVM": ("acoustic", "acoustic-svm"),
        "Whisper": ("whisper", "whisper-svm"),
    }
    for label, (feature, method) in methods.items():
        x_train, x_test = feature_sets[feature]
        if method == "centroid":
            pred = class_centroid_predict(x_train, train_y, x_test)
        else:
            clf = train_source_head(method, x_train, train_y, int(args.seed), 300)
            pred = clf.predict(x_test).astype(np.int64)
        for site_name, mask in site_masks.items():
            out[site_name][label] = float((pred[mask] == test_y[mask]).mean()) if mask.any() else float("nan")

    # Few-shot prototype: average correctness over repeated support/query splits per site.
    x_test = feature_sets["fused"][1]
    correct = {site_name: [] for site_name in site_masks}
    for split_idx in range(max(1, int(args.prototype_splits))):
        support_idx, query_idx = split_global_support_query(test_y, int(args.prototype_k), int(args.seed) + split_idx * 1009)
        pred = class_centroid_predict(x_test[support_idx], test_y[support_idx], x_test[query_idx])
        for site_name, mask in site_masks.items():
            qmask = mask[query_idx]
            if qmask.any():
                correct[site_name].extend((pred[qmask] == test_y[query_idx][qmask]).astype(float).tolist())
    for site_name, vals in correct.items():
        out[site_name]["Target prototype"] = float(np.mean(vals)) if vals else float("nan")
    return out


def site_memory_values(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    pred = pd.read_csv(args.speechsignal_predictions)
    id_to_vowel = {i: v for i, v in enumerate(VOWEL_ORDER)}
    vowel_to_id = {v: i for i, v in id_to_vowel.items()}
    pred["global_pred_label_id"] = pred["global_pred_vowel"].map(vowel_to_id)
    out: dict[str, dict[str, float]] = {}
    for name, kind, prefix in AREA_COLUMNS:
        if kind != "site":
            continue
        sub = pred[pred["site"].astype(str).str.startswith(prefix)]
        out[name] = {
            "Global-only class memory": float((sub["true_label_id"] == sub["global_pred_label_id"]).mean()),
            "PC-DLCMNet": float((sub["true_label_id"] == sub["pred_label_id"]).mean()),
        }
    return out


def main() -> None:
    args = parse_args()
    full_df = load_dataset(args)
    base = pd.read_csv(args.baseline_summary)
    mem = pd.read_csv(args.memory_summary)
    site_base = site_source_baselines(args, full_df)
    site_mem = site_memory_values(args)

    methods = [
        "MFCC-SVM",
        "Acoustic centroid",
        "Acoustic ridge classifier",
        "Acoustic linear SVM",
        "wav2vec~2.0",
        "HuBERT",
        "WavLM",
        "Whisper",
        "mHuBERT-147",
        "MR-HuBERT",
        "MS-HuBERT",
        "Qwen2-Audio",
        "SALMONN",
        "Target prototype",
        "Global-only class memory",
        "PC-DLCMNet",
    ]
    baseline_key = {
        "Acoustic centroid": "acoustic-centroid",
        "Acoustic ridge classifier": "acoustic-ridge",
        "Acoustic linear SVM": "acoustic-svm",
        "Whisper": "whisper-svm",
        "Target prototype": "target-proto-fused",
    }
    values: dict[str, list[float | None]] = {m: [] for m in methods}
    for method in methods:
        for area, kind, prefix in AREA_COLUMNS:
            value: float | None = None
            if kind == "region":
                region = region_by_prefix(full_df, prefix)
                if method in baseline_key:
                    row = base[(base["region"].astype(str) == region) & (base["method"] == baseline_key[method])]
                    if len(row):
                        value = float(row.iloc[0]["accuracy"])
                elif method == "Global-only class memory":
                    row = mem[mem["region"].astype(str) == region]
                    if len(row):
                        value = float(row.iloc[0]["global_accuracy"])
                elif method == "PC-DLCMNet":
                    row = mem[mem["region"].astype(str) == region]
                    if len(row):
                        value = float(row.iloc[0]["accuracy"])
            else:
                if method in site_base.get(area, {}):
                    value = site_base[area][method]
                elif method in site_mem.get(area, {}):
                    value = site_mem[area][method]
            values[method].append(value)

    best, runner = [], []
    for ci in range(len(AREA_COLUMNS)):
        ranked = sorted([(v[ci], m) for m, v in values.items() if v[ci] is not None and np.isfinite(v[ci])], reverse=True)
        best.append(ranked[0][1] if ranked else "")
        runner.append(ranked[1][1] if len(ranked) > 1 else "")

    def tex_cell(method: str, ci: int) -> str:
        value = values[method][ci]
        text = fmt(value)
        if text == "--":
            return text
        if method == best[ci]:
            return r"\textbf{" + text + "}"
        if method == runner[ci]:
            return r"\underline{" + text + "}"
        return text

    sections = [
        ("Acoustic feature classifiers", ["MFCC-SVM", "Acoustic centroid", "Acoustic ridge classifier", "Acoustic linear SVM"]),
        ("Pre-trained speech representation models", ["wav2vec~2.0", "HuBERT", "WavLM", "Whisper"]),
        ("Recent speech foundation models", ["mHuBERT-147", "MR-HuBERT", "MS-HuBERT", "Qwen2-Audio", "SALMONN"]),
        ("Few-shot and memory-based methods", ["Target prototype", "Global-only class memory", "PC-DLCMNet"]),
    ]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Accuracy (\%) under the cross-area $K$-shot setting. Qingyang, Tongling, Jingxian, and Nanling are target regions, while Ningguo and Lishui are target sites within the SpeechSignal holdout region. PC-DLCMNet and Global-only memory use 100 random support/query splits with one evaluation pass per split. The best result in each column is highlighted in bold, and the runner-up is underlined.}",
        r"\label{tab:main_acc}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(area for area, _, _ in AREA_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for title, rows in sections:
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        for method in rows:
            lines.append(method + " & " + " & ".join(tex_cell(method, ci) for ci in range(len(AREA_COLUMNS))) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular*}", r"\end{table*}"])

    out_tex = Path(args.output_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_rows = [{"method": m, **{AREA_COLUMNS[i][0]: values[m][i] for i in range(len(AREA_COLUMNS))}} for m in methods]
    pd.DataFrame(out_rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(out_tex)


if __name__ == "__main__":
    main()
