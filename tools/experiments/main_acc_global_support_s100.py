#!/usr/bin/env python3
"""Build the paper main Accuracy table under global target-support 100-split evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc_dlcmnet.utils.paths import (
    default_audio_root,
    default_dataset,
    default_matrix_cache,
    default_whisper_cache,
)

WORKSPACE = ROOT
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"
DATASET = default_dataset()
WHISPER_CACHE = default_whisper_cache()
MATRIX_CACHE = default_matrix_cache()
AUDIO_ROOT = default_audio_root()
OUT_DIR = ROOT / "output/0614/main_acc_global_support_s100"
PAPER_DIR = ROOT / "output/0614/paper_ready_results"
BASELINE_OUT = ROOT / "output/0614/main_acc_global_support_s100_baselines"


TARGETS = {
    "Qingyang": {
        "region_column": "region",
        "holdout": "04青阳",
        "ckpt": ROOT / "output/0614/multiregion_paper_dual_memory/04_1a770e49/04_1a770e49_stage3.pt",
        "reuse": ROOT / "output/0614/component_ablation_representative_global_support_s100/qingyang_pcdlcmnet_s100.json",
    },
    "Tongling": {
        "region_column": "region",
        "holdout": "06铜陵",
        "ckpt": ROOT / "output/0614/multiregion_paper_dual_memory/06_4b65c09c/06_4b65c09c_stage3.pt",
    },
    "Jingxian": {
        "region_column": "region",
        "holdout": "08泾县",
        "ckpt": ROOT / "output/0614/multiregion_paper_dual_memory/08_2fe038da/08_2fe038da_stage3.pt",
    },
    "Nanling": {
        "region_column": "region",
        "holdout": "10南陵",
        "ckpt": ROOT / "output/0614/multiregion_paper_dual_memory/10_a5b08243/10_a5b08243_stage3.pt",
    },
    "Ningguo": {
        "region_column": "site",
        "holdout": "12宁国",
        "ckpt": ROOT / "output/0614/site_domain_paper_dual_memory/12_ningguo/12_ningguo_stage3.pt",
        "reuse": ROOT / "output/0614/component_ablation_representative_global_support_s100/ningguo_pcdlcmnet_s100.json",
    },
    "Lishui": {
        "region_column": "site",
        "holdout": "14溧水",
        "ckpt": ROOT / "output/0614/site_domain_paper_dual_memory/14_lishui/14_lishui_stage3.pt",
        "reuse": ROOT / "output/0614/component_ablation_representative_global_support_s100/lishui_pcdlcmnet_s100.json",
    },
}


METHODS = [
    "MFCC",
    "Acoustic features",
    "wav2vec~2.0",
    "HuBERT",
    "WavLM",
    "Whisper",
    "mHuBERT-147",
    "MR-HuBERT",
    "MS-HuBERT",
    "Qwen2-Audio",
    "SALMONN",
    "PC-DLCMNet",
]


BASELINE_KEY = {
    "MFCC": "mfcc-centroid",
    "Acoustic features": "acoustic-centroid",
    "Whisper": "whisper-centroid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--baseline-max-train-rows", type=int, default=0)
    return parser.parse_args()


def cli_path(path: Path) -> str:
    try:
        return os.path.relpath(path, WORKSPACE)
    except Exception:
        return str(path)


def run_output_path(area: str) -> Path:
    return OUT_DIR / f"{area.lower()}_pcdlcmnet_s100.json"


def reusable_result_path(area: str, target: dict[str, Any]) -> Path | None:
    output = run_output_path(area)
    if output.exists():
        return output
    reuse = target.get("reuse")
    if reuse and Path(reuse).exists():
        return Path(reuse)
    return None


def command(args: argparse.Namespace, area: str, target: dict[str, Any], output: Path) -> list[str]:
    return [
        args.python,
        "-u",
        str(RUNNER),
        "--holdout-region",
        str(target["holdout"]),
        "--region-column",
        str(target["region_column"]),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--episode-length",
        "256",
        "--episode-batch-size",
        "2",
        "--support-shots",
        "2",
        "--eval-support-shots",
        str(args.k),
        "--eval-ensemble-runs",
        "1",
        "--eval-split-runs",
        str(args.splits),
        "--eval-query-only",
        "--eval-split-mode",
        "global_support",
        "--eval-query-shots-per-class",
        "1",
        "--eval-classifier",
        "memory",
        "--support-write-mode",
        "label",
        "--support-label-blend",
        "1.0",
        "--lambda-global",
        "0.5",
        "--hidden-dim",
        "256",
        "--score-dim",
        "128",
        "--prompt-dim",
        "128",
        "--num-prompts",
        "8",
        "--layers",
        "3",
        "--heads",
        "4",
        "--ffn-dim",
        "768",
        "--dropout",
        "0.1",
        "--temperature",
        "0.2",
        "--token-chunks",
        "32",
        "--aux-source",
        "acoustic",
        "--aux-representation",
        "sequence",
        "--aux-cache-dir",
        "output/0614/feature_memory_aux_cache",
        "--output",
        str(output),
        "--skip-val-eval",
        "--quiet",
        "--progress",
        "--csv",
        str(DATASET),
        "--cache-dir",
        str(WHISPER_CACHE),
        "--matrix-cache-dir",
        str(MATRIX_CACHE),
        "--audio-root",
        str(AUDIO_ROOT),
        "--max-steps",
        "0",
        "--init-checkpoint",
        cli_path(Path(target["ckpt"])),
    ]


def ensure_pcdlcmnet_results(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for area, target in TARGETS.items():
        if reusable_result_path(area, target) is not None and not args.force:
            print(f"[skip] {area}", flush=True)
            continue
        output = run_output_path(area)
        if not Path(target["ckpt"]).exists():
            raise FileNotFoundError(f"Missing checkpoint for {area}: {target['ckpt']}")
        cmd = command(args, area, target, output)
        print("[run]", area, " ".join(cmd), flush=True)
        if args.skip_run:
            continue
        start = time.perf_counter()
        subprocess.run(cmd, cwd=str(WORKSPACE), check=True)
        print(f"[done] {area} time={time.perf_counter() - start:.1f}s", flush=True)


def read_metric(path: Path) -> tuple[float, float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = float(data["metrics"]["accuracy"])
    split = data.get("evaluation_splits", {}).get("test", {})
    std = None
    if isinstance(split, dict):
        metrics_std = split.get("metrics_std", {})
        if isinstance(metrics_std, dict) and "accuracy" in metrics_std:
            std = float(metrics_std["accuracy"])
    return mean, std


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_accuracy_mean_std(y_true: np.ndarray, pred: np.ndarray, k: int, splits: int, seed: int) -> tuple[float, float]:
    from pc_dlcmnet.data.episodes import split_global_support_query

    vals = []
    for split_idx in range(max(1, int(splits))):
        _support_idx, query_idx = split_global_support_query(y_true, int(k), int(seed) + split_idx * 1009)
        vals.append(float((pred[query_idx] == y_true[query_idx]).mean()) if len(query_idx) else float("nan"))
    clean = [v for v in vals if np.isfinite(v)]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def split_fewshot_accuracy_mean_std(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    k: int,
    splits: int,
    seed: int,
) -> tuple[float, float]:
    from pc_dlcmnet.data.episodes import split_global_support_query
    from tools.experiments.baseline_comparisons import class_centroid_predict, train_source_head

    vals = []
    for split_idx in range(max(1, int(splits))):
        support_idx, query_idx = split_global_support_query(y, int(k), int(seed) + split_idx * 1009)
        if len(support_idx) == 0 or len(query_idx) == 0:
            continue
        try:
            if method.endswith("centroid"):
                pred = class_centroid_predict(x[support_idx], y[support_idx], x[query_idx])
            else:
                clf = train_source_head(method, x[support_idx], y[support_idx], int(seed) + split_idx, 300)
                pred = clf.predict(x[query_idx]).astype(np.int64)
            vals.append(float((pred == y[query_idx]).mean()))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def maybe_subsample_train(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(max_rows) <= 0 or len(y) <= int(max_rows):
        return x, y
    rng = np.random.default_rng(int(seed))
    chosen = []
    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        n = max(1, int(round(int(max_rows) * len(idx) / len(y))))
        chosen.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    return x[chosen], y[chosen]


def mfcc_mean_std(aux_seq: np.ndarray, n_mels: int = 128, n_mfcc: int = 39) -> np.ndarray:
    # Sequence auxiliary features are [mel, mfcc, delta, delta2].
    if aux_seq.ndim == 2:
        return aux_seq[:, : n_mfcc * 2].astype(np.float32)
    mfcc = aux_seq[:, :, n_mels : n_mels + n_mfcc]
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=1).astype(np.float32)


def baseline_result_path(area: str, method: str) -> Path:
    safe_method = method.lower().replace(" ", "_").replace("~", "").replace("-", "_")
    return BASELINE_OUT / "target_support_stats" / f"{area.lower()}_{safe_method}_s100.json"


def target_baselines(args: argparse.Namespace) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    from pc_dlcmnet.data.acoustic_features import apply_vector_norm, extract_aux_matrix_cached, fit_vector_norm
    from tools.experiments.baseline_comparisons import class_centroid_predict, pool_mean, pool_mean_std, train_source_head
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

    values: dict[str, dict[str, tuple[float | None, float | None]]] = {
        method: {area: (None, None) for area in TARGETS} for method in BASELINE_KEY
    }
    dataset_args = SimpleNamespace(
        csv=str(DATASET),
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        label_column="vowel",
        region_column="region",
        speaker_column="speaker_id",
        labels=None,
    )
    full_df = load_dataset(dataset_args)
    device = resolve_device(args.device)
    for area, target in TARGETS.items():
        target_args = SimpleNamespace(
            csv=str(DATASET),
            output_dir="",
            path_column="wav_path",
            start_column="start_time",
            end_column="end_time",
            label_column="vowel",
            region_column=str(target["region_column"]),
            speaker_column="speaker_id",
            labels=None,
            protocol="loro",
            holdout_region=str(target["holdout"]),
            holdout_regions=None,
            val_size=0.1,
            test_size=0.1,
            train_fraction=1.0,
            seed=0,
            cache_dir=str(WHISPER_CACHE),
            matrix_cache_dir=str(MATRIX_CACHE),
            feature_norm="global",
            matrix_cache_scan=True,
            whisper_model="base",
            device=args.device,
            target_sr=16000,
            token_chunks=32,
            audio_root=str(AUDIO_ROOT),
            aux_source="acoustic",
            aux_representation="stats",
            aux_cache_dir="output/0614/feature_memory_aux_cache_stats",
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
        train_df, _val_df, test_df, _split_info = split_data(full_df, target_args)
        train_y = labels_to_ids(train_df["vowel"])
        test_y = labels_to_ids(test_df["vowel"])

        pending = [method for method in BASELINE_KEY if args.force or not baseline_result_path(area, method).exists()]
        if pending:
            print(f"[baseline] {area}: extracting features for {', '.join(pending)}", flush=True)
            train_tokens_raw, _ = extract_matrix_cached(train_df, target_args, None, device, Path(WHISPER_CACHE), "train")
            test_tokens_raw, _ = extract_matrix_cached(test_df, target_args, None, device, Path(WHISPER_CACHE), "test")
            token_norm = fit_token_norm(train_tokens_raw, "global")
            train_tokens = apply_token_norm(train_tokens_raw, token_norm)
            test_tokens = apply_token_norm(test_tokens_raw, token_norm)
            train_aux_raw, _ = extract_aux_matrix_cached(train_df, target_args, train_tokens_raw, "train")
            test_aux_raw, _ = extract_aux_matrix_cached(test_df, target_args, test_tokens_raw, "test")
            aux_norm = fit_vector_norm(train_aux_raw, "global")
            train_aux = apply_vector_norm(train_aux_raw, aux_norm)
            test_aux = apply_vector_norm(test_aux_raw, aux_norm)
            feature_sets = {
                "mfcc": (mfcc_mean_std(train_aux), mfcc_mean_std(test_aux)),
                "acoustic": (pool_mean_std(train_aux), pool_mean_std(test_aux)),
                "whisper": (pool_mean(train_tokens), pool_mean(test_tokens)),
            }
            split_scores: dict[str, tuple[float, float]] = {}
            for method in pending:
                if method == "Acoustic features":
                    _train_x, test_x = feature_sets["acoustic"]
                    split_scores[method] = split_fewshot_accuracy_mean_std(test_x, test_y, "acoustic-centroid", args.k, args.splits, args.seed)
                elif method == "MFCC":
                    _train_x, test_x = feature_sets["mfcc"]
                    split_scores[method] = split_fewshot_accuracy_mean_std(test_x, test_y, "mfcc-centroid", args.k, args.splits, args.seed)
                elif method == "Whisper":
                    _train_x, test_x = feature_sets["whisper"]
                    split_scores[method] = split_fewshot_accuracy_mean_std(test_x, test_y, "whisper-centroid", args.k, args.splits, args.seed)
            del train_tokens_raw, test_tokens_raw, train_tokens, test_tokens, train_aux_raw, test_aux_raw, train_aux, test_aux
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for method, (mean, std) in split_scores.items():
                write_json(
                    baseline_result_path(area, method),
                    {
                        "area": area,
                        "method": method,
                        "baseline_protocol": "target_support_train",
                        "holdout": target["holdout"],
                        "region_column": target["region_column"],
                        "metrics": {"accuracy": mean},
                        "evaluation_splits": {"test": {"runs": int(args.splits), "metrics_std": {"accuracy": std}}},
                        "test_rows": int(len(test_y)),
                    },
                )

        for method in BASELINE_KEY:
            path = baseline_result_path(area, method)
            if path.exists():
                values[method][area] = read_metric(path)
    return values


def pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.2f}"


def pct_pm(value: float | None, std: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if std is None or not np.isfinite(std):
        return f"{100.0 * value:.2f}"
    return f"{100.0 * value:.2f} $\\pm$ {100.0 * std:.2f}"


def pcdlcmnet_values(full_values: dict[str, dict[str, tuple[float | None, float | None]]]) -> None:
    for area, target in TARGETS.items():
        path = reusable_result_path(area, target)
        if path is None:
            continue
        full_values["PC-DLCMNet"][area] = read_metric(path)


def build_table(args: argparse.Namespace) -> None:
    areas = list(TARGETS.keys())
    values: dict[str, dict[str, tuple[float | None, float | None]]] = {
        method: {area: (None, None) for area in areas} for method in METHODS
    }
    baseline_values = target_baselines(args)
    for method, by_area in baseline_values.items():
        for area, cell in by_area.items():
            values[method][area] = cell
    pcdlcmnet_values(values)

    best: dict[str, str] = {}
    runner: dict[str, str] = {}
    for area in areas:
        ranked = sorted(
            [(cell[0], method) for method, by_area in values.items() for cell in [by_area[area]] if cell[0] is not None and np.isfinite(cell[0])],
            reverse=True,
        )
        best[area] = ranked[0][1] if ranked else ""
        runner[area] = ranked[1][1] if len(ranked) > 1 else ""

    def tex_cell(method: str, area: str) -> str:
        mean, std = values[method][area]
        text = pct_pm(mean, std)
        if text == "--":
            return text
        if method == best[area]:
            return r"\textbf{" + text + "}"
        if method == runner[area]:
            return r"\underline{" + text + "}"
        return text

    sections = [
        ("Acoustic feature classifiers", ["MFCC", "Acoustic features"]),
        ("Pre-trained speech representation models", ["wav2vec~2.0", "HuBERT", "WavLM", "Whisper"]),
        ("Recent speech foundation models", ["mHuBERT-147", "MR-HuBERT", "MS-HuBERT", "Qwen2-Audio", "SALMONN"]),
        ("Our method", ["PC-DLCMNet"]),
    ]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Accuracy (\%) under the cross-region $K$-shot setting. PC-DLCMNet is evaluated with 100 random target-support splits and one evaluation pass per split. Each split samples $K$ labeled support examples per class from the target region and evaluates on the remaining target samples. The best result in each column is highlighted in bold, and the runner-up is underlined.}",
        r"\label{tab:main_acc}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(areas) + r" \\",
        r"\midrule",
    ]
    for section, rows in sections:
        lines.append(r"\multicolumn{7}{l}{\textit{" + section + r"}} \\")
        lines.append(r"\midrule")
        for method in rows:
            lines.append(method + " & " + " & ".join(tex_cell(method, area) for area in areas) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular*}", r"\end{table*}"])

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = PAPER_DIR / "main_acc_table_global_support_s100.tex"
    csv_path = PAPER_DIR / "main_acc_table_global_support_s100.csv"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for method in METHODS:
        row = {"Method": method}
        for area in areas:
            mean, std = values[method][area]
            row[area] = mean
            row[f"{area}_std"] = std
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(tex_path)
    print("\n".join(lines), flush=True)


def main() -> None:
    args = parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    ensure_pcdlcmnet_results(args)
    if not args.skip_run:
        build_table(args)


if __name__ == "__main__":
    main()
