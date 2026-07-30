#!/usr/bin/env python3
"""Build a protocol-consistent main Accuracy table from validated results.

The table only fills cells that can be traced to the 100-split target-support
evaluation. Cells without validated local results are left as ``--``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc_dlcmnet.utils.paths import HF_MODELS_ROOT, default_dataset
from pc_dlcmnet.data.episodes import split_global_support_query
from tools.experiments.baseline_comparisons import class_centroid_predict
from pc_dlcmnet.training.supervised import labels_to_ids, load_dataset, split_data
from tools.experiments.ssl_pretrained_baselines import cache_path

DATASET = default_dataset()
PAPER_DIR = ROOT / "output/0614/paper_ready_results"
SSL_CACHE = ROOT / "output/0614/ssl_feature_cache"
PC_RESULTS = {
    "Qingyang": ROOT / "output/0614/component_ablation_representative_global_support_s100/qingyang_pcdlcmnet_s100.json",
    "Tongling": ROOT / "output/0614/main_acc_global_support_s100/tongling_pcdlcmnet_s100.json",
    "Jingxian": ROOT / "output/0614/main_acc_global_support_s100/jingxian_pcdlcmnet_s100.json",
    "Nanling": ROOT / "output/0614/main_acc_global_support_s100/nanling_pcdlcmnet_s100.json",
    "Ningguo": ROOT / "output/0614/component_ablation_representative_global_support_s100/ningguo_pcdlcmnet_s100.json",
    "Lishui": ROOT / "output/0614/component_ablation_representative_global_support_s100/lishui_pcdlcmnet_s100.json",
}
WHISPER_RESULTS = {
    "Qingyang": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/qingyang_whisper_s100.json",
    "Tongling": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/tongling_whisper_s100.json",
    "Jingxian": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/jingxian_whisper_s100.json",
    "Nanling": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/nanling_whisper_s100.json",
    "Ningguo": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/ningguo_whisper_s100.json",
    "Lishui": ROOT / "output/0614/main_acc_global_support_s100_baselines/target_support_train/lishui_whisper_s100.json",
}

TARGETS = {
    "Qingyang": ("region", "04青阳"),
    "Tongling": ("region", "06铜陵"),
    "Jingxian": ("region", "08泾县"),
    "Nanling": ("region", "10南陵"),
    "Ningguo": ("site", "12宁国"),
    "Lishui": ("site", "14溧水"),
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

SSL_ALIASES = {
    "wav2vec~2.0": "wav2vec2-base",
    "HuBERT": "hubert-base",
    "WavLM": "wavlm-base-plus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_metric(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = data.get("metrics", {}).get("accuracy")
    split = data.get("evaluation_splits", {}).get("test", {})
    std = None
    if isinstance(split, dict):
        std = split.get("metrics_std", {}).get("accuracy")
    return (float(mean) if mean is not None else None, float(std) if std is not None else None)


def split_proto_score(x: np.ndarray, y: np.ndarray, k: int, splits: int, seed: int) -> tuple[float, float]:
    vals: list[float] = []
    for split_idx in range(int(splits)):
        support_idx, query_idx = split_global_support_query(y, int(k), int(seed) + split_idx * 1009)
        if not len(support_idx) or not len(query_idx):
            continue
        pred = class_centroid_predict(x[support_idx], y[support_idx], x[query_idx])
        vals.append(float((pred == y[query_idx]).mean()))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def base_data_args(region_column: str, holdout: str) -> SimpleNamespace:
    return SimpleNamespace(
        csv=str(DATASET),
        output_dir="",
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
        seed=0,
        require_all_source_regions=True,
    )


def ssl_args(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        label_column="vowel",
        target_sr=16000,
        cache_dir=str(SSL_CACHE),
    )


def load_ssl_test_features(test_df: pd.DataFrame, model_name: str) -> np.ndarray | None:
    args = ssl_args(model_name)
    path = cache_path(test_df, args, model_name, "test")
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def compute_ssl_values(args: argparse.Namespace) -> dict[str, dict[str, tuple[float | None, float | None]]]:
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
    out = {method: {area: (None, None) for area in TARGETS} for method in SSL_ALIASES}
    for area, (region_column, holdout) in TARGETS.items():
        data_args = base_data_args(region_column, holdout)
        _train_df, _val_df, test_df, _split_info = split_data(full_df, data_args)
        y = labels_to_ids(test_df["vowel"])
        for method, alias in SSL_ALIASES.items():
            x = load_ssl_test_features(test_df, alias)
            if x is None:
                continue
            out[method][area] = split_proto_score(x, y, args.k, args.splits, args.seed)
    return out


def pct_pm(value: float | None, std: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if std is None or not np.isfinite(std):
        return f"{100 * value:.2f}"
    return f"{100 * value:.2f} $\\pm$ {100 * std:.2f}"


def build_table(values: dict[str, dict[str, tuple[float | None, float | None]]]) -> str:
    areas = list(TARGETS)
    best: dict[str, str] = {}
    runner: dict[str, str] = {}
    for area in areas:
        ranked = sorted(
            [
                (mean, method)
                for method in METHODS
                for mean, _std in [values[method][area]]
                if mean is not None and np.isfinite(mean)
            ],
            reverse=True,
        )
        best[area] = ranked[0][1] if ranked else ""
        runner[area] = ranked[1][1] if len(ranked) > 1 else ""

    def cell(method: str, area: str) -> str:
        text = pct_pm(*values[method][area])
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
        r"\caption{Accuracy (\%) under the cross-region $K$-shot setting. Each reported cell uses 100 random target-support splits with one evaluation pass per split. Each split samples $K$ labeled support examples per class from the target region and evaluates on the remaining target samples. The best result in each column is highlighted in bold, and the runner-up is underlined.}",
        r"\label{tab:main_acc}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(areas) + r" \\",
        r"\midrule",
    ]
    for title, rows in sections:
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        for method in rows:
            lines.append(method + " & " + " & ".join(cell(method, area) for area in areas) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular*}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    values = {method: {area: (None, None) for area in TARGETS} for method in METHODS}
    for method, by_area in compute_ssl_values(args).items():
        values[method].update(by_area)
    for area, path in WHISPER_RESULTS.items():
        values["Whisper"][area] = read_metric(path)
    for area, path in PC_RESULTS.items():
        values["PC-DLCMNet"][area] = read_metric(path)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tex = build_table(values)
    tex_path = PAPER_DIR / "main_acc_table_valid_s100.tex"
    legacy_tex_path = PAPER_DIR / "main_acc_table_global_support_s100.tex"
    csv_path = PAPER_DIR / "main_acc_table_valid_s100.csv"
    legacy_csv_path = PAPER_DIR / "main_acc_table_global_support_s100.csv"
    tex_path.write_text(tex, encoding="utf-8")
    legacy_tex_path.write_text(tex, encoding="utf-8")
    rows = []
    for method in METHODS:
        row: dict[str, Any] = {"Method": method}
        for area in TARGETS:
            mean, std = values[method][area]
            row[area] = mean
            row[f"{area}_std"] = std
        rows.append(row)
    table_df = pd.DataFrame(rows)
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    table_df.to_csv(legacy_csv_path, index=False, encoding="utf-8-sig")
    print(tex_path)
    print(tex)


if __name__ == "__main__":
    main()
