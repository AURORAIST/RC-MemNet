#!/usr/bin/env python3
"""Build statistical significance tables from saved RC-MemNet artifacts.

The script uses only saved per-region/per-run artifacts. It does not infer
paired tests from aggregate standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_ROOT = PACKAGE_ROOT / "output/0722/main_table_rc_memnet_14targets_20260722_231203"
DEFAULT_BASELINE_ROOT = PACKAGE_ROOT / "output/0722/baselines_whisaid_maslora_20260725"
DEFAULT_PER_RUN_ROOT = PACKAGE_ROOT / "output/0722/baselines_whisaid_maslora_20260725_per_run"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output/0722/statistical_significance"
DEFAULT_METRICS = ["accuracy", "macro_f1", "weighted_f1"]
REGION_ALIASES = {
    "dangtu": ["dangtu", "当涂"],
    "wuhu": ["wuhu", "芜湖"],
    "chizhou": ["chizhou", "池州"],
    "qingyang": ["qingyang", "青阳"],
    "suncun": ["suncun", "孙村"],
    "tongling": ["tongling", "铜陵"],
    "xuancheng": ["xuancheng", "宣城"],
    "jingxian": ["jingxian", "泾县"],
    "fanchang": ["fanchang", "繁昌"],
    "nanling": ["nanling", "南陵"],
    "huangshan": ["huangshan", "黄山"],
    "ningguo": ["ningguo", "宁国"],
    "gaochun": ["gaochun", "高淳"],
    "lishui": ["lishui", "溧水"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--per-run-root", type=Path, default=DEFAULT_PER_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-runs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def region_key(value: Any) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    for canonical, aliases in REGION_ALIASES.items():
        if any(alias.lower() in lower for alias in aliases):
            return canonical
    stripped = lower
    while stripped and (stripped[0].isdigit() or stripped[0] in {"_", "-", " "}):
        stripped = stripped[1:]
    return stripped or lower


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_stats(values_a: np.ndarray, values_b: np.ndarray, bootstrap_runs: int, seed: int) -> dict[str, Any]:
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    n = int(len(diff))
    if n == 0:
        return {
            "n": 0,
            "mean_a": None,
            "mean_b": None,
            "mean_diff": None,
            "std_diff": None,
            "t_value": None,
            "p_value_normal_approx": None,
            "p_value_sign_test": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = std_diff / math.sqrt(n) if n > 0 else 0.0
    t_value = mean_diff / se if se > 0 else (float("inf") if mean_diff != 0 else 0.0)
    p_approx = float(2.0 * (1.0 - normal_cdf(abs(t_value)))) if math.isfinite(t_value) else 0.0

    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    zero = int((diff == 0).sum())
    effective_n = pos + neg
    if effective_n:
        k = min(pos, neg)
        tail = sum(math.comb(effective_n, i) for i in range(k + 1)) / (2**effective_n)
        p_sign = min(1.0, 2.0 * tail)
    else:
        p_sign = 1.0

    rng = np.random.default_rng(int(seed))
    if bootstrap_runs > 0:
        boot = np.empty(int(bootstrap_runs), dtype=np.float64)
        for i in range(int(bootstrap_runs)):
            sample = diff[rng.integers(0, n, size=n)]
            boot[i] = float(sample.mean())
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))
    else:
        ci_low = mean_diff
        ci_high = mean_diff

    return {
        "n": n,
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_value": float(t_value),
        "p_value_normal_approx": p_approx,
        "p_value_sign_test": p_sign,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
    }


def one_sample_against_runs(point_value: float, run_values: np.ndarray, bootstrap_runs: int, seed: int) -> dict[str, Any]:
    runs = np.asarray(run_values, dtype=np.float64)
    runs = runs[np.isfinite(runs)]
    diffs = np.full_like(runs, float(point_value)) - runs
    stats = paired_stats(np.full_like(runs, float(point_value)), runs, bootstrap_runs, seed)
    stats["comparison_note"] = "RC-MemNet point estimate minus saved baseline per-run metrics"
    stats["baseline_run_min"] = float(runs.min()) if len(runs) else None
    stats["baseline_run_max"] = float(runs.max()) if len(runs) else None
    stats["baseline_run_mean"] = float(runs.mean()) if len(runs) else None
    stats["rc_above_baseline_runs"] = int((diffs > 0).sum()) if len(runs) else 0
    return stats


def collect_main(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/result.json")):
        data = load_json(path)
        holdout = data.get("split", {}).get("holdout_region") or path.parent.name
        metrics = data.get("metrics", {})
        rows[region_key(holdout)] = {
            "method": data.get("method", "RC-MemNet"),
            "region": str(holdout),
            "folder": path.parent.name,
            "result_file": str(path),
            **{metric: metrics.get(metric) for metric in DEFAULT_METRICS},
        }
    return rows


def collect_baselines(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    baselines: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/*/result.json")):
        data = load_json(path)
        method = str(data.get("model_type") or path.parent.parent.name)
        holdout = data.get("holdout_region") or data.get("split", {}).get("holdout_region") or path.parent.name
        metrics = data.get("metrics", {})
        baselines.setdefault(method, {})[region_key(holdout)] = {
            "method": method,
            "region": str(holdout),
            "folder": path.parent.name,
            "result_file": str(path),
            **{metric: metrics.get(metric) for metric in DEFAULT_METRICS},
        }
    return baselines


def collect_per_run(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    runs: dict[str, dict[str, pd.DataFrame]] = {}
    for path in sorted(root.glob("*/*/per_run_metrics.csv")):
        method = path.parent.parent.name
        key = region_key(path.parent.name)
        runs.setdefault(method, {})[key] = pd.read_csv(path, encoding="utf-8-sig")
    return runs


def fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value) * 100.0:.2f}"


def fmt_p(value: Any) -> str:
    if value is None:
        return ""
    value = float(value)
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def stars(p_value: Any) -> str:
    if p_value is None:
        return ""
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Statistical significance of RC-MemNet improvements over saved baselines.}",
        r"\label{tab:statistical_significance}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Baseline & Metric & $n$ & Baseline & RC-MemNet & Gain & $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['baseline']} & {row['metric']} & {row['n']} & "
            f"{row['baseline_mean_pct']} & {row['main_mean_pct']} & "
            f"{row['gain_pct']} & {row['p_value']} {row['stars']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_markdown(rows: list[dict[str, Any]], region_rows: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# Statistical Significance",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Tests use saved artifacts only. Region-level rows are paired by holdout region. Per-run rows are reported only when a saved `per_run_metrics.csv` exists.",
        "",
        "## Summary",
        "",
        "| Baseline | Metric | n | Baseline | RC-MemNet | Gain | 95% bootstrap CI | Sign p | t approx p | Sig. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {row['n']} | {row['baseline_mean_pct']} | "
            f"{row['main_mean_pct']} | {row['gain_pct']} | {row['ci95_pct']} | "
            f"{row['p_value_sign_test']} | {row['p_value_normal_approx']} | {row['stars']} |"
        )
    lines.extend(
        [
            "",
            "## Region Pairs",
            "",
            "| Baseline | Region | Metric | Baseline | RC-MemNet | Gain |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in region_rows:
        lines.append(
            f"| {row['baseline']} | {row['region']} | {row['metric']} | "
            f"{fmt_pct(row['baseline_value'])} | {fmt_pct(row['main_value'])} | {fmt_pct(row['gain'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{(output_dir / 'significance_summary.csv').name}`",
            f"- `{(output_dir / 'region_level_pairs.csv').name}`",
            f"- `{(output_dir / 'significance_summary.json').name}`",
            f"- `{(output_dir / 'significance_table.tex').name}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics = [item.strip() for item in str(args.metrics).split(",") if item.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = collect_main(args.main_root)
    baselines = collect_baselines(args.baseline_root)
    per_runs = collect_per_run(args.per_run_root)

    summary_rows: list[dict[str, Any]] = []
    region_pair_rows: list[dict[str, Any]] = []
    raw_stats: dict[str, Any] = {
        "main_root": str(args.main_root),
        "baseline_root": str(args.baseline_root),
        "per_run_root": str(args.per_run_root),
        "bootstrap_runs": int(args.bootstrap_runs),
        "seed": int(args.seed),
        "comparisons": {},
    }

    for baseline, by_region in sorted(baselines.items()):
        raw_stats["comparisons"].setdefault(baseline, {})
        common_regions = sorted(set(main_rows) & set(by_region))
        for metric_idx, metric in enumerate(metrics):
            pairs = [
                (key, main_rows[key].get(metric), by_region[key].get(metric))
                for key in common_regions
                if main_rows[key].get(metric) is not None and by_region[key].get(metric) is not None
            ]
            if pairs:
                main_values = np.asarray([float(item[1]) for item in pairs], dtype=np.float64)
                baseline_values = np.asarray([float(item[2]) for item in pairs], dtype=np.float64)
                stat = paired_stats(main_values, baseline_values, int(args.bootstrap_runs), int(args.seed) + metric_idx)
                raw_stats["comparisons"][baseline][metric] = stat
                row = {
                    "comparison_type": "region_paired",
                    "baseline": baseline,
                    "metric": metric,
                    "n": stat["n"],
                    "baseline_mean_pct": fmt_pct(stat["mean_b"]),
                    "main_mean_pct": fmt_pct(stat["mean_a"]),
                    "gain_pct": fmt_pct(stat["mean_diff"]),
                    "ci95_pct": f"[{fmt_pct(stat['bootstrap_ci_low'])}, {fmt_pct(stat['bootstrap_ci_high'])}]",
                    "p_value": fmt_p(stat["p_value_sign_test"]),
                    "p_value_sign_test": fmt_p(stat["p_value_sign_test"]),
                    "p_value_normal_approx": fmt_p(stat["p_value_normal_approx"]),
                    "stars": stars(stat["p_value_sign_test"]),
                    "positive": stat["positive_count"],
                    "negative": stat["negative_count"],
                    "zero": stat["zero_count"],
                    "note": "Paired by holdout region",
                }
                summary_rows.append(row)
                for key, main_value, base_value in pairs:
                    region_pair_rows.append(
                        {
                            "baseline": baseline,
                            "region_key": key,
                            "region": main_rows[key]["region"],
                            "metric": metric,
                            "baseline_value": float(base_value),
                            "main_value": float(main_value),
                            "gain": float(main_value) - float(base_value),
                            "main_file": main_rows[key]["result_file"],
                            "baseline_file": by_region[key]["result_file"],
                        }
                    )

            for key, run_df in sorted(per_runs.get(baseline, {}).items()):
                if key not in main_rows or metric not in run_df.columns or main_rows[key].get(metric) is None:
                    continue
                stat = one_sample_against_runs(
                    float(main_rows[key][metric]),
                    run_df[metric].to_numpy(dtype=np.float64),
                    int(args.bootstrap_runs),
                    int(args.seed) + 1000 + metric_idx,
                )
                label = f"{baseline}_per_run_{key}"
                raw_stats["comparisons"].setdefault(label, {})[metric] = stat
                summary_rows.append(
                    {
                        "comparison_type": "baseline_per_run",
                        "baseline": f"{baseline} per-run ({main_rows[key]['region']})",
                        "metric": metric,
                        "n": stat["n"],
                        "baseline_mean_pct": fmt_pct(stat["mean_b"]),
                        "main_mean_pct": fmt_pct(stat["mean_a"]),
                        "gain_pct": fmt_pct(stat["mean_diff"]),
                        "ci95_pct": f"[{fmt_pct(stat['bootstrap_ci_low'])}, {fmt_pct(stat['bootstrap_ci_high'])}]",
                        "p_value": fmt_p(stat["p_value_sign_test"]),
                        "p_value_sign_test": fmt_p(stat["p_value_sign_test"]),
                        "p_value_normal_approx": fmt_p(stat["p_value_normal_approx"]),
                        "stars": stars(stat["p_value_sign_test"]),
                        "positive": stat["positive_count"],
                        "negative": stat["negative_count"],
                        "zero": stat["zero_count"],
                        "note": stat["comparison_note"],
                    }
                )

    write_csv(summary_rows, output_dir / "significance_summary.csv")
    write_csv(region_pair_rows, output_dir / "region_level_pairs.csv")
    (output_dir / "significance_summary.json").write_text(json.dumps(raw_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "significance_table.tex").write_text(build_latex_table(summary_rows), encoding="utf-8")
    (output_dir / "README.md").write_text(build_markdown(summary_rows, region_pair_rows, output_dir), encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary_rows), "region_pairs": len(region_pair_rows), "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
