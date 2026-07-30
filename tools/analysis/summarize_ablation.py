#!/usr/bin/env python3
"""Summarize code-aligned ablations from existing 0614 experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TARGET_DANGTU = "01\u5f53\u6d82"

FEATURE_RUNS = [
    {
        "label": "speech_baseline",
        "path": PACKAGE_ROOT / "output/0614/effectiveness/01dt_speech_baseline_seed0.json",
        "comparable": True,
        "notes": "Static speech-only baseline on the 01_dangtu holdout.",
    },
    {
        "label": "memory_only",
        "glob": "output/0614/feature_memory_grid/**/results/*memory_only_seed0.json",
        "holdout_region": TARGET_DANGTU,
        "comparable": True,
        "notes": "Adds shared class memory inside the Transformer; no feature stream, no prompt gate.",
    },
    {
        "label": "feature_memory_no_prompt",
        "glob": "output/0614/feature_memory_grid/**/results/*feature_memory_no_prompt_seed0.json",
        "holdout_region": TARGET_DANGTU,
        "comparable": True,
        "notes": "Adds acoustic feature fusion on top of shared class memory; prompt gate disabled.",
    },
    {
        "label": "feature_memory_prompt",
        "path": PACKAGE_ROOT / "output/0614/effectiveness/01dt_feature_memory_prompt_seed0.json",
        "comparable": True,
        "notes": "Adds prompt-gated Q/K/V attention on top of feature fusion.",
    },
    {
        "label": "region_memory_prompt",
        "path": PACKAGE_ROOT / "output/0614/region_memory_probe/dt_region_memory_800step.json",
        "comparable": False,
        "notes": "Region memory also enables region supervision and compactness loss in this run.",
    },
]

PAPER_SINGLE_REGION = PACKAGE_ROOT / "output/0614/effectiveness/01dt_paper_dual_memory_s2e32_ft2_lr5e5_seed0.json"
PAPER_SUMMARY = PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory/summary.csv"
PAPER_FINAL_STAGE3 = PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory/final_stage3_report.json"

RECURRENT_CANDIDATES = [
    PACKAGE_ROOT / "output/0614/recurrent_effectiveness/pilot_01/01dt_recurrent_online120_lamstatic05_tau0.json",
    PACKAGE_ROOT / "output/0614/recurrent_effectiveness/pilot_01_batched/01dt_recurrent_online40_batched4_lamstatic05_tau0.json",
    PACKAGE_ROOT / "output/0614/recurrent_effectiveness/pilot_01_norm/01dt_recurrent_online40_norm_lamstatic05_tau0.json",
    PACKAGE_ROOT / "output/0614/recurrent_effectiveness/two_stage_01/01dt_recurrent_from_static120_online80_tau05_lamstatic1.json",
    PACKAGE_ROOT / "output/0614/recurrent_effectiveness/two_stage_01/01dt_recurrent_from_static120_online80_tau085_lamstatic1.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/ablation_analysis"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def plus_minus(value: float) -> str:
    return f"{value:+.4f}"


def yn(value: Any) -> str:
    return "Y" if bool(value) else ""


def variant_switches(variant: str) -> dict[str, bool]:
    recurrent_variants = {"recurrent_memory_prompt", "recurrent_region_memory_prompt"}
    feature_variants = {"feature_memory_no_prompt", "feature_memory_prompt", "region_memory_prompt"} | recurrent_variants
    prompt_variants = {"feature_memory_prompt", "region_memory_prompt"} | recurrent_variants
    region_variants = {"region_memory_prompt", "recurrent_region_memory_prompt"}
    return {
        "class_memory": variant != "speech_baseline",
        "feature_stream": variant in feature_variants,
        "prompt_gate": variant in prompt_variants,
        "region_memory": variant in region_variants,
        "recurrent_memory": variant in recurrent_variants,
    }


def resolve_feature_path(spec: dict[str, Any]) -> Path:
    direct = spec.get("path")
    if direct:
        return Path(direct)
    candidates = sorted(PACKAGE_ROOT.glob(str(spec["glob"])))
    for candidate in candidates:
        if candidate.name.endswith((".debug.json", ".audit.json")):
            continue
        data = load_json(candidate)
        if str(data.get("variant", "")) != str(spec["label"]):
            continue
        expected_region = str(spec.get("holdout_region", ""))
        if expected_region and str(data.get("holdout_region", "")) != expected_region:
            continue
        return candidate
    raise FileNotFoundError(f"Could not resolve {spec['label']} with {spec['glob']}")


def collect_feature_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in FEATURE_RUNS:
        path = resolve_feature_path(spec)
        data = load_json(path)
        variant = str(data.get("variant", spec["label"]))
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        switches = variant_switches(variant)
        rows.append(
            {
                "label": spec["label"],
                "variant": variant,
                "path": str(path),
                "holdout_region": data.get("holdout_region", ""),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                "comparable": bool(spec["comparable"]),
                "notes": spec["notes"],
                "class_memory": switches["class_memory"],
                "feature_stream": switches["feature_stream"],
                "prompt_gate": switches["prompt_gate"],
                "region_memory": switches["region_memory"],
                "recurrent_memory": switches["recurrent_memory"],
                "region_supervision": variant == "region_memory_prompt",
                "region_compactness": variant == "region_memory_prompt",
                "max_steps": int(config.get("max_steps", 0) or 0),
                "lr": safe_float(config.get("lr")),
            }
        )
    baseline = next(row for row in rows if row["label"] == "speech_baseline")
    for row in rows:
        row["delta_vs_baseline"] = row["macro_f1"] - baseline["macro_f1"]
    return rows


def collect_feature_step_deltas(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {row["label"]: row for row in feature_rows}
    steps = [
        ("speech_baseline", "memory_only", "Add shared class memory"),
        ("memory_only", "feature_memory_no_prompt", "Add acoustic feature stream"),
        ("feature_memory_no_prompt", "feature_memory_prompt", "Add prompt gate"),
        ("feature_memory_prompt", "region_memory_prompt", "Add region memory"),
    ]
    rows = []
    for before_label, after_label, change in steps:
        before = by_label[before_label]
        after = by_label[after_label]
        rows.append(
            {
                "change": change,
                "before": before_label,
                "after": after_label,
                "macro_f1_delta": after["macro_f1"] - before["macro_f1"],
                "accuracy_delta": after["accuracy"] - before["accuracy"],
                "strictly_comparable": bool(before["comparable"] and after["comparable"]),
                "notes": after["notes"],
            }
        )
    return rows


def collect_paper_stage_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    with PAPER_SUMMARY.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            episode_macro = float(item["macro_f1"])
            global_macro = float(item["global_macro_f1"])
            episode_acc = float(item["accuracy"])
            global_acc = float(item["global_accuracy"])
            rows.append(
                {
                    "holdout_region": item["holdout_region"],
                    "stage": item["stage"],
                    "macro_f1": episode_macro,
                    "global_macro_f1": global_macro,
                    "episode_minus_global_macro_f1": episode_macro - global_macro,
                    "accuracy": episode_acc,
                    "global_accuracy": global_acc,
                    "episode_minus_global_accuracy": episode_acc - global_acc,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stage"])].append(row)

    means: dict[str, dict[str, float]] = {}
    for stage, stage_rows in grouped.items():
        means[stage] = {
            "count": float(len(stage_rows)),
            "macro_f1": sum(row["macro_f1"] for row in stage_rows) / len(stage_rows),
            "global_macro_f1": sum(row["global_macro_f1"] for row in stage_rows) / len(stage_rows),
            "episode_minus_global_macro_f1": sum(row["episode_minus_global_macro_f1"] for row in stage_rows) / len(stage_rows),
            "accuracy": sum(row["accuracy"] for row in stage_rows) / len(stage_rows),
            "global_accuracy": sum(row["global_accuracy"] for row in stage_rows) / len(stage_rows),
            "episode_minus_global_accuracy": sum(row["episode_minus_global_accuracy"] for row in stage_rows) / len(stage_rows),
        }
    return rows, means


def collect_paper_summary() -> dict[str, Any]:
    stage_rows, stage_means = collect_paper_stage_rows()
    stage3_report = load_json(PAPER_FINAL_STAGE3)
    single_region = load_json(PAPER_SINGLE_REGION)
    single_metrics = single_region.get("metrics", {})
    single_global = single_region.get("global_metrics", {})
    single_macro = float(single_metrics.get("macro_f1", 0.0))
    single_global_macro = float(single_global.get("macro_f1", 0.0))
    single_acc = float(single_metrics.get("accuracy", 0.0))
    single_global_acc = float(single_global.get("accuracy", 0.0))
    return {
        "all_stage_rows": stage_rows,
        "stage_means": stage_means,
        "hard4_stage3_mean": stage3_report.get("multiregion_stage3_mean", {}),
        "single_region_reference": {
            "holdout_region": single_region.get("holdout_region", TARGET_DANGTU),
            "macro_f1": single_macro,
            "global_macro_f1": single_global_macro,
            "episode_minus_global_macro_f1": single_macro - single_global_macro,
            "accuracy": single_acc,
            "global_accuracy": single_global_acc,
            "episode_minus_global_accuracy": single_acc - single_global_acc,
        },
    }


def collect_recurrent_summary() -> dict[str, Any]:
    rows = []
    for path in RECURRENT_CANDIDATES:
        if not path.exists():
            continue
        data = load_json(path)
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        rows.append(
            {
                "path": str(path),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "holdout_region": data.get("holdout_region", ""),
                "recurrent_eval_mode": config.get("recurrent_eval_mode", ""),
                "recurrent_online_threshold": safe_float(config.get("recurrent_online_threshold")),
                "max_steps": int(config.get("max_steps", 0) or 0),
            }
        )
    rows.sort(key=lambda item: item["macro_f1"], reverse=True)
    return {"runs": rows, "best": rows[0] if rows else None}


def build_conclusions(
    feature_rows: list[dict[str, Any]],
    feature_deltas: list[dict[str, Any]],
    paper: dict[str, Any],
    recurrent: dict[str, Any],
) -> list[str]:
    by_label = {row["label"]: row for row in feature_rows}
    deltas = {row["after"]: row["macro_f1_delta"] for row in feature_deltas}
    stage3 = paper["stage_means"].get("stage3", {})
    single = paper["single_region_reference"]
    best_recurrent = recurrent.get("best")
    conclusions = [
        f"\u5728 {TARGET_DANGTU} \u5bf9\u7167\u4e2d\uff0c\u5171\u4eab class memory \u662f\u9759\u6001\u7ebf\u91cc\u6700\u7a33\u7684\u589e\u76ca\u6e90\uff0cmacro-F1 \u76f8\u6bd4 speech baseline {plus_minus(deltas['memory_only'])}\u3002",
        f"\u52a0\u5165 acoustic feature stream \u540e\u6ca1\u6709\u51c0\u589e\u76ca\uff0cmacro-F1 \u76f8\u6bd4 memory_only {plus_minus(deltas['feature_memory_no_prompt'])}\u3002",
        f"prompt gate \u5728\u9759\u6001 feature-memory \u7ebf\u91cc\u4e5f\u6ca1\u6709\u8d21\u732e\u51c0\u6536\u76ca\uff0cmacro-F1 \u76f8\u6bd4 no-prompt {plus_minus(deltas['feature_memory_prompt'])}\u3002",
    ]
    if stage3:
        conclusions.append(
            "\u771f\u6b63\u660e\u663e\u8d77\u4f5c\u7528\u7684\u662f paper \u7ebf\u7684 support-driven episode class memory\uff1a"
            f"16 \u4e2a\u7559\u51fa\u65b9\u8a00 stage3 \u5747\u503c\u4e0a\uff0cepisode \u6bd4 global-only \u9ad8 {plus_minus(stage3['episode_minus_global_macro_f1'])} macro-F1\u3002"
        )
    prompt_row = by_label["feature_memory_prompt"]
    conclusions.append(
        f"{TARGET_DANGTU} \u4e0a\uff0cpaper \u53cc\u5c42 memory \u6700\u4f18\u5355\u533a run \u6bd4\u9759\u6001 feature_memory_prompt \u9ad8 {plus_minus(single['macro_f1'] - prompt_row['macro_f1'])} macro-F1\u3002"
    )
    if best_recurrent:
        conclusions.append(
            f"\u663e\u5f0f recurrent write-memory \u5728\u5df2\u6709\u8d85\u53c2\u4e0b\u4e0d\u7a33\uff0c\u6700\u597d\u7684\u73b0\u6210 run \u53ea\u6709 {best_recurrent['macro_f1']:.4f} macro-F1\u3002"
        )
    return conclusions


def make_report(
    feature_rows: list[dict[str, Any]],
    feature_deltas: list[dict[str, Any]],
    paper: dict[str, Any],
    recurrent: dict[str, Any],
) -> str:
    conclusions = build_conclusions(feature_rows, feature_deltas, paper, recurrent)
    lines: list[str] = [
        "# 0614 \u6d88\u878d\u6c47\u603b",
        "",
        f"\u751f\u6210\u65f6\u95f4: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. Feature-Memory \u4ee3\u7801\u5f00\u5173\u6d88\u878d",
        "",
        f"\u4e3b\u8981\u5bf9\u7167\u533a\u57df: `{TARGET_DANGTU}`\u3002\u6a21\u5757\u5f00\u5173\u6839\u636e `pc_dlcmnet.models.feature_memory::build_model_from_variant` \u63a8\u65ad\u3002",
        "",
        "| variant | macro_f1 | delta_vs_baseline | acc | class_memory | feature_stream | prompt | region_memory | recurrent |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in feature_rows:
        lines.append(
            "| {label} | {macro_f1:.4f} | {delta} | {accuracy:.4f} | {class_memory} | {feature_stream} | {prompt_gate} | {region_memory} | {recurrent_memory} |".format(
                label=row["label"],
                macro_f1=row["macro_f1"],
                delta=plus_minus(row["delta_vs_baseline"]),
                accuracy=row["accuracy"],
                class_memory=yn(row["class_memory"]),
                feature_stream=yn(row["feature_stream"]),
                prompt_gate=yn(row["prompt_gate"]),
                region_memory=yn(row["region_memory"]),
                recurrent_memory=yn(row["recurrent_memory"]),
            )
        )
    lines.extend(
        [
            "",
            "### \u9010\u6b65\u589e\u91cf",
            "",
            "| change | macro_f1_delta | acc_delta | strict_control |",
            "|---|---:|---:|---|",
        ]
    )
    for row in feature_deltas:
        lines.append(
            "| {change} | {mf1} | {acc} | {strict} |".format(
                change=row["change"],
                mf1=plus_minus(row["macro_f1_delta"]),
                acc=plus_minus(row["accuracy_delta"]),
                strict="Y" if row["strictly_comparable"] else "approx",
            )
        )
    lines.extend(
        [
            "",
            "\u5907\u6ce8: `region_memory_prompt` \u8fd9\u6761\u73b0\u6709 run \u540c\u65f6\u542f\u7528\u4e86 region supervision \u548c compactness loss\uff0c\u56e0\u6b64\u53ea\u80fd\u4f5c\u4e3a\u8fd1\u4f3c\u5bf9\u7167\u3002",
            "",
            "## 2. Paper \u53cc\u5c42 Memory \u529f\u80fd\u6d88\u878d",
            "",
            "`metrics` = global + episode memory\uff1b`global_metrics` = global-only\u3002\u540c\u4e00\u6a21\u578b\u5185\u7684\u8fd9\u4e2a\u5dee\u503c\u6700\u80fd\u8bf4\u660e episode memory \u672c\u8eab\u7684\u4f5c\u7528\u3002",
            "",
            "| stage | mean_macro_f1 | mean_global_macro_f1 | episode_minus_global | mean_acc | mean_global_acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in ["stage1", "stage2", "stage3"]:
        row = paper["stage_means"].get(stage)
        if row:
            lines.append(
                "| {stage} | {macro:.4f} | {global_macro:.4f} | {delta:.4f} | {acc:.4f} | {global_acc:.4f} |".format(
                    stage=stage,
                    macro=row["macro_f1"],
                    global_macro=row["global_macro_f1"],
                    delta=row["episode_minus_global_macro_f1"],
                    acc=row["accuracy"],
                    global_acc=row["global_accuracy"],
                )
            )
    hard4 = paper["hard4_stage3_mean"]
    single = paper["single_region_reference"]
    if hard4:
        lines.extend(
            [
                "",
                "Hard-4 stage3 \u5747\u503c: "
                f"macro-F1={hard4.get('macro_f1', 0.0):.4f}, "
                f"global-only macro-F1={hard4.get('global_macro_f1', 0.0):.4f}, "
                f"delta={plus_minus(float(hard4.get('macro_f1', 0.0)) - float(hard4.get('global_macro_f1', 0.0)))}.",
            ]
        )
    lines.extend(
        [
            "",
            f"{TARGET_DANGTU} \u5355\u533a\u6700\u4f18 paper run: episode macro-F1={single['macro_f1']:.4f}, global-only macro-F1={single['global_macro_f1']:.4f}, delta={plus_minus(single['episode_minus_global_macro_f1'])}.",
            "",
            "## 3. Recurrent Memory \u73b0\u72b6",
            "",
        ]
    )
    if recurrent["runs"]:
        lines.extend(["| file | macro_f1 | acc | eval_mode | max_steps |", "|---|---:|---:|---|---:|"])
        for row in recurrent["runs"]:
            lines.append(
                "| {file} | {macro_f1:.4f} | {accuracy:.4f} | {mode} | {max_steps} |".format(
                    file=Path(row["path"]).name,
                    macro_f1=row["macro_f1"],
                    accuracy=row["accuracy"],
                    mode=row["recurrent_eval_mode"],
                    max_steps=row["max_steps"],
                )
            )
    else:
        lines.append("\u6ca1\u6709\u53d1\u73b0\u53ef\u7528\u7684 recurrent \u7ed3\u679c\u3002")
    lines.extend(["", "## 4. \u7ed3\u8bba", ""])
    lines.extend(f"- {text}" for text in conclusions)
    lines.extend(
        [
            "",
            "## 5. \u6ce8\u610f\u4e8b\u9879",
            "",
            "- \u8fd9\u4efd\u6c47\u603b\u57fa\u4e8e\u4ed3\u5e93\u73b0\u6709\u7ed3\u679c\uff0c\u672c\u673a\u5f53\u524d\u53ea\u6709 CPU\uff0c\u672a\u65b0\u589e\u957f\u65f6\u95f4\u5b8c\u6574\u91cd\u8bad\u7f51\u683c\u3002",
            "- \u82e5\u8981\u4e25\u683c\u786e\u8ba4 region memory \u6216 recurrent memory\uff0c\u9700\u8981\u540c holdout\u3001\u540c\u6b65\u6570\u3001\u540c\u91c7\u6837\u7b56\u7565\u8865\u9f50\u5bf9\u7167\u3002",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = collect_feature_runs()
    feature_deltas = collect_feature_step_deltas(feature_rows)
    paper = collect_paper_summary()
    recurrent = collect_recurrent_summary()

    report = make_report(feature_rows, feature_deltas, paper, recurrent)
    (output_dir / "ablation_report.md").write_text(report, encoding="utf-8")
    write_csv(output_dir / "feature_ablation.csv", feature_rows)
    write_csv(output_dir / "feature_step_deltas.csv", feature_deltas)
    write_csv(output_dir / "paper_stage_rows.csv", paper["all_stage_rows"])
    payload = {
        "feature_ablation": feature_rows,
        "feature_step_deltas": feature_deltas,
        "paper_dual_memory": paper,
        "recurrent_memory": recurrent,
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report": str(output_dir / "ablation_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
