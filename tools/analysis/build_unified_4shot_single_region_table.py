#!/usr/bin/env python3
"""Build a unified 4-shot table for one target region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def fmt(mean: Any, std: Any) -> str:
    try:
        return f"{float(mean) * 100.0:.2f} ± {float(std) * 100.0:.2f}"
    except Exception:
        return ""


def load_rc_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    metrics_std = data.get("metrics_std", {})
    return {
        "method": data.get("ablation_variant", data.get("method", "RC-MemNet")),
        "display_method": {
            "rc_memnet": "RC-MemNet",
            "rc_dmnet": "RC-MemNet",
            "no_context_prompt": "No context prompt",
            "no_memory_adapter": "No memory adapter",
            "no_prompt_no_memory": "No prompt + no memory",
            "no_prompt_routing": "No context prompt",
            "count_based_adaptation": "No memory adapter",
            "support_prototype": "Support prototype",
        }.get(str(data.get("ablation_variant", "")), data.get("method", "RC-MemNet")),
        "accuracy": metrics.get("accuracy", ""),
        "accuracy_std": metrics_std.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", ""),
        "macro_f1_std": metrics_std.get("macro_f1", 0.0),
        "source": str(path),
    }


def load_summary(path: Path, region_col: str, region_value: str, method_col: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    if region_col not in df.columns:
        raise ValueError(f"missing region column {region_col!r} in {path}")
    part = df[df[region_col].astype(str) == str(region_value)].copy()
    rows = []
    for _, row in part.iterrows():
        rows.append(
            {
                "method": str(row[method_col]),
                "display_method": str(row[method_col]),
                "accuracy": row.get("accuracy", ""),
                "accuracy_std": row.get("accuracy_std", 0.0),
                "macro_f1": row.get("macro_f1", ""),
                "macro_f1_std": row.get("macro_f1_std", 0.0),
                "source": str(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-label", required=True)
    parser.add_argument("--traditional-summary", default="")
    parser.add_argument("--hf-summary", default="")
    parser.add_argument("--rc-result", default="")
    parser.add_argument("--extra-results", nargs="*", default=[])
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.traditional_summary:
        rows.extend(load_summary(Path(args.traditional_summary), "area", args.region_label, "method"))
    if args.hf_summary:
        rows.extend(load_summary(Path(args.hf_summary), "area", args.region_label, "model"))
    if args.rc_result:
        rows.append(load_rc_result(Path(args.rc_result)))
    for item in args.extra_results:
        rows.append(load_rc_result(Path(item)))

    if not rows:
        raise SystemExit("no rows collected")

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["accuracy", "macro_f1"], ascending=False, na_position="last")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "single_region_results.csv", index=False, encoding="utf-8-sig")

    table = frame[["display_method", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std"]].copy()
    table.columns = ["Method", "Accuracy", "Accuracy Std", "Macro-F1", "Macro-F1 Std"]
    table["Accuracy"] = table.apply(lambda r: fmt(r["Accuracy"], r["Accuracy Std"]), axis=1)
    table["Macro-F1"] = table.apply(lambda r: fmt(r["Macro-F1"], r["Macro-F1 Std"]), axis=1)
    table = table.drop(columns=["Accuracy Std", "Macro-F1 Std"])
    table.to_csv(out_dir / "single_region_table.csv", index=False, encoding="utf-8-sig")

    lines = [f"# Unified 4-shot results for {args.region_label}", ""]
    lines.append("| Method | Accuracy | Macro-F1 |")
    lines.append("|---|---:|---:|")
    for _, row in table.iterrows():
        lines.append(f"| {row['Method']} | {row['Accuracy']} | {row['Macro-F1']} |")
    (out_dir / "single_region_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "rows": len(table)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
