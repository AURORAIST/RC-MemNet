#!/usr/bin/env python3
"""Build Accuracy/Macro-F1 tables from SSL fine-tuning summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def fmt(mean: float, std: float) -> str:
    if mean is None or std is None:
        return "—"
    try:
        if math.isnan(float(mean)) or math.isnan(float(std)):
            return "—"
    except Exception:
        return "—"
    return f"{mean * 100.0:.2f} ± {std * 100.0:.2f}"


def load_summary(summary_path: Path) -> pd.DataFrame:
    return pd.read_csv(summary_path, encoding="utf-8-sig")


def load_from_root(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(root.glob("**/result.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("metrics", {})
        metrics_std = data.get("metrics_std", {})
        zero_shot = data.get("zero_shot", {})
        rows.append(
            {
                "model": data.get("model"),
                "resolved_model": data.get("resolved_model"),
                "target": data.get("target"),
                "protocol": data.get("protocol"),
                "k": data.get("k"),
                "splits": data.get("splits"),
                "train_scheme": data.get("train_scheme"),
                "accuracy": metrics.get("accuracy"),
                "accuracy_std": metrics_std.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_f1_std": metrics_std.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "weighted_f1_std": metrics_std.get("weighted_f1"),
                "zero_shot_accuracy": zero_shot.get("accuracy"),
                "zero_shot_macro_f1": zero_shot.get("macro_f1"),
                "time_sec": data.get("time_sec"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no result.json files found under {root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary")
    parser.add_argument("--root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if bool(args.summary) == bool(args.root):
        raise SystemExit("provide exactly one of --summary or --root")

    df = load_summary(Path(args.summary)) if args.summary else load_from_root(Path(args.root))
    if "protocol" in df.columns:
        protocols = sorted({str(value) for value in df["protocol"].dropna().tolist()})
        if len(protocols) > 1:
            raise SystemExit(f"mixed protocols detected: {protocols}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(["model", "target"], inplace=True)
    df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    for metric in ["accuracy", "macro_f1"]:
        value_std = f"{metric}_std"
        pivot = df.pivot(index="model", columns="target", values=metric).copy()
        pivot_std = df.pivot(index="model", columns="target", values=value_std).copy()
        table = pivot.copy()
        for col in table.columns:
            table[col] = [fmt(float(pivot.loc[row, col]), float(pivot_std.loc[row, col])) for row in table.index]
        table.reset_index().rename(columns={"model": "Method"}).to_csv(out_dir / f"{metric}_table.csv", index=False, encoding="utf-8-sig")
        lines = [f"# {metric.upper()} table", ""]
        lines.append("| Method | " + " | ".join(table.columns) + " |")
        lines.append("|---|" + "---:|" * len(table.columns))
        for model in table.index:
            row = [str(model)] + [table.loc[model, col] for col in table.columns]
            lines.append("| " + " | ".join(row) + " |")
        (out_dir / f"{metric}_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
