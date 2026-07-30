#!/usr/bin/env python3
"""Export prompt routing weights from training curves into one long CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="Experiment directory containing */*/curve.csv files.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to run_root/prompt_route_weights_by_iteration.csv.",
    )
    return parser.parse_args()


def numeric_or_blank(value: str | None) -> float | str:
    if value is None or value == "":
        return ""
    try:
        parsed = float(value)
    except ValueError:
        return ""
    if math.isnan(parsed):
        return ""
    return parsed


def parse_setting(curve_path: Path, run_root: Path) -> tuple[str, str, str]:
    rel = curve_path.relative_to(run_root)
    setting = rel.parts[0] if len(rel.parts) >= 3 else ""
    area = rel.parts[1] if len(rel.parts) >= 3 else curve_path.parent.name
    if "_" in setting:
        sweep, value = setting.rsplit("_", 1)
    else:
        sweep, value = setting, ""
    return setting, sweep, value, area


def prompt_columns(fieldnames: list[str]) -> list[str]:
    cols = [name for name in fieldnames if name.startswith("prompt_route_mean_")]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[-1]))


def export(run_root: Path, output: Path) -> int:
    rows = 0
    fieldnames = [
        "setting",
        "sweep",
        "value",
        "area",
        "step",
        "epoch",
        "prompt",
        "routing_weight",
        "prompt_route_entropy",
        "val_prompt_route_entropy",
        "curve_file",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for curve_path in sorted(run_root.glob("*/*/curve.csv")):
            setting, sweep, value, area = parse_setting(curve_path, run_root)
            with curve_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                cols = prompt_columns(reader.fieldnames)
                for row in reader:
                    for col in cols:
                        alpha = numeric_or_blank(row.get(col))
                        if alpha == "":
                            continue
                        writer.writerow(
                            {
                                "setting": setting,
                                "sweep": sweep,
                                "value": value,
                                "area": area,
                                "step": numeric_or_blank(row.get("step")),
                                "epoch": numeric_or_blank(row.get("epoch")),
                                "prompt": int(col.rsplit("_", 1)[-1]),
                                "routing_weight": alpha,
                                "prompt_route_entropy": numeric_or_blank(row.get("prompt_route_entropy")),
                                "val_prompt_route_entropy": numeric_or_blank(row.get("val_prompt_route_entropy")),
                                "curve_file": str(curve_path),
                            }
                        )
                        rows += 1
    return rows


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output = Path(args.output) if args.output else run_root / "prompt_route_weights_by_iteration.csv"
    rows = export(run_root, output)
    print(f"wrote {rows} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
