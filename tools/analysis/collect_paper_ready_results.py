#!/usr/bin/env python3
"""Collect paper-ready tables and figures into one folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PACKAGE_ROOT / "output/0614/paper_ready_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, manifest: list[dict[str, Any]], description: str) -> None:
    if not src.exists():
        manifest.append({"source": str(src), "destination": str(dst), "status": "missing", "description": description})
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "source": str(src),
            "destination": str(dst),
            "status": "copied",
            "bytes": int(dst.stat().st_size),
            "description": description,
        }
    )


def copy_tree_files(src_dir: Path, dst_dir: Path, manifest: list[dict[str, Any]], description: str, suffixes: set[str] | None = None) -> None:
    if not src_dir.exists():
        manifest.append({"source": str(src_dir), "destination": str(dst_dir), "status": "missing", "description": description})
        return
    for src in sorted(path for path in src_dir.rglob("*") if path.is_file()):
        if suffixes is not None and src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(src_dir)
        copy_file(src, dst_dir / rel, manifest, description)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def build_master_summary(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    significance = output_dir / "05_supplementary_diagnostics" / "paired_significance.json"
    if significance.exists():
        stats = load_json(significance)
        for metric, row in stats.items():
            rows.append(
                {
                    "section": "paired_significance",
                    "item": metric,
                    "global": pct(row.get("mean_b")),
                    "episode_rw": pct(row.get("mean_a")),
                    "gain": pct(row.get("mean_diff")),
                    "ci95": f"[{pct(row.get('bootstrap_ci_low'))}, {pct(row.get('bootstrap_ci_high'))}]",
                    "positive_regions": f"{row.get('positive_count')}/{row.get('n')}",
                    "note": "Paired region-level test: Episode RW vs Global-only",
                }
            )

    region_csv = output_dir / "03_region_memory_gain" / "region_memory_gain.csv"
    if region_csv.exists():
        with region_csv.open(encoding="utf-8-sig", newline="") as f:
            done = [row for row in csv.DictReader(f) if row.get("status") == "done"]
        if done:
            mean_macro = sum(float(row["macro_f1"]) for row in done) / len(done)
            mean_global = sum(float(row["global_macro_f1"]) for row in done) / len(done)
            mean_acc = sum(float(row["accuracy"]) for row in done) / len(done)
            mean_global_acc = sum(float(row["global_accuracy"]) for row in done) / len(done)
            rows.append(
                {
                    "section": "region_memory_gain",
                    "item": "mean_16_regions_macro_f1",
                    "global": pct(mean_global),
                    "episode_rw": pct(mean_macro),
                    "gain": pct(mean_macro - mean_global),
                    "ci95": "",
                    "positive_regions": f"{sum(float(row['gain_macro_f1']) > 0 for row in done)}/{len(done)}",
                    "note": "Stage3 Global-only vs Global + Episode RW",
                }
            )
            rows.append(
                {
                    "section": "region_memory_gain",
                    "item": "mean_16_regions_accuracy",
                    "global": pct(mean_global_acc),
                    "episode_rw": pct(mean_acc),
                    "gain": pct(mean_acc - mean_global_acc),
                    "ci95": "",
                    "positive_regions": f"{sum(float(row['gain_accuracy']) > 0 for row in done)}/{len(done)}",
                    "note": "Stage3 Global-only vs Global + Episode RW",
                }
            )

    write_mode = output_dir / "04_memory_write_mode_ablation" / "summary.csv"
    if write_mode.exists():
        with write_mode.open(encoding="utf-8-sig", newline="") as f:
            rows_raw = [row for row in csv.DictReader(f) if row.get("status") in {"done", "skipped"} and row.get("macro_f1")]
        modes = sorted(set(row["mode"] for row in rows_raw))
        for mode in modes:
            part = [row for row in rows_raw if row["mode"] == mode]
            if not part:
                continue
            macro = sum(float(row["macro_f1"]) for row in part) / len(part)
            global_macro = sum(float(row["global_macro_f1"]) for row in part) / len(part)
            acc = sum(float(row["accuracy"]) for row in part) / len(part)
            global_acc = sum(float(row["global_accuracy"]) for row in part) / len(part)
            rows.append(
                {
                    "section": "memory_write_mode",
                    "item": mode,
                    "global": pct(global_macro),
                    "episode_rw": pct(macro),
                    "gain": pct(macro - global_macro),
                    "ci95": "",
                    "positive_regions": f"{sum(float(row['gain_macro_f1']) > 0 for row in part)}/{len(part)}",
                    "note": f"Accuracy {pct(acc)} vs global {pct(global_acc)}",
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row.keys()}) if rows else ["section", "item"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir: Path, manifest: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    copied = [row for row in manifest if row.get("status") == "copied"]
    missing = [row for row in manifest if row.get("status") == "missing"]
    figures = [row for row in copied if str(row.get("destination", "")).lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))]
    tables = [row for row in copied if str(row.get("destination", "")).lower().endswith((".csv", ".md", ".tex", ".json"))]

    lines = [
        "# Paper-Ready Results",
        "",
        f"Collected: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder consolidates the effective results used for the PC-DLCMNet paper experiments.",
        "",
        "## Folder Map",
        "",
        "| Folder | Contents |",
        "|---|---|",
        "| `01_main_ablation` | Paper ablation tables and module evidence. |",
        "| `02_parameter_sensitivity` | Parameter sensitivity tables and plots. |",
        "| `03_region_memory_gain` | 16-region Global-only vs Episode RW gain analysis. |",
        "| `04_memory_write_mode_ablation` | Pseudo/blend/label memory write strategy ablation. |",
        "| `05_supplementary_diagnostics` | Significance tests, prompt/feature diagnostics, and confusion matrices. |",
        "| `99_manifest` | Copy manifest, master summary, and image list. |",
        "",
        "## Key Numbers",
        "",
        "| Section | Item | Global | Episode RW | Gain | CI / Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('section', '')} | {row.get('item', '')} | {row.get('global', '')} | "
            f"{row.get('episode_rw', '')} | {row.get('gain', '')} | {row.get('ci95') or row.get('note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    for row in figures:
        dst = Path(str(row["destination"]))
        rel = dst.relative_to(output_dir)
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Tables And Reports",
            "",
        ]
    )
    for row in tables:
        dst = Path(str(row["destination"]))
        rel = dst.relative_to(output_dir)
        lines.append(f"- `{rel}`")
    if missing:
        lines.extend(["", "## Missing Sources", ""])
        for row in missing:
            lines.append(f"- `{row.get('source')}` ({row.get('description')})")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    manifest: list[dict[str, Any]] = []
    source_root = PACKAGE_ROOT / "output/0614"

    # 1. Ablation and paper tables.
    ablation = source_root / "ablation_analysis"
    for name in [
        "ablation_report.md",
        "ablation_summary.json",
        "feature_ablation.csv",
        "feature_step_deltas.csv",
        "paper_ablation_tables.md",
        "paper_ablation_tables.tex",
        "paper_stage_rows.csv",
    ]:
        copy_file(ablation / name, output_dir / "01_main_ablation" / name, manifest, "Ablation tables and reports")

    # 2. Parameter sensitivity.
    sensitivity = source_root / "parameter_sensitivity_grid_full"
    for name in [
        "summary.csv",
        "summary.json",
        "summary.md",
        "paper_parameter_sensitivity_full.md",
        "paper_parameter_sensitivity_full.tex",
    ]:
        copy_file(sensitivity / name, output_dir / "02_parameter_sensitivity" / name, manifest, "Parameter sensitivity tables")
    copy_tree_files(sensitivity / "plots", output_dir / "02_parameter_sensitivity" / "plots", manifest, "Parameter sensitivity figures", {".png", ".csv"})

    # 3. Region memory gain.
    region_gain = source_root / "region_memory_gain"
    copy_tree_files(region_gain, output_dir / "03_region_memory_gain", manifest, "Region memory gain analysis", {".csv", ".json", ".md", ".png"})

    # 4. Memory write mode ablation.
    write_mode = source_root / "memory_write_mode_ablation"
    for name in ["summary.csv", "summary.json", "summary.md"]:
        copy_file(write_mode / name, output_dir / "04_memory_write_mode_ablation" / name, manifest, "Memory write mode ablation")
    copy_tree_files(write_mode / "plots", output_dir / "04_memory_write_mode_ablation" / "plots", manifest, "Memory write mode figures", {".png"})

    # 5. Supplementary diagnostics.
    supplementary = source_root / "supplementary_diagnostics"
    copy_tree_files(supplementary, output_dir / "05_supplementary_diagnostics", manifest, "Supplementary diagnostics", {".csv", ".json", ".md", ".png"})

    # Keep pure prototype k-shot as appendix/diagnostic, not main evidence.
    kshot = source_root / "kshot_single_region_01dt_k1_10_split100_global"
    for name in ["summary.csv", "summary.json", "summary.md"]:
        copy_file(kshot / name, output_dir / "06_appendix_pure_prototype_kshot" / name, manifest, "Pure prototype k-shot diagnostic")
    copy_tree_files(kshot / "plots", output_dir / "06_appendix_pure_prototype_kshot" / "plots", manifest, "Pure prototype k-shot figures", {".png", ".csv"})

    summary_rows = build_master_summary(output_dir)
    write_csv(summary_rows, output_dir / "99_manifest" / "master_summary.csv")
    write_csv(manifest, output_dir / "99_manifest" / "copy_manifest.csv")
    image_rows = [
        {
            "figure": str(Path(str(row["destination"])).relative_to(output_dir)),
            "description": row.get("description", ""),
            "bytes": row.get("bytes", ""),
        }
        for row in manifest
        if row.get("status") == "copied" and str(row.get("destination", "")).lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    write_csv(image_rows, output_dir / "99_manifest" / "figure_index.csv")
    write_readme(output_dir, manifest, summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "copied": sum(row.get("status") == "copied" for row in manifest), "missing": sum(row.get("status") == "missing" for row in manifest)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
