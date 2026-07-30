#!/usr/bin/env python3
"""Run and summarize feature-memory experiment grids."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
RUNNER = PACKAGE_ROOT / "pc_dlcmnet/training/supervised.py"

from pc_dlcmnet.utils.paths import default_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(default_dataset()))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--regions", nargs="*", default=["01当涂", "08泾县", "11黄山", "13高淳"])
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "speech_baseline",
            "memory_only",
            "feature_memory_no_prompt",
            "feature_memory_prompt",
            "region_memory_prompt",
        ],
    )
    parser.add_argument("--train-fractions", nargs="*", type=float, default=[1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--anchor-kind", default="onehot", choices=["none", "random", "onehot", "ipa"])
    parser.add_argument("--aux-source", default="auto", choices=["auto", "acoustic", "whisper_stats", "old_fusion_cache"])
    parser.add_argument("--aux-representation", default="auto", choices=["auto", "sequence", "stats"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def safe_name(value: object) -> str:
    text = str(value)
    for ch in "\\/:*?\"<>| ":
        text = text.replace(ch, "_")
    return text


def resolve_regions(args: argparse.Namespace) -> list[str]:
    if len(args.regions) == 1 and args.regions[0].lower() == "all":
        df = pd.read_csv(args.csv)
        if "region" not in df.columns:
            raise ValueError("CSV has no region column.")
        return sorted(df["region"].dropna().astype(str).unique().tolist())
    return [str(region) for region in args.regions]


def run_job(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(process.returncode)


def collect_result(path: Path, status: str, returncode: int, log_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "returncode": int(returncode),
        "result_file": str(path),
        "log_file": str(log_path),
    }
    if not path.exists():
        return row
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = str(exc)
        return row
    metrics = data.get("metrics", {})
    val_metrics = data.get("val_metrics", {})
    diag = data.get("diagnostics", {}).get("test", {})
    cache = data.get("cache", {})
    aux_train_meta = cache.get("aux", {}).get("train", {})
    aux_source = aux_train_meta.get("source", "")
    aux_representation = aux_train_meta.get("representation", "")
    split_audit = data.get("split_audit", {})
    method_audit = data.get("method_audit", {})
    row.update(
        {
            "task": data.get("task", ""),
            "method": data.get("method", ""),
            "variant": data.get("variant", ""),
            "anchor_kind": data.get("anchor_kind", ""),
            "holdout_region": data.get("holdout_region", ""),
            "train_fraction": data.get("train_fraction", ""),
            "seed": data.get("seed", ""),
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "micro_f1": metrics.get("micro_f1", ""),
            "val_accuracy": val_metrics.get("accuracy", ""),
            "val_macro_f1": val_metrics.get("macro_f1", ""),
            "memory_residual_norm": diag.get("memory_residual_norm", ""),
            "candidate_memory_norm": diag.get("candidate_memory_norm", ""),
            "write_gate_mean": diag.get("write_gate_mean", ""),
            "write_update_norm": diag.get("write_update_norm", ""),
            "write_weight_active": diag.get("write_weight_active", ""),
            "feature_bias_norm": diag.get("feature_bias_norm", ""),
            "feature_gate_mean": diag.get("feature_gate_mean", ""),
            "region_attention_entropy": diag.get("region_attention_entropy", ""),
            "region_attention_max": diag.get("region_attention_max", ""),
            "region_context_norm": diag.get("region_context_norm", ""),
            "prompt_q_gate_std": diag.get("prompt_q_gate_std", ""),
            "prompt_k_gate_std": diag.get("prompt_k_gate_std", ""),
            "prompt_v_gate_std": diag.get("prompt_v_gate_std", ""),
            "aux_source": aux_source,
            "aux_representation": aux_representation,
            "train_size": data.get("split_sizes", {}).get("train", ""),
            "val_size": data.get("split_sizes", {}).get("val", ""),
            "test_size": data.get("split_sizes", {}).get("test", ""),
            "num_source_pool_regions": split_audit.get("num_source_pool_regions", ""),
            "num_train_regions": split_audit.get("num_train_regions", ""),
            "target_rows_in_train": split_audit.get("target_rows_in_train", ""),
            "target_rows_in_val": split_audit.get("target_rows_in_val", ""),
            "test_is_holdout_only": split_audit.get("test_is_holdout_only", ""),
            "train_regions_cover_all_non_target_regions": split_audit.get("train_regions_cover_all_non_target_regions", ""),
            "class_memory_inside_transformer": method_audit.get("class_memory_inside_transformer", ""),
            "feature_stream_enabled": method_audit.get("feature_stream_enabled", ""),
            "prompt_attention_gate_enabled": method_audit.get("prompt_attention_gate_enabled", ""),
            "region_memory_enabled": method_audit.get("region_memory_enabled", ""),
            "recurrent_memory_enabled": method_audit.get("recurrent_memory_enabled", ""),
            "gated_memory_write_enabled": method_audit.get("gated_memory_write_enabled", ""),
            "online_test_time_memory_enabled": method_audit.get("online_test_time_memory_enabled", ""),
            "region_conditioned_prompt_enabled": method_audit.get("region_conditioned_prompt_enabled", ""),
            "region_supervision_enabled": method_audit.get("region_supervision_enabled", ""),
            "domain_balanced_sampling_enabled": method_audit.get("domain_balanced_sampling_enabled", ""),
            "class_balanced_sampling_enabled": method_audit.get("class_balanced_sampling_enabled", ""),
            "sampler": data.get("training", {}).get("sampler", {}).get("sampler", ""),
        }
    )
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "| region | fraction | variant | acc | macro_f1 | val_macro_f1 | status |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("holdout_region", "")), str(r.get("train_fraction", "")), str(r.get("variant", "")))):
        md_lines.append(
            "| {region} | {frac} | {variant} | {acc} | {mf1} | {vmf1} | {status} |".format(
                region=row.get("holdout_region", ""),
                frac=row.get("train_fraction", ""),
                variant=row.get("variant", ""),
                acc=format_float(row.get("accuracy", "")),
                mf1=format_float(row.get("macro_f1", "")),
                vmf1=format_float(row.get("val_macro_f1", "")),
                status=row.get("status", ""),
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    try:
        make_plots(rows, output_dir)
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def format_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import warnings

    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    valid = [row for row in rows if row.get("status") in {"done", "skipped"} and row.get("macro_f1") != ""]
    if not valid:
        return
    df = pd.DataFrame(valid)
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    for metric in ["macro_f1", "accuracy"]:
        pivot = df.pivot_table(index="holdout_region", columns="variant", values=metric, aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(11, 4), rot=30)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(0.05, min(1.0, float(pivot.max().max()) + 0.1)))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_region.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PACKAGE_ROOT / f"output/0614/feature_memory_grid/{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = resolve_regions(args)
    rows: list[dict[str, Any]] = []
    for region in regions:
        for fraction in args.train_fractions:
            for variant in args.variants:
                tag = f"{safe_name(region)}_frac{fraction:g}_{variant}_seed{args.seed}"
                result_path = output_dir / "results" / f"{tag}.json"
                log_path = output_dir / "logs" / f"{tag}.log"
                if args.skip_existing and result_path.exists():
                    rows.append(collect_result(result_path, "skipped", 0, log_path))
                    write_summary(rows, output_dir)
                    continue
                command = [
                    args.python,
                    "-u",
                    str(RUNNER),
                    "--csv",
                    str(args.csv),
                    "--output",
                    str(result_path),
                    "--holdout-region",
                    str(region),
                    "--variant",
                    str(variant),
                    "--train-fraction",
                    str(fraction),
                    "--seed",
                    str(args.seed),
                    "--device",
                    str(args.device),
                    "--max-steps",
                    str(args.max_steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--eval-every",
                    str(args.eval_every),
                    "--anchor-kind",
                    str(args.anchor_kind),
                    "--aux-source",
                    str(args.aux_source),
                    "--aux-representation",
                    str(args.aux_representation),
                ]
                command.extend(args.extra_args)
                print(f"[grid] running {tag}", flush=True)
                returncode = run_job(command, log_path, args.dry_run)
                status = "dry_run" if args.dry_run else ("done" if returncode == 0 and result_path.exists() else "failed")
                row = collect_result(result_path, status, returncode, log_path)
                row.setdefault("holdout_region", str(region))
                row.setdefault("train_fraction", float(fraction))
                row.setdefault("variant", str(variant))
                row.setdefault("seed", int(args.seed))
                rows.append(row)
                write_summary(rows, output_dir)
                if returncode != 0:
                    print(f"[grid] failed {tag}; see {log_path}", flush=True)
    write_summary(rows, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
