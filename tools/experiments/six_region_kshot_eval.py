#!/usr/bin/env python3
"""Run six-region evaluation-time k-shot sensitivity curves.

This script reuses trained stage-3 checkpoints and only varies
``eval_support_shots``. By default it fixes the query set, averages
labeled support features into class prototypes, and reports query-only
metrics, which matches the standard k-shot evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"
DEFAULT_REGIONS = ["01当涂", "03池州", "06铜陵", "08泾县", "11黄山", "13高淳"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/kshot_six_region_eval"))
    parser.add_argument("--checkpoint-root", default=str(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 8, 12])
    parser.add_argument("--eval-ensemble-runs", type=int, default=5)
    parser.add_argument("--support-write-mode", choices=["label", "pseudo", "blend"], default="label")
    parser.add_argument("--support-label-blend", type=float, default=1.0)
    parser.add_argument("--eval-query-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-split-mode", choices=["rolling", "fixed_query"], default="fixed_query")
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--eval-classifier", choices=["memory", "prototype"], default="prototype")
    parser.add_argument("--prototype-fallback-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prototype-support-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def region_stem(region: str) -> str:
    prefix_match = re.match(r"^(\d+)", str(region).strip())
    prefix = prefix_match.group(1) if prefix_match else "region"
    digest = hashlib.sha1(str(region).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_is_usable(path: Path, expected_k: int, expected_region: str, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    metrics = data.get("metrics", {})
    config = data.get("config", {})
    if "macro_f1" not in metrics:
        return False
    diag = data.get("diagnostics", {}).get("test", {})
    return (
        str(config.get("holdout_region", "")) == expected_region
        and int(config.get("eval_support_shots", -1)) == expected_k
        and str(config.get("support_write_mode", diag.get("support_write_mode", ""))) == str(args.support_write_mode)
        and bool(config.get("eval_query_only", diag.get("eval_query_only", False))) == bool(args.eval_query_only)
        and str(config.get("eval_split_mode", diag.get("eval_split_mode", ""))) == str(args.eval_split_mode)
        and str(config.get("eval_classifier", diag.get("eval_classifier", ""))) == str(args.eval_classifier)
        and abs(float(config.get("prototype_support_weight", diag.get("prototype_support_weight", 1.0))) - float(args.prototype_support_weight)) <= 1e-9
    )


def find_stage3_artifacts(region: str, checkpoint_root: Path) -> tuple[Path, Path]:
    stem = region_stem(region)
    direct_json = checkpoint_root / stem / f"{stem}_stage3.json"
    direct_pt = checkpoint_root / stem / f"{stem}_stage3.pt"
    if direct_json.exists() and direct_pt.exists():
        return direct_json, direct_pt

    for json_path in checkpoint_root.glob("*/*_stage3.json"):
        try:
            data = load_json(json_path)
        except Exception:
            continue
        config = data.get("config", {})
        split = data.get("split", {})
        if config.get("holdout_region") == region or split.get("holdout_region") == region:
            checkpoint = data.get("files", {}).get("checkpoint") or str(json_path.with_suffix(".pt"))
            pt_path = Path(checkpoint)
            if not pt_path.is_absolute():
                pt_path = PACKAGE_ROOT / pt_path
            if pt_path.exists():
                return json_path, pt_path
    raise FileNotFoundError(f"stage3 checkpoint not found for region: {region}")


def cfg(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get("config", {}).get(key, default)


def build_eval_command(
    args: argparse.Namespace,
    region: str,
    k: int,
    stage3_json: Path,
    stage3_pt: Path,
    output_json: Path,
) -> list[str]:
    data = load_json(stage3_json)
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--max-steps",
        "0",
        "--init-checkpoint",
        str(stage3_pt),
        "--holdout-region",
        region,
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--episode-length",
        str(cfg(data, "episode_length", 32)),
        "--episode-batch-size",
        str(cfg(data, "episode_batch_size", 2)),
        "--support-shots",
        str(cfg(data, "support_shots", 2)),
        "--eval-support-shots",
        str(k),
        "--eval-ensemble-runs",
        str(args.eval_ensemble_runs),
        "--support-write-mode",
        str(args.support_write_mode),
        "--support-label-blend",
        str(args.support_label_blend),
        "--eval-split-mode",
        str(args.eval_split_mode),
        "--eval-query-shots-per-class",
        str(args.eval_query_shots_per_class),
        "--eval-classifier",
        str(args.eval_classifier),
        "--prototype-support-weight",
        str(args.prototype_support_weight),
        "--lambda-global",
        str(cfg(data, "lambda_global", 0.5)),
        "--hidden-dim",
        str(cfg(data, "hidden_dim", 256)),
        "--score-dim",
        str(cfg(data, "score_dim", 128)),
        "--prompt-dim",
        str(cfg(data, "prompt_dim", 128)),
        "--num-prompts",
        str(cfg(data, "num_prompts", 8)),
        "--layers",
        str(cfg(data, "layers", 3)),
        "--heads",
        str(cfg(data, "heads", 4)),
        "--ffn-dim",
        str(cfg(data, "ffn_dim", 768)),
        "--dropout",
        str(cfg(data, "dropout", 0.1)),
        "--temperature",
        str(cfg(data, "temperature", 0.2)),
        "--token-chunks",
        str(cfg(data, "token_chunks", 32)),
        "--aux-source",
        str(cfg(data, "aux_source", "acoustic")),
        "--aux-representation",
        str(cfg(data, "aux_representation", "sequence")),
        "--aux-cache-dir",
        str(cfg(data, "aux_cache_dir", "output/0614/feature_memory_aux_cache")),
        "--output",
        str(output_json),
    ]
    command.append("--eval-query-only" if bool(args.eval_query_only) else "--no-eval-query-only")
    command.append("--prototype-fallback-global" if bool(args.prototype_fallback_global) else "--no-prototype-fallback-global")
    for optional in ["csv", "cache_dir", "matrix_cache_dir", "audio_root"]:
        value = cfg(data, optional, "")
        if value:
            command.extend([f"--{optional.replace('_', '-')}", str(value)])
    return command


def collect_result(region: str, k: int, output_json: Path, log_path: Path, status: str, returncode: int, command: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": region,
        "k": int(k),
        "status": status,
        "returncode": int(returncode),
        "result_file": str(output_json),
        "log_file": str(log_path),
        "command": subprocess.list2cmdline(command),
    }
    if not output_json.exists():
        return row
    try:
        data = load_json(output_json)
    except Exception as exc:
        row["error"] = str(exc)
        return row
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    diag = data.get("diagnostics", {}).get("test", {})
    row.update(
        {
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "global_accuracy": global_metrics.get("accuracy", ""),
            "global_macro_f1": global_metrics.get("macro_f1", ""),
            "episode_gain_macro_f1": (
                float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0))
                if metrics and global_metrics
                else ""
            ),
            "support_fraction": diag.get("support_fraction", ""),
            "query_fraction": diag.get("query_fraction", ""),
            "mean_query_size": diag.get("mean_query_size", ""),
            "evaluated_fraction": diag.get("evaluated_fraction", ""),
            "eval_query_only": diag.get("eval_query_only", ""),
            "eval_split_mode": diag.get("eval_split_mode", ""),
            "eval_classifier": diag.get("eval_classifier", ""),
            "prototype_support_weight": diag.get("prototype_support_weight", ""),
            "support_write_mode": diag.get("support_write_mode", ""),
            "support_label_blend": diag.get("support_label_blend", ""),
            "write_gate_mean": diag.get("write_gate_mean", ""),
            "prompt_entropy": diag.get("episode_prompt_route_entropy", ""),
            "stage3_checkpoint": data.get("init_checkpoint", {}).get("path", ""),
        }
    )
    return row


def upsert_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    replaced = False
    key = (str(new_row.get("region")), int(new_row.get("k", -1)))
    for row in rows:
        row_key = (str(row.get("region")), int(row.get("k", -1)))
        if row_key == key:
            merged.append(new_row)
            replaced = True
        else:
            merged.append(row)
    if not replaced:
        merged.append(new_row)
    return merged


def write_csv_json(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def write_markdown(rows: list[dict[str, Any]], output_dir: Path) -> None:
    done = [row for row in rows if row.get("status") in {"done", "skipped"}]
    lines = [
        "# Six-Region K-Shot Evaluation",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This experiment loads each region's stage-3 checkpoint and varies only `eval_support_shots`.",
        "By default, each k uses the same fixed query set, support labels are averaged into class prototypes, and metrics are computed on query items only.",
        "",
        "## Region Curves",
        "",
        "| Region | k | Acc | Macro-F1 | Global Macro-F1 | Episode Gain | Support Fraction | Query Fraction | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("region")), int(r.get("k", 0)))):
        lines.append(
            "| {region} | {k} | {acc} | {mf1} | {gmf1} | {gain} | {support} | {query} | {status} |".format(
                region=row.get("region", ""),
                k=row.get("k", ""),
                acc=pct(row.get("accuracy", "")),
                mf1=pct(row.get("macro_f1", "")),
                gmf1=pct(row.get("global_macro_f1", "")),
                gain=pct(row.get("episode_gain_macro_f1", "")),
                support=pct(row.get("support_fraction", "")),
                query=pct(row.get("query_fraction", row.get("evaluated_fraction", ""))),
                status=row.get("status", ""),
            )
        )
    if done:
        frame = pd.DataFrame(done)
        frame["macro_f1"] = pd.to_numeric(frame["macro_f1"], errors="coerce")
        frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
        frame["episode_gain_macro_f1"] = pd.to_numeric(frame["episode_gain_macro_f1"], errors="coerce")
        frame["k"] = pd.to_numeric(frame["k"], errors="coerce")
        mean = frame.groupby("k", as_index=False)[["macro_f1", "accuracy", "episode_gain_macro_f1"]].mean()
        lines.extend(["", "## Mean Across Regions", "", "| k | Acc | Macro-F1 | Episode Gain |", "|---:|---:|---:|---:|"])
        for _, row in mean.sort_values("k").iterrows():
            lines.append(
                f"| {int(row['k'])} | {row['accuracy'] * 100:.2f} | {row['macro_f1'] * 100:.2f} | {row['episode_gain_macro_f1'] * 100:.2f} |"
            )
        best_idx = mean["macro_f1"].idxmax()
        best = mean.loc[best_idx]
        lines.extend(
            [
                "",
                "## Paper Note",
                "",
                (
                    f"Across the selected six regions, the best average k-shot setting is `k={int(best['k'])}` "
                    f"with Macro-F1 `{best['macro_f1'] * 100:.2f}%`. The curve uses a fixed query set, "
                    "so changes across k mainly reflect the number of labeled support prototypes."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Figures:",
            "- `plots/kshot_macro_f1_by_region.png`",
            "- `plots/kshot_macro_f1_six_region_panels.png`",
            "- `plots/kshot_macro_f1_mean.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame([row for row in rows if row.get("status") in {"done", "skipped"}])
    if frame.empty:
        return
    for col in ["k", "macro_f1", "accuracy", "global_macro_f1", "episode_gain_macro_f1"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for region, part in frame.groupby("region", sort=False):
        part = part.sort_values("k")
        ax.plot(part["k"], part["macro_f1"] * 100.0, marker="o", linewidth=2.0, label=region)
    ax.set_title("K-shot sensitivity across six holdout regions")
    ax.set_xlabel("Evaluation support shots (k)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(plot_dir / "kshot_macro_f1_by_region.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    regions = list(dict.fromkeys(frame["region"].astype(str).tolist()))
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=True)
    axes_flat = axes.flatten()
    for ax, region in zip(axes_flat, regions):
        part = frame[frame["region"] == region].sort_values("k")
        ax.plot(part["k"], part["macro_f1"] * 100.0, marker="o", linewidth=2.1, label="Episode RW")
        ax.plot(part["k"], part["global_macro_f1"] * 100.0, marker="s", linewidth=1.8, label="Global-only")
        best = part.loc[part["macro_f1"].idxmax()]
        ax.scatter([best["k"]], [best["macro_f1"] * 100.0], s=80, facecolors="white", edgecolors="black", zorder=5)
        ax.set_title(str(region))
        ax.set_xlabel("k")
        ax.set_ylabel("Macro-F1 (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(regions) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower right", bbox_to_anchor=(0.98, 0.01))
    fig.suptitle("Six-region k-shot sensitivity panels", y=0.995)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])
    fig.savefig(plot_dir / "kshot_macro_f1_six_region_panels.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    mean = frame.groupby("k", as_index=False)["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = mean["k"].to_numpy(dtype=float)
    y = mean["mean"].to_numpy(dtype=float) * 100.0
    std = mean["std"].fillna(0.0).to_numpy(dtype=float) * 100.0
    ax.plot(x, y, marker="o", linewidth=2.3, color="#1f77b4")
    ax.fill_between(x, y - std, y + std, color="#1f77b4", alpha=0.16, linewidth=0)
    best = mean.loc[mean["mean"].idxmax()]
    ax.scatter([best["k"]], [best["mean"] * 100.0], s=90, facecolors="white", edgecolors="black", zorder=5)
    ax.set_title("Mean k-shot sensitivity across six regions")
    ax.set_xlabel("Evaluation support shots (k)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(plot_dir / "kshot_macro_f1_mean.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def run_job(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return int(process.returncode)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3: dict[str, tuple[Path, Path]] = {}
    for region in args.regions:
        stage3[region] = find_stage3_artifacts(region, checkpoint_root)

    commands: list[str] = []
    jobs: list[dict[str, Any]] = []
    for region in args.regions:
        stage3_json, stage3_pt = stage3[region]
        safe_region = region_stem(region)
        for k in args.k_values:
            output_json = output_dir / "results" / safe_region / f"{safe_region}_k{k}.json"
            log_path = output_dir / "logs" / safe_region / f"{safe_region}_k{k}.log"
            command = build_eval_command(args, region, int(k), stage3_json, stage3_pt, output_json)
            commands.append(subprocess.list2cmdline(command))
            jobs.append(
                {
                    "region": region,
                    "k": int(k),
                    "output_json": output_json,
                    "log_path": log_path,
                    "command": command,
                }
            )
    (output_dir / "commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")

    rows = load_existing_rows(output_dir)
    for idx, job in enumerate(jobs, start=1):
        region = str(job["region"])
        k = int(job["k"])
        output_json = Path(job["output_json"])
        log_path = Path(job["log_path"])
        command = list(job["command"])
        if bool(args.skip_existing) and result_is_usable(output_json, k, region, args):
            row = collect_result(region, k, output_json, log_path, "skipped", 0, command)
            rows = upsert_row(rows, row)
            print(f"[{idx}/{len(jobs)}] skipped {region} k={k}", flush=True)
        else:
            print(f"[{idx}/{len(jobs)}] running {region} k={k}", flush=True)
            returncode = run_job(command, log_path, bool(args.dry_run))
            status = "dry_run" if bool(args.dry_run) else ("done" if returncode == 0 and output_json.exists() else "failed")
            row = collect_result(region, k, output_json, log_path, status, returncode, command)
            rows = upsert_row(rows, row)
            if returncode != 0 and bool(args.stop_on_failure):
                write_csv_json(rows, output_dir)
                write_markdown(rows, output_dir)
                plot_results(rows, output_dir)
                raise RuntimeError(f"failed {region} k={k}; see {log_path}")
        write_csv_json(rows, output_dir)
        write_markdown(rows, output_dir)
        plot_results(rows, output_dir)

    write_csv_json(rows, output_dir)
    write_markdown(rows, output_dir)
    plot_results(rows, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "jobs": len(jobs),
                "summary": str(output_dir / "summary.csv"),
                "figures": str(output_dir / "plots"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
