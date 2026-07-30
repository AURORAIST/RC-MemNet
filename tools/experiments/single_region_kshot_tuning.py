#!/usr/bin/env python3
"""Tune fixed-query prototype k-shot evaluation on one holdout region."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"
DEFAULT_REGION = "01当涂"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/kshot_single_region_tuning"))
    parser.add_argument("--checkpoint-root", default=str(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 8, 12])
    parser.add_argument("--prototype-support-weights", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--eval-ensemble-runs", type=int, default=1)
    parser.add_argument("--eval-split-runs", type=int, default=1)
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--child-quiet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-child-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-val-eval", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_value(value: object) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch in {"p", "m"} else "_" for ch in text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def region_stem(region: str) -> str:
    import hashlib
    import re

    prefix_match = re.match(r"^(\d+)", str(region).strip())
    prefix = prefix_match.group(1) if prefix_match else "region"
    digest = hashlib.sha1(str(region).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


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


def build_command(
    args: argparse.Namespace,
    stage3_json: Path,
    stage3_pt: Path,
    k: int,
    support_weight: float,
    temperature: float,
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
        str(args.region),
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
        "--eval-split-runs",
        str(args.eval_split_runs),
        "--support-write-mode",
        "label",
        "--support-label-blend",
        "1.0",
        "--eval-query-only",
        "--eval-split-mode",
        "global_support",
        "--eval-query-shots-per-class",
        str(args.eval_query_shots_per_class),
        "--eval-classifier",
        "prototype",
        "--prototype-fallback-global",
        "--prototype-support-weight",
        str(support_weight),
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
        str(temperature),
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
    if bool(getattr(args, "child_quiet", True)):
        command.append("--quiet")
    command.append("--progress" if bool(getattr(args, "show_child_progress", True)) else "--no-progress")
    command.append("--skip-val-eval" if bool(getattr(args, "skip_val_eval", True)) else "--no-skip-val-eval")
    for optional in ["csv", "cache_dir", "matrix_cache_dir", "audio_root"]:
        value = cfg(data, optional, "")
        if value:
            command.extend([f"--{optional.replace('_', '-')}", str(value)])
    return command


def result_is_usable(path: Path, args: argparse.Namespace, k: int, support_weight: float, temperature: float) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    if "macro_f1" not in data.get("metrics", {}):
        return False
    config = data.get("config", {})
    diag = data.get("diagnostics", {}).get("test", {})
    split_runs = data.get("evaluation_splits", {}).get("test", {}).get("runs", config.get("eval_split_runs", diag.get("eval_split_runs", 1)))
    return (
        str(config.get("holdout_region")) == str(args.region)
        and int(config.get("eval_support_shots", -1)) == int(k)
        and int(split_runs or 1) == int(args.eval_split_runs)
        and bool(config.get("skip_val_eval", False)) == bool(args.skip_val_eval)
        and str(config.get("eval_split_mode", diag.get("eval_split_mode", ""))) == "global_support"
        and str(config.get("eval_classifier", diag.get("eval_classifier", ""))) == "prototype"
        and bool(config.get("eval_query_only", diag.get("eval_query_only", False)))
        and abs(float(config.get("prototype_support_weight", diag.get("prototype_support_weight", 1.0))) - float(support_weight)) <= 1e-9
        and abs(float(config.get("temperature", 0.0)) - float(temperature)) <= 1e-9
    )


def collect_result(
    args: argparse.Namespace,
    k: int,
    support_weight: float,
    temperature: float,
    output_json: Path,
    log_path: Path,
    status: str,
    returncode: int,
    command: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": args.region,
        "k": int(k),
        "prototype_support_weight": float(support_weight),
        "temperature": float(temperature),
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
    split_summary = data.get("evaluation_splits", {}).get("test", {})
    metrics_std = split_summary.get("metrics_std", {}) if isinstance(split_summary, dict) else {}
    global_metrics_std = split_summary.get("global_metrics_std", {}) if isinstance(split_summary, dict) else {}
    split_metrics = split_summary.get("metrics", []) if isinstance(split_summary, dict) else []
    split_global_metrics = split_summary.get("global_metrics", []) if isinstance(split_summary, dict) else []
    paired_gains = []
    if isinstance(split_metrics, list) and isinstance(split_global_metrics, list):
        for metric_row, global_row in zip(split_metrics, split_global_metrics):
            try:
                paired_gains.append(float(metric_row["macro_f1"]) - float(global_row["macro_f1"]))
            except Exception:
                pass
    row.update(
        {
            "accuracy": metrics.get("accuracy", ""),
            "accuracy_std": metrics_std.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "macro_f1_std": metrics_std.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "weighted_f1_std": metrics_std.get("weighted_f1", ""),
            "global_accuracy": global_metrics.get("accuracy", ""),
            "global_accuracy_std": global_metrics_std.get("accuracy", ""),
            "global_macro_f1": global_metrics.get("macro_f1", ""),
            "global_macro_f1_std": global_metrics_std.get("macro_f1", ""),
            "episode_gain_macro_f1": (
                float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0))
                if metrics and global_metrics
                else ""
            ),
            "episode_gain_macro_f1_std": statistics.stdev(paired_gains) if len(paired_gains) > 1 else (0.0 if paired_gains else ""),
            "eval_split_runs": split_summary.get("runs", diag.get("eval_split_runs", "")) if isinstance(split_summary, dict) else diag.get("eval_split_runs", ""),
            "support_fraction": diag.get("support_fraction", ""),
            "query_fraction": diag.get("query_fraction", ""),
            "evaluated_rows": diag.get("evaluated_rows", ""),
            "mean_support_size": diag.get("mean_support_size", ""),
            "mean_query_size": diag.get("mean_query_size", ""),
        }
    )
    return row


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("k")), str(row.get("prototype_support_weight")), str(row.get("temperature")))


def upsert_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    key = row_key(new_row)
    out = []
    replaced = False
    for row in rows:
        if row_key(row) == key:
            out.append(new_row)
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(new_row)
    return out


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_dir)
    plot_summary(rows, output_dir / "plots")


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def pct_pm(mean_value: Any, std_value: Any) -> str:
    mean_text = pct(mean_value)
    if not mean_text:
        return ""
    std_text = pct(std_value)
    if not std_text:
        return mean_text
    return f"{mean_text} +/- {std_text}"


def write_markdown(rows: list[dict[str, Any]], output_dir: Path) -> None:
    done = [row for row in rows if row.get("status") in {"done", "skipped"}]
    lines = [
        "# Single-Region K-Shot Prototype Tuning",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Protocol: global target-region support split, labeled support prototypes, query-only metrics. Each split samples k labeled support items per class and evaluates on the remaining target-region rows.",
        "",
    ]
    if done:
        best = max(done, key=lambda row: float(row.get("macro_f1", -1.0) or -1.0))
        lines.extend(
            [
                "## Best Result",
                "",
                "| Region | k | Split runs | Support weight | Temperature | Acc | Macro-F1 | Global Macro-F1 | Gain | Query Fraction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| {best.get('region')} | {best.get('k')} | {best.get('eval_split_runs', '')} | "
                    f"{best.get('prototype_support_weight')} | {best.get('temperature')} | "
                    f"{pct_pm(best.get('accuracy'), best.get('accuracy_std'))} | "
                    f"{pct_pm(best.get('macro_f1'), best.get('macro_f1_std'))} | "
                    f"{pct_pm(best.get('global_macro_f1'), best.get('global_macro_f1_std'))} | "
                    f"{pct_pm(best.get('episode_gain_macro_f1'), best.get('episode_gain_macro_f1_std'))} | "
                    f"{pct(best.get('query_fraction'))} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Complete Results",
            "",
            "| k | Split runs | Support weight | Temperature | Acc | Macro-F1 | Global Macro-F1 | Gain | Query Fraction | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda r: (int(r.get("k", 0)), float(r.get("temperature", 0)), float(r.get("prototype_support_weight", 0)))):
        lines.append(
            "| {k} | {runs} | {w} | {temp} | {acc} | {mf1} | {gmf1} | {gain} | {qfrac} | {status} |".format(
                k=row.get("k", ""),
                runs=row.get("eval_split_runs", ""),
                w=row.get("prototype_support_weight", ""),
                temp=row.get("temperature", ""),
                acc=pct_pm(row.get("accuracy", ""), row.get("accuracy_std", "")),
                mf1=pct_pm(row.get("macro_f1", ""), row.get("macro_f1_std", "")),
                gmf1=pct_pm(row.get("global_macro_f1", ""), row.get("global_macro_f1_std", "")),
                gain=pct_pm(row.get("episode_gain_macro_f1", ""), row.get("episode_gain_macro_f1_std", "")),
                qfrac=pct(row.get("query_fraction", "")),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Figures:",
            "- `plots/macro_f1_by_k_weight_panels.png`",
            "- `plots/best_macro_f1_by_k.png`",
            "- `plots/best_temperature_weight_heatmap.csv`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    frame = pd.DataFrame([row for row in rows if row.get("status") in {"done", "skipped"}])
    if frame.empty:
        return
    for col in [
        "k",
        "prototype_support_weight",
        "temperature",
        "macro_f1",
        "macro_f1_std",
        "accuracy",
        "accuracy_std",
        "global_macro_f1",
        "global_macro_f1_std",
    ]:
        if col not in frame.columns:
            frame[col] = float("nan")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    plot_dir.mkdir(parents=True, exist_ok=True)
    temperatures = sorted(frame["temperature"].dropna().unique().tolist())
    ncols = min(2, max(1, len(temperatures)))
    nrows = (len(temperatures) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, temp in zip(axes_flat, temperatures):
        part_t = frame[frame["temperature"] == temp]
        for weight, part in part_t.groupby("prototype_support_weight", sort=True):
            part = part.sort_values("k")
            yerr = part["macro_f1_std"] * 100.0
            if yerr.notna().any() and yerr.fillna(0.0).abs().sum() > 0:
                ax.errorbar(
                    part["k"],
                    part["macro_f1"] * 100.0,
                    yerr=yerr.fillna(0.0),
                    marker="o",
                    linewidth=2.0,
                    capsize=3,
                    label=f"w={weight:g}",
                )
            else:
                ax.plot(part["k"], part["macro_f1"] * 100.0, marker="o", linewidth=2.0, label=f"w={weight:g}")
        ax.set_title(f"temperature={temp:g}")
        ax.set_xlabel("k")
        ax.set_ylabel("Macro-F1 (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    for ax in axes_flat[len(temperatures) :]:
        ax.axis("off")
    fig.suptitle("Single-region k-shot tuning", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_dir / "macro_f1_by_k_weight_panels.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    best_by_k = frame.sort_values("macro_f1", ascending=False).groupby("k", as_index=False).first().sort_values("k")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    yerr = best_by_k["macro_f1_std"] * 100.0
    if yerr.notna().any() and yerr.fillna(0.0).abs().sum() > 0:
        ax.errorbar(
            best_by_k["k"],
            best_by_k["macro_f1"] * 100.0,
            yerr=yerr.fillna(0.0),
            marker="o",
            linewidth=2.4,
            capsize=3,
        )
    else:
        ax.plot(best_by_k["k"], best_by_k["macro_f1"] * 100.0, marker="o", linewidth=2.4)
    for _, row in best_by_k.iterrows():
        ax.annotate(f"w={row['prototype_support_weight']:g},t={row['temperature']:g}", (row["k"], row["macro_f1"] * 100.0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Best tuned Macro-F1 at each k")
    ax.set_xlabel("k")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(plot_dir / "best_macro_f1_by_k.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    best_by_k.to_csv(plot_dir / "best_temperature_weight_heatmap.csv", index=False, encoding="utf-8-sig")


def load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def run_job(command: list[str], log_path: Path, dry_run: bool, show_child_progress: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log_file.flush()
        stderr_target = None if bool(show_child_progress) else subprocess.STDOUT
        process = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            stdout=log_file,
            stderr=stderr_target,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return int(process.returncode)


def format_pct_pm(mean_value: Any, std_value: Any) -> str:
    mean_text = pct(mean_value)
    if not mean_text:
        return "NA"
    std_text = pct(std_value)
    return f"{mean_text}+/-{std_text}" if std_text else mean_text


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage3_json, stage3_pt = find_stage3_artifacts(args.region, Path(args.checkpoint_root))

    jobs: list[dict[str, Any]] = []
    commands: list[str] = []
    for k in args.k_values:
        for temperature in args.temperatures:
            for support_weight in args.prototype_support_weights:
                tag = f"k{k}_w{safe_value(support_weight)}_t{safe_value(temperature)}"
                output_json = output_dir / "results" / f"{tag}.json"
                log_path = output_dir / "logs" / f"{tag}.log"
                command = build_command(args, stage3_json, stage3_pt, int(k), float(support_weight), float(temperature), output_json)
                jobs.append(
                    {
                        "k": int(k),
                        "support_weight": float(support_weight),
                        "temperature": float(temperature),
                        "output_json": output_json,
                        "log_path": log_path,
                        "command": command,
                    }
                )
                commands.append(subprocess.list2cmdline(command))
    (output_dir / "commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")

    rows = load_existing_rows(output_dir)
    for idx, job in enumerate(jobs, start=1):
        k = int(job["k"])
        support_weight = float(job["support_weight"])
        temperature = float(job["temperature"])
        output_json = Path(job["output_json"])
        log_path = Path(job["log_path"])
        command = list(job["command"])
        if bool(args.skip_existing) and result_is_usable(output_json, args, k, support_weight, temperature):
            status = "skipped"
            returncode = 0
            job_seconds = 0.0
            print(f"[{idx}/{len(jobs)}] skip k={k} w={support_weight:g} t={temperature:g}", flush=True)
        else:
            print(f"[{idx}/{len(jobs)}] run  k={k} w={support_weight:g} t={temperature:g}", flush=True)
            job_start = time.perf_counter()
            returncode = run_job(
                command,
                log_path,
                bool(args.dry_run),
                show_child_progress=bool(args.show_child_progress),
            )
            job_seconds = time.perf_counter() - job_start
            status = "dry_run" if bool(args.dry_run) else ("done" if returncode == 0 and output_json.exists() else "failed")
        row = collect_result(args, k, support_weight, temperature, output_json, log_path, status, returncode, command)
        row["job_time_sec"] = float(job_seconds)
        rows = upsert_row(rows, row)
        write_tables(rows, output_dir)
        print(
            (
                f"[{idx}/{len(jobs)}] {status} k={k} "
                f"mf1={format_pct_pm(row.get('macro_f1'), row.get('macro_f1_std'))} "
                f"acc={format_pct_pm(row.get('accuracy'), row.get('accuracy_std'))} "
                f"time={job_seconds:.1f}s"
            ),
            flush=True,
        )
        if returncode != 0 and bool(args.stop_on_failure):
            raise RuntimeError(f"failed k={k} w={support_weight:g} t={temperature:g}; see {log_path}")

    write_tables(rows, output_dir)
    print(
        json.dumps(
            {
                "region": args.region,
                "jobs": len(jobs),
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.csv"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
