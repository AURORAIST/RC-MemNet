#!/usr/bin/env python3
"""Run one-factor-at-a-time parameter sensitivity sweeps for paper memory model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"
TARGET_DANGTU = "01\u5f53\u6d82"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/parameter_sensitivity_grid"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--holdout-region", default=TARGET_DANGTU)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sweeps",
        nargs="*",
        default=[
            "num_prompts",
            "lambda_global",
            "support_shots",
            "episode_length",
            "temperature",
            "eval_support_shots",
            "eval_ensemble_runs",
        ],
        choices=[
            "num_prompts",
            "lambda_global",
            "support_shots",
            "episode_length",
            "temperature",
            "eval_support_shots",
            "eval_ensemble_runs",
        ],
    )

    parser.add_argument("--train-max-steps", type=int, default=3000)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=500)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=800)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--eval-ensemble-runs", type=int, default=5)
    parser.add_argument("--base-checkpoint", default=str(PACKAGE_ROOT / "output/0614/effectiveness/01dt_paper_dual_memory_s2e32_seed0.pt"))

    parser.add_argument("--base-num-prompts", type=int, default=8)
    parser.add_argument("--base-support-shots", type=int, default=2)
    parser.add_argument("--base-eval-support-shots", type=int, default=4)
    parser.add_argument("--base-episode-length", type=int, default=32)
    parser.add_argument("--base-lambda-global", type=float, default=0.5)
    parser.add_argument("--base-temperature", type=float, default=0.2)

    parser.add_argument("--num-prompts-values", nargs="*", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--lambda-global-values", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--support-shots-values", nargs="*", type=int, default=[1, 2, 3, 4, 6])
    parser.add_argument("--episode-length-values", nargs="*", type=int, default=[8, 16, 24, 32, 48])
    parser.add_argument("--temperature-values", nargs="*", type=float, default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--eval-support-shots-values", nargs="*", type=int, default=[1, 2, 4, 8, 12])
    parser.add_argument("--eval-ensemble-runs-values", nargs="*", type=int, default=[1, 3, 5, 10])
    return parser.parse_args()


def safe_value(value: object) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch in {"p", "m"} else "_" for ch in text)


def base_train_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_prompts": int(args.base_num_prompts),
        "support_shots": int(args.base_support_shots),
        "eval_support_shots": int(args.base_eval_support_shots),
        "episode_length": int(args.base_episode_length),
        "lambda_global": float(args.base_lambda_global),
        "temperature": float(args.base_temperature),
        "eval_ensemble_runs": int(args.eval_ensemble_runs),
    }


def base_eval_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_prompts": int(args.base_num_prompts),
        "support_shots": int(args.base_support_shots),
        "eval_support_shots": int(args.base_eval_support_shots),
        "episode_length": int(args.base_episode_length),
        "lambda_global": float(args.base_lambda_global),
        "temperature": float(args.base_temperature),
        "eval_ensemble_runs": int(args.eval_ensemble_runs),
    }


def add_common_args(command: list[str], args: argparse.Namespace, params: dict[str, Any], output_json: Path) -> None:
    command.extend(
        [
            "--holdout-region",
            str(args.holdout_region),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--episode-length",
            str(params["episode_length"]),
            "--episode-batch-size",
            str(args.episode_batch_size),
            "--support-shots",
            str(params["support_shots"]),
            "--eval-support-shots",
            str(params["eval_support_shots"]),
            "--eval-ensemble-runs",
            str(params["eval_ensemble_runs"]),
            "--lambda-global",
            str(params["lambda_global"]),
            "--num-prompts",
            str(params["num_prompts"]),
            "--temperature",
            str(params["temperature"]),
            "--log-every",
            str(args.log_every),
            "--eval-every",
            str(args.eval_every),
            "--output",
            str(output_json),
        ]
    )


def build_train_command(args: argparse.Namespace, params: dict[str, Any], output_json: Path) -> list[str]:
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--max-steps",
        str(args.train_max_steps),
        "--lr",
        str(args.train_lr),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--early-stop-min-steps",
        str(args.early_stop_min_steps),
    ]
    add_common_args(command, args, params, output_json)
    if bool(args.save_checkpoints):
        command.append("--save-checkpoint")
    return command


def build_eval_command(args: argparse.Namespace, params: dict[str, Any], output_json: Path) -> list[str]:
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--max-steps",
        "0",
        "--init-checkpoint",
        str(args.base_checkpoint),
    ]
    add_common_args(command, args, params, output_json)
    return command


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add_train_sweep(sweep: str, values: list[Any], param_name: str) -> None:
        for value in values:
            params = base_train_params(args)
            params[param_name] = value
            tag = f"{sweep}_{safe_value(value)}"
            output_json = Path(args.output_dir) / "results" / sweep / f"{tag}.json"
            jobs.append(
                {
                    "sweep": sweep,
                    "mode": "train",
                    "parameter": param_name,
                    "value": value,
                    "params": params,
                    "output_json": output_json,
                    "log_path": Path(args.output_dir) / "logs" / sweep / f"{tag}.log",
                    "command_builder": "train",
                }
            )

    def add_eval_sweep(sweep: str, values: list[Any], param_name: str) -> None:
        for value in values:
            params = base_eval_params(args)
            params[param_name] = value
            tag = f"{sweep}_{safe_value(value)}"
            output_json = Path(args.output_dir) / "results" / sweep / f"{tag}.json"
            jobs.append(
                {
                    "sweep": sweep,
                    "mode": "eval",
                    "parameter": param_name,
                    "value": value,
                    "params": params,
                    "output_json": output_json,
                    "log_path": Path(args.output_dir) / "logs" / sweep / f"{tag}.log",
                    "command_builder": "eval",
                }
            )

    if "num_prompts" in args.sweeps:
        add_train_sweep("num_prompts", list(args.num_prompts_values), "num_prompts")
    if "lambda_global" in args.sweeps:
        add_train_sweep("lambda_global", list(args.lambda_global_values), "lambda_global")
    if "support_shots" in args.sweeps:
        add_train_sweep("support_shots", list(args.support_shots_values), "support_shots")
    if "episode_length" in args.sweeps:
        add_train_sweep("episode_length", list(args.episode_length_values), "episode_length")
    if "temperature" in args.sweeps:
        add_train_sweep("temperature", list(args.temperature_values), "temperature")
    if "eval_support_shots" in args.sweeps:
        add_eval_sweep("eval_support_shots", list(args.eval_support_shots_values), "eval_support_shots")
    if "eval_ensemble_runs" in args.sweeps:
        add_eval_sweep("eval_ensemble_runs", list(args.eval_ensemble_runs_values), "eval_ensemble_runs")
    return jobs


def command_for_job(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    if job["command_builder"] == "eval":
        return build_eval_command(args, job["params"], job["output_json"])
    return build_train_command(args, job["params"], job["output_json"])


def collect_result(job: dict[str, Any], status: str, returncode: int, command: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sweep": job["sweep"],
        "mode": job["mode"],
        "parameter": job["parameter"],
        "value": job["value"],
        "status": status,
        "returncode": int(returncode),
        "result_file": str(job["output_json"]),
        "log_file": str(job["log_path"]),
        "command": subprocess.list2cmdline(command),
    }
    for key, value in job["params"].items():
        row[key] = value
    path = Path(job["output_json"])
    if not path.exists():
        return row
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = str(exc)
        return row
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    val_metrics = data.get("val_metrics", {})
    training = data.get("training", {})
    row.update(
        {
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "global_accuracy": global_metrics.get("accuracy", ""),
            "global_macro_f1": global_metrics.get("macro_f1", ""),
            "val_accuracy": val_metrics.get("accuracy", ""),
            "val_macro_f1": val_metrics.get("macro_f1", ""),
            "episode_gain_macro_f1": (
                float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0))
                if metrics and global_metrics
                else ""
            ),
            "best_val_macro_f1": training.get("best_val_macro_f1", ""),
            "best_step": training.get("best_step", ""),
            "final_step": training.get("final_step", ""),
            "stopped_early": training.get("stopped_early", ""),
        }
    )
    return row


def load_result_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-9
        except Exception:
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except Exception:
            return False
    return str(actual) == str(expected)


def existing_result_is_usable(args: argparse.Namespace, job: dict[str, Any]) -> tuple[bool, str]:
    data = load_result_json(Path(job["output_json"]))
    if data is None:
        return False, "missing_or_unreadable_json"
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict) or "macro_f1" not in metrics:
        return False, "missing_metrics"

    config = data.get("config", {})
    if not isinstance(config, dict):
        return False, "missing_config"
    expected_config = {
        "holdout_region": str(args.holdout_region),
        "seed": int(args.seed),
        "episode_length": int(job["params"]["episode_length"]),
        "episode_batch_size": int(args.episode_batch_size),
        "support_shots": int(job["params"]["support_shots"]),
        "eval_support_shots": int(job["params"]["eval_support_shots"]),
        "eval_ensemble_runs": int(job["params"]["eval_ensemble_runs"]),
        "lambda_global": float(job["params"]["lambda_global"]),
        "num_prompts": int(job["params"]["num_prompts"]),
        "temperature": float(job["params"]["temperature"]),
    }
    for key, expected in expected_config.items():
        if not values_match(config.get(key), expected):
            return False, f"config_mismatch:{key}"

    if job["mode"] == "eval":
        if not values_match(config.get("max_steps"), 0):
            return False, "eval_result_not_eval_only"
        return True, "usable_eval_result"

    training = data.get("training", {})
    if not isinstance(training, dict):
        return False, "missing_training_summary"
    try:
        configured_max_steps = int(config.get("max_steps", 0) or 0)
        final_step = int(training.get("final_step", 0) or 0)
    except Exception:
        return False, "invalid_training_steps"
    stopped_early = bool(training.get("stopped_early", False))
    best_step = training.get("best_step", None)
    if best_step in ("", None):
        return False, "missing_best_step"
    if configured_max_steps < int(args.train_max_steps):
        return False, "old_short_max_steps"
    if stopped_early:
        return True, "usable_early_stopped_result"
    if final_step >= int(args.train_max_steps):
        return True, "usable_full_length_result"
    return False, "not_converged_or_full_length"


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_dir / "summary.md")
    try:
        plot_summary(csv_path, output_dir / "plots")
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def format_metric(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return ""


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Parameter Sensitivity Grid",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| sweep | value | mode | Macro-F1 | Acc | Global Macro-F1 | Episode Gain | best step | final step | status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("sweep")), float(r.get("value", 0) or 0), str(r.get("mode")))):
        lines.append(
            "| {sweep} | {value} | {mode} | {mf1} | {acc} | {gmf1} | {gain} | {best_step} | {final_step} | {status} |".format(
                sweep=row.get("sweep", ""),
                value=row.get("value", ""),
                mode=row.get("mode", ""),
                mf1=format_metric(row.get("macro_f1", "")),
                acc=format_metric(row.get("accuracy", "")),
                gmf1=format_metric(row.get("global_macro_f1", "")),
                gain=format_metric(row.get("episode_gain_macro_f1", "")),
                best_step=row.get("best_step", ""),
                final_step=row.get("final_step", ""),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Train sweeps vary one parameter at a time and retrain from scratch.",
            "- Eval sweeps load the base checkpoint and only change evaluation-time parameters.",
            "- `--skip-existing` only skips compatible results that reached the requested training budget or stopped by early stopping.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(csv_path: Path, plot_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or "macro_f1" not in df.columns:
        return
    ok = df[df["status"].isin(["done", "skipped"])].copy()
    if ok.empty:
        return
    for metric in ["macro_f1", "accuracy", "global_macro_f1", "global_accuracy", "episode_gain_macro_f1"]:
        ok[metric] = pd.to_numeric(ok[metric], errors="coerce") * 100.0
    ok["value_num"] = pd.to_numeric(ok["value"], errors="coerce")
    plot_dir.mkdir(parents=True, exist_ok=True)
    for sweep, part in ok.groupby("sweep", sort=True):
        part = part.sort_values("value_num")
        if part["macro_f1"].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        ax.plot(part["value_num"], part["macro_f1"], marker="o", linewidth=2.2, label="Episode RW")
        if part["global_macro_f1"].notna().sum() > 0:
            ax.plot(part["value_num"], part["global_macro_f1"], marker="s", linewidth=2.0, label="Global-only")
        ax.set_title(f"{sweep} sensitivity")
        ax.set_xlabel(str(part["parameter"].iloc[0]))
        ax.set_ylabel("Macro-F1 (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(plot_dir / f"{sweep}_macro_f1.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def load_existing_summary(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("sweep", "")), str(row.get("mode", "")), str(row.get("value", "")))


def upsert_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    key = row_key(new_row)
    merged = []
    replaced = False
    for row in rows:
        if row_key(row) == key:
            merged.append(new_row)
            replaced = True
        else:
            merged.append(row)
    if not replaced:
        merged.append(new_row)
    return merged


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
        )
        return int(process.returncode)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    rows: list[dict[str, Any]] = load_existing_summary(output_dir)
    commands_path = output_dir / "commands.ps1"
    commands_path.write_text(
        "\n".join(subprocess.list2cmdline(command_for_job(args, job)) for job in jobs) + "\n",
        encoding="utf-8",
    )

    for idx, job in enumerate(jobs, start=1):
        command = command_for_job(args, job)
        result_path = Path(job["output_json"])
        if bool(args.skip_existing) and result_path.exists():
            usable, reason = existing_result_is_usable(args, job)
            if usable:
                row = collect_result(job, "skipped", 0, command)
                row["skip_reason"] = reason
                rows = upsert_row(rows, row)
                write_summary(rows, output_dir)
                print(f"[{idx}/{len(jobs)}] skipped {job['sweep']}={job['value']} ({reason})", flush=True)
                continue
            print(f"[{idx}/{len(jobs)}] rerun stale {job['sweep']}={job['value']} ({reason})", flush=True)
        print(f"[{idx}/{len(jobs)}] running {job['sweep']}={job['value']} ({job['mode']})", flush=True)
        returncode = run_job(command, Path(job["log_path"]), bool(args.dry_run))
        status = "dry_run" if bool(args.dry_run) else ("done" if returncode == 0 and result_path.exists() else "failed")
        row = collect_result(job, status, returncode, command)
        rows = upsert_row(rows, row)
        write_summary(rows, output_dir)
        if returncode != 0:
            print(f"[warn] failed {job['sweep']}={job['value']}; see {job['log_path']}", flush=True)

    write_summary(rows, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(jobs), "summary": str(output_dir / "summary.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
