#!/usr/bin/env python3
"""Run fixed-4-shot hyperparameter sensitivity sweeps on eight target regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/experiments/paper_dual_memory.py"

TARGETS = [
    ("Qingyang", "04青阳", "region"),
    ("Tongling", "06铜陵", "region"),
    ("Jingxian", "08泾县", "region"),
    ("Nanling", "10南陵", "region"),
    ("Ningguo", "12宁国", "site"),
    ("Lishui", "14溧水", "site"),
    ("Chizhou", "03池州", "region"),
    ("Huangshan", "11黄山", "region"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / f"output/0614/hyperparam_sensitivity_8targets_k4_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--csv", default="data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-stems", nargs="*", default=None)
    parser.add_argument("--sweeps", nargs="*", default=["num_prompts", "lambda_global", "temperature", "episode_length"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-prompts-values", nargs="*", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--lambda-global-values", nargs="*", type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--temperature-values", nargs="*", type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--episode-length-values", nargs="*", type=int, default=[16, 32, 48])

    parser.add_argument("--base-num-prompts", type=int, default=8)
    parser.add_argument("--base-lambda-global", type=float, default=0.5)
    parser.add_argument("--base-temperature", type=float, default=0.5)
    parser.add_argument("--base-episode-length", type=int, default=32)

    parser.add_argument("--k-shot", type=int, default=4)
    parser.add_argument("--train-support-shots", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--early-stop-patience", type=int, default=200)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--mechanism-k-values", nargs="*", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--mechanism-max-episodes", type=int, default=0)
    parser.add_argument("--export-prompt-route-curves", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_value(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def selected_targets(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if not args.target_stems:
        return TARGETS
    wanted = set(args.target_stems)
    targets = [item for item in TARGETS if item[0] in wanted]
    missing = sorted(wanted.difference({item[0] for item in targets}))
    if missing:
        raise ValueError(f"unknown target stems: {missing}")
    return targets


def build_sweep_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = {
        "num_prompts": ("num_prompts", list(args.num_prompts_values)),
        "lambda_global": ("lambda_global", list(args.lambda_global_values)),
        "temperature": ("temperature", list(args.temperature_values)),
        "episode_length": ("episode_length", list(args.episode_length_values)),
    }
    base = {
        "num_prompts": int(args.base_num_prompts),
        "lambda_global": float(args.base_lambda_global),
        "temperature": float(args.base_temperature),
        "episode_length": int(args.base_episode_length),
    }
    settings = []
    for sweep in args.sweeps:
        if sweep not in specs:
            raise ValueError(f"unknown sweep: {sweep}")
        param, values = specs[sweep]
        for value in values:
            params = dict(base)
            params[param] = value
            settings.append({"sweep": sweep, "parameter": param, "value": value, **params})
    return settings


def result_is_usable(path: Path, args: argparse.Namespace, setting: dict[str, Any], holdout: str, column: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    cfg = data.get("config", {})
    metrics = data.get("metrics", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    return (
        metrics.get("accuracy") is not None
        and str(cfg.get("holdout_region")) == holdout
        and str(cfg.get("region_column")) == column
        and int(cfg.get("eval_support_shots", -1)) == int(args.k_shot)
        and int(split.get("runs", cfg.get("eval_split_runs", -1))) == int(args.num_tasks)
        and int(cfg.get("num_prompts", -1)) == int(setting["num_prompts"])
        and abs(float(cfg.get("lambda_global", -1.0)) - float(setting["lambda_global"])) < 1e-9
        and abs(float(cfg.get("temperature", -1.0)) - float(setting["temperature"])) < 1e-9
        and int(cfg.get("episode_length", -1)) == int(setting["episode_length"])
    )


def build_command(args: argparse.Namespace, setting: dict[str, Any], holdout: str, column: str, out_dir: Path) -> list[str]:
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--csv",
        str(args.csv),
        "--region-column",
        column,
        "--holdout-region",
        holdout,
        "--device",
        str(args.device),
        "--eval-support-shots",
        str(args.k_shot),
        "--eval-split-runs",
        str(args.num_tasks),
        "--eval-split-mode",
        "global_support",
        "--support-shots",
        str(args.train_support_shots),
        "--max-steps",
        str(args.max_steps),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--early-stop-min-steps",
        str(args.early_stop_min_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--lambda-global",
        str(setting["lambda_global"]),
        "--hidden-dim",
        "256",
        "--score-dim",
        "128",
        "--prompt-dim",
        "128",
        "--num-prompts",
        str(setting["num_prompts"]),
        "--layers",
        "3",
        "--heads",
        "4",
        "--ffn-dim",
        "768",
        "--dropout",
        "0.1",
        "--temperature",
        str(setting["temperature"]),
        "--episode-length",
        str(setting["episode_length"]),
        "--episode-batch-size",
        str(args.episode_batch_size),
        "--eval-ensemble-runs",
        "1",
        "--log-every",
        "50",
        "--eval-every",
        "50",
        "--save-checkpoint",
        "--output",
        str(out_dir / "result.json"),
        "--curve-output",
        str(out_dir / "curve.csv"),
        "--debug-output",
        str(out_dir / "debug.json"),
        "--audit-output",
        str(out_dir / "audit.json"),
        "--predictions-output",
        str(out_dir / "predictions.csv"),
        "--mechanism-output-dir",
        str(out_dir / "csv"),
        "--mechanism-max-episodes",
        str(args.mechanism_max_episodes),
    ]
    if args.mechanism_k_values:
        command.extend(["--mechanism-k-values", *[str(v) for v in args.mechanism_k_values]])
    return command


def csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


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


def export_prompt_route_curves(jobs: list[dict[str, Any]], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        curve = job["out_dir"] / "curve.csv"
        if not curve.exists():
            continue
        with curve.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            prompt_cols = sorted(
                [name for name in reader.fieldnames if name.startswith("prompt_route_mean_")],
                key=lambda name: int(name.rsplit("_", 1)[-1]),
            )
            for curve_row in reader:
                step = numeric_or_blank(curve_row.get("step"))
                epoch = numeric_or_blank(curve_row.get("epoch"))
                route_entropy = numeric_or_blank(curve_row.get("prompt_route_entropy"))
                val_route_entropy = numeric_or_blank(curve_row.get("val_prompt_route_entropy"))
                for col in prompt_cols:
                    alpha = numeric_or_blank(curve_row.get(col))
                    if alpha == "":
                        continue
                    rows.append(
                        {
                            "sweep": job["setting"]["sweep"],
                            "parameter": job["setting"]["parameter"],
                            "value": job["setting"]["value"],
                            "area": job["area"],
                            "holdout": job["holdout"],
                            "region_column": job["column"],
                            "num_prompts": job["setting"]["num_prompts"],
                            "lambda_global": job["setting"]["lambda_global"],
                            "temperature": job["setting"]["temperature"],
                            "episode_length": job["setting"]["episode_length"],
                            "step": step,
                            "epoch": epoch,
                            "prompt": int(col.rsplit("_", 1)[-1]),
                            "routing_weight": alpha,
                            "prompt_route_entropy": route_entropy,
                            "val_prompt_route_entropy": val_route_entropy,
                            "curve_file": str(curve),
                        }
                    )
    path = output_dir / "prompt_route_weights_by_iteration.csv"
    fieldnames = [
        "sweep",
        "parameter",
        "value",
        "area",
        "holdout",
        "region_column",
        "num_prompts",
        "lambda_global",
        "temperature",
        "episode_length",
        "step",
        "epoch",
        "prompt",
        "routing_weight",
        "prompt_route_entropy",
        "val_prompt_route_entropy",
        "curve_file",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_row(job: dict[str, Any], status: str, rc: int, command: list[str]) -> dict[str, Any]:
    row = {
        "sweep": job["setting"]["sweep"],
        "parameter": job["setting"]["parameter"],
        "value": job["setting"]["value"],
        "area": job["area"],
        "holdout": job["holdout"],
        "region_column": job["column"],
        "status": status,
        "returncode": rc,
        "result_file": str(job["result"]),
        "command": subprocess.list2cmdline(command),
        "curve_rows": csv_rows(job["out_dir"] / "curve.csv"),
        "routing_rows": csv_rows(job["out_dir"] / "csv/prompt_routing.csv"),
    }
    row.update({k: job["setting"][k] for k in ["num_prompts", "lambda_global", "temperature", "episode_length"]})
    if job["result"].exists():
        try:
            data = json.loads(job["result"].read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            std = data.get("evaluation_splits", {}).get("test", {}).get("metrics_std", {})
            row.update(
                {
                    "accuracy": metrics.get("accuracy", ""),
                    "accuracy_std": std.get("accuracy", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "macro_f1_std": std.get("macro_f1", ""),
                    "weighted_f1": metrics.get("weighted_f1", ""),
                    "weighted_f1_std": std.get("weighted_f1", ""),
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    done = [r for r in rows if r.get("status") in {"done", "skipped"} and r.get("accuracy") not in ("", None)]
    if done:
        import pandas as pd

        df = pd.DataFrame(done)
        for col in ["accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "weighted_f1", "weighted_f1_std"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        agg = (
            df.groupby(["sweep", "parameter", "value"], as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                accuracy_area_std=("accuracy", "std"),
                macro_f1=("macro_f1", "mean"),
                macro_f1_area_std=("macro_f1", "std"),
                weighted_f1=("weighted_f1", "mean"),
                weighted_f1_area_std=("weighted_f1", "std"),
                areas=("area", "nunique"),
            )
            .sort_values(["sweep", "value"])
        )
        agg.to_csv(output_dir / "summary_across_8areas.csv", index=False, encoding="utf-8-sig")


def run_one(args: argparse.Namespace, job: dict[str, Any]) -> dict[str, Any]:
    out_dir = job["out_dir"]
    result = job["result"]
    command = build_command(args, job["setting"], job["holdout"], job["column"], out_dir)
    if args.skip_existing and result_is_usable(result, args, job["setting"], job["holdout"], job["column"]):
        return collect_row(job, "skipped", 0, command)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (out_dir / "train.log").write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        return collect_row(job, "dry_run", 0, command)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_MODULE_LOADING"] = "EAGER"
    if str(args.device).startswith("cuda"):
        env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
    with (out_dir / "train.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        rc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
    status = "done" if rc == 0 and result_is_usable(result, args, job["setting"], job["holdout"], job["column"]) else "failed"
    return collect_row(job, status, int(rc), command)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = build_sweep_settings(args)
    targets = selected_targets(args)
    jobs = []
    for setting in settings:
        tag = f"{setting['sweep']}_{safe_value(setting['value'])}"
        for area, holdout, column in targets:
            out_dir = output_dir / tag / area
            jobs.append({"setting": setting, "area": area, "holdout": holdout, "column": column, "out_dir": out_dir, "result": out_dir / "result.json"})

    commands = [subprocess.list2cmdline(build_command(args, job["setting"], job["holdout"], job["column"], job["out_dir"])) for job in jobs]
    (output_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(jobs), "workers": int(args.workers)}, ensure_ascii=False), flush=True)

    rows: list[dict[str, Any]] = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(run_one, args, job): job for job in jobs}
        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            rows.append(row)
            write_summary(rows, output_dir)
            if args.export_prompt_route_curves:
                export_prompt_route_curves(jobs, output_dir)
            print(f"[{idx}/{len(jobs)}] {row['status']} {row['sweep']}={row['value']} {row['area']}", flush=True)
            if row["status"] == "failed":
                failures.append(row)
                break
    write_summary(rows, output_dir)
    if args.export_prompt_route_curves:
        export_prompt_route_curves(jobs, output_dir)
    if failures:
        raise SystemExit(f"failed: {failures[:1]}")
    print(json.dumps({"summary": str(output_dir / "summary_across_8areas.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
