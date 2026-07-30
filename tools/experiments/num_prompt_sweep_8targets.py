#!/usr/bin/env python3
"""Run num_prompt sweeps on the eight target regions and export diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
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
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / f"output/0614/num_prompt_sweep_8targets_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--python", default="/home/ustc1958/miniconda3/envs/graph/bin/python")
    parser.add_argument("--csv", default="data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-stems", nargs="*", default=None)
    parser.add_argument("--num-prompts-values", nargs="*", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--k-shot", type=int, default=4)
    parser.add_argument("--train-support-shots", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--early-stop-patience", type=int, default=200)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--lambda-global", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mechanism-k-values", nargs="*", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--mechanism-max-episodes", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-chizhou", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--chizhou-source-dir",
        default=str(ROOT / "output/0614/num_prompt_sweep_chizhou_20260629"),
        help="Existing single-region Chizhou sweep to copy when compatible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_targets(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if not args.target_stems:
        return TARGETS
    wanted = set(args.target_stems)
    selected = [item for item in TARGETS if item[0] in wanted]
    missing = sorted(wanted.difference({item[0] for item in selected}))
    if missing:
        raise ValueError(f"unknown target stems: {missing}")
    return selected


def result_is_usable(path: Path, args: argparse.Namespace, num_prompts: int, holdout: str, column: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    cfg = data.get("config", {})
    metrics = data.get("metrics", {})
    split = data.get("evaluation_splits", {}).get("test", {})
    mechanism = data.get("diagnostics", {}).get("mechanism", {})
    return (
        metrics.get("macro_f1") is not None
        and str(cfg.get("holdout_region")) == holdout
        and str(cfg.get("region_column")) == column
        and int(cfg.get("num_prompts", -1)) == int(num_prompts)
        and int(cfg.get("eval_support_shots", -1)) == int(args.k_shot)
        and int(split.get("runs", cfg.get("eval_split_runs", -1))) == int(args.num_tasks)
        and (path.parent / "result.pt").exists()
        and (path.parent / "curve.csv").exists()
        and bool(mechanism.get("output_dir"))
        and (path.parent / "csv/gema_gate_diagnostics.csv").exists()
        and (path.parent / "csv/memory_trajectory.csv").exists()
        and (path.parent / "csv/pcmr_decomposition.csv").exists()
        and (path.parent / "csv/prompt_routing.csv").exists()
        and (path.parent / "csv/predictions.csv").exists()
    )


def build_command(args: argparse.Namespace, num_prompts: int, holdout: str, column: str, out_dir: Path) -> list[str]:
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
        str(args.lambda_global),
        "--hidden-dim",
        "256",
        "--score-dim",
        "128",
        "--prompt-dim",
        "128",
        "--num-prompts",
        str(num_prompts),
        "--layers",
        "3",
        "--heads",
        "4",
        "--ffn-dim",
        "768",
        "--dropout",
        "0.1",
        "--temperature",
        str(args.temperature),
        "--episode-length",
        str(args.episode_length),
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


def copy_existing_chizhou(args: argparse.Namespace, out_dir: Path, num_prompts: int) -> bool:
    if not args.reuse_chizhou:
        return False
    src = Path(args.chizhou_source_dir) / f"num_prompts_{num_prompts}"
    src_result = src / "result.json"
    if not result_is_usable(src_result, args, num_prompts, "03池州", "region"):
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["result.json", "result.pt", "audit.json", "curve.csv", "debug.json", "predictions.csv"]:
        if (src / name).exists():
            shutil.copy2(src / name, out_dir / name)
    src_csv = src / "csv"
    dst_csv = out_dir / "csv"
    if src_csv.exists():
        if dst_csv.exists():
            shutil.rmtree(dst_csv)
        shutil.copytree(src_csv, dst_csv)
    return result_is_usable(out_dir / "result.json", args, num_prompts, "03池州", "region")


def csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def collect_row(job: dict[str, Any], status: str, returncode: int, command: list[str]) -> dict[str, Any]:
    out_dir = job["out_dir"]
    row: dict[str, Any] = {
        "area": job["area"],
        "holdout": job["holdout"],
        "region_column": job["column"],
        "num_prompts": job["num_prompts"],
        "status": status,
        "returncode": returncode,
        "run_dir": str(out_dir),
        "result_file": str(out_dir / "result.json"),
        "command": subprocess.list2cmdline(command),
        "gate_rows": csv_rows(out_dir / "csv/gema_gate_diagnostics.csv"),
        "memory_rows": csv_rows(out_dir / "csv/memory_trajectory.csv"),
        "pcmr_rows": csv_rows(out_dir / "csv/pcmr_decomposition.csv"),
        "routing_rows": csv_rows(out_dir / "csv/prompt_routing.csv"),
        "prediction_rows": csv_rows(out_dir / "csv/predictions.csv"),
        "curve_rows": csv_rows(out_dir / "curve.csv"),
    }
    result = out_dir / "result.json"
    if result.exists():
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            global_metrics = data.get("global_metrics", {})
            training = data.get("training", {})
            row.update(
                {
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "micro_f1": metrics.get("micro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "global_accuracy": global_metrics.get("accuracy"),
                    "global_macro_f1": global_metrics.get("macro_f1"),
                    "global_weighted_f1": global_metrics.get("weighted_f1"),
                    "best_val_macro_f1": training.get("best_val_macro_f1"),
                    "best_step": training.get("best_step"),
                    "final_step": training.get("final_step"),
                    "stopped_early": training.get("stopped_early"),
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "num_prompt_sweep_8targets_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "num_prompt_sweep_8targets_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    done = [row for row in rows if row.get("status") in {"done", "skipped", "copied"} and row.get("macro_f1") is not None]
    if done:
        import pandas as pd

        df = pd.DataFrame(done)
        for col in ["accuracy", "macro_f1", "weighted_f1", "global_macro_f1"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        agg = (
            df.groupby("num_prompts", as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                accuracy_area_std=("accuracy", "std"),
                macro_f1=("macro_f1", "mean"),
                macro_f1_area_std=("macro_f1", "std"),
                weighted_f1=("weighted_f1", "mean"),
                weighted_f1_area_std=("weighted_f1", "std"),
                global_macro_f1=("global_macro_f1", "mean"),
                areas=("area", "nunique"),
            )
            .sort_values("num_prompts")
        )
        agg.to_csv(output_dir / "num_prompt_sweep_8targets_across_area.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    targets = selected_targets(args)
    jobs = []
    for num_prompts in args.num_prompts_values:
        for area, holdout, column in targets:
            out_dir = output_dir / f"num_prompts_{num_prompts}" / area
            jobs.append(
                {
                    "num_prompts": int(num_prompts),
                    "area": area,
                    "holdout": holdout,
                    "column": column,
                    "out_dir": out_dir,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [subprocess.list2cmdline(build_command(args, job["num_prompts"], job["holdout"], job["column"], job["out_dir"])) for job in jobs]
    (output_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "jobs": len(jobs)}, ensure_ascii=False), flush=True)

    rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        command = build_command(args, job["num_prompts"], job["holdout"], job["column"], job["out_dir"])
        result = job["out_dir"] / "result.json"
        if args.skip_existing and result_is_usable(result, args, job["num_prompts"], job["holdout"], job["column"]):
            row = collect_row(job, "skipped", 0, command)
            rows.append(row)
            write_summary(rows, output_dir)
            print(f"[{idx}/{len(jobs)}] skipped num_prompts={job['num_prompts']} {job['area']}", flush=True)
            continue
        if job["area"] == "Chizhou" and copy_existing_chizhou(args, job["out_dir"], job["num_prompts"]):
            row = collect_row(job, "copied", 0, command)
            rows.append(row)
            write_summary(rows, output_dir)
            print(f"[{idx}/{len(jobs)}] copied num_prompts={job['num_prompts']} Chizhou", flush=True)
            continue
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            (job["out_dir"] / "train.log").write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
            row = collect_row(job, "dry_run", 0, command)
            rows.append(row)
            write_summary(rows, output_dir)
            print(f"[{idx}/{len(jobs)}] dry_run num_prompts={job['num_prompts']} {job['area']}", flush=True)
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["CUDA_MODULE_LOADING"] = "EAGER"
        if str(args.device).startswith("cuda"):
            env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
        with (job["out_dir"] / "train.log").open("w", encoding="utf-8", errors="replace") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log.flush()
            rc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
        status = "done" if rc == 0 and result_is_usable(result, args, job["num_prompts"], job["holdout"], job["column"]) else "failed"
        row = collect_row(job, status, int(rc), command)
        rows.append(row)
        write_summary(rows, output_dir)
        print(f"[{idx}/{len(jobs)}] {status} num_prompts={job['num_prompts']} {job['area']}", flush=True)
        if status == "failed":
            raise SystemExit(f"failed: num_prompts={job['num_prompts']} area={job['area']} rc={rc}")

    write_summary(rows, output_dir)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "num_prompt_sweep_8targets_summary.csv"),
                "across_area": str(output_dir / "num_prompt_sweep_8targets_across_area.csv"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
