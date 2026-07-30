#!/usr/bin/env python3
"""Run the tuned paper dual-memory recipe over multiple holdout regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import pandas as pd

from pc_dlcmnet.utils.paths import default_project_root


def default_old_root() -> Path:
    return default_project_root()


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--regions", nargs="*", default=["03池州", "08泾县", "11黄山", "13高淳"])
    parser.add_argument("--use-all-regions", action="store_true", default=False)
    parser.add_argument("--output-dir", default="output/0614/multiregion_paper_dual_memory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--support-shots", type=int, default=2)
    parser.add_argument("--eval-support-shots", type=int, default=4)
    parser.add_argument("--eval-ensemble-runs", type=int, default=5)
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--lambda-global", type=float, default=0.5)
    parser.add_argument("--stage1-steps", type=int, default=800)
    parser.add_argument("--stage2-steps", type=int, default=400)
    parser.add_argument("--stage3-steps", type=int, default=200)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--skip-completed", action="store_true", default=False)
    parser.add_argument("--cuda-init-retries", type=int, default=5)
    parser.add_argument("--cuda-init-wait", type=float, default=3.0)
    return parser.parse_args()


def discover_regions(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df["region"].astype(str).unique().tolist())


def cuda_preflight(python: str, env: dict[str, str], retries: int, wait_sec: float) -> None:
    for attempt in range(1, max(1, int(retries)) + 1):
        check = subprocess.run(
            [
                python,
                "-c",
                (
                    "import torch; "
                    "torch.cuda.init(); "
                    "print(torch.cuda.get_device_name(0), flush=True)"
                ),
            ],
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check.returncode == 0:
            return
        print(f"[cuda-preflight] attempt {attempt} failed: {check.stdout.strip()}", flush=True)
        time.sleep(float(wait_sec))
    raise RuntimeError("CUDA preflight failed; refusing to run on CPU")


def run_command(command: list[str], log_path: Path, args: argparse.Namespace) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    if str(args.device).startswith("cuda"):
        env["PC_DLCMNET_REQUIRE_CUDA"] = "1"
        cuda_preflight(args.python, env, args.cuda_init_retries, args.cuda_init_wait)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                print(line, end="")
            except UnicodeEncodeError:
                safe_line = line.encode("gbk", errors="replace").decode("gbk", errors="replace")
                print(safe_line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def region_stem(region: str) -> str:
    prefix_match = re.match(r"^(\d+)", str(region).strip())
    prefix = prefix_match.group(1) if prefix_match else "region"
    digest = hashlib.sha1(str(region).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def build_stage_command(
    args: argparse.Namespace,
    holdout_region: str,
    output_json: Path,
    max_steps: int,
    lr: float,
    init_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        "-u",
        str(PACKAGE_ROOT / "tools/experiments/paper_dual_memory.py"),
        "--csv",
        str(args.csv),
        "--holdout-region",
        holdout_region,
        "--device",
        str(args.device),
        "--episode-length",
        str(args.episode_length),
        "--episode-batch-size",
        str(args.episode_batch_size),
        "--support-shots",
        str(args.support_shots),
        "--eval-support-shots",
        str(args.eval_support_shots),
        "--eval-ensemble-runs",
        str(args.eval_ensemble_runs),
        "--eval-split-runs",
        str(args.eval_split_runs),
        "--lambda-global",
        str(args.lambda_global),
        "--max-steps",
        str(max_steps),
        "--lr",
        str(lr),
        "--seed",
        str(args.seed),
        "--log-every",
        str(args.log_every),
        "--eval-every",
        str(args.eval_every),
        "--save-checkpoint",
        "--output",
        str(output_json),
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    return command


def summarize_region(data: dict[str, Any], region: str, stage: str) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    global_metrics = data.get("global_metrics", {})
    return {
        "holdout_region": region,
        "stage": stage,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
        "global_accuracy": float(global_metrics.get("accuracy", 0.0)),
        "global_macro_f1": float(global_metrics.get("macro_f1", 0.0)),
    }


def maybe_append_existing_stage(
    summary_rows: list[dict[str, Any]],
    region: str,
    stage: str,
    json_path: Path,
) -> bool:
    if not json_path.exists():
        return False
    summary_rows.append(summarize_region(load_metrics(json_path), region, stage))
    return True


def write_summary(rows: list[dict[str, Any]], summary_csv: Path, summary_json: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    available_regions = discover_regions(args.csv)
    if args.use_all_regions:
        regions = available_regions
    else:
        regions = [region for region in args.regions if region in available_regions]
    if not regions:
        raise ValueError("No valid regions selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    started_at = time.perf_counter()
    for region in regions:
        safe_region = region_stem(region)
        region_dir = output_dir / safe_region
        region_dir.mkdir(parents=True, exist_ok=True)
        print(f"[multi-region] holdout={region}", flush=True)

        stage1_json = region_dir / f"{safe_region}_stage1.json"
        stage1_log = region_dir / f"{safe_region}_stage1.log"
        if args.skip_completed and stage1_json.exists():
            maybe_append_existing_stage(summary_rows, region, "stage1", stage1_json)
        else:
            run_command(
                build_stage_command(
                    args=args,
                    holdout_region=region,
                    output_json=stage1_json,
                    max_steps=int(args.stage1_steps),
                    lr=float(args.stage1_lr),
                    init_checkpoint=None,
                ),
                log_path=stage1_log,
                args=args,
            )
            summary_rows.append(summarize_region(load_metrics(stage1_json), region, "stage1"))

        stage2_json = region_dir / f"{safe_region}_stage2.json"
        stage2_log = region_dir / f"{safe_region}_stage2.log"
        stage1_ckpt = stage1_json.with_suffix(".pt")
        if args.skip_completed and stage2_json.exists():
            maybe_append_existing_stage(summary_rows, region, "stage2", stage2_json)
        else:
            run_command(
                build_stage_command(
                    args=args,
                    holdout_region=region,
                    output_json=stage2_json,
                    max_steps=int(args.stage2_steps),
                    lr=float(args.stage2_lr),
                    init_checkpoint=stage1_ckpt,
                ),
                log_path=stage2_log,
                args=args,
            )
            summary_rows.append(summarize_region(load_metrics(stage2_json), region, "stage2"))

        stage3_json = region_dir / f"{safe_region}_stage3.json"
        stage3_log = region_dir / f"{safe_region}_stage3.log"
        stage2_ckpt = stage2_json.with_suffix(".pt")
        if args.skip_completed and stage3_json.exists():
            maybe_append_existing_stage(summary_rows, region, "stage3", stage3_json)
        else:
            run_command(
                build_stage_command(
                    args=args,
                    holdout_region=region,
                    output_json=stage3_json,
                    max_steps=int(args.stage3_steps),
                    lr=float(args.stage3_lr),
                    init_checkpoint=stage2_ckpt,
                ),
                log_path=stage3_log,
                args=args,
            )
            summary_rows.append(summarize_region(load_metrics(stage3_json), region, "stage3"))

        write_summary(
            summary_rows,
            summary_csv=output_dir / "summary.csv",
            summary_json=output_dir / "summary.json",
        )

    elapsed = time.perf_counter() - started_at
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "regions": regions,
                "elapsed_sec": elapsed,
                "summary_csv": str(output_dir / "summary.csv"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
