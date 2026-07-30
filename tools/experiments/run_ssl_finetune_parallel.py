#!/usr/bin/env python3
"""Run SSL fine-tuning jobs in parallel with a small worker pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MODELS = [
    "wav2vec2-base",
    "hubert-base-ls960",
    "wavlm-base",
    "whisper-base",
    "wav2vec2-large-robust-hfcache",
    "wav2vec2-xls-r-300m-hfcache",
    "mHuBERT-147",
    "MR-HuBERT",
    "MS-HuBERT",
    "allophant-hierarchical-hfcache",
]

TARGETS = [
    "Dangtu",
    "Wuhu",
    "Chizhou",
    "Qingyang",
    "Suncun",
    "Tongling",
    "Xuancheng",
    "Jingxian",
    "Fanchang",
    "Nanling",
    "Huangshan",
    "Ningguo",
    "Gaochun",
    "Lishui",
]


@dataclass
class Job:
    model: str
    target: str
    output_dir: Path
    log_path: Path
    device: str
    attempt: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/0722/data/wu_vowel_segments.paper_regions.csv")
    parser.add_argument("--output-dir", default="output/0722/ssl_finetune_paper_models_14targets_parallel")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed-on-cpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def build_jobs(out_root: Path, device: str) -> list[Job]:
    jobs: list[Job] = []
    for model in MODELS:
        for target in TARGETS:
            job_dir = out_root / safe_name(model) / target
            jobs.append(Job(model=model, target=target, output_dir=job_dir, log_path=job_dir / "run.log", device=device))
    return jobs


def job_priority(job: Job) -> tuple[int, str, str]:
    return (1 if job.model == "wavlm-base" else 0, job.model, job.target)


def result_exists(job: Job) -> bool:
    for result_path in job.output_dir.glob(f"results/{job.model}/{job.target}/result.json"):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return data.get("model") == job.model and data.get("target") == job.target and "metrics" in data
    return False


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out_root, str(args.device))
    if bool(args.resume):
        jobs = [job for job in jobs if not result_exists(job)]
    jobs.sort(key=job_priority)
    manifest = {
        "csv": str(Path(args.csv).resolve()),
        "device": str(args.device),
        "concurrency": int(args.concurrency),
        "models": MODELS,
        "targets": TARGETS,
        "jobs": [
            {
                "model": job.model,
                "target": job.target,
                "output_dir": str(job.output_dir),
                "log": str(job.log_path),
            }
            for job in jobs
        ],
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    pending = jobs[:]
    running: dict[subprocess.Popen[str], Job] = {}
    started = time.time()
    script = Path(__file__).with_name("ssl_finetune_source_then_4shot.py")

    while pending or running:
        while pending and len(running) < max(1, int(args.concurrency)):
            job = pending.pop(0)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                str(script),
                "--csv",
                str(args.csv),
                "--output-dir",
                str(job.output_dir),
                "--models",
                job.model,
                "--targets",
                job.target,
                "--device",
                str(job.device),
                "--train-scheme",
                "last_layer",
                "--max-steps",
                str(args.max_steps),
                "--eval-every",
                str(args.eval_every),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--k",
                str(args.k),
                "--splits",
                str(args.splits),
            ]
            if args.force:
                command.append("--force")
            log_file = job.log_path.open("a" if job.attempt > 1 else "w", encoding="utf-8")
            if job.attempt > 1:
                log_file.write(f"\n--- retry attempt {job.attempt} device={job.device} ---\n")
            log_file.write(" ".join(command) + "\n")
            log_file.flush()
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(Path.cwd()))
            running[process] = job
            print(f"[start] {job.model} {job.target} pid={process.pid} device={job.device} attempt={job.attempt}", flush=True)

        finished = []
        for process, job in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            finished.append((process, job, code))
        if not finished:
            time.sleep(5)
            continue
        for process, job, code in finished:
            running.pop(process, None)
            print(f"[done] {job.model} {job.target} rc={code}", flush=True)
            if code != 0:
                if bool(args.retry_failed_on_cpu) and job.device != "cpu" and job.attempt < 2:
                    pending.insert(0, Job(job.model, job.target, job.output_dir, job.log_path, "cpu", job.attempt + 1))
                    print(f"[retry-cpu] {job.model} {job.target}", flush=True)
                    continue
                raise SystemExit(f"job failed: {job.model} {job.target}; see {job.log_path}")

    print(json.dumps({"done": len(jobs), "elapsed_sec": round(time.time() - started, 2)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
