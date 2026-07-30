#!/usr/bin/env python3
"""Evaluate episode-memory write modes from existing stage checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=str(PACKAGE_ROOT / "output/0614/multiregion_paper_dual_memory"))
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "output/0614/memory_write_mode_ablation"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", default="stage3")
    parser.add_argument("--regions", nargs="*", default=[])
    parser.add_argument("--modes", nargs="*", default=["pseudo", "blend", "label"])
    parser.add_argument("--blend-value", type=float, default=0.5)
    parser.add_argument("--eval-support-shots", type=int, default=4)
    parser.add_argument("--eval-ensemble-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_value(value: object) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch in {"p", "m"} else "_" for ch in text)


def region_stem(region: str) -> str:
    prefix = "".join(ch for ch in str(region) if ch.isdigit())[:2] or "region"
    digest = hashlib.sha1(str(region).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def discover_checkpoints(root: Path, stage: str, regions: list[str]) -> list[tuple[str, Path, Path]]:
    wanted = set(regions)
    found: list[tuple[str, Path, Path]] = []
    for json_path in sorted(root.glob(f"*/*_{stage}.json")):
        try:
            data = load_json(json_path)
        except Exception:
            continue
        region = data.get("split", {}).get("holdout_region") or data.get("config", {}).get("holdout_region") or json_path.parent.name
        if wanted and region not in wanted:
            continue
        checkpoint = data.get("files", {}).get("checkpoint") or str(json_path.with_suffix(".pt"))
        pt_path = Path(checkpoint)
        if not pt_path.is_absolute():
            pt_path = PACKAGE_ROOT / pt_path
        if pt_path.exists():
            found.append((str(region), json_path, pt_path))
    return found


def cfg(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get("config", {}).get(key, default)


def mode_blend(mode: str, blend_value: float) -> float:
    if mode == "pseudo":
        return 0.0
    if mode == "label":
        return 1.0
    if mode == "blend":
        return float(blend_value)
    raise ValueError(f"unknown write mode: {mode}")


def build_command(
    args: argparse.Namespace,
    region: str,
    stage_json: Path,
    stage_pt: Path,
    mode: str,
    output_json: Path,
) -> list[str]:
    data = load_json(stage_json)
    command = [
        str(args.python),
        "-u",
        str(RUNNER),
        "--max-steps",
        "0",
        "--init-checkpoint",
        str(stage_pt),
        "--holdout-region",
        str(region),
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
        str(args.eval_support_shots),
        "--eval-ensemble-runs",
        str(args.eval_ensemble_runs),
        "--eval-split-runs",
        "1",
        "--support-write-mode",
        mode,
        "--support-label-blend",
        str(mode_blend(mode, args.blend_value)),
        "--eval-query-only",
        "--eval-split-mode",
        "rolling",
        "--eval-classifier",
        "memory",
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
        "--quiet",
        "--no-progress",
        "--skip-val-eval",
    ]
    for optional in ["csv", "cache_dir", "matrix_cache_dir", "audio_root"]:
        value = cfg(data, optional, "")
        if value:
            command.extend([f"--{optional.replace('_', '-')}", str(value)])
    return command


def result_is_usable(path: Path, region: str, mode: str, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    return (
        "macro_f1" in metrics
        and str(config.get("holdout_region")) == str(region)
        and str(config.get("support_write_mode")) == str(mode)
        and str(config.get("eval_classifier")) == "memory"
        and int(config.get("eval_support_shots", -1)) == int(args.eval_support_shots)
        and int(config.get("eval_ensemble_runs", -1)) == int(args.eval_ensemble_runs)
    )


def run_job(command: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + subprocess.list2cmdline(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        f.flush()
        process = subprocess.run(
            command,
            cwd=str(PACKAGE_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return int(process.returncode)


def collect_result(region: str, mode: str, output_json: Path, log_path: Path, status: str, returncode: int, command: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "region": region,
        "mode": mode,
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
    row.update(
        {
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "global_accuracy": global_metrics.get("accuracy", ""),
            "global_macro_f1": global_metrics.get("macro_f1", ""),
            "gain_macro_f1": float(metrics.get("macro_f1", 0.0)) - float(global_metrics.get("macro_f1", 0.0)) if metrics and global_metrics else "",
            "gain_accuracy": float(metrics.get("accuracy", 0.0)) - float(global_metrics.get("accuracy", 0.0)) if metrics and global_metrics else "",
            "evaluated_rows": data.get("diagnostics", {}).get("test", {}).get("evaluated_rows", ""),
        }
    )
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
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


def write_markdown(rows: list[dict[str, Any]], output_dir: Path) -> None:
    done = [row for row in rows if row.get("status") in {"done", "skipped"} and row.get("macro_f1") not in {"", None}]
    lines = [
        "# Memory Write Mode Ablation",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Protocol: load existing stage checkpoint, evaluate memory classifier with different support write modes.",
        "",
    ]
    if done:
        frame = pd.DataFrame(done)
        for col in ["macro_f1", "accuracy", "global_macro_f1", "global_accuracy", "gain_macro_f1", "gain_accuracy"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        grouped = frame.groupby("mode", as_index=False).mean(numeric_only=True).sort_values("macro_f1", ascending=False)
        lines.extend(
            [
                "## Mean Results",
                "",
                "| Mode | Macro-F1 | Global Macro-F1 | Gain | Acc | Global Acc | Gain |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['mode']} | {pct(row['macro_f1'])} | {pct(row['global_macro_f1'])} | {pct(row['gain_macro_f1'])} | "
                f"{pct(row['accuracy'])} | {pct(row['global_accuracy'])} | {pct(row['gain_accuracy'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Per-Region Results",
            "",
            "| Region | Mode | Macro-F1 | Global Macro-F1 | Gain | Acc | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item.get("region", "")), str(item.get("mode", "")))):
        lines.append(
            f"| {row.get('region', '')} | {row.get('mode', '')} | {pct(row.get('macro_f1'))} | "
            f"{pct(row.get('global_macro_f1'))} | {pct(row.get('gain_macro_f1'))} | {pct(row.get('accuracy'))} | {row.get('status', '')} |"
        )
    lines.extend(["", "Figure:", "- `plots/write_mode_macro_f1.png`"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    frame = pd.DataFrame([row for row in rows if row.get("status") in {"done", "skipped"} and row.get("macro_f1") not in {"", None}])
    if frame.empty:
        return
    for col in ["macro_f1", "global_macro_f1", "gain_macro_f1", "accuracy"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    grouped = frame.groupby("mode", as_index=False).mean(numeric_only=True).sort_values("macro_f1", ascending=False)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(grouped["mode"], grouped["macro_f1"] * 100.0, color="#2e86ab", label="Episode RW")
    ax.plot(grouped["mode"], grouped["global_macro_f1"] * 100.0, marker="o", color="#7f8c8d", label="Global-only")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Memory write mode ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "write_mode_macro_f1.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("region", "")), str(row.get("mode", "")))


def upsert(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(Path(args.checkpoint_root), str(args.stage), [str(r) for r in args.regions])
    jobs = []
    commands = []
    for region, stage_json, stage_pt in checkpoints:
        for mode in args.modes:
            tag = f"{region_stem(region)}_{safe_value(mode)}"
            output_json = output_dir / "results" / f"{tag}.json"
            log_path = output_dir / "logs" / f"{tag}.log"
            command = build_command(args, region, stage_json, stage_pt, str(mode), output_json)
            jobs.append((region, str(mode), output_json, log_path, command))
            commands.append(subprocess.list2cmdline(command))
    (output_dir / "commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")

    rows = load_existing_rows(output_dir)
    for idx, (region, mode, output_json, log_path, command) in enumerate(jobs, start=1):
        if bool(args.skip_existing) and result_is_usable(output_json, region, mode, args):
            status = "skipped"
            returncode = 0
            print(f"[{idx}/{len(jobs)}] skip {region} mode={mode}", flush=True)
        else:
            print(f"[{idx}/{len(jobs)}] run  {region} mode={mode}", flush=True)
            returncode = run_job(command, log_path, bool(args.dry_run))
            status = "dry_run" if bool(args.dry_run) else ("done" if returncode == 0 and output_json.exists() else "failed")
        row = collect_result(region, mode, output_json, log_path, status, returncode, command)
        rows = upsert(rows, row)
        write_summary(rows, output_dir)
        if returncode != 0 and bool(args.stop_on_failure):
            raise RuntimeError(f"failed region={region} mode={mode}; see {log_path}")
    write_summary(rows, output_dir)
    print(json.dumps({"jobs": len(jobs), "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
