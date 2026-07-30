"""Plot PC-DLCMNet mechanism visualizations from prepared diagnostic CSV files.

This script does not run inference or collect diagnostics. It only reads CSV
tables and calls the reusable plotting functions in ``pc_dlcmnet.visualization``.

Expected input files under ``--input-dir``:
- ``gema_gate_diagnostics.csv``: region,class,k,gate_global_weight
- ``memory_trajectory.csv``: class,memory_type,x,y or class,memory_type,f0,f1,...
- ``pcmr_decomposition.csv``: query_id,true_class,class,component,value
- ``prompt_routing.csv``: sample_id,region,prompt,alpha,correct
- ``predictions.csv``: true_class,baseline_pred,ours_pred
- ``support_corruption.csv``: noise_ratio,method,macro_f1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import pandas as pd

from pc_dlcmnet.visualization import (
    plot_confusion_matrix_comparison,
    plot_confusion_matrix_delta,
    plot_gema_adaptation_ratio_boxplot,
    plot_gema_class_region_gate_heatmap,
    plot_gema_gate_distribution,
    plot_gema_gate_lines_by_class,
    plot_gema_gate_lines_by_region,
    plot_memory_adaptation_trajectory,
    plot_memory_movement_lines,
    plot_pcmr_decomposition,
    plot_pcmr_prompt_shift_heatmap,
    plot_pcmr_true_class_attention,
    plot_prompt_entropy_boxplot,
    plot_prompt_usage_bar,
    plot_prompt_usage_region_heatmap,
    plot_support_corruption_curve,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing prepared diagnostic CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures will be written.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats, e.g. png pdf.")
    parser.add_argument("--true-col", default="true_class", help="True-label column in predictions.csv.")
    parser.add_argument("--baseline-col", default="baseline_pred", help="Baseline prediction column in predictions.csv.")
    parser.add_argument("--ours-col", default="ours_pred", help="PC-DLCMNet prediction column in predictions.csv.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for UMAP/t-SNE fallback projection.")
    return parser.parse_args()


def read_csv_if_exists(input_dir: Path, name: str) -> pd.DataFrame | None:
    path = input_dir / name
    if not path.exists():
        print(f"[skip] {name} not found", flush=True)
        return None
    print(f"[read] {path}", flush=True)
    return pd.read_csv(path)


def write(fig, output_dir: Path, name: str, formats: list[str]) -> None:
    save_figure(fig, output_dir, name, formats=formats)
    print(f"[write] {name} ({', '.join(formats)})", flush=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_df = read_csv_if_exists(input_dir, "gema_gate_diagnostics.csv")
    if gate_df is not None:
        write(plot_gema_class_region_gate_heatmap(gate_df), output_dir, "fig_gema_class_region_gate_heatmap", args.formats)
        if "k" in gate_df.columns:
            write(plot_gema_gate_distribution(gate_df), output_dir, "fig_gema_gate_distribution_1shot_5shot", args.formats)
            write(plot_gema_gate_lines_by_class(gate_df), output_dir, "fig_gema_gate_lines_by_class", args.formats)
            write(plot_gema_gate_lines_by_region(gate_df), output_dir, "fig_gema_support_injection_lines_by_region", args.formats)
        else:
            print("[skip] fig_gema_gate_distribution_1shot_5shot requires column: k", flush=True)

    memory_df = read_csv_if_exists(input_dir, "memory_trajectory.csv")
    if memory_df is not None:
        write(plot_memory_adaptation_trajectory(memory_df, seed=args.seed), output_dir, "fig_memory_adaptation_trajectory", args.formats)
        if "k" in memory_df.columns:
            write(plot_memory_movement_lines(memory_df), output_dir, "fig_memory_movement_lines", args.formats)
            write(plot_gema_adaptation_ratio_boxplot(memory_df), output_dir, "fig_gema_adaptation_ratio_boxplot", args.formats)

    pcmr_df = read_csv_if_exists(input_dir, "pcmr_decomposition.csv")
    if pcmr_df is not None:
        write(plot_pcmr_decomposition(pcmr_df), output_dir, "fig_pcmr_decomposition", args.formats)
        if "attention_shift" in set(pcmr_df.get("component", [])):
            write(plot_pcmr_prompt_shift_heatmap(pcmr_df), output_dir, "fig_pcmr_prompt_shift_heatmap", args.formats)
            write(plot_pcmr_true_class_attention(pcmr_df), output_dir, "fig_pcmr_true_class_attention", args.formats)

    routing_df = read_csv_if_exists(input_dir, "prompt_routing.csv")
    if routing_df is not None:
        write(plot_prompt_usage_bar(routing_df), output_dir, "fig_prompt_usage_bar", args.formats)
        if "region" in routing_df.columns:
            write(plot_prompt_usage_region_heatmap(routing_df), output_dir, "fig_prompt_usage_region_heatmap", args.formats)
        else:
            print("[skip] fig_prompt_usage_region_heatmap requires column: region", flush=True)
        if "correct" in routing_df.columns:
            write(plot_prompt_entropy_boxplot(routing_df), output_dir, "fig_prompt_entropy_boxplot", args.formats)
        else:
            print("[skip] fig_prompt_entropy_boxplot requires column: correct", flush=True)

    pred_df = read_csv_if_exists(input_dir, "predictions.csv")
    if pred_df is not None:
        required = [args.true_col, args.baseline_col, args.ours_col]
        missing = [col for col in required if col not in pred_df.columns]
        if missing:
            print(f"[skip] confusion matrix requires columns: {missing}", flush=True)
        else:
            write(
                plot_confusion_matrix_comparison(
                    pred_df[args.true_col].astype(str).tolist(),
                    pred_df[args.baseline_col].astype(str).tolist(),
                    pred_df[args.ours_col].astype(str).tolist(),
                ),
                output_dir,
                "fig_confusion_matrix_comparison",
                args.formats,
            )
            write(
                plot_confusion_matrix_delta(
                    pred_df[args.true_col].astype(str).tolist(),
                    pred_df[args.baseline_col].astype(str).tolist(),
                    pred_df[args.ours_col].astype(str).tolist(),
                ),
                output_dir,
                "fig_confusion_matrix_delta",
                args.formats,
            )

    corruption_df = read_csv_if_exists(input_dir, "support_corruption.csv")
    if corruption_df is not None:
        write(plot_support_corruption_curve(corruption_df), output_dir, "fig_support_corruption_curve", args.formats)

    print(f"[done] figures saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
