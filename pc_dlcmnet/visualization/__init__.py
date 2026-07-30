"""Reusable visualization helpers for PC-DLCMNet paper figures.

The functions in this package are intentionally side-effect free: importing
them does not read experiment outputs or create figures. Callers pass prepared
diagnostic tables and decide when/where to save the returned figures.
"""

from pc_dlcmnet.visualization.paper_diagnostics import (
    DEFAULT_REGION_ORDER,
    DEFAULT_VOWEL_ORDER,
    compute_prompt_entropy,
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
    setup_paper_style,
)

__all__ = [
    "DEFAULT_REGION_ORDER",
    "DEFAULT_VOWEL_ORDER",
    "compute_prompt_entropy",
    "plot_confusion_matrix_comparison",
    "plot_confusion_matrix_delta",
    "plot_gema_adaptation_ratio_boxplot",
    "plot_gema_class_region_gate_heatmap",
    "plot_gema_gate_distribution",
    "plot_gema_gate_lines_by_class",
    "plot_gema_gate_lines_by_region",
    "plot_memory_adaptation_trajectory",
    "plot_memory_movement_lines",
    "plot_pcmr_decomposition",
    "plot_pcmr_prompt_shift_heatmap",
    "plot_pcmr_true_class_attention",
    "plot_prompt_entropy_boxplot",
    "plot_prompt_usage_bar",
    "plot_prompt_usage_region_heatmap",
    "plot_support_corruption_curve",
    "save_figure",
    "setup_paper_style",
]
