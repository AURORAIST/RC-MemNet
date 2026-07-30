#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 1. Configuration
# =========================

ROOT = Path("/home/ustc1958/lxy/graph/tone/complete_package0614")
SWEEP_ROOT = (
    ROOT
    / "output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps"
)
OUTPUT_DIR = ROOT / "pc_dlcmnet_figures_hybrid/confusion_matrices_8regions_best_global"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_BEST_GLOBAL_SETTING = True
PREDICTION_COLUMN = "global_pred_vowel"

AREAS = [
    "Qingyang",
    "Tongling",
    "Jingxian",
    "Nanling",
    "Ningguo",
    "Lishui",
    "Chizhou",
    "Huangshan",
]

LABELS = ["a", "e", "i", "o", "u", "y", "ɔ", "ə", "ɛ"]
HEIGHT_LABELS = ["High", "Mid-High", "Mid-Low", "Low"]
VOWEL_TO_HEIGHT = {
    "i": "High",
    "u": "High",
    "y": "High",
    "e": "Mid-High",
    "o": "Mid-High",
    "ɔ": "Mid-Low",
    "ə": "Mid-Low",
    "ɛ": "Mid-Low",
    "a": "Low",
}
BACKNESS_LABELS = ["Front", "Central", "Back"]
VOWEL_TO_BACKNESS = {
    "e": "Front",
    "i": "Front",
    "y": "Front",
    "ɛ": "Front",
    "a": "Central",
    "ə": "Central",
    "o": "Back",
    "u": "Back",
    "ɔ": "Back",
}
ROUNDING_LABELS = ["Rounded", "Unrounded"]
VOWEL_TO_ROUNDING = {
    "o": "Rounded",
    "u": "Rounded",
    "y": "Rounded",
    "ɔ": "Rounded",
    "a": "Unrounded",
    "e": "Unrounded",
    "i": "Unrounded",
    "ə": "Unrounded",
    "ɛ": "Unrounded",
}

DPI = 600
FIGSIZE = (12.8, 6.0)
CMAP = "Blues"
DIAG_LOW = np.array([236, 246, 255], dtype=float) / 255.0
DIAG_HIGH = np.array([8, 81, 156], dtype=float) / 255.0
OFF_LOW = np.array([255, 242, 230], dtype=float) / 255.0
OFF_HIGH = np.array([198, 58, 42], dtype=float) / 255.0


# =========================
# 2. Helpers
# =========================

def read_metric(result_path: Path, metric_group: str, metric_name: str) -> float | None:
    import json

    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = (data.get(metric_group) or {}).get(metric_name)
    return float(value) if value is not None else None


def select_prediction_files() -> tuple[dict[str, Path], pd.DataFrame]:
    selected: dict[str, Path] = {}
    selected_rows = []

    if not USE_BEST_GLOBAL_SETTING:
        fixed_root = SWEEP_ROOT / "num_prompts_8"
        for area in AREAS:
            selected[area] = fixed_root / area / "predictions.csv"
            selected_rows.append(
                {
                    "area": area,
                    "setting": "num_prompts_8",
                    "global_accuracy": np.nan,
                    "metric_accuracy": np.nan,
                    "prediction_file": str(selected[area]),
                }
            )
        return selected, pd.DataFrame(selected_rows)

    for area in AREAS:
        candidates = []
        for result_path in SWEEP_ROOT.glob(f"*/{area}/result.json"):
            pred_path = result_path.parent / "predictions.csv"
            if not pred_path.exists():
                continue
            global_acc = read_metric(result_path, "global_metrics", "accuracy")
            metric_acc = read_metric(result_path, "metrics", "accuracy")
            if global_acc is None:
                continue
            candidates.append(
                {
                    "area": area,
                    "setting": result_path.parent.parent.name,
                    "global_accuracy": global_acc,
                    "metric_accuracy": metric_acc,
                    "prediction_file": str(pred_path),
                }
            )
        if not candidates:
            raise FileNotFoundError(f"No candidate predictions found for {area} under {SWEEP_ROOT}")

        best = max(candidates, key=lambda row: row["global_accuracy"])
        selected[area] = Path(best["prediction_file"])
        selected_rows.append(best)

    return selected, pd.DataFrame(selected_rows)


PREDICTION_FILES, SELECTED_SETTINGS = select_prediction_files()


def load_predictions(area: str) -> pd.DataFrame:
    csv_path = PREDICTION_FILES[area]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"vowel", PREDICTION_COLUMN}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_cols)}")

    return df.dropna(subset=["vowel", PREDICTION_COLUMN]).copy()


def confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cm = pd.crosstab(df["vowel"], df[PREDICTION_COLUMN])
    return cm.reindex(index=LABELS, columns=LABELS, fill_value=0).astype(int)


def grouped_confusion_matrix(df: pd.DataFrame, mapping: dict[str, str], labels: list[str]) -> pd.DataFrame:
    grouped = df.copy()
    grouped["true_group"] = grouped["vowel"].map(mapping)
    grouped["pred_group"] = grouped[PREDICTION_COLUMN].map(mapping)
    grouped = grouped.dropna(subset=["true_group", "pred_group"])
    cm = pd.crosstab(grouped["true_group"], grouped["pred_group"])
    return cm.reindex(index=labels, columns=labels, fill_value=0).astype(int)


def row_normalize(cm: pd.DataFrame) -> pd.DataFrame:
    row_sums = cm.sum(axis=1).replace(0, np.nan)
    return cm.div(row_sums, axis=0).fillna(0.0) * 100.0


def diagonal_focus_rgb(values: np.ndarray) -> np.ndarray:
    rgb = np.zeros(values.shape + (3,), dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            if i == j:
                t = np.clip(value / 100.0, 0.0, 1.0)
                rgb[i, j, :] = DIAG_LOW * (1.0 - t) + DIAG_HIGH * t
            else:
                # Off-diagonal cells are errors, so high confusion should stay visually strong.
                t = np.clip(value / 100.0, 0.0, 1.0)
                rgb[i, j, :] = OFF_LOW * (1.0 - t) + OFF_HIGH * t
    return rgb


def build_grouped_outputs(
    group_name: str,
    mapping: dict[str, str],
    labels: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    long_rows = []
    metric_rows = []
    grouped_matrices = {}
    grouped_normalized_matrices = {}

    for area in AREAS:
        df_area = load_predictions(area)
        cm = grouped_confusion_matrix(df_area, mapping, labels)
        cm_norm = row_normalize(cm) / 100.0
        grouped_matrices[area] = cm
        grouped_normalized_matrices[area] = cm_norm

        cm.to_csv(OUTPUT_DIR / f"confusion_counts_{group_name}_{area}.csv", encoding="utf-8-sig")
        cm_norm.to_csv(
            OUTPUT_DIR / f"confusion_row_fraction_{group_name}_{area}.csv",
            encoding="utf-8-sig",
            float_format="%.4f",
        )

        total = int(cm.to_numpy().sum())
        correct = int(np.trace(cm.to_numpy()))
        accuracy = correct / total if total else np.nan
        metric_rows.append(
            {
                "area": area,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            }
        )

        for true_label in labels:
            for pred_label in labels:
                long_rows.append(
                    {
                        "area": area,
                        "true_group": true_label,
                        "pred_group": pred_label,
                        "count": int(cm.loc[true_label, pred_label]),
                        "row_fraction": float(cm_norm.loc[true_label, pred_label]),
                    }
                )

    long_df = pd.DataFrame(long_rows)
    metrics_df = pd.DataFrame(metric_rows)
    long_df.to_csv(
        OUTPUT_DIR / f"confusion_matrices_8regions_{group_name}_long.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    metrics_df.to_csv(
        OUTPUT_DIR / f"confusion_matrices_8regions_{group_name}_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    return grouped_matrices, grouped_normalized_matrices, long_df, metrics_df


def plot_grouped_matrix(
    group_name: str,
    labels: list[str],
    normalized_matrices: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    figsize: tuple[float, float] = (13.4, 6.2),
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=DPI, constrained_layout=False)
    axes = axes.ravel()

    for panel_idx, (ax, area) in enumerate(zip(axes, AREAS)):
        values = normalized_matrices[area].to_numpy()
        ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"{area}", fontsize=11, pad=5)

        if panel_idx >= 4:
            ax.set_xlabel("Predicted", fontsize=9, labelpad=4)
        else:
            ax.set_xlabel("")
        if panel_idx % 4 == 0:
            ax.set_ylabel("True", fontsize=9, labelpad=4)
        else:
            ax.set_ylabel("")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.2)
        ax.set_yticklabels(labels, fontsize=8.2)
        ax.tick_params(axis="x", rotation=0, length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=3)

        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        font_size = 8.8 if len(labels) <= 4 else 6.4
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = values[i, j]
                text_color = "white" if value >= 0.50 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=font_size)

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.085, top=0.93, wspace=0.22, hspace=0.28)
    output_path = OUTPUT_DIR / f"confusion_matrices_8regions_{group_name}_2x4.jpg"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =========================
# 3. Build matrices and CSV
# =========================

long_rows = []
metric_rows = []
matrices = {}
normalized_matrices = {}

for area in AREAS:
    df_area = load_predictions(area)
    cm = confusion_matrix(df_area)
    cm_norm = row_normalize(cm)

    matrices[area] = cm
    normalized_matrices[area] = cm_norm

    cm.to_csv(OUTPUT_DIR / f"confusion_counts_{area}.csv", encoding="utf-8-sig")
    cm_norm.to_csv(
        OUTPUT_DIR / f"confusion_row_percent_{area}.csv",
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    total = int(cm.to_numpy().sum())
    correct = int(np.trace(cm.to_numpy()))
    accuracy = correct / total if total else np.nan

    metric_rows.append(
        {
            "area": area,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }
    )

    for true_label in LABELS:
        for pred_label in LABELS:
            long_rows.append(
                {
                    "area": area,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(cm.loc[true_label, pred_label]),
                    "row_percent": float(cm_norm.loc[true_label, pred_label]),
                }
            )

long_df = pd.DataFrame(long_rows)
metrics_df = pd.DataFrame(metric_rows)

long_df.to_csv(
    OUTPUT_DIR / "confusion_matrices_8regions_long.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f",
)
metrics_df.to_csv(
    OUTPUT_DIR / "confusion_matrices_8regions_accuracy.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)
SELECTED_SETTINGS.to_csv(
    OUTPUT_DIR / "confusion_matrices_8regions_selected_settings.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)


# =========================
# 4. Plot 2 x 4 subplots
# =========================

fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, dpi=DPI, constrained_layout=False)
axes = axes.ravel()

for panel_idx, (ax, area) in enumerate(zip(axes, AREAS)):
    cm = matrices[area]
    cm_norm = normalized_matrices[area]
    accuracy = metrics_df.loc[metrics_df["area"] == area, "accuracy"].iloc[0]

    im = ax.imshow(cm_norm.to_numpy(), cmap=CMAP, vmin=0, vmax=100, aspect="equal")

    ax.set_title(f"{area} ({accuracy * 100:.1f}%)", fontsize=8.5, pad=4)
    if panel_idx >= 4:
        ax.set_xlabel("Predicted", fontsize=7.5, labelpad=2)
    else:
        ax.set_xlabel("")
    if panel_idx % 4 == 0:
        ax.set_ylabel("True", fontsize=7.5, labelpad=2)
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=6.4)
    ax.set_yticklabels(LABELS, fontsize=6.4)
    ax.tick_params(axis="x", rotation=0, length=1.5, pad=1.5, width=0.45)
    ax.tick_params(axis="y", length=1.5, pad=1.5, width=0.45)

    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            value = cm_norm.loc[true_label, pred_label]
            if value == 0:
                text = "0%"
                text_color = "0.72"
                font_size = 4.1
            else:
                text = f"{value:.0f}%"
                text_color = "white" if value >= 55 else ("0.25" if value < 8 else "black")
                font_size = 4.3
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.25")

cbar_ax = fig.add_axes([0.935, 0.18, 0.012, 0.64])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
cbar.set_label("Row-normalized (%)", fontsize=7.5)

fig.subplots_adjust(left=0.055, right=0.915, bottom=0.095, top=0.92, wspace=0.25, hspace=0.30)

output_path = OUTPUT_DIR / "confusion_matrices_8regions_2x4.jpg"
fig.savefig(output_path, bbox_inches="tight")
clean_output_path = OUTPUT_DIR / "confusion_matrices_8regions_2x4_clean.jpg"
fig.savefig(clean_output_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {output_path}")
print(f"Saved clean figure: {clean_output_path}")


# =========================
# 5. Plot diagonal-focused version
# =========================

fig2, axes2 = plt.subplots(2, 4, figsize=FIGSIZE, dpi=DPI, constrained_layout=False)
axes2 = axes2.ravel()

for panel_idx, (ax, area) in enumerate(zip(axes2, AREAS)):
    cm_norm = normalized_matrices[area]
    accuracy = metrics_df.loc[metrics_df["area"] == area, "accuracy"].iloc[0]
    values = cm_norm.to_numpy()

    ax.imshow(diagonal_focus_rgb(values), vmin=0, vmax=1, aspect="equal")
    ax.set_title(f"{area} ({accuracy * 100:.1f}%)", fontsize=8.5, pad=4)

    if panel_idx >= 4:
        ax.set_xlabel("Predicted", fontsize=7.5, labelpad=2)
    else:
        ax.set_xlabel("")
    if panel_idx % 4 == 0:
        ax.set_ylabel("True", fontsize=7.5, labelpad=2)
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=6.4)
    ax.set_yticklabels(LABELS, fontsize=6.4)
    ax.tick_params(axis="x", rotation=0, length=1.5, pad=1.5, width=0.45)
    ax.tick_params(axis="y", length=1.5, pad=1.5, width=0.45)

    # Thin separators make the matrix readable without turning it into a heavy table.
    ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            value = values[i, j]
            if value == 0:
                text = "0%"
                text_color = "0.76"
                font_size = 3.8
            else:
                text = f"{value:.0f}%"
                if i == j and value >= 58:
                    text_color = "white"
                elif i != j and value >= 45:
                    text_color = "white"
                else:
                    text_color = "0.18"
                font_size = 4.3
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=font_size)

    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.25")

legend_ax = fig2.add_axes([0.932, 0.25, 0.025, 0.50])
legend_ax.axis("off")
legend_ax.add_patch(plt.Rectangle((0.0, 0.62), 0.32, 0.12, color=DIAG_HIGH, transform=legend_ax.transAxes))
legend_ax.text(0.42, 0.68, "Correct", transform=legend_ax.transAxes, fontsize=7.5, va="center", rotation=90)
legend_ax.add_patch(plt.Rectangle((0.0, 0.30), 0.32, 0.12, color=OFF_HIGH, transform=legend_ax.transAxes))
legend_ax.text(0.42, 0.36, "Confused", transform=legend_ax.transAxes, fontsize=7.5, va="center", rotation=90)

fig2.subplots_adjust(left=0.055, right=0.915, bottom=0.095, top=0.92, wspace=0.25, hspace=0.30)

diag_output_path = OUTPUT_DIR / "confusion_matrices_8regions_2x4_diag_focus.jpg"
fig2.savefig(diag_output_path, bbox_inches="tight")
plt.close(fig2)

print(f"Saved diagonal-focused figure: {diag_output_path}")


# =========================
# 6. Plot paper-example style
# =========================

fig3, axes3 = plt.subplots(2, 4, figsize=(13.4, 6.2), dpi=DPI, constrained_layout=False)
axes3 = axes3.ravel()

for panel_idx, (ax, area) in enumerate(zip(axes3, AREAS)):
    cm_norm = normalized_matrices[area] / 100.0
    accuracy = metrics_df.loc[metrics_df["area"] == area, "accuracy"].iloc[0]
    values = cm_norm.to_numpy()

    ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(f"{area}", fontsize=11, pad=5)

    if panel_idx >= 4:
        ax.set_xlabel("Predicted", fontsize=9, labelpad=4)
    else:
        ax.set_xlabel("")
    if panel_idx % 4 == 0:
        ax.set_ylabel("True", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=7.6)
    ax.set_yticklabels(LABELS, fontsize=7.6)
    ax.tick_params(axis="x", rotation=0, length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)

    ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = values[i, j]
            text_color = "white" if value >= 0.50 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.4,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

fig3.subplots_adjust(left=0.055, right=0.99, bottom=0.085, top=0.93, wspace=0.22, hspace=0.28)

example_output_path = OUTPUT_DIR / "confusion_matrices_8regions_2x4_example_style.jpg"
fig3.savefig(example_output_path, bbox_inches="tight")
plt.close(fig3)

print(f"Saved example-style figure: {example_output_path}")


# =========================
# 7. Plot vowel-height grouped version
# =========================

height_long_rows = []
height_metric_rows = []
height_matrices = {}
height_normalized_matrices = {}

for area in AREAS:
    df_area = load_predictions(area)
    cm = grouped_confusion_matrix(df_area, VOWEL_TO_HEIGHT, HEIGHT_LABELS)
    cm_norm = row_normalize(cm) / 100.0
    height_matrices[area] = cm
    height_normalized_matrices[area] = cm_norm

    total = int(cm.to_numpy().sum())
    correct = int(np.trace(cm.to_numpy()))
    accuracy = correct / total if total else np.nan
    height_metric_rows.append(
        {
            "area": area,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }
    )

    for true_label in HEIGHT_LABELS:
        for pred_label in HEIGHT_LABELS:
            height_long_rows.append(
                {
                    "area": area,
                    "true_group": true_label,
                    "pred_group": pred_label,
                    "count": int(cm.loc[true_label, pred_label]),
                    "row_fraction": float(cm_norm.loc[true_label, pred_label]),
                }
            )

height_long_df = pd.DataFrame(height_long_rows)
height_metrics_df = pd.DataFrame(height_metric_rows)
height_long_df.to_csv(
    OUTPUT_DIR / "confusion_matrices_8regions_height_long.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f",
)
height_metrics_df.to_csv(
    OUTPUT_DIR / "confusion_matrices_8regions_height_accuracy.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)

fig4, axes4 = plt.subplots(2, 4, figsize=(13.4, 6.2), dpi=DPI, constrained_layout=False)
axes4 = axes4.ravel()

for panel_idx, (ax, area) in enumerate(zip(axes4, AREAS)):
    values = height_normalized_matrices[area].to_numpy()
    accuracy = height_metrics_df.loc[height_metrics_df["area"] == area, "accuracy"].iloc[0]
    ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(f"{area}", fontsize=11, pad=5)

    if panel_idx >= 4:
        ax.set_xlabel("Predicted", fontsize=9, labelpad=4)
    else:
        ax.set_xlabel("")
    if panel_idx % 4 == 0:
        ax.set_ylabel("True", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(len(HEIGHT_LABELS)))
    ax.set_yticks(np.arange(len(HEIGHT_LABELS)))
    ax.set_xticklabels(HEIGHT_LABELS, fontsize=8.2)
    ax.set_yticklabels(HEIGHT_LABELS, fontsize=8.2)
    ax.tick_params(axis="x", rotation=0, length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)

    ax.set_xticks(np.arange(-0.5, len(HEIGHT_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(HEIGHT_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(HEIGHT_LABELS)):
        for j in range(len(HEIGHT_LABELS)):
            value = values[i, j]
            text_color = "white" if value >= 0.50 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

fig4.subplots_adjust(left=0.06, right=0.99, bottom=0.085, top=0.93, wspace=0.22, hspace=0.28)

height_output_path = OUTPUT_DIR / "confusion_matrices_8regions_height_2x4.jpg"
fig4.savefig(height_output_path, bbox_inches="tight")
plt.close(fig4)

print(f"Saved vowel-height figure: {height_output_path}")
print(f"Saved vowel-height long data: {OUTPUT_DIR / 'confusion_matrices_8regions_height_long.csv'}")
print(f"Saved vowel-height metrics: {OUTPUT_DIR / 'confusion_matrices_8regions_height_accuracy.csv'}")


# =========================
# 8. Plot backness and rounding grouped versions
# =========================

_, backness_normalized_matrices, _, backness_metrics_df = build_grouped_outputs(
    "backness",
    VOWEL_TO_BACKNESS,
    BACKNESS_LABELS,
)
backness_output_path = plot_grouped_matrix(
    "backness",
    BACKNESS_LABELS,
    backness_normalized_matrices,
    backness_metrics_df,
    figsize=(13.4, 5.8),
)

_, rounding_normalized_matrices, _, rounding_metrics_df = build_grouped_outputs(
    "rounding",
    VOWEL_TO_ROUNDING,
    ROUNDING_LABELS,
)
rounding_output_path = plot_grouped_matrix(
    "rounding",
    ROUNDING_LABELS,
    rounding_normalized_matrices,
    rounding_metrics_df,
    figsize=(13.4, 5.6),
)

print(f"Saved vowel-backness figure: {backness_output_path}")
print(f"Saved vowel-backness long data: {OUTPUT_DIR / 'confusion_matrices_8regions_backness_long.csv'}")
print(f"Saved vowel-backness metrics: {OUTPUT_DIR / 'confusion_matrices_8regions_backness_accuracy.csv'}")
print(f"Saved vowel-rounding figure: {rounding_output_path}")
print(f"Saved vowel-rounding long data: {OUTPUT_DIR / 'confusion_matrices_8regions_rounding_long.csv'}")
print(f"Saved vowel-rounding metrics: {OUTPUT_DIR / 'confusion_matrices_8regions_rounding_accuracy.csv'}")
print(f"Saved long data: {OUTPUT_DIR / 'confusion_matrices_8regions_long.csv'}")
print(f"Saved metrics: {OUTPUT_DIR / 'confusion_matrices_8regions_accuracy.csv'}")
