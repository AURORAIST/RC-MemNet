#!/usr/bin/env python3
"""Visualize acoustic representation differences across SSL front-end models.

This script uses existing cached features from local_hf_ssl_feature_cache.
It does not run model inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

ROOT = Path("/home/ustc1958/lxy/graph/tone/complete_package0614")
OUTPUT_DIR = ROOT / "pc_dlcmnet_figures_hybrid" / "ssl_acoustic_model_comparison"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CACHE_ROOT = ROOT / "output/0614/local_hf_ssl_feature_cache"

MODELS = [
    ("wav2vec2-base", "wav2vec2_base"),
    ("HuBERT", "hubert_base_ls960"),
    ("WavLM", "wavlm_base"),
    ("Whisper", "whisper_base"),
    ("mHuBERT-147", "mHuBERT_147"),
    ("MR-HuBERT", "MR_HuBERT"),
    ("MS-HuBERT", "MS_HuBERT"),
    ("SALMONN-proxy", "SALMONN_proxy"),
]

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

COLORS = {
    "Qingyang": "#4C78A8",
    "Tongling": "#F58518",
    "Jingxian": "#54A24B",
    "Nanling": "#E45756",
    "Ningguo": "#72B7B2",
    "Lishui": "#B279A2",
    "Chizhou": "#FF9DA6",
    "Huangshan": "#9D755D",
}

MAX_PER_AREA = 220
RANDOM_SEED = 7


def load_cache_array(model_safe: str, area: str) -> np.ndarray:
    model_dir = CACHE_ROOT / model_safe
    candidates = sorted(model_dir.glob(f"{area}_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No cache found for {model_safe}/{area}")
    # Prefer files with matching json metadata.
    for path in candidates:
        meta = path.with_suffix(".json")
        if meta.exists():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                if str(payload.get("area", area)) == area:
                    return np.load(path).astype(np.float32)
            except Exception:
                pass
    return np.load(candidates[0]).astype(np.float32)


def load_model_frame(model_name: str, model_safe: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for area in AREAS:
        x = load_cache_array(model_safe, area)
        n = len(x)
        if n > MAX_PER_AREA:
            idx = rng.choice(n, size=MAX_PER_AREA, replace=False)
            x = x[idx]
        for i in range(len(x)):
            rows.append({"model": model_name, "area": area, "feature": x[i]})
    return pd.DataFrame(rows)


def pca_points(frame: pd.DataFrame) -> pd.DataFrame:
    x = np.stack(frame["feature"].to_list()).astype(np.float32)
    x = StandardScaler().fit_transform(x)
    z = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)
    out = frame[["model", "area"]].copy()
    out["pc1"] = z[:, 0]
    out["pc2"] = z[:, 1]
    return out


def separation_score(frame: pd.DataFrame) -> dict[str, float]:
    x = np.stack(frame["feature"].to_list()).astype(np.float32)
    y = frame["area"].to_numpy()
    x = StandardScaler().fit_transform(x)
    global_mean = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for area in AREAS:
        xa = x[y == area]
        if len(xa) == 0:
            continue
        center = xa.mean(axis=0)
        between += len(xa) * float(np.sum((center - global_mean) ** 2))
        within += float(np.sum((xa - center) ** 2))
    ratio = between / max(within, 1e-8)
    return {"between_within_ratio": ratio, "between": between, "within": within}


def plot_region_pca_grid(points_by_model: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), dpi=300, sharex=False, sharey=False)
    flat_axes = axes.ravel()
    for ax, (model_name, pts) in zip(flat_axes, points_by_model.items()):
        for area in AREAS:
            d = pts[pts["area"] == area]
            ax.scatter(
                d["pc1"],
                d["pc2"],
                s=8,
                color=COLORS[area],
                alpha=0.45,
                linewidth=0,
            )
            center = d[["pc1", "pc2"]].mean()
            ax.scatter(center["pc1"], center["pc2"], s=44, color=COLORS[area], edgecolor="white", linewidth=0.6)
            ax.text(center["pc1"], center["pc2"], " " + area, fontsize=6.8, va="center")
        ax.text(0.02, 0.95, model_name, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.25)
    for ax in flat_axes[len(points_by_model) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[a], markersize=6, label=a)
        for a in AREAS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUTPUT_DIR / "ssl_models_region_pca_2x4.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def add_cov_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 3:
        return
    pts = np.column_stack([x, y])
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # 1.5-sigma ellipse: compact enough for comparing region spread.
    width, height = 2 * 1.5 * np.sqrt(np.maximum(vals, 1e-8))
    ell = Ellipse(mean, width=width, height=height, angle=angle, facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.0)
    ax.add_patch(ell)


def plot_region_ellipse_grid(points_by_model: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), dpi=300, sharex=False, sharey=False)
    flat_axes = axes.ravel()
    for ax, (model_name, pts) in zip(flat_axes, points_by_model.items()):
        for area in AREAS:
            d = pts[pts["area"] == area]
            add_cov_ellipse(ax, d["pc1"].to_numpy(), d["pc2"].to_numpy(), COLORS[area])
            center = d[["pc1", "pc2"]].mean()
            ax.scatter(center["pc1"], center["pc2"], s=48, color=COLORS[area], edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(0.02, 0.95, model_name, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.25)
        ax.autoscale()
    for ax in flat_axes[len(points_by_model) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[a], markersize=6, label=a)
        for a in AREAS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUTPUT_DIR / "ssl_models_region_ellipses_2x4.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_separation_lollipop(scores: pd.DataFrame) -> Path:
    scores = scores.sort_values("between_within_ratio")
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    y = np.arange(len(scores))
    vals = scores["between_within_ratio"].to_numpy()
    ax.hlines(y, 0, vals, color="#9AA0A6", linewidth=1.5)
    ax.scatter(vals, y, s=52, color="#0072B2", zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + vals.max() * 0.018, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(scores["model"])
    ax.set_xlabel("Region separability: between-region / within-region scatter")
    ax.set_ylabel("SSL acoustic model")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_models_region_separability_lollipop.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_centroid_paths(points_by_model: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), dpi=300)
    flat_axes = axes.ravel()
    for ax, (model_name, pts) in zip(flat_axes, points_by_model.items()):
        centers = []
        for area in AREAS:
            d = pts[pts["area"] == area]
            c = d[["pc1", "pc2"]].mean().to_numpy()
            centers.append(c)
            ax.scatter(c[0], c[1], s=56, color=COLORS[area], edgecolor="white", linewidth=0.7)
            ax.text(c[0], c[1], " " + area, fontsize=7, va="center")
        centers = np.asarray(centers)
        ax.plot(centers[:, 0], centers[:, 1], color="#5F6368", linewidth=0.9, alpha=0.6)
        ax.text(0.02, 0.95, model_name, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.set_xlabel("Centroid PC1", fontsize=8)
        ax.set_ylabel("Centroid PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.25)
    for ax in flat_axes[len(points_by_model) :]:
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "ssl_models_region_centroid_paths_2x4.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points_by_model: dict[str, pd.DataFrame] = {}
    score_rows = []
    for model_name, model_safe in MODELS:
        print(f"Loading {model_name}")
        try:
            frame = load_model_frame(model_name, model_safe)
        except FileNotFoundError as exc:
            print(f"Skip {model_name}: {exc}")
            continue
        points_by_model[model_name] = pca_points(frame)
        score_rows.append({"model": model_name, **separation_score(frame)})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUTPUT_DIR / "ssl_models_region_separability.csv", index=False, encoding="utf-8-sig")
    paths = [
        plot_region_pca_grid(points_by_model),
        plot_region_ellipse_grid(points_by_model),
        plot_centroid_paths(points_by_model),
        plot_separation_lollipop(scores),
    ]
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
