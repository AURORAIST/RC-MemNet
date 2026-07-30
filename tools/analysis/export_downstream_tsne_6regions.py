#!/usr/bin/env python3
"""Export and plot six-region downstream representation t-SNE panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.models.rc_dmnet import RCDMNet
from pc_dlcmnet.training.supervised import (
    extract_matrix_cached,
    labels_to_ids,
    load_dataset,
    resolve_device,
    set_seed,
    split_data,
)
from pc_dlcmnet.utils.tensor import masked_mean
from tools.experiments.eval_rc_checkpoint import (
    apply_cli_overrides,
    build_model_config,
    checkpoint_state,
    merged_runtime_config,
    resolve_runtime_paths,
    torch_load,
)
from tools.experiments.run_rc_dmnet import adapt_fusion, load_features
from pc_dlcmnet.data.episodes import split_global_support_query


TARGET_REGIONS = [
    ("04_Qingyang", "04青阳", "Qingyang"),
    ("06_Tongling", "06铜陵", "Tongling"),
    ("08_Jingxian", "08泾县", "Jingxian"),
    ("10_Nanling", "10南陵", "Nanling"),
    ("12_Ningguo", "12宁国", "Ningguo"),
    ("14_Lishui", "14溧水", "Lishui"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-root",
        default=str(PACKAGE_ROOT / "output/0722/main_table_rc_memnet_14targets_20260722_231203"),
    )
    parser.add_argument(
        "--rpl-root",
        default=str(PACKAGE_ROOT / "output/0722/rpl_only_6regions_20260727"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_ROOT / "output/0722/downstream_tsne_6regions_20260727"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--support-shots", type=int, default=4)
    parser.add_argument("--per-region-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def make_cli(checkpoint_path: Path, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint_path),
        output="",
        predictions_output="",
        csv="",
        device=device,
        eval_split_runs=None,
        target_support_shots=None,
        target_adapt_steps=None,
        target_adapt_lr=None,
        whisper_model="",
        cache_dir="",
        matrix_cache_dir="",
        quiet=True,
    )


def load_model_and_data(checkpoint_path: Path, device: torch.device) -> tuple[RCDMNet, argparse.Namespace, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported checkpoint: {checkpoint_path}")
    state = checkpoint_state(checkpoint)
    args = apply_cli_overrides(merged_runtime_config(checkpoint, checkpoint_path), make_cli(checkpoint_path, str(device)))
    resolve_runtime_paths(args)
    args.device = str(device)
    df = load_dataset(args)
    train_df, val_df, test_df, _split_info = split_data(df, args)
    train_x, _val_x, test_x, train_aux, _val_aux, test_aux, _matrix_meta, _aux_meta, _train_aux_meta = load_features(
        train_df,
        val_df,
        test_df,
        args,
        device,
    )
    model_config = build_model_config(checkpoint, state, args)
    model = RCDMNet(model_config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    y = labels_to_ids(test_df[args.label_column])
    return model, args, test_df, train_x, test_x, test_aux, y


def raw_backbone_features(args: argparse.Namespace, test_df: Any, query_indices: np.ndarray, device: torch.device) -> np.ndarray:
    raw, _meta = extract_matrix_cached(test_df, args, None, device, Path(args.cache_dir), "test")
    if raw.ndim == 3:
        raw = raw.mean(axis=1)
    return raw[query_indices].astype(np.float32)


def choose_query_indices(y: np.ndarray, query_seed: int, support_shots: int, per_region_class: int) -> np.ndarray:
    _support_idx, query_idx = split_global_support_query(y, support_shots=support_shots, seed=query_seed)
    rng = np.random.default_rng(query_seed)
    chosen: list[np.ndarray] = []
    for label_id in range(len(VOWEL_ORDER)):
        candidates = query_idx[y[query_idx] == label_id]
        if candidates.size == 0:
            continue
        take = min(int(per_region_class), int(candidates.size))
        chosen.append(rng.choice(candidates, size=take, replace=False))
    if not chosen:
        return np.zeros((0,), dtype=np.int64)
    out = np.concatenate(chosen).astype(np.int64)
    return out[np.argsort(out)]


@torch.no_grad()
def downstream_features(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    query_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    support_idx, _query_idx = split_global_support_query(
        y,
        support_shots=int(getattr(args, "target_support_shots", 4)),
        seed=42,
    )
    fusion_logits = adapt_fusion(model, x[support_idx], aux[support_idx], y[support_idx], args, device)
    features: list[np.ndarray] = []
    for start in range(0, len(query_indices), batch_size):
        batch_idx = query_indices[start : start + batch_size]
        speech = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device)
        aux_t = torch.as_tensor(aux[batch_idx], dtype=torch.float32, device=device)
        out = model.forward_target(speech, aux_t, fusion_logits, audio_mask=None, use_prompt=True)
        h = out["h"]
        if bool(model.config.use_memory_read):
            values = F.normalize(model.global_values, dim=-1)
            memory_read = torch.einsum("bcj,cjh->bh", out["address_weights"], values)
            feat = model.read_fusion(torch.cat([h, memory_read], dim=-1))
        else:
            feat = h
        features.append(feat.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0) if features else np.zeros((0, model.config.hidden_dim), dtype=np.float32)


def embed_2d(x: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    x_norm = normalize(x, norm="l2")
    n_components = min(50, x_norm.shape[1], x_norm.shape[0] - 1)
    x_pca = PCA(n_components=n_components, random_state=seed).fit_transform(x_norm)
    perp = min(float(perplexity), max(2.0, (x_pca.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(x_pca)


def high_dim_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(normalize(x, norm="l2"), labels, metric="cosine"))


def plot_panels_single_encoding(
    coords: dict[str, np.ndarray],
    category_values: np.ndarray,
    category_names: list[str],
    silhouettes: dict[str, float],
    out_pdf: Path,
    out_png: Path,
    legend_title: str,
    palette: list[tuple[float, float, float]] | list[tuple[float, float, float, float]],
    figure_title: str,
) -> None:
    panels = [
        ("backbone", "(a) Frozen Backbone"),
        ("rpl", "(b) RPL Only"),
        ("full", "(c) RC-MemNet"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=False)
    for ax, (key, title) in zip(axes, panels):
        z = coords[key]
        for idx, name in enumerate(category_names):
            mask = category_values == idx
            if not np.any(mask):
                continue
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=8,
                c=[palette[idx]],
                marker="o",
                linewidths=0.0,
                alpha=0.72,
            )
        ax.set_title(f"{title}\nSilhouette = {silhouettes[key]:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#B0B0B0")
    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[i], markersize=6, label=name)
        for i, name in enumerate(category_names)
    ]
    fig.legend(
        handles=color_handles,
        loc="lower center",
        ncol=min(9, len(category_names)),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=0.9,
        handletextpad=0.35,
        title=legend_title,
        title_fontsize=8,
    )
    fig.suptitle(figure_title, y=0.995, fontsize=12)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.88, bottom=0.16, wspace=0.05)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids: list[str] = []
    all_labels: list[int] = []
    all_regions: list[str] = []
    backbone_parts: list[np.ndarray] = []
    rpl_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}

    for folder, _region_zh, region_name in TARGET_REGIONS:
        print(f"[region] {region_name}", flush=True)
        rpl_ckpt = Path(args.rpl_root) / folder / "result.pt"
        full_ckpt = Path(args.full_root) / folder / "result.pt"
        if not rpl_ckpt.exists():
            raise FileNotFoundError(f"missing RPL-only checkpoint: {rpl_ckpt}")
        if not full_ckpt.exists():
            raise FileNotFoundError(f"missing full checkpoint: {full_ckpt}")

        rpl_model, rpl_args, rpl_test_df, _rpl_train_x, rpl_x, rpl_aux, y = load_model_and_data(rpl_ckpt, device)
        full_model, _full_args, full_test_df, _full_train_x, full_x, full_aux, full_y = load_model_and_data(full_ckpt, device)
        if rpl_test_df["sample_id"].astype(str).tolist() != full_test_df["sample_id"].astype(str).tolist():
            raise RuntimeError(f"sample order mismatch for {region_name}")
        if not np.array_equal(y, full_y):
            raise RuntimeError(f"label mismatch for {region_name}")

        query_idx = choose_query_indices(y, args.query_seed, args.support_shots, args.per_region_class)
        base = raw_backbone_features(rpl_args, rpl_test_df, query_idx, device)
        rpl = downstream_features(rpl_model, rpl_x, rpl_aux, y, query_idx, rpl_args, device, args.batch_size)
        full = downstream_features(full_model, full_x, full_aux, y, query_idx, _full_args, device, args.batch_size)

        sample_ids = rpl_test_df.iloc[query_idx]["sample_id"].astype(str).to_numpy()
        labels = y[query_idx]
        all_sample_ids.extend(sample_ids.tolist())
        all_labels.extend(labels.astype(int).tolist())
        all_regions.extend([region_name] * len(query_idx))
        backbone_parts.append(base)
        rpl_parts.append(rpl)
        full_parts.append(full)
        counts[region_name] = {
            vowel: int(np.sum(labels == idx))
            for idx, vowel in enumerate(VOWEL_ORDER)
        }
        print(f"  selected={len(query_idx)}", flush=True)

    labels_arr = np.asarray(all_labels, dtype=np.int64)
    regions_arr = np.asarray(all_regions)
    sample_ids_arr = np.asarray(all_sample_ids)
    backbone = np.concatenate(backbone_parts, axis=0)
    rpl = np.concatenate(rpl_parts, axis=0)
    full = np.concatenate(full_parts, axis=0)

    npz_path = out_dir / "downstream_features_6regions_seed42.npz"
    np.savez_compressed(
        npz_path,
        sample_ids=sample_ids_arr,
        labels=labels_arr,
        regions=regions_arr,
        backbone_features=backbone,
        rpl_features=rpl,
        full_features=full,
    )
    silhouettes = {
        "backbone": high_dim_silhouette(backbone, labels_arr),
        "rpl": high_dim_silhouette(rpl, labels_arr),
        "full": high_dim_silhouette(full, labels_arr),
    }
    coords = {
        "backbone": embed_2d(backbone, args.query_seed, args.perplexity),
        "rpl": embed_2d(rpl, args.query_seed, args.perplexity),
        "full": embed_2d(full, args.query_seed, args.perplexity),
    }
    coord_path = out_dir / "downstream_tsne_coords_6regions_seed42.npz"
    np.savez_compressed(coord_path, labels=labels_arr, regions=regions_arr, sample_ids=sample_ids_arr, **coords)
    vowel_palette = list(plt.get_cmap("tab10").colors[: len(VOWEL_ORDER)])
    region_order = [name for _folder, _zh, name in TARGET_REGIONS]
    region_to_idx = {name: idx for idx, name in enumerate(region_order)}
    region_ids = np.asarray([region_to_idx[name] for name in regions_arr], dtype=np.int64)
    region_palette = list(plt.get_cmap("tab20").colors[: len(region_order)])
    plot_panels_single_encoding(
        coords,
        labels_arr,
        list(VOWEL_ORDER),
        silhouettes,
        out_dir / "fig_downstream_representation_vowels_6regions.pdf",
        out_dir / "fig_downstream_representation_vowels_6regions.png",
        "Vowels",
        vowel_palette,
        "Downstream t-SNE colored by vowels",
    )
    plot_panels_single_encoding(
        coords,
        region_ids,
        region_order,
        silhouettes,
        out_dir / "fig_downstream_representation_regions_6regions.pdf",
        out_dir / "fig_downstream_representation_regions_6regions.png",
        "Regions",
        region_palette,
        "Downstream t-SNE colored by regions",
    )
    manifest = {
        "sample_count": int(labels_arr.size),
        "feature_shapes": {
            "backbone": list(backbone.shape),
            "rpl": list(rpl.shape),
            "full": list(full.shape),
        },
        "target_regions": TARGET_REGIONS,
        "query_seed": int(args.query_seed),
        "support_shots": int(args.support_shots),
        "per_region_class": int(args.per_region_class),
        "counts": counts,
        "silhouette_cosine_high_dim": silhouettes,
        "dimensionality_note": (
            "Frozen Whisper pooled features and RC-MemNet hidden features have different original dimensions; "
            "each panel uses the same PCA/t-SNE hyperparameters but a separate PCA+t-SNE fit."
        ),
        "files": {
            "features": str(npz_path),
            "coords": str(coord_path),
            "vowel_pdf": str(out_dir / "fig_downstream_representation_vowels_6regions.pdf"),
            "vowel_png": str(out_dir / "fig_downstream_representation_vowels_6regions.png"),
            "region_pdf": str(out_dir / "fig_downstream_representation_regions_6regions.pdf"),
            "region_png": str(out_dir / "fig_downstream_representation_regions_6regions.png"),
        },
    }
    manifest_path = out_dir / "downstream_tsne_6regions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
