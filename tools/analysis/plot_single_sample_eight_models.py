#!/usr/bin/env python3
"""Visualize one fixed speech sample through eight trained regional models.

Outputs:
- acoustic_single_sample_mel_mfcc_f0.jpg
- single_sample_eight_models_probabilities.jpg
- single_sample_eight_models_prompt_routing.jpg
- single_sample_eight_models_pcmr_attention.jpg
- single_sample_eight_models_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

OUTPUT_DIR = PACKAGE_ROOT / "pc_dlcmnet_figures_hybrid" / "single_sample_eight_models"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUTPUT_DIR / ".numba"))

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from pc_dlcmnet.data.acoustic_features import apply_vector_norm, extract_aux_matrix_cached
from pc_dlcmnet.data.episodes import make_episode_tensors, split_global_support_query
from pc_dlcmnet.evaluation.mechanism import _attention_without_prompt, _mean_class_key_tensor
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.models.network import build_paper_model
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    extract_matrix_cached,
    infer_aux_branch_dims,
    labels_to_ids,
    load_dataset,
    load_model_init_checkpoint,
    resolve_device,
    split_data,
)


RUN_DIR = PACKAGE_ROOT / "output/0614/hyperparam_sensitivity_8targets_k4_full_20260703_4sweeps"
SETTING = "num_prompts_8"
DEVICE = "cuda"
SUPPORT_K = 4
SEED = 0

# 固定同一条 query。也可以改成 None，让脚本自动选一条。
QUERY_SAMPLE_ID = "row_00009625"

REGIONS = [
    ("Qingyang", "04青阳"),
    ("Tongling", "06铜陵"),
    ("Jingxian", "08泾县"),
    ("Nanling", "10南陵"),
    ("Ningguo", "12宁国"),
    ("Lishui", "14溧水"),
    ("Chizhou", "03池州"),
    ("Huangshan", "11黄山"),
]


def namespace_from_checkpoint_config(config: dict, device: str) -> SimpleNamespace:
    cfg = dict(config)
    cfg["device"] = device
    cfg.setdefault("path_column", "wav_path")
    cfg.setdefault("start_column", "start_time")
    cfg.setdefault("end_column", "end_time")
    cfg.setdefault("label_column", "vowel")
    cfg.setdefault("region_column", "region")
    cfg.setdefault("speaker_column", "speaker_id")
    cfg.setdefault("labels", None)
    cfg.setdefault("feature_norm", "global")
    cfg.setdefault("matrix_cache_scan", True)
    cfg.setdefault("aux_source", "acoustic")
    cfg.setdefault("aux_representation", "sequence")
    cfg.setdefault("aux_n_mfcc", 39)
    cfg.setdefault("aux_n_mels", 128)
    cfg.setdefault("aux_f0_segments", 5)
    cfg.setdefault("aux_use_delta_mfcc", True)
    cfg.setdefault("aux_use_formants", True)
    cfg.setdefault("support_write_mode", "pseudo")
    cfg.setdefault("support_label_blend", 0.0)
    cfg.setdefault("num_prompts", 8)
    cfg.setdefault("hidden_dim", 256)
    cfg.setdefault("score_dim", 128)
    cfg.setdefault("prompt_dim", 128)
    cfg.setdefault("layers", 3)
    cfg.setdefault("heads", 4)
    cfg.setdefault("ffn_dim", 768)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("temperature", 0.5)
    return SimpleNamespace(**cfg)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_query_row(df: pd.DataFrame, sample_id: str | None) -> pd.DataFrame:
    if sample_id:
        match = df[df["sample_id"].astype(str) == str(sample_id)]
        if not match.empty:
            return match.iloc[[0]].reset_index(drop=True)
        print(f"[warn] sample_id={sample_id} not found; using first row.")
    return df.iloc[[0]].reset_index(drop=True)


def load_region_model(region_dir: Path, device: torch.device):
    ckpt_path = region_dir / "result.pt"
    ckpt = load_checkpoint(ckpt_path, device)
    args = namespace_from_checkpoint_config(ckpt["config"], str(device))

    df = load_dataset(args)
    train_df, _val_df, test_df, _split_info = split_data(df, args)
    train_x_raw, _ = extract_matrix_cached(train_df, args, None, device, Path(args.cache_dir), "train")
    test_x_raw, _ = extract_matrix_cached(test_df, args, None, device, Path(args.cache_dir), "test")
    train_x = apply_token_norm(train_x_raw, ckpt["token_norm"])
    test_x = apply_token_norm(test_x_raw, ckpt["token_norm"])

    train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, args, train_x_raw, "train")
    test_aux_raw, _ = extract_aux_matrix_cached(test_df, args, test_x_raw, "test")
    train_aux = apply_vector_norm(train_aux_raw, ckpt["aux_norm"])
    test_aux = apply_vector_norm(test_aux_raw, ckpt["aux_norm"])

    model = build_paper_model(
        input_dim=train_x.shape[-1],
        aux_dim=train_aux.shape[-1],
        aux_branch_dims=infer_aux_branch_dims(train_aux_meta, train_aux),
        hidden_dim=args.hidden_dim,
        score_dim=args.score_dim,
        prompt_dim=args.prompt_dim,
        num_prompts=args.num_prompts,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_audio_tokens=train_x.shape[1],
        temperature=args.temperature,
    )
    load_model_init_checkpoint(model, str(ckpt_path), device)
    model.to(device).eval()
    return args, model, df, test_df.reset_index(drop=True), test_x, test_aux


def extract_query_features(query_row: pd.DataFrame, args, device: torch.device, ckpt: dict):
    query_x_raw, _ = extract_matrix_cached(query_row, args, None, device, Path(args.cache_dir), "single_query")
    query_x = apply_token_norm(query_x_raw, ckpt["token_norm"])
    query_aux_raw, _ = extract_aux_matrix_cached(query_row, args, query_x_raw, "single_query")
    query_aux = apply_vector_norm(query_aux_raw, ckpt["aux_norm"])
    return query_x, query_aux


def plot_acoustic(query_row: pd.DataFrame, out_dir: Path) -> Path:
    row = query_row.iloc[0]
    wav_path = Path(row["wav_path"])
    start = float(row["start_time"])
    end = float(row["end_time"])
    y, sr = librosa.load(wav_path, sr=16000, mono=True, offset=start, duration=max(0.05, end - start))
    y, _ = librosa.effects.trim(y, top_db=35)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=80, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=128)
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=512, hop_length=128)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=128)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), dpi=300)
    librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, hop_length=128, ax=axes[0], cmap="magma")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel freq.")
    librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=128, ax=axes[1], cmap="coolwarm")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MFCC")
    axes[2].plot(times, f0, color="#1f77b4", linewidth=1.2)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("F0 (Hz)")
    axes[2].grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    path = out_dir / "acoustic_single_sample_mel_mfcc_f0.jpg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def infer_one_model(region_name: str, region_code: str, query_row: pd.DataFrame, device: torch.device) -> dict:
    region_dir = RUN_DIR / SETTING / region_name
    ckpt = load_checkpoint(region_dir / "result.pt", device)
    args, model, _df, test_df, test_x, test_aux = load_region_model(region_dir, device)
    query_x, query_aux = extract_query_features(query_row, args, device, ckpt)
    test_y = labels_to_ids(test_df[args.label_column])
    support_idx, _ = split_global_support_query(test_y, SUPPORT_K, seed=int(args.seed) + 700_000 + SUPPORT_K * 9_973)

    support_speech, support_aux, support_mask = make_episode_tensors(test_x, test_aux, support_idx, device)
    support_labels = torch.tensor(test_y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
    query_speech, query_aux_t, query_mask = make_episode_tensors(query_x, query_aux, np.asarray([0], dtype=np.int64), device)
    label_blend = 0.0 if args.support_write_mode == "pseudo" else (float(args.support_label_blend) if args.support_write_mode == "blend" else 1.0)

    with torch.no_grad():
        build = model.build_episode_memory(
            support_speech,
            support_aux,
            support_mask,
            support_labels=support_labels,
            label_blend=label_blend,
        )
        out = model.predict(query_speech, query_aux_t, query_mask, build["episode_memory"])
        prob = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()[0]
        route_tokens = out["route_weights"].detach().cpu().numpy()[0]
        route = route_tokens.mean(axis=0)
        pooled_audio = out["pooled_audio"].detach().cpu().numpy()[0]
        query_vec = out["query"].detach().cpu().numpy()[0]
        h0_tokens = out["h0"].detach().cpu().numpy()[0]
        audio_state_tokens = out["audio_state"].detach().cpu().numpy()[0]
        feature_weights = out.get("feature_weights")
        if feature_weights is not None:
            feature_weights_np = feature_weights.detach().cpu().numpy()[0]
        else:
            feature_weights_np = np.zeros((h0_tokens.shape[0], 1), dtype=np.float32)
        token_count = query_speech.shape[1]
        attn_no_prompt = _mean_class_key_tensor(_attention_without_prompt(out["acoustic_scores"]), token_count, len(VOWEL_ORDER))[0]
        attn_prompt = _mean_class_key_tensor(out["attention"], token_count, len(VOWEL_ORDER))[0]
        acoustic_score = _mean_class_key_tensor(out["acoustic_scores"], token_count, len(VOWEL_ORDER))[0]
        prompt_bias = _mean_class_key_tensor(out["prompt_bias"], token_count, len(VOWEL_ORDER))[0]

    pred_idx = int(prob.argmax())
    true_class = str(query_row.iloc[0][args.label_column])
    return {
        "model": region_name,
        "region_code": region_code,
        "true_class": true_class,
        "pred_class": VOWEL_ORDER[pred_idx],
        "confidence": float(prob[pred_idx]),
        "prob": prob,
        "route": route,
        "route_tokens": route_tokens,
        "attention_without_prompt": attn_no_prompt,
        "attention_with_prompt": attn_prompt,
        "attention_shift": attn_prompt - attn_no_prompt,
        "acoustic_score": acoustic_score,
        "prompt_bias": prompt_bias,
        "pooled_audio": pooled_audio,
        "query_vec": query_vec,
        "h0_tokens": h0_tokens,
        "audio_state_tokens": audio_state_tokens,
        "feature_weights": feature_weights_np,
    }


def save_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], ylabel: str, xlabel: str, path: Path, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_bars(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.2), dpi=300, sharey=True)
    colors = ["#8E8E93"] * len(VOWEL_ORDER)
    for ax, row in zip(axes.ravel(), rows):
        prob = row["prob"]
        pred_idx = int(np.argmax(prob))
        true_idx = list(VOWEL_ORDER).index(row["true_class"]) if row["true_class"] in VOWEL_ORDER else -1
        bar_colors = colors.copy()
        bar_colors[pred_idx] = "#D55E00"
        if true_idx >= 0:
            bar_colors[true_idx] = "#0072B2" if true_idx != pred_idx else "#009E73"
        ax.bar(np.arange(len(VOWEL_ORDER)), prob, color=bar_colors, width=0.68)
        ax.axhline(1.0 / len(VOWEL_ORDER), color="#A0A0A0", linestyle="--", linewidth=0.7)
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"pred={row['pred_class']}, conf={row['confidence']:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_xticks(np.arange(len(VOWEL_ORDER)))
        ax.set_xticklabels(VOWEL_ORDER)
        ax.set_xlabel("Vowel class", fontsize=8)
        ax.set_ylabel("Probability", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_probability_bars_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_prompt_lollipops(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.2), dpi=300, sharey=True)
    for ax, row in zip(axes.ravel(), rows):
        route = row["route"]
        x = np.arange(len(route))
        main = int(np.argmax(route))
        ax.vlines(x, 0, route, color="#9AA0A6", linewidth=1.0)
        ax.scatter(x, route, s=28, color="#4C78A8", zorder=3)
        ax.scatter([main], [route[main]], s=54, color="#E45756", zorder=4)
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"main prompt={main}", transform=ax.transAxes, fontsize=8)
        ax.set_xticks(x)
        ax.set_xlabel("Prompt", fontsize=8)
        ax.set_ylabel("Routing weight", fontsize=8)
        ax.set_ylim(0, max(0.6, float(route.max()) * 1.15))
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_prompt_lollipop_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_embedding_pca(rows: list[dict], out_dir: Path) -> None:
    model_labels = [r["model"] for r in rows]
    pooled = np.stack([r["pooled_audio"] for r in rows])
    query = np.stack([r["query_vec"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), dpi=300)
    for ax, mat, title in zip(axes, [pooled, query], ["Pooled audio representation", "Projected query representation"]):
        z = PCA(n_components=2, random_state=0).fit_transform(mat)
        for i, label in enumerate(model_labels):
            ax.scatter(z[i, 0], z[i, 1], s=46, color="#0072B2")
            ax.text(z[i, 0], z[i, 1], " " + label, fontsize=8, va="center")
        ax.axhline(0, color="#D0D0D0", linewidth=0.7)
        ax.axvline(0, color="#D0D0D0", linewidth=0.7)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_embedding_pca.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_decision_summary(rows: list[dict], out_dir: Path) -> None:
    labels = [r["model"] for r in rows]
    conf = np.asarray([r["confidence"] for r in rows], dtype=float)
    correct = [r["pred_class"] == r["true_class"] for r in rows]
    colors = ["#009E73" if ok else "#D55E00" for ok in correct]
    fig, ax = plt.subplots(figsize=(7.0, 3.3), dpi=300)
    y = np.arange(len(rows))
    ax.barh(y, conf, color=colors, height=0.62)
    for pos, row in enumerate(rows):
        ax.text(conf[pos] + 0.008, pos, f"{row['pred_class']}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Regional model")
    ax.set_xlim(0, max(0.45, float(conf.max()) * 1.25))
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_decision_summary.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_prompt_temporal_stack(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), dpi=300, sharex=True, sharey=True)
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]
    for ax, row in zip(axes.ravel(), rows):
        route = np.asarray(row["route_tokens"], dtype=float)
        x = np.linspace(0.0, 1.0, route.shape[0])
        ax.stackplot(x, route.T, colors=palette[: route.shape[1]], alpha=0.88, linewidth=0)
        dominant = route.argmax(axis=1)
        changes = int(np.sum(dominant[1:] != dominant[:-1])) if len(dominant) > 1 else 0
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"main={int(row['route'].argmax())}, switches={changes}", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Normalized speech time", fontsize=8)
        ax.set_ylabel("Prompt share", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.35, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=palette[i], linewidth=5, label=f"P{i}")
        for i in range(min(8, rows[0]["route_tokens"].shape[1]))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "single_sample_eight_models_prompt_temporal_stack_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_probability_lollipops(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4), dpi=300, sharey=True)
    for ax, row in zip(axes.ravel(), rows):
        prob = np.asarray(row["prob"], dtype=float)
        x = np.arange(len(VOWEL_ORDER))
        pred_idx = int(prob.argmax())
        true_idx = list(VOWEL_ORDER).index(row["true_class"]) if row["true_class"] in VOWEL_ORDER else -1
        colors = np.array(["#9AA0A6"] * len(VOWEL_ORDER), dtype=object)
        colors[pred_idx] = "#D55E00"
        if true_idx >= 0:
            colors[true_idx] = "#009E73" if true_idx == pred_idx else "#0072B2"
        ax.vlines(x, 0, prob, color=colors, linewidth=1.8)
        ax.scatter(x, prob, color=colors, s=42, zorder=3)
        ax.axhline(1.0 / len(VOWEL_ORDER), color="#A0A0A0", linestyle="--", linewidth=0.7)
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"{row['true_class']} -> {row['pred_class']}", transform=ax.transAxes, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(VOWEL_ORDER)
        ax.set_xlabel("Vowel class", fontsize=8)
        ax.set_ylabel("Probability", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_probability_lollipop_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_acoustic_state_trajectories(rows: list[dict], out_dir: Path) -> None:
    all_tokens = np.concatenate([np.asarray(r["audio_state_tokens"], dtype=float) for r in rows], axis=0)
    z_all = PCA(n_components=2, random_state=0).fit_transform(all_tokens)
    cursor = 0
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), dpi=300, sharex=True, sharey=True)
    for ax, row in zip(axes.ravel(), rows):
        n = int(np.asarray(row["audio_state_tokens"]).shape[0])
        z = z_all[cursor : cursor + n]
        cursor += n
        t = np.linspace(0.0, 1.0, n)
        ax.plot(z[:, 0], z[:, 1], color="#4C78A8", linewidth=1.4, alpha=0.9)
        sc = ax.scatter(z[:, 0], z[:, 1], c=t, cmap="viridis", s=24, edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(z[0, 0], z[0, 1], marker="o", s=46, color="#009E73", edgecolor="white", linewidth=0.5, zorder=4)
        ax.scatter(z[-1, 0], z[-1, 1], marker="s", s=46, color="#D55E00", edgecolor="white", linewidth=0.5, zorder=4)
        path_len = float(np.linalg.norm(np.diff(z, axis=0), axis=1).sum()) if n > 1 else 0.0
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"path={path_len:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Acoustic PC1", fontsize=8)
        ax.set_ylabel("Acoustic PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.25)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.98, bottom=0.16, wspace=0.18, hspace=0.28)
    cax = fig.add_axes([0.32, 0.06, 0.36, 0.018])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Normalized speech time", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_dir / "single_sample_eight_models_acoustic_state_trajectories_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_acoustic_fingerprint_curves(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), dpi=300, sharex=True, sharey=True)
    for ax, row in zip(axes.ravel(), rows):
        vec = np.asarray(row["pooled_audio"], dtype=float)
        x = np.arange(len(vec))
        ax.plot(x, vec, color="#4C78A8", linewidth=0.9)
        ax.axhline(0, color="#A0A0A0", linewidth=0.7)
        top = np.argsort(np.abs(vec))[-8:]
        ax.scatter(top, vec[top], s=18, color="#D55E00", zorder=3)
        energy = float(np.linalg.norm(vec))
        spread = float(np.std(vec))
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(0.02, 0.80, f"norm={energy:.2f}, std={spread:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("Hidden acoustic dimension", fontsize=8)
        ax.set_ylabel("Activation", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_acoustic_fingerprint_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_acoustic_branch_weights(rows: list[dict], out_dir: Path) -> None:
    branch_dim = int(np.asarray(rows[0]["feature_weights"]).shape[-1])
    if branch_dim <= 1:
        return
    labels = [f"B{i}" for i in range(branch_dim)]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4), dpi=300, sharey=True)
    for ax, row in zip(axes.ravel(), rows):
        weights = np.asarray(row["feature_weights"], dtype=float)
        mean = weights.mean(axis=0) if weights.ndim == 2 else weights
        x = np.arange(branch_dim)
        ax.vlines(x, 0, mean, color="#9AA0A6", linewidth=1.5)
        ax.scatter(x, mean, color="#0072B2", s=38, zorder=3)
        ax.text(0.02, 0.92, row["model"], transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Acoustic branch", fontsize=8)
        ax.set_ylabel("Mean gate weight", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "single_sample_eight_models_acoustic_branch_weights_2x4.jpg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(DEVICE)

    first_ckpt = load_checkpoint(RUN_DIR / SETTING / REGIONS[0][0] / "result.pt", device)
    first_args = namespace_from_checkpoint_config(first_ckpt["config"], str(device))
    full_df = load_dataset(first_args)
    query_row = choose_query_row(full_df, QUERY_SAMPLE_ID)
    query_row.to_csv(OUTPUT_DIR / "single_sample_metadata.csv", index=False, encoding="utf-8-sig")
    plot_acoustic(query_row, OUTPUT_DIR)

    rows = [infer_one_model(name, code, query_row, device) for name, code in REGIONS]
    model_labels = [r["model"] for r in rows]
    prompt_labels = [str(i) for i in range(len(rows[0]["route"]))]

    prob = np.stack([r["prob"] for r in rows])
    route = np.stack([r["route"] for r in rows])
    attn = np.stack([r["attention_with_prompt"] for r in rows])
    shift = np.stack([r["attention_shift"] for r in rows])

    save_heatmap(prob, model_labels, list(VOWEL_ORDER), "Regional model", "Vowel class", OUTPUT_DIR / "single_sample_eight_models_probabilities.jpg", "magma")
    save_heatmap(route, model_labels, prompt_labels, "Regional model", "Prompt", OUTPUT_DIR / "single_sample_eight_models_prompt_routing.jpg", "YlGnBu")
    save_heatmap(attn, model_labels, list(VOWEL_ORDER), "Regional model", "Memory class", OUTPUT_DIR / "single_sample_eight_models_pcmr_attention.jpg", "viridis")
    save_heatmap(shift, model_labels, list(VOWEL_ORDER), "Regional model", "Memory class", OUTPUT_DIR / "single_sample_eight_models_attention_shift.jpg", "coolwarm")
    plot_probability_bars(rows, OUTPUT_DIR)
    plot_prompt_lollipops(rows, OUTPUT_DIR)
    plot_embedding_pca(rows, OUTPUT_DIR)
    plot_decision_summary(rows, OUTPUT_DIR)
    plot_prompt_temporal_stack(rows, OUTPUT_DIR)
    plot_probability_lollipops(rows, OUTPUT_DIR)
    plot_acoustic_state_trajectories(rows, OUTPUT_DIR)
    plot_acoustic_fingerprint_curves(rows, OUTPUT_DIR)
    plot_acoustic_branch_weights(rows, OUTPUT_DIR)

    summary = []
    for r in rows:
        row = {
            "model": r["model"],
            "region_code": r["region_code"],
            "query_sample_id": str(query_row.iloc[0]["sample_id"]),
            "query_region": str(query_row.iloc[0][first_args.region_column]),
            "query_site": str(query_row.iloc[0].get("site", "")),
            "true_class": r["true_class"],
            "pred_class": r["pred_class"],
            "confidence": r["confidence"],
        }
        row.update({f"prob_{label}": float(r["prob"][i]) for i, label in enumerate(VOWEL_ORDER)})
        row.update({f"prompt_{i}": float(v) for i, v in enumerate(r["route"])})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "single_sample_eight_models_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
