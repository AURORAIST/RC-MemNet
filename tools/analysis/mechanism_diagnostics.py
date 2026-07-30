"""Mechanism visualizations for PC-DLCMNet.

This script extracts fine-grained evidence from trained checkpoints:
- GEMA write gate by class, region, and support K.
- PCMR query-to-class-memory attention distributions.
- t-SNE representations for Whisper-only, prototype, and PC-DLCMNet features.

The figures are diagnostic. They should be interpreted from the measured
values, not forced to match an expected trend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from pc_dlcmnet.data.acoustic_features import (
    apply_vector_norm,
    extract_aux_matrix_cached,
    fit_vector_norm,
)
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.data.episodes import (
    build_episode_groups,
    make_episode_tensors,
    split_support_query,
)
from pc_dlcmnet.models.network import build_paper_model
from pc_dlcmnet.utils.tensor import masked_mean
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    default_old_root,
    extract_matrix_cached,
    fit_token_norm,
    infer_aux_branch_dims,
    labels_to_ids,
    load_dataset,
    load_model_init_checkpoint,
    resolve_device,
    set_seed,
    split_data,
)


OUT_DEFAULT = Path("output/0614/paper_ready_results/08_mechanism_diagnostics")


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="output/0614/multiregion_paper_dual_memory")
    parser.add_argument("--paper-ready-root", default="output/0614/paper_ready_results")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--regions", nargs="*", default=["01当涂", "02芜湖", "06铜陵", "08泾县", "11黄山", "13高淳"])
    parser.add_argument("--k-values", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--max-episodes", type=int, default=24)
    parser.add_argument("--max-tsne-samples", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def region_plot_label(region: str) -> str:
    mapping = {
        "01当涂": "R01",
        "02芜湖": "R02",
        "06铜陵": "R06",
        "08泾县": "R08",
        "11黄山": "R11",
        "13高淳": "R13",
    }
    if str(region) in mapping:
        return mapping[str(region)]
    prefix = str(region)[:2]
    return f"R{prefix}" if prefix.isdigit() else str(region)


def find_region_run(result_root: Path, region: str) -> tuple[Path, Path]:
    for json_path in sorted(result_root.glob("*/*_stage3.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {})
        if str(cfg.get("holdout_region", "")) == str(region):
            ckpt = json_path.with_suffix(".pt")
            if ckpt.exists():
                return json_path, ckpt
    raise FileNotFoundError(f"No stage3 result/checkpoint found for region: {region}")


def namespace_from_config(config: dict, csv_path: str, device: str) -> SimpleNamespace:
    old_root = default_old_root()
    defaults = {
        "csv": csv_path,
        "path_column": "wav_path",
        "start_column": "start_time",
        "end_column": "end_time",
        "label_column": "vowel",
        "region_column": "region",
        "speaker_column": "speaker_id",
        "labels": None,
        "protocol": "loro",
        "holdout_region": "",
        "val_size": 0.1,
        "test_size": 0.1,
        "train_fraction": 1.0,
        "seed": 0,
        "cache_dir": str(old_root / "output/salmonn_style_whisper_cache_base"),
        "matrix_cache_dir": str(old_root / "output/ppm_supervised_matrix_cache"),
        "feature_norm": "global",
        "matrix_cache_scan": True,
        "whisper_model": "base",
        "device": device,
        "target_sr": 16000,
        "token_chunks": 32,
        "audio_root": "",
        "aux_source": "acoustic",
        "aux_representation": "sequence",
        "aux_cache_dir": "output/0614/feature_memory_aux_cache",
        "old_feature_cache_dir": str(old_root / "output/feature_cache_vowel"),
        "aux_n_mfcc": 39,
        "aux_n_mels": 128,
        "aux_f0_segments": 5,
        "aux_use_delta_mfcc": True,
        "aux_use_formants": True,
        "episode_column": "",
        "episode_length": 32,
        "episode_batch_size": 2,
        "support_shots": 2,
        "eval_support_shots": 4,
        "support_write_mode": "pseudo",
        "support_label_blend": 0.0,
        "eval_query_only": True,
        "eval_split_mode": "rolling",
        "eval_query_shots_per_class": 1,
        "eval_classifier": "memory",
        "prototype_fallback_global": True,
        "prototype_support_weight": 1.0,
        "hidden_dim": 256,
        "score_dim": 128,
        "prompt_dim": 128,
        "num_prompts": 8,
        "layers": 3,
        "heads": 4,
        "ffn_dim": 768,
        "dropout": 0.1,
        "temperature": 0.2,
    }
    defaults.update(config)
    defaults["csv"] = csv_path
    defaults["device"] = device
    return SimpleNamespace(**defaults)


def load_region_bundle(json_path: Path, ckpt_path: Path, csv_path: str, device: torch.device) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    args = namespace_from_config(payload.get("config", {}), csv_path, str(device))
    df = load_dataset(args)
    train_df, _val_df, test_df, _split_info = split_data(df, args)
    cache_dir = Path(args.cache_dir)
    train_x_raw, _ = extract_matrix_cached(train_df, args, None, device, cache_dir, "train")
    test_x_raw, _ = extract_matrix_cached(test_df, args, None, device, cache_dir, "test")
    token_norm = fit_token_norm(train_x_raw, args.feature_norm)
    train_x = apply_token_norm(train_x_raw, token_norm)
    test_x = apply_token_norm(test_x_raw, token_norm)

    train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, args, train_x_raw, "train")
    test_aux_raw, _test_aux_meta = extract_aux_matrix_cached(test_df, args, test_x_raw, "test")
    aux_norm = fit_vector_norm(train_aux_raw, "global")
    train_aux = apply_vector_norm(train_aux_raw, aux_norm)
    test_aux = apply_vector_norm(test_aux_raw, aux_norm)
    test_y = labels_to_ids(test_df[args.label_column])
    groups, _ = build_episode_groups(test_df, args)

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
    model.to(device)
    model.eval()
    return {
        "args": args,
        "model": model,
        "test_df": test_df.reset_index(drop=True),
        "x": test_x,
        "aux": test_aux,
        "y": test_y,
        "groups": groups,
    }


def memory_attention_distribution(attention: torch.Tensor, token_count: int, num_classes: int) -> torch.Tensor:
    # attention: [B, H, T, T+C]. Average heads and audio-query tokens, keep class-memory keys.
    mem_attn = attention[:, :, :, token_count : token_count + num_classes]
    dist = mem_attn.mean(dim=(1, 2))
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def collect_region_diagnostics(bundle: dict, region: str, k_values: list[int], max_episodes: int, seed: int, device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    model = bundle["model"]
    x = bundle["x"]
    aux = bundle["aux"]
    y = bundle["y"]
    groups = list(bundle["groups"])[:max_episodes]
    gate_rows: list[dict] = []
    attn_rows: list[dict] = []
    embed_rows: list[dict] = []
    with torch.no_grad():
        for k in k_values:
            for ep_pos, indices in enumerate(groups):
                support_idx, query_idx = split_support_query(indices, y, k, seed=seed + ep_pos * 100 + k)
                if len(query_idx) == 0:
                    continue
                support_speech, support_aux, support_mask = make_episode_tensors(x, aux, support_idx, device)
                support_labels = torch.tensor(y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
                query_speech, query_aux, query_mask = make_episode_tensors(x, aux, query_idx, device)

                build = model.build_episode_memory(
                    support_speech,
                    support_aux,
                    support_mask,
                    support_labels=support_labels,
                    label_blend=1.0,
                )
                class_gate = build["write_gate"].mean(dim=-1).detach().cpu().numpy()
                support_counts = np.bincount(y[support_idx], minlength=len(VOWEL_ORDER)) if len(support_idx) else np.zeros(len(VOWEL_ORDER), dtype=int)
                for c, label in enumerate(VOWEL_ORDER):
                    gate_rows.append(
                        {
                            "region": region,
                            "episode": ep_pos,
                            "k": k,
                            "class_id": c,
                            "class": label,
                            "gate_global_weight": float(class_gate[c]),
                            "support_injection_weight": float(1.0 - class_gate[c]),
                            "support_count": int(support_counts[c]),
                        }
                    )

                ep_out = model.predict(query_speech, query_aux, query_mask, build["episode_memory"])
                gl_out = model.predict(query_speech, query_aux, query_mask, model.global_memory)
                proto_features = model.encode_inputs(query_speech, query_aux, query_mask)
                whisper_vec = query_speech.mean(dim=1).detach().cpu().numpy()
                proto_vec = masked_mean(proto_features["h0"], query_mask).detach().cpu().numpy()
                pcd_vec = ep_out["pooled_audio"].detach().cpu().numpy()
                ep_pred = ep_out["logits"].argmax(dim=-1).detach().cpu().numpy()
                gl_pred = gl_out["logits"].argmax(dim=-1).detach().cpu().numpy()
                ep_prob = torch.softmax(ep_out["logits"], dim=-1).detach().cpu().numpy()
                attn = memory_attention_distribution(ep_out["attention"], query_speech.shape[1], len(VOWEL_ORDER)).detach().cpu().numpy()
                for local_i, row_idx in enumerate(query_idx.tolist()):
                    true_id = int(y[row_idx])
                    for c, label in enumerate(VOWEL_ORDER):
                        attn_rows.append(
                            {
                                "region": region,
                                "episode": ep_pos,
                                "k": k,
                                "row_index": int(row_idx),
                                "true_id": true_id,
                                "true_class": VOWEL_ORDER[true_id],
                                "memory_class_id": c,
                                "memory_class": label,
                                "attention": float(attn[local_i, c]),
                                "probability": float(ep_prob[local_i, c]),
                                "episode_pred": int(ep_pred[local_i]),
                                "global_pred": int(gl_pred[local_i]),
                            }
                        )
                    if k == max(k_values):
                        for name, vec in [("whisper", whisper_vec[local_i]), ("prototype_h0", proto_vec[local_i]), ("pcdlcmnet", pcd_vec[local_i])]:
                            embed_rows.append(
                                {
                                    "region": region,
                                    "row_index": int(row_idx),
                                    "true_id": true_id,
                                    "true_class": VOWEL_ORDER[true_id],
                                    "feature_type": name,
                                    **{f"f{j}": float(v) for j, v in enumerate(vec.tolist())},
                                }
                            )
    return gate_rows, attn_rows, embed_rows


def plot_gate_figures(gate_df: pd.DataFrame, out_dir: Path) -> None:
    summary = gate_df.groupby(["k", "class"], as_index=False)["support_injection_weight"].mean()
    pivot = summary.pivot(index="class", columns="k", values="support_injection_weight").reindex(VOWEL_ORDER)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    values = pivot.values
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad = max(0.01, (vmax - vmin) * 0.08)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=max(0.0, vmin - pad), vmax=min(1.0, vmax + pad))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"K={c}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("GEMA support injection by class and K")
    ax.set_xlabel("Support shots per class")
    ax.set_ylabel("Vowel class")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Support injection weight (1 - gate)")
    save(fig, out_dir, "fig7_gema_gate_class_k_heatmap")

    region_summary = gate_df.groupby(["region", "k"], as_index=False)["support_injection_weight"].mean()
    region_summary["region_plot"] = region_summary["region"].map(region_plot_label)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for region, group in region_summary.groupby("region_plot"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["support_injection_weight"], marker="o", label=region)
    ax.set_title("GEMA support injection changes with K")
    ax.set_xlabel("Support shots per class (K)")
    ax.set_ylabel("Mean support injection weight")
    ymin = float(region_summary["support_injection_weight"].min())
    ymax = float(region_summary["support_injection_weight"].max())
    pad = max(0.005, (ymax - ymin) * 0.25)
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))
    ax.legend(frameon=False, ncol=2)
    save(fig, out_dir, "fig8_gema_gate_region_k_lines")


def plot_attention_figures(attn_df: pd.DataFrame, out_dir: Path) -> None:
    k = int(attn_df["k"].max())
    d = attn_df[attn_df["k"] == k]
    matrix = d.groupby(["true_class", "memory_class"], as_index=False)["attention"].mean()
    pivot = matrix.pivot(index="true_class", columns="memory_class", values="attention").reindex(index=VOWEL_ORDER, columns=VOWEL_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(VOWEL_ORDER)))
    ax.set_xticklabels(VOWEL_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(VOWEL_ORDER)))
    ax.set_yticklabels(VOWEL_ORDER)
    ax.set_title(f"PCMR attention to class memory (K={k})")
    ax.set_xlabel("Memory class")
    ax.set_ylabel("True query class")
    fig.colorbar(im, ax=ax, label="Attention")
    save(fig, out_dir, "fig9_pcmr_attention_true_to_memory_heatmap")

    confusable = [c for c in ["a", "e", "o", "ɔ", "ə", "ɛ"] if c in VOWEL_ORDER]
    if confusable:
        sub = pivot.loc[confusable, confusable]
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        im = ax.imshow(sub.values, cmap="magma", aspect="auto")
        ax.set_xticks(np.arange(len(confusable)))
        ax.set_xticklabels(confusable)
        ax.set_yticks(np.arange(len(confusable)))
        ax.set_yticklabels(confusable)
        ax.set_title("PCMR attention on confusable vowels")
        ax.set_xlabel("Memory class")
        ax.set_ylabel("True query class")
        fig.colorbar(im, ax=ax, label="Attention")
        save(fig, out_dir, "fig10_pcmr_attention_confusable_vowels")


def plot_confusion_delta(attn_df: pd.DataFrame, out_dir: Path) -> None:
    k = int(attn_df["k"].max())
    one = attn_df[attn_df["k"] == k].drop_duplicates(["region", "row_index"])
    y_true = one["true_id"].to_numpy(dtype=int)
    ep = one["episode_pred"].to_numpy(dtype=int)
    gl = one["global_pred"].to_numpy(dtype=int)
    labels = list(range(len(VOWEL_ORDER)))
    gl_cm = confusion_matrix(y_true, gl, labels=labels, normalize="true")
    ep_cm = confusion_matrix(y_true, ep, labels=labels, normalize="true")
    delta = ep_cm - gl_cm
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-np.max(np.abs(delta)), vmax=np.max(np.abs(delta)))
    ax.set_xticks(np.arange(len(VOWEL_ORDER)))
    ax.set_xticklabels(VOWEL_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(VOWEL_ORDER)))
    ax.set_yticklabels(VOWEL_ORDER)
    ax.set_title("Confusion change: Episode RW - Global-only")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax, label="Change in normalized confusion")
    save(fig, out_dir, "fig11_confusion_delta_episode_minus_global")


def plot_tsne(embed_df: pd.DataFrame, out_dir: Path, max_samples: int, seed: int) -> None:
    all_feature_cols = [c for c in embed_df.columns if re.fullmatch(r"f\d+", str(c))]
    rng = np.random.default_rng(seed)
    base_ids = embed_df[["region", "row_index", "true_class"]].drop_duplicates()
    if len(base_ids) > max_samples:
        base_ids = base_ids.iloc[rng.choice(len(base_ids), size=max_samples, replace=False)]
    sampled = embed_df.merge(base_ids, on=["region", "row_index", "true_class"], how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    for ax, feature_type, title in zip(axes, ["whisper", "prototype_h0", "pcdlcmnet"], ["Whisper-only", "Prototype/H0", "PC-DLCMNet"]):
        d = sampled[sampled["feature_type"] == feature_type].copy()
        feature_cols = [c for c in all_feature_cols if c in d.columns and not d[c].isna().all()]
        x = d[feature_cols].to_numpy(dtype=np.float32)
        x = StandardScaler().fit_transform(x)
        perplexity = min(30, max(5, (len(d) - 1) // 4))
        z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(x)
        for class_id, label in enumerate(VOWEL_ORDER):
            mask = d["true_class"].to_numpy() == label
            if mask.any():
                ax.scatter(z[mask, 0], z[mask, 1], s=8, alpha=0.7, label=label)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 9), frameon=False)
    fig.suptitle("Representation visualization by vowel class", y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out_dir, "fig12_tsne_representation_comparison")


def main() -> None:
    args = parse_args()
    style()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.result_root)

    if args.plot_only:
        gate_df = pd.read_csv(out_dir / "gema_gate_diagnostics.csv")
        attn_df = pd.read_csv(out_dir / "pcmr_attention_diagnostics.csv")
        embed_df = pd.read_csv(out_dir / "representation_embeddings.csv")
        plot_gate_figures(gate_df, out_dir)
        plot_attention_figures(attn_df, out_dir)
        plot_confusion_delta(attn_df, out_dir)
        plot_tsne(embed_df, out_dir, args.max_tsne_samples, args.seed)
        print(f"[plot-only] regenerated figures in {out_dir}", flush=True)
        return

    all_gate: list[dict] = []
    all_attn: list[dict] = []
    all_embed: list[dict] = []
    for region in args.regions:
        json_path, ckpt_path = find_region_run(result_root, region)
        print(f"[diagnostics] region={region} checkpoint={ckpt_path}", flush=True)
        bundle = load_region_bundle(json_path, ckpt_path, args.csv, device)
        gate_rows, attn_rows, embed_rows = collect_region_diagnostics(
            bundle=bundle,
            region=region,
            k_values=args.k_values,
            max_episodes=args.max_episodes,
            seed=args.seed,
            device=device,
        )
        all_gate.extend(gate_rows)
        all_attn.extend(attn_rows)
        all_embed.extend(embed_rows)

    gate_df = pd.DataFrame(all_gate)
    attn_df = pd.DataFrame(all_attn)
    embed_df = pd.DataFrame(all_embed)
    gate_df.to_csv(out_dir / "gema_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    attn_df.to_csv(out_dir / "pcmr_attention_diagnostics.csv", index=False, encoding="utf-8-sig")
    embed_df.to_csv(out_dir / "representation_embeddings.csv", index=False, encoding="utf-8-sig")

    plot_gate_figures(gate_df, out_dir)
    plot_attention_figures(attn_df, out_dir)
    plot_confusion_delta(attn_df, out_dir)
    plot_tsne(embed_df, out_dir, args.max_tsne_samples, args.seed)

    summary = {
        "regions": args.regions,
        "k_values": args.k_values,
        "gate_rows": int(len(gate_df)),
        "attention_rows": int(len(attn_df)),
        "embedding_rows": int(len(embed_df)),
        "mean_gate_global_weight": float(gate_df["gate_global_weight"].mean()) if len(gate_df) else None,
        "mean_support_injection_weight": float(gate_df["support_injection_weight"].mean()) if len(gate_df) else None,
    }
    (out_dir / "mechanism_diagnostics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
