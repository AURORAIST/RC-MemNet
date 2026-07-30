#!/usr/bin/env python3
"""Run the paper RC-MemNet leave-one-region experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pc_dlcmnet.data.acoustic_features import (
    apply_vector_norm,
    aux_rows_hash,
    extract_aux_matrix_cached,
    fit_vector_norm,
    serializable_norm,
)
from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.models.rc_dmnet import RCDMNet, RCDMNetConfig
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    build_split_audit,
    class_loss_weight,
    configure_stdout,
    dataframe_hash,
    default_old_root,
    extract_matrix_cached,
    fit_token_norm,
    infer_aux_branch_dims,
    label_counts,
    labels_to_ids,
    load_dataset,
    load_model_init_checkpoint,
    prediction_frame,
    resolve_device,
    serializable_token_norm,
    set_seed,
    split_data,
    summarize_confusions,
)
from tools.experiments.baseline_comparisons import load_whisper_encoder


@torch.no_grad()
def prompt_token_distribution(model: RCDMNet) -> dict[str, float]:
    """Normalized magnitude of the learned context prompt tokens.

    The 0722 RC-MemNet uses prompt tokens directly instead of a softmax router
    over a prompt library, so this is the closest per-prompt training dynamic
    available from the main model.
    """
    prompts = model.region_prompts.detach()
    token_norm = prompts.norm(dim=-1).mean(dim=0)
    token_weight = token_norm / token_norm.sum().clamp_min(1e-8)
    return {
        f"prompt_token_weight_{idx}": float(value.detach().cpu())
        for idx, value in enumerate(token_weight)
    }


@torch.no_grad()
def prompt_route_distribution(out: dict[str, torch.Tensor]) -> dict[str, float]:
    weights = out.get("prompt_route_weights")
    if weights is None or weights.numel() == 0:
        return {}
    mean_weights = weights.detach().mean(dim=0).cpu()
    return {
        f"prompt_route_mean_{idx}": float(value)
        for idx, value in enumerate(mean_weights)
    }


@torch.no_grad()
def prompt_interaction_distribution(out: dict[str, torch.Tensor]) -> dict[str, float]:
    interactions = out.get("prompt_acoustic_interactions")
    if interactions is None or interactions.numel() == 0:
        return {}
    mean_interactions = interactions.detach().mean(dim=0).cpu()
    return {
        f"prompt_interaction_mean_{idx}": float(value)
        for idx, value in enumerate(mean_interactions)
    }


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--curve-output", default="")
    parser.add_argument("--audit-output", default="")
    parser.add_argument("--checkpoint-output", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--protocol", choices=["loro", "random"], default="loro")
    parser.add_argument("--holdout-region", required=True)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(old_root / "output/salmonn_style_whisper_cache_base"))
    parser.add_argument("--matrix-cache-dir", default="output/0722/matrix_cache")
    parser.add_argument("--feature-norm", choices=["none", "global"], default="global")
    parser.add_argument("--matrix-cache-scan", action="store_true", default=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--token-chunks", type=int, default=32)
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--aux-source", choices=["auto", "acoustic", "whisper_stats", "old_fusion_cache"], default="acoustic")
    parser.add_argument("--aux-representation", choices=["auto", "sequence", "stats"], default="sequence")
    parser.add_argument("--aux-cache-dir", default="output/0722/feature_memory_aux_cache")
    parser.add_argument("--old-feature-cache-dir", default=str(old_root / "output/feature_cache_vowel"))
    parser.add_argument("--aux-n-mfcc", type=int, default=39)
    parser.add_argument("--aux-n-mels", type=int, default=128)
    parser.add_argument("--aux-f0-segments", type=int, default=5)
    parser.add_argument("--aux-use-delta-mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-use-formants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--sampler-domain-power", type=float, default=1.0)
    parser.add_argument("--sampler-class-power", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--score-dim", type=int, default=128)
    parser.add_argument("--prompt-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--address-temperature", type=float, default=0.2)
    parser.add_argument("--classifier-temperature", type=float, default=0.2)
    parser.add_argument("--memory-mix", type=float, default=0.5)
    parser.add_argument("--source-write-kappa", type=float, default=4.0)
    parser.add_argument("--target-write-kappa", type=float, default=1.0)
    parser.add_argument("--global-momentum", type=float, default=0.1)
    parser.add_argument(
        "--ablation-variant",
        choices=[
            "rc_dmnet",
            "rc_memnet",
            "support_prototype",
            "single_global",
            "multi_global",
            "dual_level",
            "uniform_prompt_fusion",
            "no_context_prompt",
            "no_memory_adapter",
            "no_prompt_no_memory",
            "no_prompt_routing",
            "count_based_adaptation",
            "no_slow_consolidation",
            "full_pclr",
            "uniform_query_routing",
            "uniform_source_composition",
            "single_source_prompt",
            "unconstrained_composition",
        ],
        default="rc_memnet",
    )
    parser.add_argument("--write-top-k", type=int, default=1)
    parser.add_argument("--target-support-shots", type=int, default=4)
    parser.add_argument("--target-adapt-steps", type=int, default=50)
    parser.add_argument("--target-adapt-lr", type=float, default=0.05)
    parser.add_argument("--eval-split-runs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--require-all-source-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def apply_ablation_overrides(args: argparse.Namespace) -> None:
    variant = str(getattr(args, "ablation_variant", "rc_dmnet"))
    if variant in {"rc_dmnet", "rc_memnet"}:
        args.ablation_variant = "rc_memnet"
    elif variant == "no_prompt_routing":
        args.ablation_variant = "no_context_prompt"
        variant = "no_context_prompt"
    elif variant == "count_based_adaptation":
        args.ablation_variant = "no_memory_adapter"
        variant = "no_memory_adapter"
    if variant == "single_global":
        args.num_slots = 1
        args.memory_mix = 1.0
        args.target_adapt_steps = 0
    elif variant == "multi_global":
        args.memory_mix = 1.0
        args.target_adapt_steps = 0
    elif variant in {"dual_level", "uniform_prompt_fusion"}:
        args.target_adapt_steps = 0
    elif variant in {"no_context_prompt", "no_prompt_no_memory"}:
        args.target_adapt_steps = 0
    elif variant in {"no_memory_adapter", "no_slow_consolidation", "full_pclr"}:
        pass


def build_region_ids(rows: pd.DataFrame, region_column: str, region_to_id: dict[str, int]) -> np.ndarray:
    return rows[region_column].astype(str).map(region_to_id).to_numpy(dtype=np.int64)


def build_sampler(y: np.ndarray, region_ids: np.ndarray, args: argparse.Namespace) -> WeightedRandomSampler:
    class_counts = np.bincount(y, minlength=len(VOWEL_ORDER)).astype(np.float64)
    region_counts = np.bincount(region_ids, minlength=int(region_ids.max()) + 1).astype(np.float64)
    class_weights = np.ones_like(class_counts)
    region_weights = np.ones_like(region_counts)
    class_weights[class_counts > 0] = class_counts[class_counts > 0] ** (-float(args.sampler_class_power))
    region_weights[region_counts > 0] = region_counts[region_counts > 0] ** (-float(args.sampler_domain_power))
    weights = class_weights[y] * region_weights[region_ids]
    weights = weights / max(float(weights.mean()), 1e-12)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(VOWEL_ORDER))), average="macro", zero_division=0)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
    }


@torch.no_grad()
def predict_source(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    region_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    for start in range(0, len(x), int(args.eval_batch_size)):
        end = min(start + int(args.eval_batch_size), len(x))
        speech = torch.tensor(x[start:end], dtype=torch.float32, device=device)
        aux_t = torch.tensor(aux[start:end], dtype=torch.float32, device=device)
        r_t = torch.tensor(region_ids[start:end], dtype=torch.long, device=device)
        out = model.forward_source(speech, aux_t, r_t)
        probs.append(torch.softmax(out["logits"], dim=-1).detach().cpu().numpy())
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(VOWEL_ORDER)), dtype=np.float32)
    return prob.argmax(axis=1).astype(np.int64), prob.astype(np.float32)


def set_model_trainable(model: RCDMNet, trainable: bool) -> list[bool]:
    old = [bool(p.requires_grad) for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(trainable)
    return old


def restore_trainable(model: RCDMNet, old: list[bool]) -> None:
    for param, flag in zip(model.parameters(), old):
        param.requires_grad_(flag)


def adapt_fusion(
    model: RCDMNet,
    support_x: np.ndarray,
    support_aux: np.ndarray,
    support_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    fusion_logits = torch.zeros(model.num_regions, dtype=torch.float32, device=device, requires_grad=True)
    if len(support_y) == 0 or int(args.target_adapt_steps) <= 0:
        return fusion_logits.detach()
    if getattr(model.config, "source_composition_mode", "convex") in {"uniform", "single"}:
        return fusion_logits.detach()
    old_flags = set_model_trainable(model, False)
    model.eval()
    optimizer = torch.optim.Adam([fusion_logits], lr=float(args.target_adapt_lr))
    speech = torch.tensor(support_x, dtype=torch.float32, device=device)
    aux_t = torch.tensor(support_aux, dtype=torch.float32, device=device)
    labels = torch.tensor(support_y, dtype=torch.long, device=device)
    try:
        for _step in range(int(args.target_adapt_steps)):
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_global_with_fusion(speech, aux_t, fusion_logits)
            loss = F.cross_entropy(out["logits"], labels)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
    finally:
        restore_trainable(model, old_flags)
    return fusion_logits.detach()


@torch.no_grad()
def encode_z_batches(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    zs = []
    model.eval()
    for start in range(0, len(x), int(args.eval_batch_size)):
        end = min(start + int(args.eval_batch_size), len(x))
        speech = torch.tensor(x[start:end], dtype=torch.float32, device=device)
        aux_t = torch.tensor(aux[start:end], dtype=torch.float32, device=device)
        zs.append(model.encode(speech, aux_t, None)["z"].detach())
    if not zs:
        return torch.zeros((0, model.config.hidden_dim), dtype=torch.float32, device=device)
    return torch.cat(zs, dim=0)


@torch.no_grad()
def evaluate_support_prototype_splits(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_metrics = []
    first_pred = np.full(len(y), -1, dtype=np.int64)
    first_prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    first_global_prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    first_support_flag = np.zeros(len(y), dtype=np.int64)
    first_query_flag = np.zeros(len(y), dtype=np.int64)
    z_all = encode_z_batches(model, x, aux, args, device)
    fallback = F.normalize(model.global_values.mean(dim=1), dim=-1)
    for run_idx in range(max(1, int(args.eval_split_runs))):
        support_idx, query_idx = split_global_support_query(
            y,
            support_shots=int(args.target_support_shots),
            seed=int(args.seed) + run_idx * 1009,
        )
        prototypes = fallback.clone()
        support_labels = torch.tensor(y[support_idx], dtype=torch.long, device=device)
        support_z = z_all[torch.tensor(support_idx, dtype=torch.long, device=device)]
        for c in support_labels.unique(sorted=True).tolist():
            mask = support_labels == int(c)
            prototypes[int(c)] = F.normalize(support_z[mask].mean(dim=0), dim=0)
        query_z = z_all[torch.tensor(query_idx, dtype=torch.long, device=device)]
        logits = torch.matmul(F.normalize(query_z, dim=-1), F.normalize(prototypes, dim=-1).transpose(0, 1))
        logits = logits / float(model.config.classifier_temperature)
        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        pred = prob.argmax(axis=1).astype(np.int64)
        metrics = metric_dict(y[query_idx], pred)
        metrics["support_size"] = int(len(support_idx))
        metrics["query_size"] = int(len(query_idx))
        split_metrics.append(metrics)
        if run_idx == 0:
            first_pred[query_idx] = pred
            first_prob[query_idx] = prob
            first_global_prob[query_idx] = prob
            first_support_flag[support_idx] = 1
            first_query_flag[query_idx] = 1
    keys = ["accuracy", "macro_f1", "weighted_f1", "micro_f1", "support_size", "query_size"]
    summary = {
        "runs": int(len(split_metrics)),
        "support_shots": int(args.target_support_shots),
        "metrics": split_metrics,
        "mean": {key: float(np.mean([m[key] for m in split_metrics])) for key in keys},
        "std": {key: float(np.std([m[key] for m in split_metrics], ddof=1)) if len(split_metrics) > 1 else 0.0 for key in keys},
    }
    return summary, first_pred, first_prob, first_global_prob, first_support_flag, first_query_flag


@torch.no_grad()
def predict_target_batches(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    fusion_logits: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    use_prompt: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = []
    global_probs = []
    model.eval()
    for start in range(0, len(x), int(args.eval_batch_size)):
        end = min(start + int(args.eval_batch_size), len(x))
        speech = torch.tensor(x[start:end], dtype=torch.float32, device=device)
        aux_t = torch.tensor(aux[start:end], dtype=torch.float32, device=device)
        out = model.forward_target(speech, aux_t, fusion_logits, use_prompt=use_prompt)
        probs.append(torch.softmax(out["logits"], dim=-1).detach().cpu().numpy())
        global_probs.append(torch.softmax(out["global_logits"], dim=-1).detach().cpu().numpy())
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(VOWEL_ORDER)), dtype=np.float32)
    global_prob = np.concatenate(global_probs, axis=0) if global_probs else np.zeros_like(prob)
    return prob.argmax(axis=1).astype(np.int64), prob.astype(np.float32), global_prob.astype(np.float32)


def evaluate_target_splits(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if str(getattr(args, "ablation_variant", "rc_dmnet")) == "support_prototype":
        return evaluate_support_prototype_splits(model, x, aux, y, args, device)
    use_target_prompt = str(getattr(args, "ablation_variant", "rc_memnet")) not in {
        "dual_level",
        "no_context_prompt",
        "no_prompt_no_memory",
    }
    split_metrics = []
    first_pred = np.full(len(y), -1, dtype=np.int64)
    first_prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    first_global_prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    first_support_flag = np.zeros(len(y), dtype=np.int64)
    first_query_flag = np.zeros(len(y), dtype=np.int64)
    for run_idx in range(max(1, int(args.eval_split_runs))):
        support_idx, query_idx = split_global_support_query(
            y,
            support_shots=int(args.target_support_shots),
            seed=int(args.seed) + run_idx * 1009,
        )
        fusion_logits = adapt_fusion(model, x[support_idx], aux[support_idx], y[support_idx], args, device)
        pred, prob, global_prob = predict_target_batches(
            model,
            x[query_idx],
            aux[query_idx],
            fusion_logits,
            args,
            device,
            use_prompt=use_target_prompt,
        )
        metrics = metric_dict(y[query_idx], pred)
        metrics["support_size"] = int(len(support_idx))
        metrics["query_size"] = int(len(query_idx))
        split_metrics.append(metrics)
        if run_idx == 0:
            first_pred[query_idx] = pred
            first_prob[query_idx] = prob
            first_global_prob[query_idx] = global_prob
            first_support_flag[support_idx] = 1
            first_query_flag[query_idx] = 1
    keys = ["accuracy", "macro_f1", "weighted_f1", "micro_f1", "support_size", "query_size"]
    summary = {
        "runs": int(len(split_metrics)),
        "support_shots": int(args.target_support_shots),
        "metrics": split_metrics,
        "mean": {key: float(np.mean([m[key] for m in split_metrics])) for key in keys},
        "std": {key: float(np.std([m[key] for m in split_metrics], ddof=1)) if len(split_metrics) > 1 else 0.0 for key in keys},
    }
    return summary, first_pred, first_prob, first_global_prob, first_support_flag, first_query_flag


def evaluate_target_splits_safely(
    model: RCDMNet,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.device]:
    try:
        result = evaluate_target_splits(model, x, aux, y, args, device)
        return (*result, device)
    except RuntimeError as exc:
        message = str(exc)
        if device.type != "cuda" or ("CUDA" not in message and "CUBLAS" not in message):
            raise
        cpu_device = torch.device("cpu")
        print(f"[target-eval] CUDA failed during target adaptation; retrying on CPU: {message}", flush=True)
        model.to(cpu_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = evaluate_target_splits(model, x, aux, y, args, cpu_device)
        return (*result, cpu_device)


def load_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = None
    try:
        train_x_raw, train_meta = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, val_meta = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, test_meta = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")
    except (AttributeError, FileNotFoundError) as exc:
        print(f"[matrix-cache] miss needs Whisper extraction: {exc}", flush=True)
        try:
            import whisper

            whisper_model = whisper.load_model(args.whisper_model, device=str(device))
        except Exception:
            whisper_model = load_whisper_encoder(args.whisper_model, device)
        whisper_model.eval()
        for param in whisper_model.parameters():
            param.requires_grad = False
        train_x_raw, train_meta = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, val_meta = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, test_meta = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")

    token_norm = fit_token_norm(train_x_raw, args.feature_norm)
    train_x = apply_token_norm(train_x_raw, token_norm)
    val_x = apply_token_norm(val_x_raw, token_norm)
    test_x = apply_token_norm(test_x_raw, token_norm)

    train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, args, train_x_raw, "train")
    val_aux_raw, val_aux_meta = extract_aux_matrix_cached(val_df, args, val_x_raw, "val")
    test_aux_raw, test_aux_meta = extract_aux_matrix_cached(test_df, args, test_x_raw, "test")
    aux_norm = fit_vector_norm(train_aux_raw, "global")
    train_aux = apply_vector_norm(train_aux_raw, aux_norm)
    val_aux = apply_vector_norm(val_aux_raw, aux_norm)
    test_aux = apply_vector_norm(test_aux_raw, aux_norm)

    matrix_meta = {"train": train_meta, "val": val_meta, "test": test_meta, "token_norm": serializable_token_norm(token_norm)}
    aux_meta = {
        "train": train_aux_meta,
        "val": val_aux_meta,
        "test": test_aux_meta,
        "norm": serializable_norm(aux_norm),
        "rows_hash": {
            "train": aux_rows_hash(train_df, args),
            "val": aux_rows_hash(val_df, args),
            "test": aux_rows_hash(test_df, args),
        },
    }
    return train_x, val_x, test_x, train_aux, val_aux, test_aux, matrix_meta, aux_meta, train_aux_meta


def build_checkpoint_manifest(
    args: argparse.Namespace,
    split_info: dict[str, Any],
    split_hash: dict[str, Any],
    matrix_meta: dict[str, Any],
    aux_meta: dict[str, Any],
    source_regions: list[str],
    train_summary: dict[str, Any],
    init_checkpoint_info: dict[str, Any],
    model: RCDMNet,
    model_config: RCDMNetConfig,
) -> dict[str, Any]:
    whisper_model = str(args.whisper_model)
    whisper_path = Path(whisper_model)
    if whisper_model and whisper_model not in {"base", "tiny", "small", "medium", "large"} and whisper_path.exists():
        whisper_model = str(whisper_path.resolve())
    return {
        "schema_version": 1,
        "task": "leave_one_region_out_vowel_classification",
        "method": "RC-MemNet",
        "script": "tools/experiments/run_rc_dmnet.py",
        "data": {
            "csv": str(Path(args.csv).resolve()),
            "path_column": str(args.path_column),
            "start_column": str(args.start_column),
            "end_column": str(args.end_column),
            "label_column": str(args.label_column),
            "region_column": str(args.region_column),
            "speaker_column": str(args.speaker_column),
            "labels": list(args.labels) if args.labels else None,
            "protocol": str(args.protocol),
            "holdout_region": str(args.holdout_region),
            "val_size": float(args.val_size),
            "test_size": float(args.test_size),
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
            "target_sr": int(args.target_sr),
        },
        "feature_pipeline": {
            "whisper_model": whisper_model,
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "matrix_cache_dir": str(Path(args.matrix_cache_dir).resolve()),
            "feature_norm": str(args.feature_norm),
            "matrix_cache_scan": bool(args.matrix_cache_scan),
            "token_chunks": int(args.token_chunks),
            "audio_root": str(args.audio_root),
            "aux_source": str(args.aux_source),
            "aux_representation": str(args.aux_representation),
            "aux_cache_dir": str(Path(args.aux_cache_dir).resolve()),
            "old_feature_cache_dir": str(Path(args.old_feature_cache_dir).resolve()),
            "aux_n_mfcc": int(args.aux_n_mfcc),
            "aux_n_mels": int(args.aux_n_mels),
            "aux_f0_segments": int(args.aux_f0_segments),
            "aux_use_delta_mfcc": bool(args.aux_use_delta_mfcc),
            "aux_use_formants": bool(args.aux_use_formants),
        },
        "evaluation": {
            "target_support_shots": int(args.target_support_shots),
            "target_adapt_steps": int(args.target_adapt_steps),
            "target_adapt_lr": float(args.target_adapt_lr),
            "eval_split_runs": int(args.eval_split_runs),
            "ablation_variant": str(args.ablation_variant),
        },
        "model": {
            "rcdmnet_config": asdict(model_config),
            "source_regions": list(source_regions),
            "num_source_regions": int(len(source_regions)),
            "parameters": {
                "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "total": int(sum(p.numel() for p in model.parameters())),
            },
        },
        "split": split_info,
        "split_hash": split_hash,
        "cache": {
            "matrix": matrix_meta,
            "aux": aux_meta,
        },
        "training": train_summary,
        "init_checkpoint": init_checkpoint_info,
        "labels": list(VOWEL_ORDER),
        "runtime": {
            "device": str(args.device),
            "torch_version": str(torch.__version__),
        },
    }


def train_model(
    model: RCDMNet,
    train_x: np.ndarray,
    train_aux: np.ndarray,
    train_y: np.ndarray,
    train_region_ids: np.ndarray,
    val_x: np.ndarray,
    val_aux: np.ndarray,
    val_y: np.ndarray,
    val_region_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    model.to(device)
    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    train_aux_t = torch.tensor(train_aux, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    train_r_t = torch.tensor(train_region_ids, dtype=torch.long)
    model.initialize_memories(train_x_t, train_aux_t, train_y_t, train_r_t, batch_size=int(args.eval_batch_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = class_loss_weight(train_y, float(args.class_weight_power), device)
    region_to_indices = {
        int(region_id): np.flatnonzero(train_region_ids == int(region_id))
        for region_id in sorted(np.unique(train_region_ids).astype(np.int64).tolist())
    }
    curve: list[dict[str, float]] = []
    start_time = time.time()
    step = 0
    best_val = -1.0
    best_state = None
    while step < int(args.max_steps):
        region_order = list(region_to_indices)
        random.shuffle(region_order)
        for region_id in region_order:
            region_indices = region_to_indices[region_id].copy()
            np.random.default_rng(int(args.seed) + step + region_id * 1009).shuffle(region_indices)
            last_loss = 0.0
            for start in range(0, len(region_indices), int(args.batch_size)):
                batch_indices = region_indices[start : start + int(args.batch_size)]
                if len(batch_indices) == 0:
                    continue
                step += 1
                model.train()
                idx_t = torch.tensor(batch_indices, dtype=torch.long)
                speech = train_x_t[idx_t].to(device)
                aux_t = train_aux_t[idx_t].to(device)
                labels = train_y_t[idx_t].to(device)
                regions = train_r_t[idx_t].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model.forward_source(speech, aux_t, regions)
                loss = F.cross_entropy(out["logits"], labels, weight=weights, label_smoothing=float(args.label_smoothing))
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                if step % max(1, int(args.eval_every)) == 0 or step == int(args.max_steps):
                    model.consolidate_global()
                    val_pred, _val_prob = predict_source(model, val_x, val_aux, val_region_ids, args, device)
                    val_metrics = metric_dict(val_y, val_pred)
                    row = {"step": float(step), "loss": last_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
                    row.update(prompt_interaction_distribution(out))
                    row.update(prompt_route_distribution(out))
                    row.update(prompt_token_distribution(model))
                    curve.append(row)
                    if val_metrics["macro_f1"] > best_val:
                        best_val = val_metrics["macro_f1"]
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if not bool(args.quiet):
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                elif step % max(1, int(args.log_every)) == 0 and not bool(args.quiet):
                    print(json.dumps({"step": step, "loss": last_loss, "region_id": region_id}, ensure_ascii=False), flush=True)
                if step >= int(args.max_steps):
                    break
            full_idx_t = torch.tensor(region_indices, dtype=torch.long)
            model.write_region_evidence(
                train_x_t[full_idx_t],
                train_aux_t[full_idx_t],
                train_y_t[full_idx_t],
                train_r_t[full_idx_t],
                batch_size=int(args.eval_batch_size),
            )
            if step >= int(args.max_steps):
                break
    model.consolidate_global()
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    summary = {
        "steps": int(step),
        "best_val_macro_f1": float(best_val) if best_val >= 0 else None,
        "training_time_sec": float(time.time() - start_time),
    }
    return curve, summary


def write_curve(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    configure_stdout()
    args = parse_args()
    apply_ablation_overrides(args)
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device(args.device)
    if not bool(args.quiet):
        print(
            json.dumps(
                {
                    "method": "RC-MemNet",
                    "ablation_variant": str(args.ablation_variant),
                    "device": str(device),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "torch_version": torch.__version__,
                    "holdout_region": args.holdout_region,
                    "region_column": args.region_column,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output = Path(args.output)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output.with_suffix(".predictions.csv")
    curve_output = Path(args.curve_output) if args.curve_output else output.with_suffix(".curve.csv")
    audit_output = Path(args.audit_output) if args.audit_output else output.with_suffix(".audit.json")
    checkpoint_output = Path(args.checkpoint_output) if args.checkpoint_output else output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    split_audit = build_split_audit(df, train_df, val_df, test_df, args, split_info)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    source_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    region_to_id = {region: idx for idx, region in enumerate(source_regions)}

    train_x, val_x, test_x, train_aux, val_aux, test_aux, matrix_meta, aux_meta, train_aux_meta = load_features(train_df, val_df, test_df, args, device)
    train_y = labels_to_ids(train_df[args.label_column])
    val_y = labels_to_ids(val_df[args.label_column])
    test_y = labels_to_ids(test_df[args.label_column])
    train_region_ids = build_region_ids(train_df, args.region_column, region_to_id)
    val_region_ids = build_region_ids(val_df, args.region_column, region_to_id)

    config = RCDMNetConfig(
        input_dim=int(train_x.shape[-1]),
        aux_dim=int(train_aux.shape[-1]),
        aux_branch_dims=infer_aux_branch_dims(train_aux_meta, train_aux),
        num_classes=len(VOWEL_ORDER),
        num_regions=len(source_regions),
        hidden_dim=int(args.hidden_dim),
        score_dim=int(args.score_dim),
        prompt_dim=int(args.prompt_dim),
        num_slots=int(args.num_slots),
        num_prompts=int(args.num_prompts),
        dropout=float(args.dropout),
        address_temperature=float(args.address_temperature),
        classifier_temperature=float(args.classifier_temperature),
        memory_mix=float(args.memory_mix),
        source_write_kappa=float(args.source_write_kappa),
        target_write_kappa=float(args.target_write_kappa),
        global_momentum=float(args.global_momentum),
        use_context_prompts=str(args.ablation_variant) not in {"no_context_prompt", "no_prompt_no_memory"},
        use_memory_read=str(args.ablation_variant) not in {"no_memory_adapter", "no_prompt_no_memory"},
        use_prompt_routing=str(args.ablation_variant) not in {"no_context_prompt", "no_prompt_no_memory", "uniform_query_routing"},
        count_based_adaptation=False,
        slow_consolidation=str(args.ablation_variant) != "no_slow_consolidation",
        write_top_k=int(args.write_top_k),
        source_composition_mode=(
            "uniform" if str(args.ablation_variant) == "uniform_source_composition"
            else "single" if str(args.ablation_variant) == "single_source_prompt"
            else "unconstrained" if str(args.ablation_variant) == "unconstrained_composition"
            else "convex"
        ),
    )
    model = RCDMNet(config)
    init_checkpoint_info = load_model_init_checkpoint(model, args.init_checkpoint, device)
    curve, train_summary = train_model(
        model=model,
        train_x=train_x,
        train_aux=train_aux,
        train_y=train_y,
        train_region_ids=train_region_ids,
        val_x=val_x,
        val_aux=val_aux,
        val_y=val_y,
        val_region_ids=val_region_ids,
        args=args,
        device=device,
    )
    write_curve(curve, curve_output)

    val_pred, val_prob = predict_source(model, val_x, val_aux, val_region_ids, args, device)
    val_metrics = metric_dict(val_y, val_pred)
    eval_summary, test_pred, test_prob, test_global_prob, support_flag, query_flag, eval_device = evaluate_target_splits_safely(
        model, test_x, test_aux, test_y, args, device
    )
    valid = test_pred >= 0
    metrics = eval_summary["mean"]
    metrics_std = eval_summary["std"]
    report = classification_report(
        test_y[valid],
        test_pred[valid],
        labels=list(range(len(VOWEL_ORDER))),
        target_names=VOWEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_y[valid], test_pred[valid], labels=list(range(len(VOWEL_ORDER))))
    pred_df = prediction_frame(test_df.loc[valid].reset_index(drop=True), test_y[valid], test_pred[valid], test_prob[valid], args)
    pred_df["global_pred_vowel"] = [VOWEL_ORDER[idx] for idx in test_global_prob[valid].argmax(axis=-1).astype(np.int64)]
    pred_df["global_pred_confidence"] = test_global_prob[valid].max(axis=1)
    pred_df["is_support_item"] = support_flag[valid].astype(np.int64)
    pred_df["is_query_item"] = query_flag[valid].astype(np.int64)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    checkpoint_manifest = build_checkpoint_manifest(
        args=args,
        split_info=split_info,
        split_hash={
            "train": dataframe_hash(train_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "val": dataframe_hash(val_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "test": dataframe_hash(test_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        },
        matrix_meta=matrix_meta,
        aux_meta=aux_meta,
        source_regions=source_regions,
        train_summary=train_summary,
        init_checkpoint_info=init_checkpoint_info,
        model=model,
        model_config=config,
    )

    result = {
        "task": "leave_one_region_out_vowel_classification",
        "method": "RC-MemNet",
        "ablation_variant": str(args.ablation_variant),
        "paper_alignment": {
            "shared_acoustic_encoder": True,
            "rpl_prompt_inside_encoder": True,
            "prompt_conditioned_memory_query": True,
            "category_aligned_shared_memory": True,
            "region_level_read_before_write": True,
            "label_guided_source_region_writes": True,
            "target_prompt_fusion_only": True,
            "target_memory_read_only": True,
            "read_only_query_inference": True,
        },
        "ablation_semantics": {
            "dual_level": "target/global memories with target prompt disabled",
            "uniform_prompt_fusion": "target/global memories with uniform source-prompt fusion and no support adaptation",
            "rc_memnet": "shared persistent memory with support-adapted source-prompt fusion",
            "rc_dmnet": "legacy alias for rc_memnet",
            "no_context_prompt": "remove learned context prompts from source and target encoding",
            "no_memory_adapter": "remove memory read/write adaptation and keep prompt-conditioned encoder",
            "no_prompt_no_memory": "remove both context prompts and memory adaptation",
            "no_slow_consolidation": "full PCLR without slow global-memory consolidation",
            "full_pclr": "full PCLR with prompt routing, gated adaptation, and slow consolidation",
        },
        "metrics": metrics,
        "metrics_std": metrics_std,
        "val_metrics": val_metrics,
        "evaluation_splits": {"test": eval_summary},
        "target_eval_device": str(eval_device),
        "report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_confusions": summarize_confusions(cm, VOWEL_ORDER, top_n=10),
        "split": split_info,
        "split_audit": split_audit,
        "init_checkpoint": init_checkpoint_info,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "label_counts": {
            "train": label_counts(train_df[args.label_column], VOWEL_ORDER),
            "val": label_counts(val_df[args.label_column], VOWEL_ORDER),
            "test": label_counts(test_df[args.label_column], VOWEL_ORDER),
        },
        "source_regions": source_regions,
        "cache": {"matrix": matrix_meta, "aux": aux_meta},
        "config": vars(args),
        "checkpoint_manifest": checkpoint_manifest,
        "training": train_summary,
        "files": {
            "output": str(output),
            "curve": str(curve_output),
            "predictions": str(predictions_output),
            "audit": str(audit_output),
            "checkpoint": str(checkpoint_output) if args.save_checkpoint else "",
        },
        "parameters": {
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total": int(sum(p.numel() for p in model.parameters())),
        },
        "split_hash": {
            "train": dataframe_hash(train_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "val": dataframe_hash(val_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "test": dataframe_hash(test_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "manifest": checkpoint_manifest,
                "labels": VOWEL_ORDER,
            },
            checkpoint_output,
        )
    print(json.dumps({"output": str(output), "metrics": metrics, "metrics_std": metrics_std}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
