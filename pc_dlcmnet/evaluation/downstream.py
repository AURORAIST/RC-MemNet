"""Downstream evaluation heads and split-averaged evaluation."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    tqdm = None

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.data.episodes import (
    make_episode_tensors,
    split_fixed_query_support,
    split_global_support_query,
    split_support_query,
)
from pc_dlcmnet.utils.tensor import masked_mean, safe_batched_dot, safe_mm


def route_entropy(route_weights: torch.Tensor) -> float:
    weights = route_weights.detach().clamp_min(1e-8)
    entropy = -(weights.log() * weights).sum(dim=-1).mean()
    return float(entropy.cpu())


def build_support_prototypes(
    model: torch.nn.Module,
    support_speech: torch.Tensor,
    support_aux: torch.Tensor,
    support_mask: torch.Tensor,
    support_labels: torch.Tensor | None,
    fallback_global: bool,
    support_weight: float,
) -> torch.Tensor:
    support_weight = float(max(0.0, min(1.0, support_weight)))
    base_prototypes = model.global_memory.clone() if bool(fallback_global) else model.global_memory.new_zeros(model.num_classes, model.config.hidden_dim)
    prototypes = base_prototypes.clone()
    if support_speech.shape[0] > 0 and support_labels is not None and support_labels.numel() > 0:
        support_features = model.encode_inputs(support_speech, support_aux, support_mask)
        support_vec = masked_mean(support_features["h0"], support_mask)
        one_hot = F.one_hot(support_labels.long(), num_classes=model.num_classes).to(dtype=support_vec.dtype)
        counts = one_hot.sum(dim=0).unsqueeze(-1)
        support_proto = safe_mm(one_hot.transpose(0, 1), support_vec) / counts.clamp_min(model.eps)
        has_support = (counts > 0).to(dtype=support_vec.dtype)
        blended_proto = support_weight * support_proto + (1.0 - support_weight) * base_prototypes
        prototypes = has_support * blended_proto + (1.0 - has_support) * base_prototypes
    return prototypes


def prototype_predict_from_prototypes(
    model: torch.nn.Module,
    eval_speech: torch.Tensor,
    eval_aux: torch.Tensor,
    eval_mask: torch.Tensor,
    prototypes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    eval_features = model.encode_inputs(eval_speech, eval_aux, eval_mask)
    query_vec = masked_mean(eval_features["h0"], eval_mask)
    logits = safe_mm(F.normalize(query_vec, dim=-1), F.normalize(prototypes, dim=-1).transpose(0, 1)) / float(model.config.temperature)
    return {
        **eval_features,
        "logits": logits,
        "query": query_vec,
        "route_weights": eval_speech.new_zeros(eval_speech.shape[0], model.config.num_prompts),
    }


def prototype_predict(
    model: torch.nn.Module,
    eval_speech: torch.Tensor,
    eval_aux: torch.Tensor,
    eval_mask: torch.Tensor,
    support_speech: torch.Tensor,
    support_aux: torch.Tensor,
    support_mask: torch.Tensor,
    support_labels: torch.Tensor | None,
    fallback_global: bool,
    support_weight: float,
) -> dict[str, torch.Tensor]:
    prototypes = build_support_prototypes(
        model=model,
        support_speech=support_speech,
        support_aux=support_aux,
        support_mask=support_mask,
        support_labels=support_labels,
        fallback_global=fallback_global,
        support_weight=support_weight,
    )
    return prototype_predict_from_prototypes(
        model=model,
        eval_speech=eval_speech,
        eval_aux=eval_aux,
        eval_mask=eval_mask,
        prototypes=prototypes,
    )


def memory_cosine_predict_from_memory(
    model: torch.nn.Module,
    eval_speech: torch.Tensor,
    eval_aux: torch.Tensor,
    eval_mask: torch.Tensor,
    memory: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify with GEMA-updated memory while bypassing PCMR attention."""
    eval_features = model.encode_inputs(eval_speech, eval_aux, eval_mask)
    query_vec = masked_mean(eval_features["h0"], eval_mask)
    memory_state = model._expand_memory(memory, eval_speech.shape[0])
    query = F.normalize(model.query_projection(query_vec), dim=-1)
    projected_memory = F.normalize(model.memory_projection(memory_state), dim=-1)
    logits = safe_batched_dot(query, projected_memory) / float(model.config.temperature)
    return {
        **eval_features,
        "logits": logits,
        "query": query,
        "projected_memory": projected_memory,
        "route_weights": eval_speech.new_zeros(eval_speech.shape[0], model.config.num_prompts),
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(VOWEL_ORDER))), average="macro", zero_division=0)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
    }


def evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    episode_groups: Sequence[np.ndarray],
    args: Any,
    device: torch.device,
    seed_offset: int,
    split_run_index: int = 0,
    subgroup_masks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float], dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    if int(getattr(args, "eval_episode_limit", 0) or 0) > 0:
        episode_groups = list(episode_groups)[: int(args.eval_episode_limit)]
    eval_support_shots = int(getattr(args, "eval_support_shots", 0) or 0)
    if eval_support_shots <= 0:
        eval_support_shots = int(args.support_shots)
    ensemble_runs = max(1, int(getattr(args, "eval_ensemble_runs", 1) or 1))
    eval_split_mode = str(getattr(args, "eval_split_mode", "global_support"))
    if eval_split_mode == "global_support":
        ensemble_runs = 1
    label_blend = 0.0 if args.support_write_mode == "pseudo" else (float(args.support_label_blend) if args.support_write_mode == "blend" else 1.0)
    pred = np.full(len(y), -1, dtype=np.int64)
    prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    global_prob = np.zeros((len(y), len(VOWEL_ORDER)), dtype=np.float32)
    support_flag = np.zeros(len(y), dtype=np.int64)
    query_flag = np.zeros(len(y), dtype=np.int64)
    support_sizes = []
    query_sizes = []
    gate_means = []
    route_entropies = []
    feature_weight_rows = []
    global_route_entropies = []
    global_split_cache: list[tuple[np.ndarray, np.ndarray, set[int]]] = []
    if eval_split_mode == "global_support":
        for ensemble_idx in range(ensemble_runs):
            split_seed = args.seed + seed_offset + int(split_run_index) * 1_000_003 + ensemble_idx * 9_973
            global_support_idx, global_query_idx = split_global_support_query(y, eval_support_shots, seed=split_seed)
            global_split_cache.append((global_support_idx, global_query_idx, set(global_query_idx.tolist())))
    global_prototype_cache: list[torch.Tensor] = []
    with torch.no_grad():
        if eval_split_mode == "global_support" and str(getattr(args, "eval_classifier", "memory")) == "prototype":
            for global_support_idx, _global_query_idx, _global_query_set in global_split_cache:
                support_speech, support_aux_t, support_mask = make_episode_tensors(x, aux, global_support_idx, device)
                support_labels = torch.tensor(y[global_support_idx], dtype=torch.long, device=device) if len(global_support_idx) else None
                global_prototype_cache.append(
                    build_support_prototypes(
                        model=model,
                        support_speech=support_speech,
                        support_aux=support_aux_t,
                        support_mask=support_mask,
                        support_labels=support_labels,
                        fallback_global=bool(getattr(args, "prototype_fallback_global", True)),
                        support_weight=float(getattr(args, "prototype_support_weight", 1.0)),
                    )
                )
        for ep_pos, episode_indices in enumerate(episode_groups):
            eval_speech, eval_aux_t, eval_mask = make_episode_tensors(x, aux, episode_indices, device)
            ensemble_ep_prob = None
            ensemble_gl_prob = None
            episode_support_mask = np.zeros(len(episode_indices), dtype=np.float32)
            episode_query_mask = np.zeros(len(episode_indices), dtype=np.float32)
            episode_gate_means = []
            episode_route_entropies = []
            episode_global_route_entropies = []
            episode_feature_rows = []
            episode_support_sizes = []
            episode_query_sizes = []
            for ensemble_idx in range(ensemble_runs):
                if eval_split_mode == "global_support":
                    support_idx, _global_query_idx, global_query_set = global_split_cache[ensemble_idx]
                    query_idx = np.asarray([int(idx) for idx in episode_indices.tolist() if int(idx) in global_query_set], dtype=np.int64)
                elif eval_split_mode == "fixed_query":
                    split_seed = args.seed + seed_offset + ep_pos * 100
                    support_seed = split_seed + (int(split_run_index) + 1) * 1_000_003 + ensemble_idx * 9_973
                    support_idx, query_idx = split_fixed_query_support(
                        episode_indices,
                        y,
                        eval_support_shots,
                        query_shots_per_class=int(getattr(args, "eval_query_shots_per_class", 1) or 1),
                        seed=split_seed,
                        support_seed=support_seed,
                    )
                else:
                    support_idx, query_idx = split_support_query(
                        episode_indices,
                        y,
                        eval_support_shots,
                        seed=args.seed + seed_offset + ep_pos * 100 + ensemble_idx + int(split_run_index) * 1_000_003,
                    )
                support_flag[support_idx] = 1
                query_flag[query_idx] = 1
                episode_support_sizes.append(int(len(support_idx)))
                episode_query_sizes.append(int(len(query_idx)))
                local_index = {int(idx): pos for pos, idx in enumerate(episode_indices.tolist())}
                for idx in support_idx.tolist():
                    if int(idx) in local_index:
                        episode_support_mask[local_index[int(idx)]] = 1.0
                for idx in query_idx.tolist():
                    if int(idx) in local_index:
                        episode_query_mask[local_index[int(idx)]] = 1.0
                build = None
                eval_classifier = str(getattr(args, "eval_classifier", "memory"))
                if eval_classifier == "global_memory":
                    episode_out = memory_cosine_predict_from_memory(
                        model=model,
                        eval_speech=eval_speech,
                        eval_aux=eval_aux_t,
                        eval_mask=eval_mask,
                        memory=model.global_memory,
                    )
                elif eval_classifier == "prototype":
                    if eval_split_mode == "global_support" and global_prototype_cache:
                        episode_out = prototype_predict_from_prototypes(
                            model=model,
                            eval_speech=eval_speech,
                            eval_aux=eval_aux_t,
                            eval_mask=eval_mask,
                            prototypes=global_prototype_cache[ensemble_idx],
                        )
                    else:
                        support_speech, support_aux_t, support_mask = make_episode_tensors(x, aux, support_idx, device)
                        support_labels = torch.tensor(y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
                        episode_out = prototype_predict(
                            model=model,
                            eval_speech=eval_speech,
                            eval_aux=eval_aux_t,
                            eval_mask=eval_mask,
                            support_speech=support_speech,
                            support_aux=support_aux_t,
                            support_mask=support_mask,
                            support_labels=support_labels,
                            fallback_global=bool(getattr(args, "prototype_fallback_global", True)),
                            support_weight=float(getattr(args, "prototype_support_weight", 1.0)),
                        )
                else:
                    support_speech, support_aux_t, support_mask = make_episode_tensors(x, aux, support_idx, device)
                    support_labels = torch.tensor(y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
                    build = model.build_episode_memory(
                        support_speech,
                        support_aux_t,
                        support_mask,
                        support_labels=support_labels,
                        label_blend=label_blend,
                    )
                    if eval_classifier == "gema_cosine":
                        episode_out = memory_cosine_predict_from_memory(
                            model=model,
                            eval_speech=eval_speech,
                            eval_aux=eval_aux_t,
                            eval_mask=eval_mask,
                            memory=build["episode_memory"],
                        )
                    else:
                        episode_out = model.predict(eval_speech, eval_aux_t, eval_mask, build["episode_memory"])
                global_out = model.predict(eval_speech, eval_aux_t, eval_mask, model.global_memory)
                ep_prob = torch.softmax(episode_out["logits"], dim=-1).detach().cpu().numpy()
                gl_prob = torch.softmax(global_out["logits"], dim=-1).detach().cpu().numpy()
                ensemble_ep_prob = ep_prob if ensemble_ep_prob is None else ensemble_ep_prob + ep_prob
                ensemble_gl_prob = gl_prob if ensemble_gl_prob is None else ensemble_gl_prob + gl_prob
                if build is not None:
                    episode_gate_means.append(float(build["write_gate"].mean().detach().cpu()))
                if episode_out.get("feature_weights") is not None and episode_out["feature_weights"].numel() > 0:
                    episode_feature_rows.append(episode_out["feature_weights"].mean(dim=0).detach().cpu().numpy())
                if episode_out.get("route_weights") is not None and episode_out["route_weights"].numel() > 0:
                    episode_route_entropies.append(route_entropy(episode_out["route_weights"]))
                if global_out.get("route_weights") is not None and global_out["route_weights"].numel() > 0:
                    episode_global_route_entropies.append(route_entropy(global_out["route_weights"]))
            ep_prob = ensemble_ep_prob / float(ensemble_runs)
            gl_prob = ensemble_gl_prob / float(ensemble_runs)
            prob[episode_indices] = ep_prob
            global_prob[episode_indices] = gl_prob
            pred[episode_indices] = ep_prob.argmax(axis=-1).astype(np.int64)
            support_flag[episode_indices] = np.maximum(support_flag[episode_indices], episode_support_mask.astype(np.int64))
            query_flag[episode_indices] = np.maximum(query_flag[episode_indices], episode_query_mask.astype(np.int64))
            support_sizes.append(float(np.mean(episode_support_sizes)) if episode_support_sizes else 0.0)
            query_sizes.append(float(np.mean(episode_query_sizes)) if episode_query_sizes else 0.0)
            gate_means.append(float(np.mean(episode_gate_means)) if episode_gate_means else 0.0)
            if episode_feature_rows:
                feature_weight_rows.append(np.mean(np.stack(episode_feature_rows, axis=0), axis=0))
            if episode_route_entropies:
                route_entropies.append(float(np.mean(episode_route_entropies)))
            if episode_global_route_entropies:
                global_route_entropies.append(float(np.mean(episode_global_route_entropies)))
    valid = pred >= 0
    if bool(getattr(args, "eval_query_only", False)):
        valid = np.logical_and(valid, query_flag == 1)
    global_pred = global_prob.argmax(axis=-1).astype(np.int64)
    metrics = metric_dict(y[valid], pred[valid]) if valid.any() else metric_dict(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    global_metrics = metric_dict(y[valid], global_pred[valid]) if valid.any() else metric_dict(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    if subgroup_masks:
        for name, mask in subgroup_masks.items():
            group_valid = np.logical_and(valid, np.asarray(mask, dtype=bool))
            prefix = f"subgroup_{name}"
            group_metrics = metric_dict(y[group_valid], pred[group_valid]) if group_valid.any() else metric_dict(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
            group_global = metric_dict(y[group_valid], global_pred[group_valid]) if group_valid.any() else metric_dict(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
            for key, value in group_metrics.items():
                metrics[f"{prefix}_{key}"] = value
            for key, value in group_global.items():
                global_metrics[f"{prefix}_{key}"] = value
            metrics[f"{prefix}_rows"] = float(group_valid.sum())
            global_metrics[f"{prefix}_rows"] = float(group_valid.sum())
    feature_means = (
        np.mean(np.stack(feature_weight_rows, axis=0), axis=0)
        if feature_weight_rows
        else np.zeros((0,), dtype=np.float32)
    )
    diag = {
        "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
        "write_gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
        "episode_prompt_route_entropy": float(np.mean(route_entropies)) if route_entropies else 0.0,
        "global_prompt_route_entropy": float(np.mean(global_route_entropies)) if global_route_entropies else 0.0,
        "support_fraction": float(support_flag.mean()) if len(support_flag) else 0.0,
        "query_fraction": float(query_flag.mean()) if len(query_flag) else 0.0,
        "mean_query_size": float(np.mean(query_sizes)) if query_sizes else 0.0,
        "evaluated_rows": int(valid.sum()),
        "evaluated_fraction": float(valid.mean()) if len(valid) else 0.0,
        "eval_support_shots": int(eval_support_shots),
        "eval_ensemble_runs": int(ensemble_runs),
        "eval_split_run_index": int(split_run_index),
        "eval_split_mode": eval_split_mode,
        "eval_query_shots_per_class": int(getattr(args, "eval_query_shots_per_class", 1) or 1),
        "eval_classifier": str(getattr(args, "eval_classifier", "memory")),
        "prototype_fallback_global": bool(getattr(args, "prototype_fallback_global", True)),
        "prototype_support_weight": float(getattr(args, "prototype_support_weight", 1.0)),
        "eval_query_only": bool(getattr(args, "eval_query_only", False)),
        "support_write_mode": str(getattr(args, "support_write_mode", "label")),
        "support_label_blend": float(label_blend),
    }
    for idx, value in enumerate(feature_means.tolist()):
        diag[f"feature_weight_{idx}"] = float(value)
    return metrics, pred, prob, diag, global_metrics, global_prob, support_flag, query_flag


def numeric_mean_std(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for row in rows for key in row.keys()})
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except Exception:
                pass
        if values:
            arr = np.asarray(values, dtype=np.float64)
            mean[key] = float(arr.mean())
            std[key] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def evaluate_split_runs(
    model: torch.nn.Module,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    episode_groups: Sequence[np.ndarray],
    args: Any,
    device: torch.device,
    seed_offset: int,
    split_name: str = "eval",
    subgroup_masks: dict[str, np.ndarray] | None = None,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
    dict[str, float],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    split_runs = max(1, int(getattr(args, "eval_split_runs", 1) or 1))
    metric_rows: list[dict[str, Any]] = []
    global_metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    first_payload = None
    eval_support_shots = int(getattr(args, "eval_support_shots", 0) or 0)
    if eval_support_shots <= 0:
        eval_support_shots = int(getattr(args, "support_shots", 1))
    eval_split_mode = str(getattr(args, "eval_split_mode", "global_support"))
    use_progress = bool(getattr(args, "progress", True)) and split_runs > 1 and tqdm is not None
    iterator = range(split_runs)
    progress_bar = None
    if use_progress:
        progress_bar = tqdm(
            iterator,
            total=split_runs,
            desc=f"{split_name} {eval_split_mode} k={eval_support_shots}",
            unit="split",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.3,
        )
        iterator = progress_bar
    for split_idx in iterator:
        split_start = time.perf_counter()
        payload = evaluate(
            model=model,
            x=x,
            aux=aux,
            y=y,
            episode_groups=episode_groups,
            args=args,
            device=device,
            seed_offset=seed_offset,
            split_run_index=split_idx,
            subgroup_masks=subgroup_masks,
        )
        metrics, _pred, _prob, diag, global_metrics, _global_prob, _support_flag, _query_flag = payload
        if first_payload is None:
            first_payload = payload
        metric_rows.append({"split": split_idx, **metrics})
        global_metric_rows.append({"split": split_idx, **global_metrics})
        diag_rows.append({"split": split_idx, **diag})
        split_time = time.perf_counter() - split_start
        if progress_bar is not None:
            progress_bar.set_postfix(
                {
                    "split_s": f"{split_time:.1f}",
                    "mf1": f"{metrics.get('macro_f1', 0.0) * 100.0:.2f}",
                    "acc": f"{metrics.get('accuracy', 0.0) * 100.0:.2f}",
                },
                refresh=True,
            )
    assert first_payload is not None
    metric_mean, metric_std = numeric_mean_std(metric_rows)
    global_mean, global_std = numeric_mean_std(global_metric_rows)
    diag_mean, diag_std = numeric_mean_std(diag_rows)
    for summary in [metric_mean, metric_std, global_mean, global_std, diag_mean, diag_std]:
        summary.pop("split", None)
    metrics0, pred0, prob0, diag0, global0, global_prob0, support_flag0, query_flag0 = first_payload
    diag_mean = {**diag0, **diag_mean}
    diag_mean.pop("eval_split_run_index", None)
    diag_mean["eval_split_runs"] = int(split_runs)
    split_summary = {
        "runs": int(split_runs),
        "metrics": metric_rows,
        "global_metrics": global_metric_rows,
        "diagnostics": diag_rows,
        "metrics_std": metric_std,
        "global_metrics_std": global_std,
        "diagnostics_std": diag_std,
        "first_split_metrics": metrics0,
        "first_split_global_metrics": global0,
        "first_split_diagnostics": diag0,
    }
    return metric_mean, pred0, prob0, diag_mean, global_mean, global_prob0, support_flag0, query_flag0, split_summary
