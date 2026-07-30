"""Episode construction and support/query sampling for dual-memory experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def resolve_episode_column(rows: Any, args: Any) -> str:
    requested = str(getattr(args, "episode_column", "") or "").strip()
    if requested and requested in rows.columns:
        return requested
    if args.speaker_column in rows.columns:
        return args.speaker_column
    return args.region_column


def build_episode_groups(rows: Any, args: Any) -> tuple[list[np.ndarray], dict[str, Any]]:
    max_len = max(1, int(getattr(args, "episode_length", 16) or 16))
    column = resolve_episode_column(rows, args)
    frame = rows.reset_index(drop=True).copy()
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["_episode_key"] = frame[column].fillna("__missing__").astype(str)
    sort_cols = ["_episode_key"]
    for candidate in [args.path_column, args.start_column, "sample_id"]:
        if candidate in frame.columns:
            sort_cols.append(candidate)
    frame = frame.sort_values(sort_cols, kind="mergesort")
    groups: list[np.ndarray] = []
    lengths = []
    for _key, group in frame.groupby("_episode_key", sort=False):
        indices = group["_row_index"].to_numpy(dtype=np.int64)
        lengths.append(int(len(indices)))
        for start in range(0, len(indices), max_len):
            groups.append(indices[start : start + max_len])
    meta = {
        "episode_column": column,
        "episode_length": max_len,
        "num_episodes": int(frame["_episode_key"].nunique()),
        "num_episode_groups": int(len(groups)),
        "episode_size_min": int(min(lengths)) if lengths else 0,
        "episode_size_max": int(max(lengths)) if lengths else 0,
        "episode_size_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    return groups, meta


def split_support_query(
    indices: np.ndarray,
    labels: np.ndarray,
    support_shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    support_shots = max(0, int(support_shots))
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= 1:
        return np.zeros((0,), dtype=np.int64), indices.copy()
    rng = np.random.default_rng(int(seed))
    episode_labels = labels[indices]
    support: list[int] = []
    query: list[int] = []
    for class_id in np.unique(episode_labels):
        class_indices = indices[episode_labels == class_id].copy()
        if len(class_indices) <= 1:
            query.extend(class_indices.tolist())
            continue
        rng.shuffle(class_indices)
        keep_query = 1
        support_count = min(support_shots, max(0, len(class_indices) - keep_query))
        if support_count > 0:
            support.extend(class_indices[:support_count].tolist())
        query.extend(class_indices[support_count:].tolist())
    support = list(dict.fromkeys(support))
    query = list(dict.fromkeys(query))
    if len(query) == 0 and len(support) > 1:
        query.append(support.pop())
    if len(support) == 0 and len(query) > 1:
        support.append(query.pop(0))
    order = {int(idx): pos for pos, idx in enumerate(indices.tolist())}
    support_arr = np.asarray(sorted(support, key=lambda idx: order[int(idx)]), dtype=np.int64)
    query_arr = np.asarray(sorted(query, key=lambda idx: order[int(idx)]), dtype=np.int64)
    return support_arr, query_arr


def split_fixed_query_support(
    indices: np.ndarray,
    labels: np.ndarray,
    support_shots: int,
    query_shots_per_class: int,
    seed: int,
    support_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    support_shots = max(0, int(support_shots))
    query_shots_per_class = max(1, int(query_shots_per_class))
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= 1:
        return np.zeros((0,), dtype=np.int64), indices.copy()
    query_rng = np.random.default_rng(int(seed))
    support_rng = np.random.default_rng(int(seed if support_seed is None else support_seed))
    episode_labels = labels[indices]
    support: list[int] = []
    query: list[int] = []
    for class_id in np.unique(episode_labels):
        class_indices = indices[episode_labels == class_id].copy()
        if len(class_indices) <= 1:
            query.extend(class_indices.tolist())
            continue
        query_rng.shuffle(class_indices)
        query_count = min(query_shots_per_class, max(1, len(class_indices) - 1))
        query.extend(class_indices[:query_count].tolist())
        support_pool = class_indices[query_count:].copy()
        support_rng.shuffle(support_pool)
        support_count = min(support_shots, len(support_pool))
        if support_count > 0:
            support.extend(support_pool[:support_count].tolist())
    support = list(dict.fromkeys(support))
    query = list(dict.fromkeys(query))
    order = {int(idx): pos for pos, idx in enumerate(indices.tolist())}
    support_arr = np.asarray(sorted(support, key=lambda idx: order[int(idx)]), dtype=np.int64)
    query_arr = np.asarray(sorted(query, key=lambda idx: order[int(idx)]), dtype=np.int64)
    return support_arr, query_arr


def split_global_support_query(
    labels: np.ndarray,
    support_shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one target-domain k-shot support set per class."""
    support_shots = max(0, int(support_shots))
    all_indices = np.arange(len(labels), dtype=np.int64)
    if len(all_indices) <= 1:
        return np.zeros((0,), dtype=np.int64), all_indices.copy()
    rng = np.random.default_rng(int(seed))
    support: list[int] = []
    query: list[int] = []
    for class_id in np.unique(labels):
        class_indices = all_indices[labels == class_id].copy()
        if len(class_indices) <= 1:
            query.extend(class_indices.tolist())
            continue
        rng.shuffle(class_indices)
        support_count = min(support_shots, max(0, len(class_indices) - 1))
        if support_count > 0:
            support.extend(class_indices[:support_count].tolist())
        query.extend(class_indices[support_count:].tolist())
    return np.asarray(sorted(set(support)), dtype=np.int64), np.asarray(sorted(set(query)), dtype=np.int64)


def empty_episode_tensors(x: np.ndarray, aux: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speech = torch.zeros((0, x.shape[1], x.shape[2]), dtype=torch.float32, device=device)
    aux_shape = (0,) + tuple(aux.shape[1:])
    aux_tensor = torch.zeros(aux_shape, dtype=torch.float32, device=device)
    mask = torch.zeros((0, x.shape[1]), dtype=torch.bool, device=device)
    return speech, aux_tensor, mask


def make_episode_tensors(
    x: np.ndarray,
    aux: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(indices) == 0:
        return empty_episode_tensors(x, aux, device)
    speech = torch.tensor(x[indices], dtype=torch.float32, device=device)
    aux_tensor = torch.tensor(aux[indices], dtype=torch.float32, device=device)
    mask = torch.ones((len(indices), x.shape[1]), dtype=torch.bool, device=device)
    return speech, aux_tensor, mask
