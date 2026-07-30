"""Small tensor utilities shared by the dual-memory model and tasks."""

from __future__ import annotations

from typing import Sequence

import torch


def normalize_branch_dims(aux_dim: int, branch_dims: Sequence[int] | None) -> tuple[int, ...]:
    if branch_dims:
        dims = tuple(int(v) for v in branch_dims if int(v) > 0)
        if sum(dims) == aux_dim:
            return dims
        if sum(dims) < aux_dim:
            return dims + (int(aux_dim) - sum(dims),)
    return (int(aux_dim),)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def safe_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Small 2-D matrix multiply without cuBLAS batched-GEMM paths."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"safe_mm expects 2-D inputs, got {tuple(left.shape)} and {tuple(right.shape)}")
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"safe_mm shape mismatch: {tuple(left.shape)} x {tuple(right.shape)}")
    return (left.unsqueeze(2) * right.unsqueeze(0)).sum(dim=1)


def safe_batched_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Dot [batch, dim] queries against [batch, classes, dim] memory rows."""
    if left.ndim != 2 or right.ndim != 3:
        raise ValueError(f"safe_batched_dot expects [B,D] and [B,C,D], got {tuple(left.shape)} and {tuple(right.shape)}")
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[2]:
        raise ValueError(f"safe_batched_dot shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return (left.unsqueeze(1) * right).sum(dim=-1)
