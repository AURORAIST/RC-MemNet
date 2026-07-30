"""Main acoustic encoder for PC-DLCMNet."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_dlcmnet.utils.tensor import masked_mean, normalize_branch_dims


class MultiSourceAcousticRepresentation(nn.Module):
    """Fuse speech tokens with token-aligned mel / MFCC / delta branches."""

    def __init__(
        self,
        input_dim: int,
        aux_dim: int,
        hidden_dim: int,
        branch_dims: Sequence[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_dims = normalize_branch_dims(aux_dim, branch_dims)
        self.speech_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.branch_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for dim in self.branch_dims
            ]
        )
        self.branch_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in self.branch_dims
            ]
        )
        self.fuse_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _split(self, aux: torch.Tensor) -> list[torch.Tensor]:
        parts = []
        start = 0
        for dim in self.branch_dims:
            parts.append(aux[..., start : start + dim])
            start += dim
        return parts

    @staticmethod
    def _align_tokens(x: torch.Tensor, token_count: int) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] == token_count:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=int(token_count),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, token_count, _ = speech_tokens.shape
        speech_hidden = self.speech_projection(speech_tokens)
        if aux_features.ndim == 2:
            branch_inputs = self._split(aux_features)
            branch_tokens = [
                proj(part).unsqueeze(1).expand(-1, token_count, -1)
                for proj, part in zip(self.branch_projections, branch_inputs)
            ]
        elif aux_features.ndim == 3:
            branch_inputs = self._split(aux_features)
            branch_tokens = [
                self._align_tokens(proj(part), token_count)
                for proj, part in zip(self.branch_projections, branch_inputs)
            ]
        else:
            raise ValueError(f"Expected 2D or 3D aux_features, got {tuple(aux_features.shape)}")

        pooled = [masked_mean(branch, audio_mask) for branch in branch_tokens]
        scores = torch.cat(
            [scorer(vec) for scorer, vec in zip(self.branch_scorers, pooled)],
            dim=-1,
        )
        weights = torch.softmax(scores, dim=-1)
        fused_feature = sum(
            branch * weights[:, idx].view(batch_size, 1, 1)
            for idx, branch in enumerate(branch_tokens)
        )
        gate = self.fuse_gate(torch.cat([speech_hidden, fused_feature], dim=-1))
        h0 = self.output_norm(gate * speech_hidden + (1.0 - gate) * fused_feature)
        return {
            "h0": h0,
            "speech_hidden": speech_hidden,
            "fused_feature": fused_feature,
            "feature_weights": weights,
            "feature_gate": gate,
        }
