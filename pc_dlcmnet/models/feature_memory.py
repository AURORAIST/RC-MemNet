#!/usr/bin/env python3
"""Prompt-guided class-memory Transformer.

This implementation follows the 0614 method definition:

    Z_i^0 = [M^0; H_i^0]
    Z_i^L = Transformer(Z_i^0; P)
    logits = shared_scorer(M_i^L)

The default variants keep M^0 as a shared learnable class-memory matrix.
Recurrent variants additionally carry M_{t-1} across an episode and write a
gated, class-selective update after prediction:

    [M_{t-1}; H_t^0] -> [M~_t; H_t^L] -> M_t

Prompt parameters modulate Q/K/V inside self-attention; they are not appended
as input tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_dlcmnet.utils.tensor import safe_mm


VOWEL_ORDER = ["a", "e", "i", "o", "u", "y", "\u0254", "\u0259", "\u025b"]
VOWEL_TO_ID = {label: idx for idx, label in enumerate(VOWEL_ORDER)}

VOWEL_TO_PHONOLOGY = {
    "a": {"height": "low", "backness": "central", "rounding": "unrounded"},
    "e": {"height": "mid-high", "backness": "front", "rounding": "unrounded"},
    "i": {"height": "high", "backness": "front", "rounding": "unrounded"},
    "o": {"height": "mid-high", "backness": "back", "rounding": "rounded"},
    "u": {"height": "high", "backness": "back", "rounding": "rounded"},
    "y": {"height": "high", "backness": "front", "rounding": "rounded"},
    "\u0254": {"height": "mid-low", "backness": "back", "rounding": "rounded"},
    "\u0259": {"height": "central", "backness": "central", "rounding": "unrounded"},
    "\u025b": {"height": "mid-low", "backness": "front", "rounding": "unrounded"},
}
HEIGHT_ORDER = ["central", "high", "low", "mid-high", "mid-low"]
BACKNESS_ORDER = ["back", "central", "front"]
ROUNDING_ORDER = ["rounded", "unrounded"]


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    aux_dim: int
    aux_branch_dims: tuple[int, ...] = ()
    num_classes: int = len(VOWEL_ORDER)
    num_domains: int = 0
    hidden_dim: int = 256
    score_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    max_audio_tokens: int = 32
    anchor_kind: str = "onehot"
    projection_seed: int = 42
    use_feature_fusion: bool = True
    use_prompt: bool = True
    use_region_memory: bool = False
    use_recurrent_memory: bool = False
    region_temperature: float = 0.25


def fixed_half_orthogonal_projection(prototype_dim: int, hidden_dim: int, seed: int) -> torch.Tensor:
    if prototype_dim > hidden_dim:
        raise ValueError(f"prototype_dim={prototype_dim} must be <= hidden_dim={hidden_dim}")
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(hidden_dim, prototype_dim, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.transpose(0, 1).contiguous()


def build_raw_class_prototypes(kind: str, seed: int = 0) -> torch.Tensor:
    kind = str(kind).lower()
    num_classes = len(VOWEL_ORDER)
    if kind == "none":
        return torch.zeros(num_classes, num_classes, dtype=torch.float32)
    if kind == "onehot":
        return torch.eye(num_classes, dtype=torch.float32)
    if kind == "random":
        generator = torch.Generator().manual_seed(int(seed))
        proto = torch.randn(num_classes, num_classes, generator=generator)
        return F.normalize(proto, dim=-1)
    if kind == "ipa":
        rows = []
        for vowel in VOWEL_ORDER:
            ph = VOWEL_TO_PHONOLOGY[vowel]
            row = []
            row.extend(float(ph["height"] == value) for value in HEIGHT_ORDER)
            row.extend(float(ph["backness"] == value) for value in BACKNESS_ORDER)
            row.extend(float(ph["rounding"] == value) for value in ROUNDING_ORDER)
            rows.append(row)
        return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)
    raise ValueError(f"Unsupported anchor kind: {kind}")


def build_hidden_anchors(kind: str, hidden_dim: int, seed: int) -> torch.Tensor:
    raw = build_raw_class_prototypes(kind, seed)
    if str(kind).lower() == "none":
        return torch.zeros(raw.shape[0], hidden_dim, dtype=torch.float32)
    projection = fixed_half_orthogonal_projection(raw.shape[1], hidden_dim, seed)
    return F.normalize(raw, dim=-1) @ projection


def normalize_branch_dims(aux_dim: int, branch_dims: Sequence[int] | None) -> tuple[int, ...]:
    if branch_dims:
        dims = tuple(int(v) for v in branch_dims if int(v) > 0)
        if sum(dims) == aux_dim:
            return dims
        if sum(dims) < aux_dim:
            return dims + (aux_dim - sum(dims),)
    return (int(aux_dim),)


class MultiSourceFeatureFusion(nn.Module):
    """Fuse mel/MFCC/delta feature branches into one token-aligned sequence."""

    def __init__(
        self,
        aux_dim: int,
        hidden_dim: int,
        branch_dims: Sequence[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branch_dims = normalize_branch_dims(aux_dim, branch_dims)
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
        self.branch_scores = nn.ModuleList(
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

    def forward(self, aux_features: torch.Tensor, token_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if aux_features.ndim == 2:
            branch_inputs = self._split(aux_features)
            branch_vectors = [proj(part) for proj, part in zip(self.branch_projections, branch_inputs)]
            branch_tokens = [vec.unsqueeze(1).expand(-1, token_count, -1) for vec in branch_vectors]
            branch_pooled = branch_vectors
        elif aux_features.ndim == 3:
            branch_inputs = self._split(aux_features)
            branch_tokens = [
                self._align_tokens(proj(part), token_count)
                for proj, part in zip(self.branch_projections, branch_inputs)
            ]
            branch_pooled = [tokens.mean(dim=1) for tokens in branch_tokens]
        else:
            raise ValueError(f"Expected aux_features to be 2D or 3D, got {tuple(aux_features.shape)}")

        scores = torch.cat(
            [score(pooled) for score, pooled in zip(self.branch_scores, branch_pooled)],
            dim=-1,
        )
        weights = torch.softmax(scores, dim=-1)
        fused = sum(tokens * weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(branch_tokens))
        condition = fused.mean(dim=1)
        return fused, condition, weights


class PromptGatedSelfAttention(nn.Module):
    """Multi-head self-attention with learned global and optional region-conditioned Q/K/V gates."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_prompt: bool = True,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim // num_heads)
        self.use_prompt = bool(use_prompt)
        self.condition_dim = int(condition_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.use_prompt:
            self.prompt = nn.Parameter(torch.randn(hidden_dim) * 0.02)
            self.prompt_to_gates = nn.Linear(hidden_dim, 3 * hidden_dim)
        else:
            self.register_parameter("prompt", None)
            self.prompt_to_gates = None
        if self.use_prompt and self.condition_dim > 0:
            self.condition_to_gates = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, 3 * hidden_dim),
            )
            # Start from the old global-prompt behavior and let training decide how much region context matters.
            nn.init.zeros_(self.condition_to_gates[-1].weight)
            nn.init.zeros_(self.condition_to_gates[-1].bias)
        else:
            self.condition_to_gates = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def prompt_gates(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        prompt_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_prompt or self.prompt_to_gates is None or self.prompt is None:
            gates = torch.ones(batch_size, 3, self.num_heads, self.head_dim, device=device, dtype=dtype)
            return gates, gates.new_zeros(6)
        raw = self.prompt_to_gates(self.prompt.to(device=device, dtype=dtype))
        raw = raw.unsqueeze(0).expand(batch_size, -1)
        if self.condition_to_gates is not None and prompt_context is not None:
            raw = raw + self.condition_to_gates(prompt_context.to(device=device, dtype=dtype))
        gates = (2.0 * torch.sigmoid(raw)).view(batch_size, 3, self.num_heads, self.head_dim)
        stats = torch.stack(
            [
                gates[:, 0].mean(),
                gates[:, 0].std(),
                gates[:, 1].mean(),
                gates[:, 1].std(),
                gates[:, 2].mean(),
                gates[:, 2].std(),
            ]
        )
        return gates, stats

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prompt_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        gates, stats = self.prompt_gates(x.device, x.dtype, x.shape[0], prompt_context)
        q = q * gates[:, 0].view(x.shape[0], self.num_heads, 1, self.head_dim)
        k = k * gates[:, 1].view(x.shape[0], self.num_heads, 1, self.head_dim)
        v = v * gates[:, 2].view(x.shape[0], self.num_heads, 1, self.head_dim)

        scores = (q.unsqueeze(3) * k.unsqueeze(2)).sum(dim=-1) / math.sqrt(float(self.head_dim))
        if key_padding_mask is not None:
            mask = key_padding_mask.bool().view(key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1])
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v.unsqueeze(2)).sum(dim=3)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        return self.out_proj(out), stats


class PromptedTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_prompt: bool = True,
        prompt_condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = PromptGatedSelfAttention(
            hidden_dim,
            num_heads,
            dropout,
            use_prompt=use_prompt,
            condition_dim=prompt_condition_dim,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prompt_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, prompt_stats = self.attn(h, key_padding_mask, prompt_context)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, prompt_stats


class FeatureMemoryTransformer(nn.Module):
    """One Transformer with shared class memories and fused acoustic tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(config.num_classes)
        self.hidden_dim = int(config.hidden_dim)
        self.anchor_kind = str(config.anchor_kind)
        self.use_feature_fusion = bool(config.use_feature_fusion)
        self.use_prompt = bool(config.use_prompt)
        self.use_region_memory = bool(config.use_region_memory)
        self.use_recurrent_memory = bool(config.use_recurrent_memory)
        self.num_domains = int(config.num_domains)
        self.region_temperature = float(config.region_temperature)

        anchors = build_hidden_anchors(config.anchor_kind, config.hidden_dim, config.projection_seed)
        self.register_buffer("anchors", anchors)
        self.class_memory = nn.Parameter(anchors + torch.randn_like(anchors) * 0.02)

        self.speech_norm = nn.LayerNorm(config.input_dim)
        self.speech_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.fused_input_norm = nn.LayerNorm(config.hidden_dim)
        self.audio_pos = nn.Parameter(torch.randn(1, config.max_audio_tokens, config.hidden_dim) * 0.02)
        self.audio_type = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.memory_type = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.region_type = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

        self.feature_fusion = MultiSourceFeatureFusion(
            aux_dim=config.aux_dim,
            hidden_dim=config.hidden_dim,
            branch_dims=config.aux_branch_dims,
            dropout=config.dropout,
        )
        if self.use_region_memory:
            if self.num_domains <= 1:
                raise ValueError("Region memory requires num_domains > 1.")
            self.region_memory = nn.Parameter(torch.randn(self.num_domains, config.hidden_dim) * 0.02)
            self.region_query = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.region_feature_gate = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Sigmoid(),
            )
            self.feature_residual_logit = nn.Parameter(torch.tensor(-2.2))
        else:
            self.register_parameter("region_memory", None)
            self.region_query = None
            self.region_feature_gate = None
            self.feature_residual_logit = None

        self.layers = nn.ModuleList(
            [
                PromptedTransformerLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    use_prompt=config.use_prompt,
                    prompt_condition_dim=config.hidden_dim if config.use_region_memory else 0,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.memory_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.score_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.shared_scorer = nn.Linear(config.score_dim, 1, bias=False)
        self.class_score_bias = nn.Parameter(torch.zeros(config.num_classes))
        if self.use_recurrent_memory:
            self.memory_write_gate = nn.Sequential(
                nn.LayerNorm(3 * config.hidden_dim),
                nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            )
            self.memory_state_norm = nn.LayerNorm(config.hidden_dim)
            nn.init.constant_(self.memory_write_gate[-1].bias, 2.0)
        else:
            self.memory_write_gate = None
            self.memory_state_norm = None

    def base_memory(self, class_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_ids = class_ids.to(self.class_memory.device)
        return self.class_memory[class_ids], self.anchors[class_ids]

    def expand_memory_state(
        self,
        base_memory: torch.Tensor,
        batch_size: int,
        memory_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_state is None:
            return base_memory.unsqueeze(0).expand(batch_size, -1, -1)
        state = memory_state.to(device=base_memory.device, dtype=base_memory.dtype)
        if state.ndim == 2:
            if state.shape != base_memory.shape:
                raise ValueError(f"memory_state shape {tuple(state.shape)} does not match {tuple(base_memory.shape)}")
            return state.unsqueeze(0).expand(batch_size, -1, -1)
        if state.ndim == 3:
            if state.shape[1:] != base_memory.shape:
                raise ValueError(f"memory_state shape {tuple(state.shape)} does not match class memory {tuple(base_memory.shape)}")
            if state.shape[0] == batch_size:
                return state
            if state.shape[0] == 1:
                return state.expand(batch_size, -1, -1)
            raise ValueError(f"memory_state batch={state.shape[0]} does not match input batch={batch_size}")
        raise ValueError(f"memory_state must be 2D or 3D, got {tuple(state.shape)}")

    def apply_memory_write(
        self,
        previous_memory: torch.Tensor,
        candidate_memory: torch.Tensor,
        audio_summary: torch.Tensor,
        class_ids: torch.Tensor,
        write_labels: torch.Tensor | None = None,
        write_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_recurrent_memory or self.memory_write_gate is None:
            empty_gate = previous_memory.new_empty(previous_memory.shape[0], previous_memory.shape[1], 0)
            empty_weights = previous_memory.new_empty(previous_memory.shape[0], previous_memory.shape[1])
            return previous_memory, empty_gate, empty_weights

        summary = audio_summary.unsqueeze(1).expand(-1, previous_memory.shape[1], -1)
        gate_input = torch.cat([previous_memory, candidate_memory, summary], dim=-1)
        write_gate = torch.sigmoid(self.memory_write_gate(gate_input))

        if write_weights is None:
            if write_labels is None:
                weights = previous_memory.new_zeros(previous_memory.shape[:2])
            else:
                labels = write_labels.to(device=previous_memory.device).view(-1, 1)
                ids = class_ids.to(device=previous_memory.device).view(1, -1)
                weights = (labels == ids).to(dtype=previous_memory.dtype)
        else:
            weights = write_weights.to(device=previous_memory.device, dtype=previous_memory.dtype)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            if weights.shape != previous_memory.shape[:2]:
                raise ValueError(f"write_weights shape {tuple(weights.shape)} does not match {tuple(previous_memory.shape[:2])}")
            weights = weights.clamp(0.0, 1.0)

        update = weights.unsqueeze(-1) * (1.0 - write_gate) * (candidate_memory - previous_memory)
        updated = previous_memory + update
        if self.memory_state_norm is not None:
            target_norm = previous_memory.detach().norm(dim=-1, keepdim=True).clamp(
                min=1.0,
                max=math.sqrt(float(self.hidden_dim)),
            )
            updated = self.memory_state_norm(updated)
            updated = F.normalize(updated, dim=-1) * target_norm
        return updated, write_gate, weights

    def forward(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor,
        class_ids: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        write_labels: torch.Tensor | None = None,
        write_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del domain_ids
        batch_size, token_count, _ = speech_tokens.shape
        class_count = int(class_ids.numel())
        audio_padding = ~audio_mask.bool()

        speech = self.speech_projection(self.speech_norm(speech_tokens))
        feature_sequence, acoustic_condition, feature_weights = self.feature_fusion(aux_features, token_count)
        region_context = None
        region_attention = speech.new_empty(batch_size, 0)
        region_logits = speech.new_empty(batch_size, 0)
        feature_gate = speech.new_zeros(batch_size, 1, self.hidden_dim)
        if self.use_region_memory and self.region_memory is not None:
            region_query = F.normalize(self.region_query(acoustic_condition), dim=-1)
            region_keys = F.normalize(self.region_memory, dim=-1)
            region_logits = safe_mm(region_query, region_keys.transpose(0, 1)) / max(self.region_temperature, 1e-6)
            region_attention = torch.softmax(region_logits, dim=-1)
            region_context = safe_mm(region_attention, self.region_memory)
            feature_gate = self.region_feature_gate(region_context).unsqueeze(1)
            feature_scale = torch.sigmoid(self.feature_residual_logit).view(1, 1, 1)
            feature_gate = feature_gate * feature_scale
        if self.use_feature_fusion:
            if self.use_region_memory:
                speech = self.fused_input_norm(speech + feature_gate * feature_sequence)
            else:
                speech = self.fused_input_norm(speech + feature_sequence)
        speech = speech + self.audio_pos[:, :token_count, :] + self.audio_type

        base_memory, anchors = self.base_memory(class_ids)
        previous_memory = self.expand_memory_state(base_memory, batch_size, memory_state)
        memory = previous_memory + self.memory_type
        pieces = []
        if region_context is not None:
            pieces.append(region_context.unsqueeze(1) + self.region_type)
        pieces.extend([memory, speech])
        sequence = torch.cat(pieces, dim=1)
        memory_padding = torch.zeros(batch_size, class_count, dtype=torch.bool, device=speech_tokens.device)
        padding_pieces = []
        if region_context is not None:
            padding_pieces.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=speech_tokens.device))
        padding_pieces.extend([memory_padding, audio_padding])
        padding = torch.cat(padding_pieces, dim=1)

        prompt_stats = []
        for layer in self.layers:
            sequence, stats = layer(sequence, padding, region_context)
            prompt_stats.append(stats)

        memory_start = 1 if region_context is not None else 0
        memory_end = memory_start + class_count
        conditioned_region = sequence[:, :1, :] if region_context is not None else speech.new_empty(batch_size, 0, self.hidden_dim)
        conditioned_memory = sequence[:, memory_start:memory_end, :]
        conditioned_audio = sequence[:, memory_end:, :]
        logits = self.shared_scorer(self.memory_mlp(conditioned_memory)).squeeze(-1)
        logits = logits + self.class_score_bias[class_ids.to(logits.device)]

        mask = audio_mask.bool().unsqueeze(-1)
        audio_summary = (conditioned_audio * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        prompt_stat_tensor = torch.stack(prompt_stats).mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        candidate_memory = conditioned_memory - self.memory_type
        if self.use_recurrent_memory:
            updated_memory, write_gate, applied_write_weights = self.apply_memory_write(
                previous_memory=previous_memory,
                candidate_memory=candidate_memory,
                audio_summary=audio_summary,
                class_ids=class_ids,
                write_labels=write_labels,
                write_weights=write_weights,
            )
            memory_residual = candidate_memory - previous_memory
            write_update = updated_memory - previous_memory
        else:
            updated_memory = previous_memory
            write_gate = speech.new_empty(batch_size, class_count, 0)
            applied_write_weights = speech.new_empty(batch_size, class_count)
            memory_residual = speech.new_zeros(batch_size, class_count, self.hidden_dim)
            write_update = speech.new_zeros(batch_size, class_count, self.hidden_dim)
        return {
            "logits": logits,
            "base_memory": base_memory,
            "anchors": anchors,
            "previous_memory": previous_memory,
            "candidate_memory": candidate_memory,
            "updated_memory": updated_memory,
            "conditioned_memory": conditioned_memory,
            "memory_residual": memory_residual,
            "write_gate": write_gate,
            "write_weights": applied_write_weights,
            "write_update": write_update,
            "audio_summary": audio_summary,
            "acoustic_condition": acoustic_condition,
            "feature_bias": feature_sequence,
            "feature_gate": feature_gate,
            "feature_weights": feature_weights,
            "prompt_stats": prompt_stat_tensor,
            "region_logits": region_logits,
            "region_attention": region_attention,
            "region_context": region_context if region_context is not None else speech.new_empty(batch_size, 0),
            "conditioned_region": conditioned_region,
        }


class SpeechTransformerBaseline(nn.Module):
    """Control baseline with the same speech Transformer but no memory tokens."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.1,
        max_audio_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, max_audio_tokens, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor,
        class_ids: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del aux_features, domain_ids
        padding = ~audio_mask.bool()
        x = self.proj(self.norm(speech_tokens)) + self.pos[:, : speech_tokens.shape[1], :]
        x = self.encoder(x, src_key_padding_mask=padding)
        mask = audio_mask.bool().unsqueeze(-1)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        masked = x.masked_fill(~mask, -1e4)
        max_pooled = masked.max(dim=1).values
        logits_all = self.head(torch.cat([mean, max_pooled], dim=-1))
        logits = logits_all[:, class_ids.to(logits_all.device)]
        empty = torch.empty(0, x.shape[-1], device=x.device)
        return {
            "logits": logits,
            "base_memory": empty,
            "anchors": empty,
            "conditioned_memory": empty,
            "memory_residual": torch.empty(speech_tokens.shape[0], 0, x.shape[-1], device=x.device),
            "audio_summary": mean,
            "acoustic_condition": mean,
            "feature_bias": empty,
            "feature_weights": torch.empty(speech_tokens.shape[0], 0, device=x.device),
            "prompt_stats": torch.zeros(speech_tokens.shape[0], 6, device=x.device),
        }


def anchor_alignment_loss(base_memory: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if base_memory.numel() == 0 or anchors.numel() == 0 or anchors.abs().sum() == 0:
        return base_memory.new_tensor(0.0)
    return (F.normalize(base_memory, dim=-1) - F.normalize(anchors, dim=-1)).pow(2).sum(dim=-1).mean()


def residual_norm_loss(memory_residual: torch.Tensor) -> torch.Tensor:
    if memory_residual.numel() == 0:
        return memory_residual.new_tensor(0.0)
    return memory_residual.pow(2).mean()


def region_compactness_loss(embeddings: torch.Tensor, domains: torch.Tensor | None) -> torch.Tensor:
    if domains is None or embeddings.numel() == 0:
        return embeddings.new_tensor(0.0)
    losses = []
    for domain in domains.detach().unique():
        mask = domains == domain
        if int(mask.sum()) < 2:
            continue
        group = F.normalize(embeddings[mask], dim=-1)
        center = group.mean(dim=0, keepdim=True)
        losses.append((group - center).pow(2).sum(dim=-1).mean())
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def memory_diagnostics(base_memory: torch.Tensor, anchors: torch.Tensor) -> dict[str, float]:
    if base_memory.numel() == 0 or anchors.numel() == 0:
        return {"base_delta_norm": 0.0, "base_memory_norm": 0.0, "anchor_cosine": 0.0}
    delta = base_memory - anchors
    if anchors.abs().sum() == 0:
        cosine = base_memory.new_tensor(0.0)
    else:
        cosine = F.cosine_similarity(base_memory, anchors, dim=-1).mean()
    return {
        "base_delta_norm": float(delta.norm(dim=-1).mean().detach().cpu()),
        "base_memory_norm": float(base_memory.norm(dim=-1).mean().detach().cpu()),
        "anchor_cosine": float(cosine.detach().cpu()),
    }


def build_model_from_variant(
    variant: str,
    input_dim: int,
    aux_dim: int,
    aux_branch_dims: Sequence[int] | None = None,
    num_domains: int = 0,
    hidden_dim: int = 256,
    score_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    ffn_dim: int = 768,
    dropout: float = 0.1,
    max_audio_tokens: int = 32,
    anchor_kind: str = "onehot",
    projection_seed: int = 42,
) -> nn.Module:
    variant = str(variant)
    if variant == "speech_baseline":
        return SpeechTransformerBaseline(
            input_dim=input_dim,
            num_classes=len(VOWEL_ORDER),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_audio_tokens=max_audio_tokens,
        )

    recurrent_variants = {"recurrent_memory_prompt", "recurrent_region_memory_prompt"}
    feature_variants = {"feature_memory_no_prompt", "feature_memory_prompt", "region_memory_prompt"} | recurrent_variants
    prompt_variants = {"feature_memory_prompt", "region_memory_prompt"} | recurrent_variants
    region_variants = {"region_memory_prompt", "recurrent_region_memory_prompt"}
    use_feature_fusion = variant in feature_variants
    use_prompt = variant in prompt_variants
    use_region_memory = variant in region_variants
    use_recurrent_memory = variant in recurrent_variants
    valid_variants = {"memory_only", "feature_memory_no_prompt", "feature_memory_prompt", "region_memory_prompt"} | recurrent_variants
    if variant not in valid_variants:
        raise ValueError(f"Unknown variant: {variant}")
    cfg = ModelConfig(
        input_dim=input_dim,
        aux_dim=aux_dim,
        aux_branch_dims=tuple(aux_branch_dims or ()),
        num_domains=int(num_domains),
        hidden_dim=hidden_dim,
        score_dim=score_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_audio_tokens=max_audio_tokens,
        anchor_kind=anchor_kind,
        projection_seed=projection_seed,
        use_feature_fusion=use_feature_fusion,
        use_prompt=use_prompt,
        use_region_memory=use_region_memory,
        use_recurrent_memory=use_recurrent_memory,
    )
    return FeatureMemoryTransformer(cfg)


def labels_to_ids(labels: Sequence[str]) -> torch.Tensor:
    return torch.tensor([VOWEL_TO_ID[str(label)] for label in labels], dtype=torch.long)
