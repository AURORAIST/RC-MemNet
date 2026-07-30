"""PC-DLCMNet network wrapper built from encoder and memory modules."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_dlcmnet.models.config import PCDLCMNetConfig
from pc_dlcmnet.models.encoder import MultiSourceAcousticRepresentation
from pc_dlcmnet.models.memory import PromptConditionedMemoryEncoder
from pc_dlcmnet.utils.tensor import masked_mean, safe_batched_dot, safe_mm


class PCDLCMNet(nn.Module):
    """Prompt-conditioned dual-level class memory Transformer."""

    def __init__(self, config: PCDLCMNetConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(config.num_classes)
        self.temperature = float(config.temperature)
        self.eps = float(config.eps)
        self.acoustic_representation = MultiSourceAcousticRepresentation(
            input_dim=config.input_dim,
            aux_dim=config.aux_dim,
            hidden_dim=config.hidden_dim,
            branch_dims=config.aux_branch_dims,
            dropout=config.dropout,
        )
        self.global_memory = nn.Parameter(torch.randn(config.num_classes, config.hidden_dim) * 0.02)
        self.memory_encoder = PromptConditionedMemoryEncoder(config)
        self.query_projection = nn.Linear(config.hidden_dim, config.score_dim)
        self.memory_projection = nn.Linear(config.hidden_dim, config.score_dim)
        self.write_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )

    def _expand_memory(self, memory: torch.Tensor, batch_size: int) -> torch.Tensor:
        if memory.ndim == 2:
            return memory.unsqueeze(0).expand(batch_size, -1, -1)
        if memory.ndim == 3:
            if memory.shape[0] == batch_size:
                return memory
            if memory.shape[0] == 1:
                return memory.expand(batch_size, -1, -1)
        raise ValueError(f"Unsupported memory shape: {tuple(memory.shape)}")

    def encode_inputs(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.acoustic_representation(speech_tokens, aux_features, audio_mask)

    def read_memory(
        self,
        h0: torch.Tensor,
        memory: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory_state = self._expand_memory(memory, h0.shape[0])
        encoded = self.memory_encoder(h0, memory_state, audio_mask)
        pooled_audio = masked_mean(encoded["audio_state"], audio_mask)
        query = F.normalize(self.query_projection(pooled_audio), dim=-1)
        projected_memory = F.normalize(self.memory_projection(memory_state), dim=-1)
        logits = safe_batched_dot(query, projected_memory) / max(self.temperature, self.eps)
        return {
            "audio_state": encoded["audio_state"],
            "memory_state": memory_state,
            "pooled_audio": pooled_audio,
            "query": query,
            "projected_memory": projected_memory,
            "logits": logits,
            "route_weights": encoded["route_weights"],
            "prompt_vectors": encoded["prompt_vectors"],
            "attention": encoded["attention"],
            "acoustic_scores": encoded["acoustic_scores"],
            "prompt_bias": encoded["prompt_bias"],
        }

    def predict(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor | None,
        memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_inputs(speech_tokens, aux_features, audio_mask)
        read = self.read_memory(features["h0"], memory, audio_mask)
        return {**features, **read}

    def build_episode_memory(
        self,
        support_speech: torch.Tensor,
        support_aux: torch.Tensor,
        support_mask: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
        label_blend: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        global_memory = self.global_memory
        if support_speech.shape[0] == 0:
            zeros = global_memory.new_zeros(self.num_classes, self.config.hidden_dim)
            return {
                "episode_memory": global_memory,
                "candidate_memory": global_memory,
                "write_gate": torch.ones_like(global_memory),
                "aggregate": zeros,
                "support_probs": global_memory.new_zeros(0, self.num_classes),
                "support_query": global_memory.new_zeros(0, self.config.score_dim),
                "support_h": global_memory.new_zeros(0, self.config.hidden_dim),
                "support_feature_weights": global_memory.new_zeros(0, len(self.acoustic_representation.branch_dims)),
                "support_write_weights": global_memory.new_zeros(0, self.num_classes),
            }

        features = self.encode_inputs(support_speech, support_aux, support_mask)
        read = self.read_memory(features["h0"], global_memory, support_mask)
        support_probs = torch.softmax(read["logits"], dim=-1)
        support_write_weights = support_probs
        label_blend = float(max(0.0, min(1.0, label_blend)))
        if support_labels is not None and support_labels.numel() > 0:
            hard_labels = F.one_hot(support_labels.long(), num_classes=self.num_classes).to(dtype=support_probs.dtype)
            support_write_weights = label_blend * hard_labels + (1.0 - label_blend) * support_probs
        write_content = self.write_projection(masked_mean(read["audio_state"], support_mask))
        aggregate_num = safe_mm(support_write_weights.transpose(0, 1), write_content)
        aggregate_den_raw = support_write_weights.sum(dim=0, keepdim=True).transpose(0, 1)
        aggregate_den = aggregate_den_raw.clamp_min(self.eps)
        aggregate = aggregate_num / aggregate_den
        update_input = torch.cat([global_memory, aggregate], dim=-1)
        candidate_memory = self.update_mlp(update_input)
        write_gate = torch.sigmoid(self.gate_mlp(update_input))
        episode_memory = write_gate * global_memory + (1.0 - write_gate) * candidate_memory
        if support_labels is not None and support_labels.numel() > 0 and label_blend >= 1.0:
            has_label_support = (aggregate_den_raw > 0).to(dtype=episode_memory.dtype)
            episode_memory = has_label_support * episode_memory + (1.0 - has_label_support) * global_memory
        return {
            "episode_memory": episode_memory,
            "candidate_memory": candidate_memory,
            "write_gate": write_gate,
            "aggregate": aggregate,
            "support_probs": support_probs,
            "support_write_weights": support_write_weights,
            "support_query": read["query"],
            "support_h": read["pooled_audio"],
            "support_feature_weights": features["feature_weights"],
            "support_route_weights": read["route_weights"],
        }

    def forward_episode(
        self,
        support_speech: torch.Tensor,
        support_aux: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor | None,
        query_speech: torch.Tensor,
        query_aux: torch.Tensor,
        query_mask: torch.Tensor,
        label_blend: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        episode_build = self.build_episode_memory(
            support_speech,
            support_aux,
            support_mask,
            support_labels=support_labels,
            label_blend=label_blend,
        )
        global_read = self.predict(query_speech, query_aux, query_mask, self.global_memory)
        episode_read = self.predict(query_speech, query_aux, query_mask, episode_build["episode_memory"])
        return {
            "global_logits": global_read["logits"],
            "episode_logits": episode_read["logits"],
            "global_memory": self.global_memory,
            "episode_attention": episode_read["attention"],
            "episode_acoustic_scores": episode_read["acoustic_scores"],
            "episode_prompt_bias": episode_read["prompt_bias"],
            "global_route_weights": global_read["route_weights"],
            "episode_route_weights": episode_read["route_weights"],
            "query_feature_weights": episode_read["feature_weights"],
            "query_feature_gate": episode_read["feature_gate"],
            "query_prompt_vectors": episode_read["prompt_vectors"],
            "query_audio_state": episode_read["audio_state"],
            **episode_build,
        }


def build_paper_model(
    input_dim: int,
    aux_dim: int,
    aux_branch_dims: Sequence[int] | None = None,
    hidden_dim: int = 256,
    score_dim: int = 128,
    prompt_dim: int = 128,
    num_prompts: int = 8,
    num_layers: int = 3,
    num_heads: int = 4,
    ffn_dim: int = 768,
    dropout: float = 0.1,
    max_audio_tokens: int = 32,
    temperature: float = 0.2,
) -> PCDLCMNet:
    cfg = PCDLCMNetConfig(
        input_dim=input_dim,
        aux_dim=aux_dim,
        aux_branch_dims=tuple(aux_branch_dims or ()),
        hidden_dim=hidden_dim,
        score_dim=score_dim,
        prompt_dim=prompt_dim,
        num_prompts=num_prompts,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_audio_tokens=max_audio_tokens,
        temperature=temperature,
    )
    return PCDLCMNet(cfg)


PaperDualMemoryTransformer = PCDLCMNet
