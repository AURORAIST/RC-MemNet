"""Regional Context Memory Network for cross-regional vowel recognition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_dlcmnet.models.encoder import MultiSourceAcousticRepresentation
from pc_dlcmnet.utils.tensor import masked_mean


@dataclass
class RCDMNetConfig:
    input_dim: int
    aux_dim: int
    aux_branch_dims: list[int]
    num_classes: int
    num_regions: int
    hidden_dim: int = 256
    score_dim: int = 128
    prompt_dim: int = 128
    num_slots: int = 4
    num_prompts: int = 8
    dropout: float = 0.1
    address_temperature: float = 0.2
    classifier_temperature: float = 0.2
    memory_mix: float = 0.5
    source_write_kappa: float = 4.0
    target_write_kappa: float = 1.0
    global_momentum: float = 0.1
    use_context_prompts: bool = True
    use_memory_read: bool = True
    use_prompt_routing: bool = True
    count_based_adaptation: bool = False
    slow_consolidation: bool = True
    write_top_k: int = 1
    source_composition_mode: str = "convex"


class RCDMNet(nn.Module):
    """RC-MemNet with RPL prompts and shared RGMA persistent memory.

    The public class name is kept for compatibility with 0712 scripts. The
    implementation follows the RC-MemNet paper text: source-region prompts are
    encoded together with acoustic tokens, all regions read one shared
    category-aligned memory, and memory values are updated only by explicit
    region-level writes.
    """

    def __init__(self, config: RCDMNetConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(config.num_classes)
        self.num_regions = int(config.num_regions)
        self.num_slots = int(config.num_slots)
        self.eps = 1e-8

        self.frontend = MultiSourceAcousticRepresentation(
            input_dim=int(config.input_dim),
            aux_dim=int(config.aux_dim),
            branch_dims=config.aux_branch_dims,
            hidden_dim=int(config.hidden_dim),
            dropout=float(config.dropout),
        )
        self.region_prompts = nn.Parameter(
            torch.randn(self.num_regions, int(config.num_prompts), int(config.hidden_dim)) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.hidden_dim),
            nhead=4,
            dim_feedforward=int(config.hidden_dim) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.prompt_token_router = nn.Sequential(
            nn.LayerNorm(int(config.hidden_dim) * 2),
            nn.Linear(int(config.hidden_dim) * 2, int(config.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim), 1),
        )
        self.prompt_context = nn.Sequential(
            nn.Linear(int(config.hidden_dim), int(config.prompt_dim)),
            nn.GELU(),
            nn.Linear(int(config.prompt_dim), int(config.score_dim)),
        )
        self.value_proj = nn.Linear(int(config.hidden_dim), int(config.hidden_dim), bias=False)
        self.h_to_query = nn.Linear(int(config.hidden_dim), int(config.score_dim))
        self.prompt_gamma = nn.Linear(int(config.score_dim), int(config.score_dim))
        self.prompt_beta = nn.Linear(int(config.score_dim), int(config.score_dim))
        self.query_norm = nn.LayerNorm(int(config.score_dim))

        self.memory_keys = nn.Parameter(
            torch.randn(self.num_classes, self.num_slots, int(config.score_dim)) * 0.02
        )
        self.global_values = nn.Parameter(
            torch.randn(self.num_classes, self.num_slots, int(config.hidden_dim)) * 0.02
        )
        self.read_fusion = nn.Linear(int(config.hidden_dim) * 2, int(config.hidden_dim))
        self.classifier = nn.Linear(int(config.hidden_dim), self.num_classes)

        self.register_buffer("memory_sums", torch.zeros(self.num_classes, self.num_slots, int(config.hidden_dim)))
        self.register_buffer("memory_strengths", torch.zeros(self.num_classes, self.num_slots))

    def _source_prompts(self, region_ids: torch.Tensor) -> torch.Tensor:
        return self.region_prompts[region_ids]

    def _fused_prompts(self, fusion_logits: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not bool(self.config.use_context_prompts):
            return torch.zeros(
                batch_size,
                self.config.num_prompts,
                self.config.hidden_dim,
                dtype=self.region_prompts.dtype,
                device=self.region_prompts.device,
            )
        mode = getattr(self.config, "source_composition_mode", "convex")
        if mode == "uniform":
            alpha = torch.ones_like(fusion_logits) / float(fusion_logits.shape[0])
        elif mode == "single":
            best_idx = fusion_logits.argmax()
            alpha = F.one_hot(best_idx, num_classes=fusion_logits.shape[0]).to(dtype=fusion_logits.dtype)
        elif mode == "unconstrained":
            alpha = fusion_logits
        else:  # convex
            alpha = torch.softmax(fusion_logits, dim=0)
        alpha = alpha.to(dtype=self.region_prompts.dtype, device=self.region_prompts.device)
        prompt = (alpha[:, None, None] * self.region_prompts).sum(dim=0)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def _encode_with_prompts(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        prompts: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        frontend = self.frontend(speech_tokens, aux_features, audio_mask)
        h0 = frontend["h0"]
        joint = torch.cat([prompts, h0], dim=1).contiguous()
        encoded = joint
        prompt_acoustic_interactions = []
        for layer in self.context_encoder.layers:
            encoded = layer(encoded)
            layer_prompt_tokens = encoded[:, : prompts.shape[1], :]
            layer_acoustic_tokens = encoded[:, prompts.shape[1] :, :]
            layer_prompt_repr = layer_prompt_tokens.mean(dim=1)
            layer_acoustic_repr = masked_mean(layer_acoustic_tokens, audio_mask)
            interaction = (F.cosine_similarity(layer_prompt_repr, layer_acoustic_repr, dim=-1) + 1.0) * 0.5
            prompt_acoustic_interactions.append(interaction.clamp(0.0, 1.0))
        if self.context_encoder.norm is not None:
            encoded = self.context_encoder.norm(encoded)
        prompt_tokens = encoded[:, : prompts.shape[1], :]
        acoustic_tokens = encoded[:, prompts.shape[1] :, :]
        acoustic_repr = masked_mean(acoustic_tokens, audio_mask)
        route_input = torch.cat(
            [prompt_tokens, acoustic_repr.unsqueeze(1).expand(-1, prompt_tokens.shape[1], -1)],
            dim=-1,
        )
        prompt_route_weights = torch.softmax(self.prompt_token_router(route_input).squeeze(-1), dim=-1)
        routed_prompt = (prompt_route_weights.unsqueeze(-1) * prompt_tokens).sum(dim=1)
        prompt_repr = self.prompt_context(routed_prompt)
        value_repr = F.normalize(self.value_proj(acoustic_repr), dim=-1)
        return {
            **frontend,
            "prompt_tokens": prompt_tokens,
            "prompt_route_weights": prompt_route_weights,
            "prompt_acoustic_interactions": torch.stack(prompt_acoustic_interactions, dim=1),
            "acoustic_tokens": acoustic_tokens,
            "h": acoustic_repr,
            "p": prompt_repr,
            "z": value_repr,
        }

    def encode(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prompts = self.region_prompts.mean(dim=0).unsqueeze(0).expand(speech_tokens.shape[0], -1, -1)
        return self._encode_with_prompts(speech_tokens, aux_features, prompts, audio_mask)

    def encode_source(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        region_ids: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._encode_with_prompts(speech_tokens, aux_features, self._source_prompts(region_ids), audio_mask)

    def encode_target(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        fusion_logits: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prompts = self._fused_prompts(fusion_logits, speech_tokens.shape[0])
        return self._encode_with_prompts(speech_tokens, aux_features, prompts, audio_mask)

    def address(self, h: torch.Tensor, prompt_context: torch.Tensor) -> torch.Tensor:
        q_base = self.h_to_query(h)
        if bool(self.config.use_prompt_routing):
            gamma = torch.tanh(self.prompt_gamma(prompt_context))
            beta = self.prompt_beta(prompt_context)
            query = self.query_norm((1.0 + gamma) * q_base + beta)
        else:
            query = self.query_norm(q_base)
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.memory_keys, dim=-1).reshape(self.num_classes * self.num_slots, -1)
        scores = torch.matmul(query, keys.transpose(0, 1)) / float(self.config.address_temperature)
        return torch.softmax(scores, dim=-1).view(-1, self.num_classes, self.num_slots)

    def _class_logits(self, h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if not bool(self.config.use_memory_read):
            return self.classifier(h) / float(self.config.classifier_temperature)
        values = F.normalize(self.global_values, dim=-1)
        memory_read = torch.einsum("bcj,cjh->bh", weights, values)
        fused = self.read_fusion(torch.cat([h, memory_read], dim=-1))
        return self.classifier(fused) / float(self.config.classifier_temperature)

    def forward_source(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        region_ids: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_source(speech_tokens, aux_features, region_ids, audio_mask)
        weights = self.address(encoded["h"], encoded["p"])
        logits = self._class_logits(encoded["h"], weights)
        return {**encoded, "prompt": encoded["p"], "address_weights": weights, "logits": logits, "global_logits": logits}

    def forward_target(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        fusion_logits: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        use_prompt: bool = True,
    ) -> dict[str, torch.Tensor]:
        if use_prompt:
            encoded = self.encode_target(speech_tokens, aux_features, fusion_logits, audio_mask)
        else:
            prompts = torch.zeros(
                speech_tokens.shape[0],
                self.config.num_prompts,
                self.config.hidden_dim,
                dtype=speech_tokens.dtype,
                device=speech_tokens.device,
            )
            encoded = self._encode_with_prompts(speech_tokens, aux_features, prompts, audio_mask)
        weights = self.address(encoded["h"], encoded["p"])
        logits = self._class_logits(encoded["h"], weights)
        return {**encoded, "prompt": encoded["p"], "address_weights": weights, "logits": logits, "global_logits": logits}

    def forward_global_with_fusion(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        fusion_logits: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_target(speech_tokens, aux_features, fusion_logits, audio_mask, use_prompt=True)

    @torch.no_grad()
    def initialize_memories(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        labels: torch.Tensor,
        region_ids: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        if not bool(self.config.use_memory_read):
            return
        self.eval()
        device = self.global_values.device
        encoded_z = []
        encoded_weights = []
        for start in range(0, speech_tokens.shape[0], batch_size):
            end = min(start + batch_size, speech_tokens.shape[0])
            batch_regions = region_ids[start:end].to(device)
            encoded = self.encode_source(
                speech_tokens[start:end].to(device),
                aux_features[start:end].to(device),
                batch_regions,
                None,
            )
            weights = self.address(encoded["h"], encoded["p"])
            encoded_z.append(encoded["z"].detach())
            encoded_weights.append(weights.detach())
        if not encoded_z:
            return
        z_all = torch.cat(encoded_z, dim=0)
        weights_all = torch.cat(encoded_weights, dim=0)
        labels = labels.to(device)
        values = self.global_values.clone()
        sums = torch.zeros_like(self.memory_sums)
        strengths = torch.zeros_like(self.memory_strengths)
        for c in range(self.num_classes):
            class_mask = labels == c
            if not bool(class_mask.any()):
                continue
            class_z = z_all[class_mask]
            slot_scores = weights_all[class_mask, c, :]
            slot_ids = slot_scores.argmax(dim=-1)
            for j in range(self.num_slots):
                slot_mask = slot_ids == j
                chosen = class_z[slot_mask] if bool(slot_mask.any()) else class_z
                center = F.normalize(chosen.mean(dim=0), dim=0)
                values[c, j] = center
                sums[c, j] = center
                strengths[c, j] = 1.0
        self.global_values.copy_(F.normalize(values, dim=-1))
        self.memory_sums.copy_(sums)
        self.memory_strengths.copy_(strengths)

    @torch.no_grad()
    def write_region_evidence(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        labels: torch.Tensor,
        region_ids: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        if not bool(self.config.use_memory_read):
            return
        self.eval()
        if speech_tokens.shape[0] == 0:
            return
        device = self.global_values.device
        g = torch.zeros_like(self.memory_sums)
        b = torch.zeros_like(self.memory_strengths)
        n = torch.bincount(labels.to(device), minlength=self.num_classes).to(dtype=self.global_values.dtype)
        top_k = max(1, min(int(self.config.write_top_k), self.num_slots))
        for start in range(0, speech_tokens.shape[0], batch_size):
            end = min(start + batch_size, speech_tokens.shape[0])
            batch_labels = labels[start:end].to(device)
            encoded = self.encode_source(
                speech_tokens[start:end].to(device),
                aux_features[start:end].to(device),
                region_ids[start:end].to(device),
                None,
            )
            class_weights = self.address(encoded["h"], encoded["p"])[
                torch.arange(batch_labels.shape[0], device=device), batch_labels.long(), :
            ]
            top_values, top_indices = torch.topk(class_weights, k=top_k, dim=-1)
            if bool(self.config.count_based_adaptation):
                top_values = torch.full_like(top_values, 1.0 / float(top_k))
            for row in range(batch_labels.shape[0]):
                c = int(batch_labels[row].item())
                z = encoded["z"][row]
                for value, slot in zip(top_values[row], top_indices[row]):
                    j = int(slot.item())
                    strength = value.to(dtype=g.dtype)
                    g[c, j] += strength * z
                    b[c, j] += strength
        active = b > 0
        for c in range(self.num_classes):
            for j in range(self.num_slots):
                if not bool(active[c, j]):
                    continue
                candidate = F.normalize(g[c, j] / (b[c, j] + self.eps), dim=0)
                omega = b[c, j] / (n[c].clamp_min(1.0) + self.eps)
                self.memory_sums[c, j] += omega * candidate
                self.memory_strengths[c, j] += omega
                self.global_values[c, j].copy_(
                    F.normalize(self.memory_sums[c, j] / (self.memory_strengths[c, j] + self.eps), dim=0)
                )

    @torch.no_grad()
    def write_source_batch(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        labels: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> None:
        self.write_region_evidence(speech_tokens, aux_features, labels, region_ids, batch_size=speech_tokens.shape[0])

    @torch.no_grad()
    def consolidate_global(self) -> None:
        self.global_values.copy_(F.normalize(self.global_values, dim=-1))

    @torch.no_grad()
    def build_target_memory(
        self,
        speech_tokens: torch.Tensor,
        aux_features: torch.Tensor,
        labels: torch.Tensor,
        fusion_logits: torch.Tensor,
        use_prompt: bool = True,
    ) -> torch.Tensor:
        if not bool(self.config.use_memory_read):
            return torch.zeros_like(self.global_values)
        return F.normalize(self.global_values.clone(), dim=-1)
