"""Prompt-conditioned class-memory reading layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pc_dlcmnet.models.config import PaperModelConfig
from pc_dlcmnet.utils.tensor import masked_mean, safe_mm


class PromptConditionedMemoryLayer(nn.Module):
    """Audio queries read audio+memory keys/values with prompt-generated bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        prompt_dim: int,
        num_prompts: int,
        max_audio_tokens: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim // num_heads)
        self.max_audio_tokens = int(max_audio_tokens)
        self.num_classes = int(num_classes)
        self.prompt_library = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)
        self.prompt_router = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_prompts),
        )
        self.prompt_bias = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * max_audio_tokens * (max_audio_tokens + num_classes)),
        )
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def route_prompt(
        self,
        audio_state: torch.Tensor,
        memory_state: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_audio = masked_mean(audio_state, audio_mask)
        pooled_memory = memory_state.mean(dim=1)
        route_logits = self.prompt_router(torch.cat([pooled_audio, pooled_memory], dim=-1))
        route_weights = torch.softmax(route_logits, dim=-1)
        prompt = safe_mm(route_weights, self.prompt_library)
        return prompt, route_weights

    def build_prompt_bias(self, prompt: torch.Tensor, token_count: int) -> torch.Tensor:
        raw = self.prompt_bias(prompt)
        full = raw.view(
            prompt.shape[0],
            self.num_heads,
            self.max_audio_tokens,
            self.max_audio_tokens + self.num_classes,
        )
        return full[:, :, :token_count, : token_count + self.num_classes]

    def forward(
        self,
        audio_state: torch.Tensor,
        memory_state: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, token_count, _ = audio_state.shape
        prompt, route_weights = self.route_prompt(audio_state, memory_state, audio_mask)
        prompt_bias = self.build_prompt_bias(prompt, token_count)

        audio_norm = self.audio_norm(audio_state)
        memory_norm = self.memory_norm(memory_state)
        kv_input = torch.cat([audio_norm, memory_norm], dim=1)

        q = self._shape(self.q_proj(audio_norm))
        k = self._shape(self.k_proj(kv_input))
        v = self._shape(self.v_proj(kv_input))
        acoustic_scores = (q.unsqueeze(3) * k.unsqueeze(2)).sum(dim=-1)
        acoustic_scores = acoustic_scores / math.sqrt(float(self.head_dim))
        scores = acoustic_scores + prompt_bias
        if audio_mask is not None:
            key_mask = torch.cat(
                [
                    ~audio_mask.bool(),
                    torch.zeros(
                        batch_size,
                        memory_state.shape[1],
                        dtype=torch.bool,
                        device=audio_state.device,
                    ),
                ],
                dim=1,
            )
            scores = scores.masked_fill(
                key_mask.view(batch_size, 1, 1, token_count + memory_state.shape[1]),
                torch.finfo(scores.dtype).min,
            )
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v.unsqueeze(2)).sum(dim=3)
        out = out.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_dim)
        audio_state = audio_state + self.dropout(self.out_proj(out))
        if audio_mask is not None:
            audio_state = audio_state * audio_mask.to(dtype=audio_state.dtype).unsqueeze(-1)
        audio_state = audio_state + self.dropout(self.ffn(self.ffn_norm(audio_state)))
        if audio_mask is not None:
            audio_state = audio_state * audio_mask.to(dtype=audio_state.dtype).unsqueeze(-1)
        return {
            "audio_state": audio_state,
            "route_weights": route_weights,
            "prompt": prompt,
            "attention": attn,
            "acoustic_scores": acoustic_scores,
            "prompt_bias": prompt_bias,
        }


class PromptConditionedMemoryEncoder(nn.Module):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PromptConditionedMemoryLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    prompt_dim=config.prompt_dim,
                    num_prompts=config.num_prompts,
                    max_audio_tokens=config.max_audio_tokens,
                    num_classes=config.num_classes,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        audio_state: torch.Tensor,
        memory_state: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        routes = []
        attentions = []
        prompts = []
        acoustic_scores = []
        prompt_biases = []
        for layer in self.layers:
            out = layer(audio_state, memory_state, audio_mask)
            audio_state = out["audio_state"]
            routes.append(out["route_weights"])
            prompts.append(out["prompt"])
            attentions.append(out["attention"])
            acoustic_scores.append(out["acoustic_scores"])
            prompt_biases.append(out["prompt_bias"])
        return {
            "audio_state": audio_state,
            "route_weights": torch.stack(routes, dim=1),
            "prompt_vectors": torch.stack(prompts, dim=1),
            "attention": attentions[-1],
            "acoustic_scores": acoustic_scores[-1],
            "prompt_bias": prompt_biases[-1],
        }
