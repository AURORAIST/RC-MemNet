"""Configuration for PC-DLCMNet."""

from __future__ import annotations

from dataclasses import dataclass

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER


@dataclass(frozen=True)
class PCDLCMNetConfig:
    input_dim: int
    aux_dim: int
    aux_branch_dims: tuple[int, ...] = ()
    num_classes: int = len(VOWEL_ORDER)
    hidden_dim: int = 256
    score_dim: int = 128
    prompt_dim: int = 128
    num_prompts: int = 8
    num_layers: int = 3
    num_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    max_audio_tokens: int = 32
    temperature: float = 0.2
    eps: float = 1e-6


PaperModelConfig = PCDLCMNetConfig
