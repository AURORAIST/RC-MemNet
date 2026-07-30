"""Backward-compatible exports for the modular PC-DLCMNet implementation.

New code should import from the focused modules:
- pc_dlcmnet.models.encoder for the main acoustic encoder
- pc_dlcmnet.models.memory for prompt-conditioned class-memory reading
- pc_dlcmnet.models.network for the full PC-DLCMNet wrapper
"""

from pc_dlcmnet.models.config import PCDLCMNetConfig, PaperModelConfig
from pc_dlcmnet.models.encoder import MultiSourceAcousticRepresentation
from pc_dlcmnet.models.memory import (
    PromptConditionedMemoryEncoder,
    PromptConditionedMemoryLayer,
)
from pc_dlcmnet.models.network import (
    PCDLCMNet,
    PaperDualMemoryTransformer,
    build_paper_model,
)
from pc_dlcmnet.utils.tensor import masked_mean, normalize_branch_dims

__all__ = [
    "PCDLCMNetConfig",
    "PaperModelConfig",
    "MultiSourceAcousticRepresentation",
    "PromptConditionedMemoryLayer",
    "PromptConditionedMemoryEncoder",
    "PCDLCMNet",
    "PaperDualMemoryTransformer",
    "build_paper_model",
    "masked_mean",
    "normalize_branch_dims",
]
