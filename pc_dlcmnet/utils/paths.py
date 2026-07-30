"""Shared path defaults for the standalone 0614 experiment bundle.

Copying ``complete_package0614`` to another machine should not require a
neighboring ``complete_package`` directory. Environment variables can override
the defaults without editing code on the server.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PACKAGE_ROOT / "output"


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


DATASET_CSV = env_path("TONE0614_DATASET", OUTPUT_ROOT / "datasets" / "wu_low_resource_vowel_dataset.csv")
WHISPER_CACHE_DIR = env_path("TONE0614_WHISPER_CACHE", OUTPUT_ROOT / "salmonn_style_whisper_cache_base")
MATRIX_CACHE_DIR = env_path("TONE0614_MATRIX_CACHE", OUTPUT_ROOT / "ppm_supervised_matrix_cache")
FEATURE_CACHE_DIR = env_path("TONE0614_FEATURE_CACHE", OUTPUT_ROOT / "feature_cache_vowel")
AUDIO_ROOT = env_path("TONE0614_AUDIO_ROOT", PACKAGE_ROOT / "audio")
HF_MODELS_ROOT = env_path("TONE0614_HF_MODELS", OUTPUT_ROOT / "hf_models")


def default_project_root() -> Path:
    return PACKAGE_ROOT


def default_dataset() -> Path:
    return DATASET_CSV


def default_whisper_cache() -> Path:
    return WHISPER_CACHE_DIR


def default_matrix_cache() -> Path:
    return MATRIX_CACHE_DIR


def default_feature_cache() -> Path:
    return FEATURE_CACHE_DIR


def default_audio_root() -> Path:
    return AUDIO_ROOT


def package_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path)

