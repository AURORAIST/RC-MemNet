#!/usr/bin/env python3
"""Auxiliary acoustic feature extraction and caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from pc_dlcmnet.utils.paths import default_audio_root, default_project_root


@dataclass(frozen=True)
class AuxFeatureConfig:
    sample_rate: int = 16000
    n_mfcc: int = 39
    n_mels: int = 128
    f0_segments: int = 5
    use_delta_mfcc: bool = True
    use_formants: bool = True
    f0_min: float = 75.0
    f0_max: float = 600.0
    n_fft_mfcc: int = 2048
    n_fft_mel: int = 2048


class AcousticFeatureExtractor:
    """Fixed-dimensional MFCC/Mel/F0/prosody/formant extractor."""

    def __init__(self, config: AuxFeatureConfig) -> None:
        self.config = config

    @property
    def output_dim(self) -> int:
        cfg = self.config
        dim = cfg.n_mfcc * 2
        if cfg.use_delta_mfcc:
            dim += cfg.n_mfcc * 2
        dim += cfg.n_mels * 2
        dim += 4
        dim += cfg.f0_segments
        dim += 5
        if cfg.use_formants:
            dim += 6
        return dim

    @property
    def sequence_dim(self) -> int:
        cfg = self.config
        return cfg.n_mels + cfg.n_mfcc + cfg.n_mfcc * 2

    @property
    def sequence_branch_dims(self) -> tuple[int, int, int]:
        cfg = self.config
        return (cfg.n_mels, cfg.n_mfcc, cfg.n_mfcc * 2)

    @staticmethod
    def _normalize_audio(audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None or len(audio) < 32:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_abs > 1e-8:
            audio = audio / max_abs
        return audio.astype(np.float32)

    @staticmethod
    def _align_feature_time(feature: np.ndarray, target_frames: int) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 2:
            raise ValueError(f"Expected [channels, frames], got {feature.shape}")
        channels, frames = feature.shape
        if frames == target_frames:
            return feature.T.astype(np.float32)
        if frames <= 1:
            return np.repeat(feature.T, repeats=target_frames, axis=0).astype(np.float32)
        old_x = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, int(target_frames), dtype=np.float32)
        aligned = np.stack([np.interp(new_x, old_x, feature[ch]) for ch in range(channels)], axis=1)
        return aligned.astype(np.float32)

    def extract_sequence(self, audio: np.ndarray | None, target_frames: int) -> np.ndarray:
        """Return token-aligned [T, mel+mfcc+delta+delta2] features."""
        cfg = self.config
        audio = self._normalize_audio(audio)
        if audio is None:
            return np.zeros((int(target_frames), self.sequence_dim), dtype=np.float32)
        target_frames = int(target_frames)
        try:
            n_fft_mel = min(cfg.n_fft_mel, len(audio))
            if n_fft_mel < 64:
                n_fft_mel = len(audio)
            hop_mel = max(1, n_fft_mel // 4)
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=cfg.sample_rate,
                n_mels=cfg.n_mels,
                n_fft=n_fft_mel,
                hop_length=hop_mel,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_seq = self._align_feature_time(mel_db, target_frames)
        except Exception:
            mel_seq = np.zeros((target_frames, cfg.n_mels), dtype=np.float32)

        try:
            n_fft = min(cfg.n_fft_mfcc, len(audio))
            if n_fft < 64:
                n_fft = len(audio)
            hop = max(1, n_fft // 4)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                n_fft=n_fft,
                hop_length=hop,
            )
            mfcc_seq = self._align_feature_time(mfcc, target_frames)
            if mfcc.shape[1] > 1:
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                delta_seq = self._align_feature_time(np.concatenate([delta1, delta2], axis=0), target_frames)
            else:
                delta_seq = np.zeros((target_frames, cfg.n_mfcc * 2), dtype=np.float32)
        except Exception:
            mfcc_seq = np.zeros((target_frames, cfg.n_mfcc), dtype=np.float32)
            delta_seq = np.zeros((target_frames, cfg.n_mfcc * 2), dtype=np.float32)

        out = np.concatenate([mel_seq, mfcc_seq, delta_seq], axis=-1).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def extract(self, audio: np.ndarray | None) -> np.ndarray:
        cfg = self.config
        audio = self._normalize_audio(audio)
        if audio is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        parts: list[np.ndarray] = []
        mfcc = None
        try:
            n_fft = min(cfg.n_fft_mfcc, len(audio))
            if n_fft < 64:
                n_fft = len(audio)
            hop = max(1, n_fft // 4)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                n_fft=n_fft,
                hop_length=hop,
            )
            parts.append(np.mean(mfcc, axis=1).astype(np.float32))
            parts.append(np.std(mfcc, axis=1).astype(np.float32))
        except Exception:
            parts.append(np.zeros(cfg.n_mfcc, dtype=np.float32))
            parts.append(np.zeros(cfg.n_mfcc, dtype=np.float32))

        if cfg.use_delta_mfcc:
            try:
                if mfcc is None or mfcc.shape[1] < 2:
                    raise ValueError("not enough frames for delta MFCC")
                delta = librosa.feature.delta(mfcc)
                parts.append(np.mean(delta, axis=1).astype(np.float32))
                parts.append(np.std(delta, axis=1).astype(np.float32))
            except Exception:
                parts.append(np.zeros(cfg.n_mfcc, dtype=np.float32))
                parts.append(np.zeros(cfg.n_mfcc, dtype=np.float32))

        try:
            n_fft = min(cfg.n_fft_mel, len(audio))
            if n_fft < 64:
                n_fft = len(audio)
            hop = max(1, n_fft // 4)
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=cfg.sample_rate,
                n_mels=cfg.n_mels,
                n_fft=n_fft,
                hop_length=hop,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            parts.append(np.mean(mel_db, axis=1).astype(np.float32))
            parts.append(np.std(mel_db, axis=1).astype(np.float32))
        except Exception:
            parts.append(np.zeros(cfg.n_mels, dtype=np.float32))
            parts.append(np.zeros(cfg.n_mels, dtype=np.float32))

        parts.append(self._f0_stats(audio))
        parts.append(self._f0_contour(audio))
        parts.append(self._prosody(audio))
        if cfg.use_formants:
            parts.append(self._formants(audio))

        out = np.concatenate(parts).astype(np.float32)
        if len(out) != self.output_dim:
            fixed = np.zeros(self.output_dim, dtype=np.float32)
            fixed[: min(len(out), len(fixed))] = out[: min(len(out), len(fixed))]
            return fixed
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _f0_stats(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        try:
            f0 = librosa.yin(audio, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=cfg.sample_rate)
            valid = f0[np.isfinite(f0) & (f0 > 0)]
            if len(valid) == 0:
                return np.zeros(4, dtype=np.float32)
            return np.array([valid.mean(), valid.std(), valid.max(), valid.min()], dtype=np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

    def _f0_contour(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        try:
            f0 = librosa.yin(audio, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=cfg.sample_rate)
            f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)
            if len(f0) == 0:
                return np.zeros(cfg.f0_segments, dtype=np.float32)
            segments = np.array_split(f0, cfg.f0_segments)
            return np.array([float(seg.mean()) if len(seg) else 0.0 for seg in segments], dtype=np.float32)
        except Exception:
            return np.zeros(cfg.f0_segments, dtype=np.float32)

    def _prosody(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        try:
            n_fft = min(2048, len(audio))
            if n_fft < 64:
                n_fft = len(audio)
            hop = max(1, n_fft // 4)
            rms = float(np.sqrt(np.mean(np.square(audio))))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=cfg.sample_rate, n_fft=n_fft, hop_length=hop)))
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=cfg.sample_rate, n_fft=n_fft, hop_length=hop)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=cfg.sample_rate, n_fft=n_fft, hop_length=hop)))
            return np.array([rms, zcr, centroid, bandwidth, rolloff], dtype=np.float32)
        except Exception:
            return np.zeros(5, dtype=np.float32)

    def _formants(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        try:
            order = int(2 + cfg.sample_rate / 1000)
            frame_len = min(int(0.025 * cfg.sample_rate), len(audio))
            hop = max(1, frame_len // 2)
            if frame_len < 32 or len(audio) < frame_len:
                return np.zeros(6, dtype=np.float32)
            tracks: list[list[float]] = [[], [], []]
            for start in range(0, len(audio) - frame_len + 1, hop):
                frame = audio[start : start + frame_len] * np.hamming(frame_len)
                try:
                    coeff = librosa.lpc(frame, order=order)
                    roots = np.roots(coeff)
                    roots = roots[np.imag(roots) > 0]
                    if len(roots) == 0:
                        continue
                    freqs = np.sort((np.angle(roots) * cfg.sample_rate / (2 * np.pi))[np.angle(roots) > 0])
                    freqs = freqs[freqs > 50]
                    for idx in range(min(3, len(freqs))):
                        tracks[idx].append(float(freqs[idx]))
                except Exception:
                    continue
            result: list[float] = []
            for track in tracks:
                if track:
                    result.extend([float(np.mean(track)), float(np.std(track))])
                else:
                    result.extend([0.0, 0.0])
            return np.array(result, dtype=np.float32)
        except Exception:
            return np.zeros(6, dtype=np.float32)


def whisper_token_statistics(tokens: np.ndarray) -> np.ndarray:
    """Fallback auxiliary feature when raw wav files are unavailable."""
    x = np.asarray(tokens, dtype=np.float32)
    if x.ndim != 2 or x.size == 0:
        return np.zeros(1, dtype=np.float32)
    delta = np.diff(x, axis=0)
    if delta.size == 0:
        delta = np.zeros_like(x[:1])
    temporal_energy = np.sqrt(np.mean(np.square(x), axis=1))
    parts = [
        x.mean(axis=0),
        x.std(axis=0),
        x.min(axis=0),
        x.max(axis=0),
        delta.mean(axis=0),
        delta.std(axis=0),
        np.array(
            [
                temporal_energy.mean(),
                temporal_energy.std(),
                temporal_energy.min(),
                temporal_energy.max(),
                float(x.shape[0]),
                float(x.shape[1]),
            ],
            dtype=np.float32,
        ),
    ]
    out = np.concatenate([np.asarray(part, dtype=np.float32).ravel() for part in parts]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def dataframe_hash(df: pd.DataFrame, columns: list[str]) -> str:
    present = [col for col in columns if col in df.columns]
    payload = df[present].sort_values(present).to_csv(index=False) if present else str(len(df))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aux_rows_hash(df: pd.DataFrame, args: Any) -> str:
    return dataframe_hash(df, ["sample_id", args.path_column, args.start_column, args.end_column])


def default_old_root() -> Path:
    return default_project_root()


def default_audio_roots(args: Any) -> list[Path]:
    roots: list[Path] = []
    for value in (
        getattr(args, "audio_root", ""),
        os.getenv("TONE_DATA_ROOT", ""),
        os.getenv("TONE0614_AUDIO_ROOT", ""),
        str(default_audio_root()),
        str(default_old_root()),
        str(Path.cwd()),
    ):
        if value:
            roots.append(Path(value))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def audio_path_candidates(path_value: object, args: Any) -> list[str]:
    raw = str(path_value)
    candidates = [raw]
    linux_prefixes = [
        "/home/ustc1958/lxy/graph/tone/tone/",
        "/home/ustc1958/lxy/graph/tone/tone",
    ]
    rel_parts: tuple[str, ...] | None = None
    for prefix in linux_prefixes:
        if raw.startswith(prefix):
            rel_parts = PurePosixPath(raw[len(prefix) :].lstrip("/")).parts
            break
    if rel_parts is not None:
        for root in default_audio_roots(args):
            candidates.append(str(root.joinpath(*rel_parts)))
            root_name = root.name
            for idx, part in enumerate(rel_parts):
                if part == root_name:
                    candidates.append(str(root.joinpath(*rel_parts[idx + 1 :])))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def resolve_audio_path(row: pd.Series, args: Any) -> str | None:
    for candidate in audio_path_candidates(row[args.path_column], args):
        if Path(candidate).exists():
            return candidate
    return None


def load_audio_segment(row: pd.Series, args: Any, audio_path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim == 2:
            audio = audio[:, 0]
        start = max(0, int(round(float(row[args.start_column]) * sr)))
        end = max(start + 1, int(round(float(row[args.end_column]) * sr)))
        audio = audio[start:end].astype(np.float32)
        target_sr = int(getattr(args, "target_sr", 16000))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception:
        return None


def feature_cache_key(row: pd.Series, args: Any, source: str, config: AuxFeatureConfig) -> str:
    payload = {
        "path": str(row[args.path_column]),
        "start": f"{float(row[args.start_column]):.6f}",
        "end": f"{float(row[args.end_column]):.6f}",
        "source": source,
        "config": config.__dict__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def old_fusion_cache_key(row: pd.Series, args: Any, feature_type: str = "whisper_fusion") -> str:
    payload = f"{row[args.path_column]}_{row[args.start_column]}_{row[args.end_column]}_{args.whisper_model}_{feature_type}"
    return hashlib.md5(payload.encode("utf-8", errors="ignore")).hexdigest()


def load_old_fusion_feature(row: pd.Series, args: Any) -> np.ndarray | None:
    cache_dir = Path(getattr(args, "old_feature_cache_dir", ""))
    if not str(cache_dir):
        return None
    for feature_type in ("whisper_fusion", "whisper+mfcc+mel", "whisper_combined"):
        path = cache_dir / f"{old_fusion_cache_key(row, args, feature_type)}.pkl"
        if path.exists():
            try:
                with path.open("rb") as f:
                    value = pickle.load(f)
                return np.asarray(value, dtype=np.float32).ravel()
            except Exception:
                continue
    return None


def choose_aux_source(rows: pd.DataFrame, args: Any) -> str:
    requested = str(getattr(args, "aux_source", "auto"))
    if requested != "auto":
        return requested
    sample = rows.head(min(50, len(rows)))
    found = 0
    for _, row in sample.iterrows():
        if resolve_audio_path(row, args) is not None:
            found += 1
    if len(sample) > 0 and found / max(1, len(sample)) >= 0.9:
        return "acoustic"
    return "whisper_stats"


def aux_representation(args: Any, source: str) -> str:
    requested = str(getattr(args, "aux_representation", "auto"))
    if requested != "auto":
        return requested
    return "sequence" if source == "acoustic" else "stats"


def aux_matrix_cache_key(
    rows: pd.DataFrame,
    args: Any,
    source: str,
    representation: str,
    config: AuxFeatureConfig,
    token_shape: tuple[int, ...] | None,
) -> str:
    payload = {
        "rows": aux_rows_hash(rows, args),
        "source": source,
        "representation": representation,
        "config": config.__dict__,
        "token_shape": list(token_shape) if token_shape is not None else None,
        "old_feature_cache_dir": str(getattr(args, "old_feature_cache_dir", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def extract_aux_matrix_cached(
    rows: pd.DataFrame,
    args: Any,
    token_matrix: np.ndarray | None,
    split_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = choose_aux_source(rows, args)
    representation = aux_representation(args, source)
    cfg = AuxFeatureConfig(
        sample_rate=int(getattr(args, "target_sr", 16000)),
        n_mfcc=int(getattr(args, "aux_n_mfcc", 39)),
        n_mels=int(getattr(args, "aux_n_mels", 128)),
        f0_segments=int(getattr(args, "aux_f0_segments", 5)),
        use_delta_mfcc=bool(getattr(args, "aux_use_delta_mfcc", True)),
        use_formants=bool(getattr(args, "aux_use_formants", True)),
    )
    cache_dir = Path(getattr(args, "aux_cache_dir", "output/0614/aux_feature_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_shape = tuple(token_matrix.shape[1:]) if token_matrix is not None else None
    key = aux_matrix_cache_key(rows, args, source, representation, cfg, token_shape)
    matrix_path = cache_dir / f"{split_name}_{key}.npy"
    meta_path = cache_dir / f"{split_name}_{key}.json"
    if matrix_path.exists() and meta_path.exists():
        matrix = np.load(matrix_path).astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["path"] = str(matrix_path)
        meta["meta_path"] = str(meta_path)
        print(f"[aux-cache] loaded {split_name}: {matrix_path}", flush=True)
        return matrix, meta

    extractor = AcousticFeatureExtractor(cfg)
    features: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    per_row_dir = cache_dir / "rows"
    per_row_dir.mkdir(parents=True, exist_ok=True)
    if source == "whisper_stats" and token_matrix is None:
        raise ValueError("aux_source=whisper_stats requires token_matrix")
    if token_matrix is not None and len(token_matrix) != len(rows):
        raise ValueError(f"token_matrix rows {len(token_matrix)} != dataframe rows {len(rows)}")

    for row_idx, (_, row) in enumerate(tqdm(rows.iterrows(), total=len(rows), desc=f"aux-{split_name}")):
        row_source = source
        cache_file = per_row_dir / f"{feature_cache_key(row, args, f'{source}:{representation}', cfg)}.npy"
        if cache_file.exists():
            vec = np.load(cache_file).astype(np.float32)
            features.append(vec)
            source_counts[f"{row_source}:row_cache"] = source_counts.get(f"{row_source}:row_cache", 0) + 1
            continue

        if source == "old_fusion_cache":
            vec = load_old_fusion_feature(row, args)
            if vec is None:
                raise FileNotFoundError(f"old fusion cache miss for row {row_idx}")
        elif source == "acoustic":
            path = resolve_audio_path(row, args)
            audio = load_audio_segment(row, args, path) if path else None
            if audio is None:
                if representation == "sequence":
                    target_frames = int(token_matrix.shape[1]) if token_matrix is not None else int(getattr(args, "token_chunks", 32))
                    vec = np.zeros((target_frames, extractor.sequence_dim), dtype=np.float32)
                else:
                    vec = np.zeros(extractor.output_dim, dtype=np.float32)
                row_source = "acoustic_missing_zero"
            else:
                if representation == "sequence":
                    target_frames = int(token_matrix.shape[1]) if token_matrix is not None else int(getattr(args, "token_chunks", 32))
                    vec = extractor.extract_sequence(audio, target_frames=target_frames)
                else:
                    vec = extractor.extract(audio)
        elif source == "whisper_stats":
            vec = whisper_token_statistics(token_matrix[row_idx])
        else:
            raise ValueError(f"Unsupported aux source: {source}")
        vec = np.asarray(vec, dtype=np.float32)
        np.save(cache_file, vec)
        features.append(vec)
        source_counts[row_source] = source_counts.get(row_source, 0) + 1

    if features:
        shapes = {tuple(vec.shape) for vec in features}
        if len(shapes) == 1:
            matrix = np.stack(features, axis=0).astype(np.float32)
        else:
            flat = [vec.reshape(-1).astype(np.float32) for vec in features]
            dim = max(len(vec) for vec in flat)
            matrix = np.zeros((len(flat), dim), dtype=np.float32)
            for idx, vec in enumerate(flat):
                matrix[idx, : len(vec)] = vec[:dim]
            representation = f"{representation}_flattened_mixed_shapes"
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    meta = {
        "split": split_name,
        "rows": int(len(rows)),
        "source": source,
        "representation": representation,
        "source_counts": source_counts,
        "shape": list(matrix.shape),
        "branch_dims": list(extractor.sequence_branch_dims) if source == "acoustic" and representation == "sequence" else [],
        "row_hash": aux_rows_hash(rows, args),
        "config": cfg.__dict__,
        "token_shape": list(token_shape) if token_shape is not None else None,
        "path": str(matrix_path),
        "meta_path": str(meta_path),
    }
    np.save(matrix_path, matrix.astype(np.float32))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[aux-cache] saved {split_name}: {matrix_path}", flush=True)
    return matrix, meta


def fit_vector_norm(x: np.ndarray, mode: str) -> dict[str, np.ndarray | str]:
    if mode == "none":
        return {"mode": "none"}
    if mode != "global":
        raise ValueError(f"Unsupported vector norm mode: {mode}")
    axes = (0, 1) if x.ndim == 3 else 0
    mean = x.mean(axis=axes, keepdims=True).astype(np.float32)
    std = x.std(axis=axes, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return {"mode": "global", "mean": mean, "std": std}


def apply_vector_norm(x: np.ndarray, norm: dict[str, np.ndarray | str]) -> np.ndarray:
    if norm["mode"] == "none":
        return x.astype(np.float32)
    return ((x - norm["mean"]) / norm["std"]).astype(np.float32)


def serializable_norm(norm: dict[str, np.ndarray | str]) -> dict[str, Any]:
    out: dict[str, Any] = {"mode": str(norm["mode"])}
    if "mean" in norm:
        out["mean_shape"] = list(np.asarray(norm["mean"]).shape)
        out["std_shape"] = list(np.asarray(norm["std"]).shape)
    return out
