#!/usr/bin/env python3
"""Evaluate local HuggingFace speech encoders under 4-shot target splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel, HubertConfig, HubertModel, WhisperModel, Wav2Vec2FeatureExtractor

from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.training.supervised import labels_to_ids, load_dataset, resolve_device, split_data
from pc_dlcmnet.utils.paths import default_audio_root
from tools.experiments.baseline_comparisons import class_centroid_predict
from tools.experiments.ssl_pretrained_baselines import load_segment


TARGETS = {
    "Chizhou": ("region", "03池州"),
    "Qingyang": ("region", "04青阳"),
    "Tongling": ("region", "06铜陵"),
    "Jingxian": ("region", "08泾县"),
    "Nanling": ("region", "10南陵"),
    "Ningguo": ("site", "12宁国"),
    "Lishui": ("site", "14溧水"),
    "Huangshan": ("region", "11黄山"),
}

DEFAULT_MODELS = {
    "wav2vec2-base": "/home/ustc1958/lxy/graph/tone/model/wav2vec2-base",
    "wav2vec2-xls-r-300m-hfcache": "/home/ustc1958/lxy/graph/tone/.hf_cache/prepared_wav2vec2-xls-r-300m",
    "wav2vec2-large-robust-hfcache": "/home/ustc1958/lxy/graph/tone/.hf_cache/prepared_wav2vec2-large-robust",
    "allophant-hierarchical-hfcache": "/home/ustc1958/lxy/graph/tone/.hf_cache/models--kgnlp--allophant-hierarchical/snapshots/ee8993bea7a279241dbed810e4dfc8a90a0a59bf/allophant.pt",
    "hubert-base-ls960": "/home/ustc1958/lxy/graph/tone/model/hubert-base-ls960",
    "wavlm-base": "/home/ustc1958/lxy/graph/tone/model/wavlm-base",
    "whisper-base": "/home/ustc1958/lxy/graph/tone/model/whisper-base",
    "mHuBERT-147": "/home/ustc1958/lxy/graph/tone/model/mHuBERT-147",
    "MR-HuBERT": "/home/ustc1958/lxy/graph/tone/model/MR-HuBERT/mrhubert_mono_base.pt",
    "MS-HuBERT": "/home/ustc1958/lxy/graph/tone/model/MS-HuBERT/iter3.pt",
    "SALMONN-proxy": "/home/ustc1958/lxy/graph/tone/model/SALMONN/salmonn_v1.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(PACKAGE_ROOT / "data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv"))
    parser.add_argument("--output-dir", default="output/0614/local_hf_ssl_4shot_s100")
    parser.add_argument("--cache-dir", default="output/0614/local_hf_ssl_feature_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS.keys()))
    parser.add_argument("--targets", nargs="*", default=list(TARGETS.keys()), choices=list(TARGETS.keys()))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def dataset_args(args: argparse.Namespace, region_column: str, holdout: str) -> SimpleNamespace:
    return SimpleNamespace(
        csv=str(args.csv),
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        label_column="vowel",
        region_column=region_column,
        speaker_column="speaker_id",
        labels=None,
        protocol="loro",
        holdout_region=holdout,
        holdout_regions=None,
        val_size=0.1,
        test_size=0.1,
        train_fraction=1.0,
        seed=int(args.seed),
        target_sr=16000,
        audio_root=str(default_audio_root()),
        require_all_source_regions=True,
    )


def rows_hash(rows: pd.DataFrame) -> str:
    cols = [c for c in ["sample_id", "wav_path", "start_time", "end_time", "vowel"] if c in rows.columns]
    return hashlib.sha256(rows[cols].to_csv(index=False).encode("utf-8")).hexdigest()


def feature_cache_path(args: argparse.Namespace, model_name: str, area: str, rows: pd.DataFrame) -> Path:
    payload = {
        "model": model_name,
        "rows": rows_hash(rows),
        "pooling": "mean_std_last_hidden",
        "target_sr": 16000,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    safe = "".join(ch if ch.isalnum() else "_" for ch in model_name)
    return Path(args.cache_dir) / safe / f"{area}_{digest}.npy"


def install_fairseq_checkpoint_stubs() -> None:
    class Dictionary:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)

    for module_name in ["fairseq", "fairseq.data", "fairseq.data.dictionary"]:
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)
    sys.modules["fairseq.data.dictionary"].Dictionary = Dictionary


def fairseq_hubert_config(model_cfg: dict[str, Any]) -> HubertConfig:
    return HubertConfig(
        hidden_size=int(model_cfg["encoder_embed_dim"]),
        num_hidden_layers=int(model_cfg["encoder_layers"]),
        num_attention_heads=int(model_cfg["encoder_attention_heads"]),
        intermediate_size=int(model_cfg["encoder_ffn_embed_dim"]),
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="group",
        conv_bias=False,
        hidden_dropout=float(model_cfg.get("dropout", 0.1)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
        activation_dropout=float(model_cfg.get("activation_dropout", 0.0)),
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.0,
        do_stable_layer_norm=False,
    )


def map_fairseq_hubert_key(key: str, *, mr_hubert: bool) -> str | None:
    if key.startswith("feature_extractor.conv_layers."):
        return key.replace(".0.weight", ".conv.weight").replace(".2.weight", ".layer_norm.weight").replace(".2.bias", ".layer_norm.bias")
    if key == "post_extract_proj.weight":
        return "feature_projection.projection.weight"
    if key == "post_extract_proj.bias":
        return "feature_projection.projection.bias"
    if key == "layer_norm.weight":
        return "feature_projection.layer_norm.weight"
    if key == "layer_norm.bias":
        return "feature_projection.layer_norm.bias"
    if mr_hubert and key.startswith("encoders.0."):
        key = "encoder." + key[len("encoders.0.") :]
    if key == "encoder.pos_conv.0.bias":
        return "encoder.pos_conv_embed.conv.bias"
    if key == "encoder.pos_conv.0.weight_g":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    if key == "encoder.pos_conv.0.weight_v":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    if key.startswith("encoder.layer_norm."):
        return key
    if key.startswith("encoder.layers."):
        return (
            key.replace(".self_attn.", ".attention.")
            .replace(".self_attn_layer_norm.", ".layer_norm.")
            .replace(".fc1.", ".feed_forward.intermediate_dense.")
            .replace(".fc2.", ".feed_forward.output_dense.")
        )
    return None


def load_fairseq_hubert_model(model_name: str, path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, Path]:
    install_fairseq_checkpoint_stubs()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["model_cfg"]
    source_state = checkpoint.get("model_weight", checkpoint.get("model"))
    if not isinstance(source_state, dict):
        raise RuntimeError(f"{model_name} checkpoint has no model state")
    model = HubertModel(fairseq_hubert_config(model_cfg))
    target_state = model.state_dict()
    converted = {}
    mr_hubert = model_name == "MR-HuBERT"
    for key, value in source_state.items():
        mapped = map_fairseq_hubert_key(key, mr_hubert=mr_hubert)
        if mapped and mapped in target_state and tuple(target_state[mapped].shape) == tuple(value.shape):
            converted[mapped] = value
    load_status = model.load_state_dict(converted, strict=False)
    if load_status.missing_keys or load_status.unexpected_keys:
        raise RuntimeError(f"{model_name} conversion incomplete: missing={load_status.missing_keys[:5]} unexpected={load_status.unexpected_keys[:5]}")
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return extractor, model, path


class SalmonnProxyEncoder(nn.Module):
    """SALMONN audio-branch proxy using local Whisper-base and SALMONN Q-former weights."""

    def __init__(self, ckpt_path: Path, whisper_path: Path) -> None:
        super().__init__()
        salmonn_dir = ckpt_path.parent
        if str(salmonn_dir) not in sys.path:
            sys.path.insert(0, str(salmonn_dir))
        from qformer.Qformer import BertConfig, BertLMHeadModel

        self.config = SimpleNamespace(model_type="salmonn_proxy")
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path, local_files_only=True).encoder
        self.ln_speech = nn.LayerNorm(1280)
        self.ln_audio = nn.LayerNorm(768)

        q_cfg = BertConfig()
        q_cfg.num_hidden_layers = 2
        q_cfg.encoder_width = 2048
        q_cfg.add_cross_attention = True
        q_cfg.cross_attention_freq = 1
        q_cfg.query_length = 1
        self.speech_Qformer = BertLMHeadModel(config=q_cfg)
        self.speech_query_tokens = nn.Parameter(torch.zeros(1, 1, q_cfg.hidden_size))
        self.speech_llama_proj = nn.Linear(q_cfg.hidden_size, 5120)
        self.second_per_frame = 0.333333
        self.second_stride = 0.333333

        state = torch.load(ckpt_path, map_location="cpu")["model"]
        own_state = self.state_dict()
        filtered = {key: value for key, value in state.items() if key in own_state and tuple(own_state[key].shape) == tuple(value.shape)}
        status = self.load_state_dict(filtered, strict=False)
        critical_prefixes = (
            "speech_query_tokens",
            "ln_speech",
            "ln_audio",
            "speech_llama_proj",
            "speech_Qformer.bert.encoder.layer.0.crossattention",
            "speech_Qformer.bert.encoder.layer.1.crossattention",
            "speech_Qformer.bert.encoder.layer.0.intermediate_query",
            "speech_Qformer.bert.encoder.layer.1.intermediate_query",
            "speech_Qformer.bert.encoder.layer.0.output_query",
            "speech_Qformer.bert.encoder.layer.1.output_query",
        )
        critical_missing = [key for key in status.missing_keys if key.startswith(critical_prefixes)]
        if critical_missing:
            raise RuntimeError(f"SALMONN proxy missing weights: {critical_missing[:8]}")
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_features: torch.Tensor, **_: Any) -> Any:
        speech = self.speech_encoder(input_features=input_features, return_dict=True).last_hidden_state
        speech = F.pad(speech, (0, 1280 - speech.shape[-1]))
        speech = self.ln_speech(speech)
        audio = self.ln_audio(torch.zeros(speech.shape[0], speech.shape[1], 768, dtype=speech.dtype, device=speech.device))
        speech = torch.cat([speech, audio], dim=-1)

        batch, steps, width = speech.shape
        kernel = max(1, round(steps * self.second_per_frame / 30.0))
        stride = max(1, round(steps * self.second_stride / 30.0))
        overlap = F.unfold(speech.transpose(1, 2).unsqueeze(2), kernel_size=(1, kernel), stride=(1, stride))
        _, _, windows = overlap.shape
        overlap = overlap.view(batch, width, kernel, windows).permute(0, 3, 2, 1)
        speech = overlap.reshape(-1, kernel, width)
        speech_atts = torch.ones(speech.shape[:-1], dtype=torch.long, device=speech.device)
        query_tokens = self.speech_query_tokens.expand(speech.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        projected = self.speech_llama_proj(query_output.last_hidden_state)
        projected = projected.view(batch, -1, projected.shape[-1]).contiguous()
        return SimpleNamespace(last_hidden_state=projected)


def load_salmonn_proxy_model(model_name: str, path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, Path]:
    whisper_path = Path(DEFAULT_MODELS["whisper-base"])
    extractor = AutoFeatureExtractor.from_pretrained(whisper_path, local_files_only=True, trust_remote_code=True)
    model = SalmonnProxyEncoder(path, whisper_path)
    model.eval().to(device)
    return extractor, model, path


def load_allophant_acoustic_model(model_name: str, path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, Path]:
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    base_path = Path(DEFAULT_MODELS["wav2vec2-xls-r-300m-hfcache"])
    model = AutoModel.from_pretrained(base_path, local_files_only=True, trust_remote_code=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError(f"{model_name} checkpoint has no model_state")
    prefix = "_acoustic_model._model."
    target_state = model.state_dict()
    converted = {}
    for key, value in source_state.items():
        if not key.startswith(prefix):
            continue
        mapped = key[len(prefix) :]
        mapped = mapped.replace(
            "encoder.pos_conv_embed.conv.weight_g",
            "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        ).replace(
            "encoder.pos_conv_embed.conv.weight_v",
            "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        )
        if mapped in target_state and tuple(target_state[mapped].shape) == tuple(value.shape):
            converted[mapped] = value
    status = model.load_state_dict(converted, strict=False)
    critical_missing = [
        key
        for key in status.missing_keys
        if key.startswith(("feature_extractor.", "feature_projection.", "encoder."))
    ]
    if critical_missing:
        raise RuntimeError(f"{model_name} acoustic conversion incomplete: missing={critical_missing[:8]}")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return extractor, model, path


def load_model(model_name: str, device: torch.device) -> tuple[Any, torch.nn.Module, Path]:
    path = Path(DEFAULT_MODELS.get(model_name, model_name))
    if model_name in {"MR-HuBERT", "MS-HuBERT"}:
        return load_fairseq_hubert_model(model_name, path, device)
    if model_name == "SALMONN-proxy":
        return load_salmonn_proxy_model(model_name, path, device)
    if model_name == "allophant-hierarchical-hfcache":
        return load_allophant_acoustic_model(model_name, path, device)
    extractor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return extractor, model, path


def extract_features(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    model_name: str,
    area: str,
    extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = feature_cache_path(args, model_name, area, rows)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and not args.force:
        return np.load(cache_path).astype(np.float32), {"source": "cache", "path": str(cache_path)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.reset_index(drop=True)
    dargs = SimpleNamespace(
        path_column="wav_path",
        start_column="start_time",
        end_column="end_time",
        target_sr=16000,
        audio_root=str(default_audio_root()),
    )
    chunks: list[np.ndarray] = []
    iterator = range(0, len(frame), int(args.batch_size))
    if not args.quiet:
        iterator = tqdm(iterator, desc=f"{model_name} {area}", unit="batch")
    with torch.no_grad():
        for start in iterator:
            batch = frame.iloc[start : start + int(args.batch_size)]
            audio = [load_segment(row, dargs) for _, row in batch.iterrows()]
            if getattr(model.config, "model_type", "") in {"whisper", "salmonn_proxy"}:
                inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
            else:
                inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if getattr(model.config, "model_type", "") == "whisper" and "input_features" in inputs:
                out = model.encoder(input_features=inputs["input_features"])
                hidden = out.last_hidden_state
            else:
                out = model(**inputs)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = getattr(out, "encoder_last_hidden_state", None)
            if hidden is None:
                raise RuntimeError(f"{model_name} produced no hidden states")
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean = (hidden * mask_f).sum(dim=1) / denom
            var = (((hidden - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / denom
            pooled = torch.cat([mean, var.clamp_min(0).sqrt()], dim=-1)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
    x = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(cache_path, x)
    meta = {"model": model_name, "area": area, "rows": int(len(rows)), "shape": list(x.shape), "row_hash": rows_hash(rows)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return x, {"source": "computed", "path": str(cache_path), "meta": str(meta_path)}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_4shot(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    runs = []
    for split_idx in range(int(args.splits)):
        support_idx, query_idx = split_global_support_query(y, int(args.k), int(args.seed) + split_idx * 1009)
        pred = class_centroid_predict(x[support_idx], y[support_idx], x[query_idx])
        runs.append(metric_dict(y[query_idx], pred))
    keys = ["accuracy", "macro_f1", "micro_f1", "weighted_f1"]
    mean = {key: float(np.mean([row[key] for row in runs])) for key in keys}
    std = {key: float(np.std([row[key] for row in runs], ddof=1)) if len(runs) > 1 else 0.0 for key in keys}
    return mean, std, runs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = load_dataset(
        SimpleNamespace(
            csv=str(args.csv),
            path_column="wav_path",
            start_column="start_time",
            end_column="end_time",
            label_column="vowel",
            region_column="region",
            labels=None,
        )
    )
    device = resolve_device(str(args.device))
    rows = []
    for model_name in args.models:
        extractor, model, model_path = load_model(model_name, device)
        for area in args.targets:
            region_column, holdout = TARGETS[area]
            result_path = output_dir / "results" / model_name / f"{area}.json"
            if result_path.exists() and not args.force:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                dargs = dataset_args(args, region_column, holdout)
                _train_df, _val_df, test_df, split_info = split_data(full_df, dargs)
                y = labels_to_ids(test_df["vowel"])
                started = time.perf_counter()
                x, cache_meta = extract_features(test_df, args, model_name, area, extractor, model, device)
                metrics, metrics_std, runs = evaluate_4shot(x, y, args)
                data = {
                    "model": model_name,
                    "model_path": str(model_path),
                    "area": area,
                    "holdout": holdout,
                    "region_column": region_column,
                    "config": {"k": int(args.k), "splits": int(args.splits), "seed": int(args.seed), "protocol": "target_global_support"},
                    "metrics": metrics,
                    "metrics_std": metrics_std,
                    "runs": runs,
                    "feature_cache": cache_meta,
                    "split": split_info,
                    "test_rows": int(len(y)),
                    "time_sec": float(time.perf_counter() - started),
                }
                write_json(result_path, data)
            row = {
                "model": model_name,
                "area": area,
                "macro_f1": data["metrics"]["macro_f1"],
                "macro_f1_std": data["metrics_std"]["macro_f1"],
                "accuracy": data["metrics"]["accuracy"],
                "accuracy_std": data["metrics_std"]["accuracy"],
                "weighted_f1": data["metrics"]["weighted_f1"],
                "weighted_f1_std": data["metrics_std"]["weighted_f1"],
                "test_rows": data["test_rows"],
                "time_sec": data["time_sec"],
            }
            rows.append(row)
            print(
                f"{model_name} {area}: mf1={row['macro_f1']*100:.2f}±{row['macro_f1_std']*100:.2f} "
                f"acc={row['accuracy']*100:.2f}±{row['accuracy_std']*100:.2f}",
                flush=True,
            )
        del model
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for model_name in args.models:
        group = [row for row in rows if row["model"] == model_name]
        aggregate.append(
            {
                "model": model_name,
                "regions": len(group),
                "macro_f1": float(np.mean([row["macro_f1"] for row in group])),
                "accuracy": float(np.mean([row["accuracy"] for row in group])),
                "weighted_f1": float(np.mean([row["weighted_f1"] for row in group])),
            }
        )
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)


if __name__ == "__main__":
    main()
