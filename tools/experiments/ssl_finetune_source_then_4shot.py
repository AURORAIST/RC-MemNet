#!/usr/bin/env python3
"""Fine-tune a pretrained speech encoder on source regions and evaluate 4-shot target performance."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from torch.utils.data import DataLoader, Dataset

from pc_dlcmnet.data.episodes import split_global_support_query
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.training.supervised import (
    build_split_audit,
    configure_stdout,
    default_old_root,
    labels_to_ids,
    load_dataset,
    resolve_device,
    set_seed,
    split_data,
)
from pc_dlcmnet.utils.paths import default_audio_root
from tools.experiments.local_hf_ssl_4shot_eval import DEFAULT_MODELS, load_model, load_segment


TARGETS = {
    "Dangtu": ("region", "01当涂"),
    "Wuhu": ("region", "02芜湖"),
    "Chizhou": ("region", "03池州"),
    "Qingyang": ("region", "04青阳"),
    "Suncun": ("region", "05孙村"),
    "Tongling": ("region", "06铜陵"),
    "Xuancheng": ("region", "07宣城"),
    "Jingxian": ("region", "08泾县"),
    "Fanchang": ("region", "09繁昌"),
    "Nanling": ("region", "10南陵"),
    "Huangshan": ("region", "11黄山"),
    "Ningguo": ("site", "12宁国"),
    "Gaochun": ("region", "13高淳"),
    "Lishui": ("site", "14溧水"),
}


MODEL_ALIASES = {
    "wav2vec2-base": "wav2vec2-base",
    "HuBERT Base": "hubert-base-ls960",
    "hubert-base": "hubert-base-ls960",
    "WavLM Base+": "wavlm-base",
    "wavlm-base-plus": "wavlm-base",
    "Whisper Base": "whisper-base",
    "whisper-base": "whisper-base",
    "Robust wav2vec 2.0 Large": "wav2vec2-large-robust-hfcache",
    "wav2vec2-large-robust": "wav2vec2-large-robust-hfcache",
    "XLS-R-300M": "wav2vec2-xls-r-300m-hfcache",
    "xls-r-300m": "wav2vec2-xls-r-300m-hfcache",
    "mHuBERT-147": "mHuBERT-147",
    "MR-HuBERT Multi-Base": "MR-HuBERT",
    "MR-HuBERT": "MR-HuBERT",
    "MS-HuBERT": "MS-HuBERT",
    "Allophant-Hierarchical": "allophant-hierarchical-hfcache",
    "allophant-hierarchical": "allophant-hierarchical-hfcache",
}


class SegmentDataset(Dataset[Any]):
    def __init__(self, frame: pd.DataFrame, args: argparse.Namespace) -> None:
        self.frame = frame.reset_index(drop=True)
        self.args = args
        self.label_to_id = {label: idx for idx, label in enumerate(VOWEL_ORDER)}
        self.dargs = SimpleNamespace(
            path_column="wav_path",
            start_column="start_time",
            end_column="end_time",
            target_sr=int(args.target_sr),
            audio_root=str(default_audio_root()),
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        audio = load_segment(row, self.dargs)
        return {"audio": audio, "label": int(self.label_to_id[str(row[self.args.label_column])])}


def collate_segments(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio": [np.asarray(item["audio"], dtype=np.float32) for item in items],
        "label": torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--output-dir", default="output/0722/ssl_finetune_source_then_4shot")
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--protocol", choices=["loro"], default="loro")
    parser.add_argument("--holdout-regions", nargs="*", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--models", nargs="*", default=["wav2vec2-base", "hubert-base-ls960", "wavlm-base", "whisper-base"])
    parser.add_argument("--targets", nargs="*", default=list(TARGETS.keys()), choices=list(TARGETS.keys()))
    parser.add_argument("--train-scheme", choices=["linear_probe", "last_layer", "full"], default="last_layer")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-feature-extractor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", default="output/0722/ssl_finetune_cache")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def get_hidden_states(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type == "whisper" or "input_features" in inputs:
        output = model.encoder(input_features=inputs["input_features"], return_dict=True)
        hidden = output.last_hidden_state
    else:
        output = model(**inputs, return_dict=True)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(output, "encoder_last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("encoder produced no hidden states")
    return hidden


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None or tuple(attention_mask.shape[:2]) != tuple(hidden.shape[:2]):
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def build_inputs(extractor: Any, audio: list[np.ndarray], model_type: str, device: torch.device) -> dict[str, torch.Tensor]:
    if model_type == "whisper":
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
    else:
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in inputs.items()}


def set_trainable(model: nn.Module, classifier: nn.Module, scheme: str, freeze_feature_extractor: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = True

    if scheme == "linear_probe":
        return

    if freeze_feature_extractor and hasattr(model, "feature_extractor"):
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
    elif hasattr(model, "feature_extractor"):
        for param in model.feature_extractor.parameters():
            param.requires_grad = True

    if hasattr(model, "feature_projection"):
        for param in model.feature_projection.parameters():
            param.requires_grad = True

    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        for param in model.parameters():
            if not freeze_feature_extractor:
                param.requires_grad = True
        return

    if scheme == "full":
        for param in model.parameters():
            param.requires_grad = True
        if freeze_feature_extractor and hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        return

    last_layers = 1
    for layer in encoder.layers[-last_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    for name in ["layer_norm", "final_layer_norm", "projector", "masked_spec_embed", "pos_conv_embed", "feature_projection"]:
        module = getattr(encoder, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def make_optimizer(model: nn.Module, classifier: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [param for param in list(model.parameters()) + list(classifier.parameters()) if param.requires_grad]
    return torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))


@torch.no_grad()
def extract_embeddings(
    rows: pd.DataFrame,
    extractor: Any,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    dataset = SegmentDataset(rows, args)
    loader = DataLoader(dataset, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=0, collate_fn=collate_segments)
    model.eval()
    chunks: list[np.ndarray] = []
    model_type = str(getattr(model.config, "model_type", ""))
    for batch in loader:
        audio = [np.asarray(item, dtype=np.float32) for item in batch["audio"]]
        inputs = build_inputs(extractor, audio, model_type, device)
        hidden = get_hidden_states(model, inputs)
        pooled = pool_hidden(hidden, inputs.get("attention_mask"))
        chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros((0, int(getattr(model.config, "hidden_size", 0))), dtype=np.float32)


def evaluate_support_proto(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    run_metrics = []
    for split_idx in range(int(args.splits)):
        support_idx, query_idx = split_global_support_query(y, int(args.k), int(args.seed) + split_idx * 1009)
        support_x = torch.tensor(x[support_idx], dtype=torch.float32)
        query_x = torch.tensor(x[query_idx], dtype=torch.float32)
        support_y = torch.tensor(y[support_idx], dtype=torch.long)
        prototypes = []
        for c in range(len(VOWEL_ORDER)):
            mask = support_y == c
            if bool(mask.any()):
                prototypes.append(F.normalize(support_x[mask].mean(dim=0), dim=0))
            else:
                prototypes.append(F.normalize(support_x.mean(dim=0), dim=0))
        proto = torch.stack(prototypes, dim=0)
        logits = torch.matmul(F.normalize(query_x, dim=-1), F.normalize(proto, dim=-1).transpose(0, 1))
        pred = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        run_metrics.append(metric_dict(y[query_idx], pred))
    keys = ["accuracy", "macro_f1", "micro_f1", "weighted_f1"]
    return {
        "runs": int(len(run_metrics)),
        "mean": {key: float(np.mean([row[key] for row in run_metrics])) for key in keys},
        "std": {key: float(np.std([row[key] for row in run_metrics], ddof=1)) if len(run_metrics) > 1 else 0.0 for key in keys},
        "metrics": run_metrics,
    }


def train_one_target(
    model_name: str,
    extractor: Any,
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    num_labels = len(VOWEL_ORDER)
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "d_model", 0)))
    if not hidden_size and hasattr(model.config, "encoder_embed_dim"):
        hidden_size = int(model.config.encoder_embed_dim)
    classifier = nn.Linear(hidden_size, num_labels).to(device)
    set_trainable(model, classifier, str(args.train_scheme), bool(args.freeze_feature_extractor))
    optimizer = make_optimizer(model, classifier, args)
    class_weights = torch.tensor(np.bincount(labels_to_ids(train_df[args.label_column]), minlength=num_labels), dtype=torch.float32, device=device)
    class_weights = (class_weights.clamp_min(1.0).sum() / class_weights.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)

    train_ds = SegmentDataset(train_df, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_segments,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state: dict[str, Any] | None = None
    best_macro = -1.0
    history: list[dict[str, float]] = []
    started = time.time()
    step = 0
    model_type = str(getattr(model.config, "model_type", ""))

    while step < int(args.max_steps):
        for batch in train_loader:
            audio = [np.asarray(item, dtype=np.float32) for item in batch["audio"]]
            labels = batch["label"].to(device)
            inputs = build_inputs(extractor, audio, model_type, device)
            hidden = get_hidden_states(model, inputs)
            pooled = pool_hidden(hidden, inputs.get("attention_mask"))
            logits = classifier(pooled)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad], float(args.grad_clip))
            optimizer.step()
            step += 1
            if step % max(1, int(args.eval_every)) == 0 or step >= int(args.max_steps):
                model.eval()
                classifier.eval()
                val_x = extract_embeddings(val_df, extractor, model, args, device)
                val_labels = labels_to_ids(val_df[args.label_column])
                with torch.no_grad():
                    val_logits = classifier(torch.tensor(val_x, dtype=torch.float32, device=device))
                val_pred = val_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                metrics = metric_dict(val_labels, val_pred)
                row = {"step": float(step), "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in metrics.items()}}
                history.append(row)
                if metrics["macro_f1"] > best_macro:
                    best_macro = metrics["macro_f1"]
                    best_state = {
                        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        "classifier_state_dict": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
                        "best_step": int(step),
                        "val_metrics": metrics,
                    }
                model.train()
                classifier.train()
                if not bool(args.quiet):
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            if step >= int(args.max_steps):
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        classifier.load_state_dict(best_state["classifier_state_dict"])
    train_time = float(time.time() - started)
    return {
        "steps": int(step),
        "best_val_macro_f1": float(best_macro) if best_macro >= 0 else None,
        "training_time_sec": train_time,
        "history": history,
        "backbone": model_name,
        "train_scheme": str(args.train_scheme),
        "hidden_size": int(hidden_size),
        "classifier": classifier,
        "model": model,
    }


def evaluate_model_4shot(
    model: nn.Module,
    classifier: nn.Module,
    extractor: Any,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    test_x = extract_embeddings(test_df, extractor, model, args, device)
    test_y = labels_to_ids(test_df[args.label_column])
    encoded = torch.tensor(test_x, dtype=torch.float32)
    classifier = classifier.to(device).eval()
    with torch.no_grad():
        logits = classifier(encoded.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
    support_result = evaluate_support_proto(test_x, test_y, args)
    return {
        "target_4shot": support_result,
        "test_embeddings_shape": list(test_x.shape),
        "zero_shot": metric_dict(test_y, probs.argmax(axis=1).astype(np.int64)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_stdout()
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    full_df = load_dataset(
        SimpleNamespace(
            csv=str(args.csv),
            path_column=args.path_column,
            start_column=args.start_column,
            end_column=args.end_column,
            label_column=args.label_column,
            region_column=args.region_column,
            speaker_column=args.speaker_column,
            labels=args.labels,
            protocol=args.protocol,
            holdout_regions=args.holdout_regions,
            val_size=args.val_size,
            test_size=args.test_size,
            train_fraction=args.train_fraction,
            seed=args.seed,
            target_sr=args.target_sr,
            audio_root=str(default_audio_root()),
            require_all_source_regions=True,
        )
    )
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for requested_model in args.models:
        model_name = normalize_model_name(requested_model)
        if model_name not in DEFAULT_MODELS and model_name not in {"MR-HuBERT", "MS-HuBERT", "allophant-hierarchical-hfcache", "wav2vec2-large-robust-hfcache", "wav2vec2-xls-r-300m-hfcache"}:
            raise ValueError(f"unknown model: {requested_model}")
        for target in args.targets:
            region_column, holdout = TARGETS[target]
            dargs = SimpleNamespace(
                csv=str(args.csv),
                path_column=args.path_column,
                start_column=args.start_column,
                end_column=args.end_column,
                label_column=args.label_column,
                region_column=region_column,
                speaker_column=args.speaker_column,
                labels=args.labels,
                protocol=args.protocol,
                holdout_region=holdout,
                holdout_regions=None,
                val_size=args.val_size,
                test_size=args.test_size,
                train_fraction=args.train_fraction,
                seed=args.seed,
                target_sr=args.target_sr,
                audio_root=str(default_audio_root()),
                require_all_source_regions=True,
            )
            train_df, val_df, test_df, split_info = split_data(full_df, dargs)
            if int(args.max_train_rows) > 0:
                train_df = train_df.head(int(args.max_train_rows)).reset_index(drop=True)
            if int(args.max_val_rows) > 0:
                val_df = val_df.head(int(args.max_val_rows)).reset_index(drop=True)
            if int(args.max_test_rows) > 0:
                test_df = test_df.head(int(args.max_test_rows)).reset_index(drop=True)
            result_dir = out_root / "results" / model_name / target
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / "result.json"
            ckpt_path = result_dir / "result.pt"
            if result_path.exists() and not bool(args.force):
                data = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                extractor, backbone, resolved_path = load_model(model_name, device)
                started = time.perf_counter()
                fit = train_one_target(model_name, extractor, backbone, train_df, val_df, test_df, args, device)
                eval_payload = evaluate_model_4shot(fit["model"], fit["classifier"], extractor, test_df, args, device)
                data = {
                    "model": requested_model,
                    "resolved_model": model_name,
                    "model_path": str(resolved_path),
                    "target": target,
                    "protocol": str(args.protocol),
                    "k": int(args.k),
                    "splits": int(args.splits),
                    "region_column": region_column,
                    "holdout": holdout,
                    "split": split_info,
                    "train_scheme": str(args.train_scheme),
                    "metrics": {
                        "accuracy": float(eval_payload["target_4shot"]["mean"]["accuracy"]),
                        "macro_f1": float(eval_payload["target_4shot"]["mean"]["macro_f1"]),
                        "weighted_f1": float(eval_payload["target_4shot"]["mean"]["weighted_f1"]),
                        "micro_f1": float(eval_payload["target_4shot"]["mean"]["micro_f1"]),
                    },
                    "metrics_std": {
                        "accuracy": float(eval_payload["target_4shot"]["std"]["accuracy"]),
                        "macro_f1": float(eval_payload["target_4shot"]["std"]["macro_f1"]),
                        "weighted_f1": float(eval_payload["target_4shot"]["std"]["weighted_f1"]),
                        "micro_f1": float(eval_payload["target_4shot"]["std"]["micro_f1"]),
                    },
                    "zero_shot": eval_payload["zero_shot"],
                    "test_embeddings_shape": eval_payload["test_embeddings_shape"],
                    "training": {
                        k: v for k, v in fit.items() if k not in {"classifier", "model", "history"}
                    },
                    "history": fit["history"],
                    "test_rows": int(len(test_df)),
                    "time_sec": float(time.perf_counter() - started),
                }
                write_json(result_path, data)
                if bool(args.save_checkpoint):
                    torch.save(
                        {
                            "model_state_dict": {key: value.detach().cpu() for key, value in fit["model"].state_dict().items()},
                            "classifier_state_dict": {key: value.detach().cpu() for key, value in fit["classifier"].state_dict().items()},
                            "model_name": requested_model,
                            "resolved_model": model_name,
                            "train_scheme": str(args.train_scheme),
                            "config": vars(args),
                            "split": split_info,
                        },
                        ckpt_path,
                    )
            rows.append(
                {
                    "model": requested_model,
                    "resolved_model": model_name,
                    "target": target,
                    "protocol": str(args.protocol),
                    "k": int(args.k),
                    "splits": int(args.splits),
                    "train_scheme": str(args.train_scheme),
                    "accuracy": data["metrics"]["accuracy"],
                    "accuracy_std": data["metrics_std"]["accuracy"],
                    "macro_f1": data["metrics"]["macro_f1"],
                    "macro_f1_std": data["metrics_std"]["macro_f1"],
                    "weighted_f1": data["metrics"]["weighted_f1"],
                    "weighted_f1_std": data["metrics_std"]["weighted_f1"],
                    "zero_shot_macro_f1": data["zero_shot"]["macro_f1"],
                    "zero_shot_accuracy": data["zero_shot"]["accuracy"],
                    "time_sec": data["time_sec"],
                }
            )
            print(
                f"{requested_model} {target}: acc={rows[-1]['accuracy']*100:.2f}±{rows[-1]['accuracy_std']*100:.2f} "
                f"macro={rows[-1]['macro_f1']*100:.2f}±{rows[-1]['macro_f1_std']*100:.2f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")

    if rows:
        aggregate = (
            summary.groupby("model", as_index=False)[["accuracy", "macro_f1", "weighted_f1", "zero_shot_macro_f1", "zero_shot_accuracy"]]
            .mean()
            .sort_values("macro_f1", ascending=False)
        )
        aggregate.to_csv(out_root / "aggregate.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
