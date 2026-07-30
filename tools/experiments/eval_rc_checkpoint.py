#!/usr/bin/env python3
"""Evaluate an RC-MemNet checkpoint using the protocol saved in the checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.models.rc_dmnet import RCDMNet, RCDMNetConfig
from pc_dlcmnet.training.supervised import (
    build_split_audit,
    configure_stdout,
    dataframe_hash,
    default_old_root,
    label_counts,
    labels_to_ids,
    load_dataset,
    prediction_frame,
    resolve_device,
    set_seed,
    split_data,
    summarize_confusions,
)
from tools.experiments.run_rc_dmnet import (
    apply_ablation_overrides,
    evaluate_target_splits_safely,
    load_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--csv", default="", help="Override CSV path when the saved path moved.")
    parser.add_argument("--device", default="")
    parser.add_argument("--eval-split-runs", type=int, default=None)
    parser.add_argument("--target-support-shots", type=int, default=None)
    parser.add_argument("--target-adapt-steps", type=int, default=None)
    parser.add_argument("--target-adapt-lr", type=float, default=None)
    parser.add_argument("--whisper-model", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--matrix-cache-dir", default="")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint)
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    return state


def merged_runtime_config(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    old_root = default_old_root()
    config = dict(checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {})
    manifest = checkpoint.get("manifest", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(manifest, dict):
        for section in ("data", "feature_pipeline", "evaluation"):
            values = manifest.get(section, {})
            if isinstance(values, dict):
                for key, value in values.items():
                    config.setdefault(key, value)
        runtime = manifest.get("runtime", {})
        if isinstance(runtime, dict):
            config.setdefault("device", runtime.get("device", "cuda"))

    defaults = {
        "csv": "",
        "output": str(checkpoint_path.with_suffix(".eval.json")),
        "predictions_output": str(checkpoint_path.with_suffix(".eval.predictions.csv")),
        "path_column": "wav_path",
        "start_column": "start_time",
        "end_column": "end_time",
        "label_column": "vowel",
        "region_column": "region",
        "speaker_column": "speaker_id",
        "labels": None,
        "protocol": "loro",
        "holdout_region": "",
        "val_size": 0.1,
        "test_size": 0.1,
        "train_fraction": 1.0,
        "seed": 0,
        "cache_dir": str(old_root / "output/salmonn_style_whisper_cache_base"),
        "matrix_cache_dir": "output/0722/matrix_cache",
        "feature_norm": "global",
        "matrix_cache_scan": True,
        "whisper_model": "base",
        "device": "cuda",
        "target_sr": 16000,
        "token_chunks": 32,
        "audio_root": "",
        "aux_source": "acoustic",
        "aux_representation": "sequence",
        "aux_cache_dir": "output/0722/feature_memory_aux_cache",
        "old_feature_cache_dir": str(old_root / "output/feature_cache_vowel"),
        "aux_n_mfcc": 39,
        "aux_n_mels": 128,
        "aux_f0_segments": 5,
        "aux_use_delta_mfcc": True,
        "aux_use_formants": True,
        "batch_size": 64,
        "eval_batch_size": 256,
        "max_steps": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
        "class_weight_power": 0.5,
        "sampler_domain_power": 1.0,
        "sampler_class_power": 0.5,
        "hidden_dim": 256,
        "score_dim": 128,
        "prompt_dim": 128,
        "num_slots": 4,
        "num_prompts": 8,
        "dropout": 0.1,
        "address_temperature": 0.2,
        "classifier_temperature": 0.2,
        "memory_mix": 0.5,
        "source_write_kappa": 4.0,
        "target_write_kappa": 1.0,
        "global_momentum": 0.1,
        "ablation_variant": "rc_memnet",
        "write_top_k": 1,
        "target_support_shots": 4,
        "target_adapt_steps": 50,
        "target_adapt_lr": 0.05,
        "eval_split_runs": 100,
        "log_every": 50,
        "eval_every": 200,
        "require_all_source_regions": True,
        "quiet": False,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    config["output"] = str(checkpoint_path.with_suffix(".eval.json"))
    config["predictions_output"] = str(checkpoint_path.with_suffix(".eval.predictions.csv"))
    return config


def apply_cli_overrides(config: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    if cli.csv:
        config["csv"] = cli.csv
    if cli.device:
        config["device"] = cli.device
    if cli.output:
        config["output"] = cli.output
    if cli.predictions_output:
        config["predictions_output"] = cli.predictions_output
    if cli.eval_split_runs is not None:
        config["eval_split_runs"] = int(cli.eval_split_runs)
    if cli.target_support_shots is not None:
        config["target_support_shots"] = int(cli.target_support_shots)
    if cli.target_adapt_steps is not None:
        config["target_adapt_steps"] = int(cli.target_adapt_steps)
    if cli.target_adapt_lr is not None:
        config["target_adapt_lr"] = float(cli.target_adapt_lr)
    if cli.whisper_model:
        config["whisper_model"] = cli.whisper_model
    if cli.cache_dir:
        config["cache_dir"] = cli.cache_dir
    if cli.matrix_cache_dir:
        config["matrix_cache_dir"] = cli.matrix_cache_dir
    config["quiet"] = bool(cli.quiet)
    args = argparse.Namespace(**config)
    apply_ablation_overrides(args)
    return args


def resolve_runtime_paths(args: argparse.Namespace) -> None:
    existing_keys = ["csv", "audio_root", "whisper_model"]
    directory_keys = ["cache_dir", "matrix_cache_dir", "aux_cache_dir", "old_feature_cache_dir"]
    for key in existing_keys:
        value = str(getattr(args, key, "") or "")
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() or path.exists():
            setattr(args, key, str(path.resolve()) if path.exists() else value)
            continue
        package_path = PACKAGE_ROOT / path
        if package_path.exists():
            setattr(args, key, str(package_path.resolve()))
    for key in directory_keys:
        value = str(getattr(args, key, "") or "")
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        if path.exists():
            setattr(args, key, str(path.resolve()))
        else:
            setattr(args, key, str((PACKAGE_ROOT / path).resolve()))


def infer_branch_dims(state: dict[str, torch.Tensor]) -> list[int]:
    dims: list[int] = []
    index = 0
    while True:
        key = f"frontend.branch_projections.{index}.1.weight"
        if key not in state:
            break
        dims.append(int(state[key].shape[1]))
        index += 1
    return dims


def build_model_config(checkpoint: dict[str, Any], state: dict[str, torch.Tensor], args: argparse.Namespace) -> RCDMNetConfig:
    manifest = checkpoint.get("manifest", {}) if isinstance(checkpoint, dict) else {}
    model_manifest = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    saved_config = model_manifest.get("rcdmnet_config", {}) if isinstance(model_manifest, dict) else {}
    valid_keys = {field.name for field in fields(RCDMNetConfig)}
    if isinstance(saved_config, dict) and saved_config:
        cleaned = {key: value for key, value in saved_config.items() if key in valid_keys}
        return RCDMNetConfig(**cleaned)

    region_prompts = state["region_prompts"]
    memory_keys = state["memory_keys"]
    branch_dims = infer_branch_dims(state)
    return RCDMNetConfig(
        input_dim=int(state["frontend.speech_projection.1.weight"].shape[1]),
        aux_dim=int(sum(branch_dims)),
        aux_branch_dims=branch_dims,
        num_classes=int(memory_keys.shape[0]),
        num_regions=int(region_prompts.shape[0]),
        hidden_dim=int(region_prompts.shape[2]),
        score_dim=int(memory_keys.shape[2]),
        prompt_dim=int(state["prompt_context.1.weight"].shape[0]),
        num_slots=int(memory_keys.shape[1]),
        num_prompts=int(region_prompts.shape[1]),
        dropout=float(getattr(args, "dropout", 0.1)),
        address_temperature=float(getattr(args, "address_temperature", 0.2)),
        classifier_temperature=float(getattr(args, "classifier_temperature", 0.2)),
        memory_mix=float(getattr(args, "memory_mix", 0.5)),
        source_write_kappa=float(getattr(args, "source_write_kappa", 4.0)),
        target_write_kappa=float(getattr(args, "target_write_kappa", 1.0)),
        global_momentum=float(getattr(args, "global_momentum", 0.1)),
        use_context_prompts=str(getattr(args, "ablation_variant", "rc_memnet")) not in {"no_context_prompt", "no_prompt_no_memory"},
        use_memory_read=str(getattr(args, "ablation_variant", "rc_memnet")) not in {"no_memory_adapter", "no_prompt_no_memory"},
        use_prompt_routing=str(getattr(args, "ablation_variant", "rc_memnet")) not in {"no_context_prompt", "no_prompt_no_memory"},
        count_based_adaptation=False,
        slow_consolidation=str(getattr(args, "ablation_variant", "")) != "no_slow_consolidation",
        write_top_k=int(getattr(args, "write_top_k", 1)),
    )


def source_regions_from_manifest(checkpoint: dict[str, Any]) -> list[str]:
    manifest = checkpoint.get("manifest", {}) if isinstance(checkpoint, dict) else {}
    model_manifest = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    regions = model_manifest.get("source_regions", []) if isinstance(model_manifest, dict) else []
    return [str(item) for item in regions] if isinstance(regions, list) else []


def main() -> None:
    configure_stdout()
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")
    state = checkpoint_state(checkpoint)
    args = apply_cli_overrides(merged_runtime_config(checkpoint, checkpoint_path), cli)
    resolve_runtime_paths(args)
    if not args.csv:
        raise ValueError("checkpoint has no saved CSV path; pass --csv")

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output = Path(args.output)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output.with_suffix(".predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.quiet):
        print(
            json.dumps(
                {
                    "mode": "eval_rc_checkpoint",
                    "checkpoint": str(checkpoint_path),
                    "csv": str(args.csv),
                    "holdout_region": str(args.holdout_region),
                    "region_column": str(args.region_column),
                    "device": str(device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    split_audit = build_split_audit(df, train_df, val_df, test_df, args, split_info)
    train_x, val_x, test_x, train_aux, val_aux, test_aux, matrix_meta, aux_meta, _train_aux_meta = load_features(
        train_df,
        val_df,
        test_df,
        args,
        device,
    )
    del train_x, val_x, train_aux, val_aux

    model_config = build_model_config(checkpoint, state, args)
    model = RCDMNet(model_config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    test_y = labels_to_ids(test_df[args.label_column])
    eval_summary, test_pred, test_prob, test_global_prob, support_flag, query_flag, eval_device = evaluate_target_splits_safely(
        model,
        test_x,
        test_aux,
        test_y,
        args,
        device,
    )
    valid = test_pred >= 0
    metrics = eval_summary["mean"]
    metrics_std = eval_summary["std"]
    report = classification_report(
        test_y[valid],
        test_pred[valid],
        labels=list(range(len(VOWEL_ORDER))),
        target_names=VOWEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_y[valid], test_pred[valid], labels=list(range(len(VOWEL_ORDER))))
    pred_df = prediction_frame(test_df.loc[valid].reset_index(drop=True), test_y[valid], test_pred[valid], test_prob[valid], args)
    pred_df["global_pred_vowel"] = [VOWEL_ORDER[idx] for idx in test_global_prob[valid].argmax(axis=-1).astype(np.int64)]
    pred_df["global_pred_confidence"] = test_global_prob[valid].max(axis=1)
    pred_df["is_support_item"] = support_flag[valid].astype(np.int64)
    pred_df["is_query_item"] = query_flag[valid].astype(np.int64)
    pred_df.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    split_hash = {
        "train": dataframe_hash(train_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        "val": dataframe_hash(val_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        "test": dataframe_hash(test_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
    }
    saved_regions = source_regions_from_manifest(checkpoint)
    current_regions = sorted(train_df[args.region_column].astype(str).unique().tolist())
    result = {
        "task": "leave_one_region_out_vowel_classification",
        "method": "RC-MemNet",
        "ablation_variant": str(args.ablation_variant),
        "mode": "checkpoint_eval",
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "metrics_std": metrics_std,
        "evaluation_splits": {"test": eval_summary},
        "target_eval_device": str(eval_device),
        "report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_confusions": summarize_confusions(cm, VOWEL_ORDER, top_n=10),
        "split": split_info,
        "split_audit": split_audit,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "label_counts": {
            "train": label_counts(train_df[args.label_column], VOWEL_ORDER),
            "val": label_counts(val_df[args.label_column], VOWEL_ORDER),
            "test": label_counts(test_df[args.label_column], VOWEL_ORDER),
        },
        "source_regions": current_regions,
        "saved_source_regions": saved_regions,
        "source_regions_match": (not saved_regions) or saved_regions == current_regions,
        "cache": {"matrix": matrix_meta, "aux": aux_meta},
        "config": vars(args),
        "checkpoint_manifest": checkpoint.get("manifest", {}),
        "files": {
            "output": str(output),
            "predictions": str(predictions_output),
            "checkpoint": str(checkpoint_path),
        },
        "parameters": {
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total": int(sum(p.numel() for p in model.parameters())),
        },
        "split_hash": split_hash,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics, "metrics_std": metrics_std}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
