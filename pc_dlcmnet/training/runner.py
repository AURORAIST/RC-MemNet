"""Command-line runner for the modular PC-DLCMNet experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from pc_dlcmnet.data.acoustic_features import (
    apply_vector_norm,
    aux_rows_hash,
    extract_aux_matrix_cached,
    fit_vector_norm,
    serializable_norm,
)
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER
from pc_dlcmnet.evaluation.downstream import evaluate_split_runs
from pc_dlcmnet.evaluation.mechanism import collect_mechanism_diagnostics
from pc_dlcmnet.data.episodes import build_episode_groups, resolve_episode_column
from pc_dlcmnet.models.network import build_paper_model
from pc_dlcmnet.training.trainer import train_model, write_curve
from pc_dlcmnet.training.supervised import (
    apply_token_norm,
    build_split_audit,
    configure_stdout,
    dataframe_hash,
    default_old_root,
    extract_matrix_cached,
    fit_token_norm,
    infer_aux_branch_dims,
    label_counts,
    labels_to_ids,
    load_dataset,
    load_model_init_checkpoint,
    prediction_frame,
    resolve_device,
    serializable_token_norm,
    set_seed,
    split_data,
    summarize_confusions,
)


def parse_args() -> argparse.Namespace:
    old_root = default_old_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(old_root / "output/datasets/wu_low_resource_vowel_dataset.csv"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--curve-output", default="")
    parser.add_argument("--debug-output", default="")
    parser.add_argument("--audit-output", default="")
    parser.add_argument("--checkpoint-output", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--path-column", default="wav_path")
    parser.add_argument("--start-column", default="start_time")
    parser.add_argument("--end-column", default="end_time")
    parser.add_argument("--label-column", default="vowel")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--protocol", choices=["loro", "random"], default="loro")
    parser.add_argument("--holdout-region", default="")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(old_root / "output/salmonn_style_whisper_cache_base"))
    parser.add_argument("--matrix-cache-dir", default=str(old_root / "output/ppm_supervised_matrix_cache"))
    parser.add_argument("--feature-norm", choices=["none", "global"], default="global")
    parser.add_argument("--matrix-cache-scan", action="store_true", default=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--token-chunks", type=int, default=32)
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--aux-source", choices=["auto", "acoustic", "whisper_stats", "old_fusion_cache"], default="acoustic")
    parser.add_argument("--aux-representation", choices=["auto", "sequence", "stats"], default="sequence")
    parser.add_argument("--aux-cache-dir", default="output/0614/feature_memory_aux_cache")
    parser.add_argument("--old-feature-cache-dir", default=str(old_root / "output/feature_cache_vowel"))
    parser.add_argument("--aux-n-mfcc", type=int, default=39)
    parser.add_argument("--aux-n-mels", type=int, default=128)
    parser.add_argument("--aux-f0-segments", type=int, default=5)
    parser.add_argument("--aux-use-delta-mfcc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-use-formants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episode-column", default="")
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--episode-batch-size", type=int, default=4)
    parser.add_argument("--support-shots", type=int, default=1)
    parser.add_argument("--eval-support-shots", type=int, default=0)
    parser.add_argument("--support-write-mode", choices=["label", "pseudo", "blend"], default="label")
    parser.add_argument("--support-label-blend", type=float, default=1.0)
    parser.add_argument("--eval-query-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-split-mode", choices=["rolling", "fixed_query", "global_support"], default="global_support")
    parser.add_argument("--eval-query-shots-per-class", type=int, default=1)
    parser.add_argument("--eval-classifier", choices=["memory", "prototype", "gema_cosine", "global_memory"], default="memory")
    parser.add_argument("--eval-subgroup-column", default="")
    parser.add_argument("--eval-subgroups", nargs="*", default=None)
    parser.add_argument("--prototype-fallback-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prototype-support-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--lambda-global", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--score-dim", type=int, default=128)
    parser.add_argument("--prompt-dim", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=500)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=800)
    parser.add_argument("--eval-episode-limit", type=int, default=0)
    parser.add_argument("--eval-ensemble-runs", type=int, default=1)
    parser.add_argument("--eval-split-runs", type=int, default=1)
    parser.add_argument("--mechanism-output-dir", default="")
    parser.add_argument("--mechanism-k-values", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--mechanism-max-episodes", type=int, default=0)
    parser.add_argument("--skip-val-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-all-source-regions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def skipped_val_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    val_metrics: dict[str, Any] = {}
    val_global_metrics: dict[str, Any] = {}
    val_diag: dict[str, Any] = {
        "skipped": True,
        "reason": "skip_val_eval",
        "eval_split_runs": 0,
    }
    val_split_summary: dict[str, Any] = {
        "skipped": True,
        "runs": 0,
        "metrics": [],
        "global_metrics": [],
        "diagnostics": [],
        "metrics_std": {},
        "global_metrics_std": {},
        "diagnostics_std": {},
    }
    return val_metrics, val_global_metrics, val_diag, val_split_summary


def build_eval_subgroup_masks(test_df, args: argparse.Namespace) -> dict[str, np.ndarray]:
    column = str(getattr(args, "eval_subgroup_column", "") or "")
    specs = getattr(args, "eval_subgroups", None) or []
    if not column or not specs:
        return {}
    if column not in test_df.columns:
        raise ValueError(f"--eval-subgroup-column {column!r} is not in the test dataframe")
    values = test_df[column].astype(str)
    masks: dict[str, np.ndarray] = {}
    for spec in specs:
        text = str(spec)
        if "=" in text:
            name, prefix = text.split("=", 1)
        else:
            name, prefix = text, text
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_")
        if not safe_name:
            safe_name = f"group{len(masks) + 1}"
        masks[safe_name] = values.str.startswith(prefix, na=False).to_numpy(dtype=bool)
    return masks


def main() -> None:
    configure_stdout()
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    if not bool(getattr(args, "quiet", False)):
        print(
            json.dumps(
                {
                    "device": str(device),
                    "requested_device": args.device,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()),
                    "torch_version": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output.with_suffix(".predictions.csv")
    curve_output = Path(args.curve_output) if args.curve_output else output.with_suffix(".curve.csv")
    debug_output = Path(args.debug_output) if args.debug_output else output.with_suffix(".debug.json")
    audit_output = Path(args.audit_output) if args.audit_output else output.with_suffix(".audit.json")
    checkpoint_output = Path(args.checkpoint_output) if args.checkpoint_output else output.with_suffix(".pt")

    df = load_dataset(args)
    train_df, val_df, test_df, split_info = split_data(df, args)
    split_audit = build_split_audit(df, train_df, val_df, test_df, args, split_info)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(getattr(args, "quiet", False)):
        print(
            json.dumps(
                {
                    "split": split_info,
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df),
                    "episode_column": resolve_episode_column(train_df, args),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = None
    matrix_meta: dict[str, Any] = {}
    try:
        train_x_raw, matrix_meta["train"] = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, matrix_meta["val"] = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, matrix_meta["test"] = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")
    except (AttributeError, FileNotFoundError) as exc:
        if not bool(getattr(args, "quiet", False)):
            print(f"[matrix-cache] miss needs Whisper extraction: {exc}", flush=True)
        try:
            import whisper
        except ImportError as whisper_exc:
            raise RuntimeError(
                "Whisper token cache missed and on-the-fly extraction is required, "
                "but importing whisper failed. Reuse an existing matrix cache or "
                "install compatible whisper dependencies before rerunning."
            ) from whisper_exc

        whisper_model = whisper.load_model(args.whisper_model, device=str(device))
        whisper_model.eval()
        for param in whisper_model.parameters():
            param.requires_grad = False
        train_x_raw, matrix_meta["train"] = extract_matrix_cached(train_df, args, whisper_model, device, cache_dir, "train")
        val_x_raw, matrix_meta["val"] = extract_matrix_cached(val_df, args, whisper_model, device, cache_dir, "val")
        test_x_raw, matrix_meta["test"] = extract_matrix_cached(test_df, args, whisper_model, device, cache_dir, "test")

    token_norm = fit_token_norm(train_x_raw, args.feature_norm)
    train_x = apply_token_norm(train_x_raw, token_norm)
    val_x = apply_token_norm(val_x_raw, token_norm)
    test_x = apply_token_norm(test_x_raw, token_norm)

    train_aux_raw, train_aux_meta = extract_aux_matrix_cached(train_df, args, train_x_raw, "train")
    val_aux_raw, val_aux_meta = extract_aux_matrix_cached(val_df, args, val_x_raw, "val")
    test_aux_raw, test_aux_meta = extract_aux_matrix_cached(test_df, args, test_x_raw, "test")
    aux_norm = fit_vector_norm(train_aux_raw, "global")
    train_aux = apply_vector_norm(train_aux_raw, aux_norm)
    val_aux = apply_vector_norm(val_aux_raw, aux_norm)
    test_aux = apply_vector_norm(test_aux_raw, aux_norm)

    train_y = labels_to_ids(train_df[args.label_column])
    val_y = labels_to_ids(val_df[args.label_column])
    test_y = labels_to_ids(test_df[args.label_column])

    train_episodes, train_episode_meta = build_episode_groups(train_df, args)
    val_episodes, val_episode_meta = build_episode_groups(val_df, args)
    test_episodes, test_episode_meta = build_episode_groups(test_df, args)

    model = build_paper_model(
        input_dim=train_x.shape[-1],
        aux_dim=train_aux.shape[-1],
        aux_branch_dims=infer_aux_branch_dims(train_aux_meta, train_aux),
        hidden_dim=args.hidden_dim,
        score_dim=args.score_dim,
        prompt_dim=args.prompt_dim,
        num_prompts=args.num_prompts,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_audio_tokens=train_x.shape[1],
        temperature=args.temperature,
    )
    init_checkpoint_info = load_model_init_checkpoint(model, args.init_checkpoint, device)

    if int(args.max_steps) >= 0:
        model, curve, train_summary = train_model(
            model=model,
            train_x=train_x,
            train_aux=train_aux,
            train_y=train_y,
            val_x=val_x,
            val_aux=val_aux,
            val_y=val_y,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            args=args,
            device=device,
        )
    else:
        model.to(device)
        curve = []
        train_summary = {"best_val_macro_f1": None, "training_time_sec": 0.0}
    write_curve(curve, curve_output)

    if bool(getattr(args, "skip_val_eval", False)):
        val_metrics, val_global_metrics, val_diag, val_split_summary = skipped_val_payload()
    else:
        val_metrics, _val_pred, _val_prob, val_diag, val_global_metrics, _val_global_prob, _val_support_flag, _val_query_flag, val_split_summary = evaluate_split_runs(
            model=model,
            x=val_x,
            aux=val_aux,
            y=val_y,
            episode_groups=val_episodes,
            args=args,
            device=device,
            seed_offset=200_000,
            split_name="val",
        )
    test_subgroup_masks = build_eval_subgroup_masks(test_df, args)
    test_metrics, test_pred, test_prob, test_diag, test_global_metrics, test_global_prob, test_support_flag, test_query_flag, test_split_summary = evaluate_split_runs(
        model=model,
        x=test_x,
        aux=test_aux,
        y=test_y,
        episode_groups=test_episodes,
        args=args,
        device=device,
        seed_offset=300_000,
        split_name="test",
        subgroup_masks=test_subgroup_masks,
    )

    valid_test = test_pred >= 0
    if bool(getattr(args, "eval_query_only", False)):
        valid_test = np.logical_and(valid_test, test_query_flag == 1)
    labels = list(range(len(VOWEL_ORDER)))
    report = classification_report(
        test_y[valid_test],
        test_pred[valid_test],
        labels=labels,
        target_names=VOWEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_y[valid_test], test_pred[valid_test], labels=labels)
    pred_df = prediction_frame(test_df.loc[valid_test].reset_index(drop=True), test_y[valid_test], test_pred[valid_test], test_prob[valid_test], args)
    pred_df["global_pred_vowel"] = [VOWEL_ORDER[idx] for idx in test_global_prob[valid_test].argmax(axis=-1).astype(np.int64)]
    pred_df["global_pred_confidence"] = test_global_prob[valid_test].max(axis=1)
    pred_df["is_support_item"] = test_support_flag[valid_test].astype(np.int64)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_output, index=False, encoding="utf-8-sig")
    mechanism_summary: dict[str, Any] = {}
    if str(getattr(args, "mechanism_output_dir", "") or "").strip():
        mechanism_summary = collect_mechanism_diagnostics(
            model=model,
            x=test_x,
            aux=test_aux,
            y=test_y,
            rows=test_df.reset_index(drop=True),
            episode_groups=test_episodes,
            args=args,
            device=device,
            output_dir=args.mechanism_output_dir,
            k_values=args.mechanism_k_values,
            max_episodes=int(getattr(args, "mechanism_max_episodes", 0) or 0),
        )

    result = {
        "task": "leave_one_region_out_vowel_classification",
        "method": "PromptConditionedDualLevelClassMemoryTransformer",
        "algorithm_flow": [
            "project Whisper token sequence into the shared hidden space",
            "extract and align Log-Mel, MFCC, and MFCC delta feature sequences",
            "adaptively fuse multi-source acoustic branches into H_f",
            "gate speech and acoustic streams into H_0",
            "read global class memory through prompt-conditioned attention bias",
            "classify by cosine matching between pooled query representation and class memory",
            "build episode memory from support speech using global-memory predictions as soft write weights",
            "fuse global memory and episode acoustic aggregate with a learned gate",
            "classify episode queries with L_episode + lambda_g * L_global",
        ],
        "paper_alignment": {
            "multi_source_acoustic_representation": True,
            "global_class_memory": True,
            "prompt_router_over_prompt_library": True,
            "prompt_to_attention_bias": True,
            "audio_query_memory_key_value_reading": True,
            "cosine_memory_matching_classifier": True,
            "order_invariant_episode_write": True,
            "support_labels_used_for_memory_write": str(args.support_write_mode) in {"label", "blend"},
            "pseudo_label_memory_write": str(args.support_write_mode) == "pseudo",
            "dual_loss_global_plus_episode": True,
        },
        "implementation_assumptions": [
            "Fuse(H_s, H_f) uses a learned hidden-space gate because the paper does not specify the exact fusion operator.",
            "Prompt bias is generated for the cached fixed token length token_chunks and sliced to the active sequence length.",
            "Episodes are built from speaker_id when available, otherwise region, and long episodes are chunked by episode_length.",
            "Support/query splitting can use per-episode sampling or global_support evaluation, where each target-domain class contributes k labeled support samples and all remaining rows are queries.",
            "Support memory writing uses labeled class-wise aggregation by default; pseudo-label writing is retained through --support-write-mode pseudo for old ablations.",
            "Evaluation can report either all episode items or query-only metrics through --eval-query-only.",
        ],
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "global_metrics": test_global_metrics,
        "val_global_metrics": val_global_metrics,
        "evaluation_splits": {
            "val": val_split_summary,
            "test": test_split_summary,
        },
        "report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_confusions": summarize_confusions(cm, VOWEL_ORDER, top_n=10),
        "diagnostics": {
            "val": val_diag,
            "test": test_diag,
            "mechanism": mechanism_summary,
            "episodes": {
                "train": train_episode_meta,
                "val": val_episode_meta,
                "test": test_episode_meta,
            },
        },
        "split": split_info,
        "split_audit": split_audit,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "label_counts": {
            "train": label_counts(train_df[args.label_column], VOWEL_ORDER),
            "val": label_counts(val_df[args.label_column], VOWEL_ORDER),
            "test": label_counts(test_df[args.label_column], VOWEL_ORDER),
        },
        "cache": {
            "matrix": matrix_meta,
            "aux": {"train": train_aux_meta, "val": val_aux_meta, "test": test_aux_meta},
            "token_norm": serializable_token_norm(token_norm),
            "aux_norm": serializable_norm(aux_norm),
            "aux_rows_hash": {
                "train": aux_rows_hash(train_df, args),
                "val": aux_rows_hash(val_df, args),
                "test": aux_rows_hash(test_df, args),
            },
        },
        "config": vars(args),
        "training": train_summary,
        "init_checkpoint": init_checkpoint_info,
        "files": {
            "output": str(output),
            "curve": str(curve_output),
            "predictions": str(predictions_output),
            "debug": str(debug_output),
            "audit": str(audit_output),
            "checkpoint": str(checkpoint_output) if args.save_checkpoint else "",
        },
        "parameters": {
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total": int(sum(p.numel() for p in model.parameters())),
        },
        "split_hash": {
            "train": dataframe_hash(train_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "val": dataframe_hash(val_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
            "test": dataframe_hash(test_df, ["sample_id", args.label_column, args.region_column, args.speaker_column]),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_payload = {
        "result_file": str(output),
        "metrics": test_metrics,
        "global_metrics": test_global_metrics,
        "evaluation_splits": result["evaluation_splits"],
        "diagnostics": result["diagnostics"],
        "cache": result["cache"],
        "split": result["split"],
        "split_audit": result["split_audit"],
        "training": result["training"],
        "command": " ".join(sys.argv),
        "init_checkpoint": init_checkpoint_info,
    }
    debug_output.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "token_norm": token_norm,
                "aux_norm": aux_norm,
                "labels": VOWEL_ORDER,
            },
            checkpoint_output,
        )
    print(json.dumps({"output": str(output), "metrics": test_metrics, "global_metrics": test_global_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
