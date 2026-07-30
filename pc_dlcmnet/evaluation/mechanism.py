"""Mechanism-diagnostic CSV export for PC-DLCMNet.

The functions here run a lightweight post-training evaluation pass and persist
the intermediate tensors required by the paper visualizations. They are called
only when the runner receives ``--mechanism-output-dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from pc_dlcmnet.data.episodes import make_episode_tensors, split_global_support_query
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER


def _mean_class_key_tensor(tensor: torch.Tensor, token_count: int, num_classes: int) -> np.ndarray:
    """Average [B,H,T,T+C] tensors over heads/tokens and keep class keys."""

    class_part = tensor[:, :, :, token_count : token_count + num_classes]
    return class_part.mean(dim=(1, 2)).detach().cpu().numpy()


def _attention_without_prompt(acoustic_scores: torch.Tensor) -> torch.Tensor:
    return torch.softmax(acoustic_scores, dim=-1)


def _memory_rows(memory: torch.Tensor, region: str, k: int, episode: int, memory_type: str) -> list[dict[str, float | int | str]]:
    arr = memory.detach().cpu().numpy()
    rows: list[dict[str, float | int | str]] = []
    for class_id, label in enumerate(VOWEL_ORDER):
        row: dict[str, float | int | str] = {
            "region": region,
            "k": int(k),
            "episode": int(episode),
            "class_id": int(class_id),
            "class": label,
            "memory_type": memory_type,
        }
        row.update({f"f{idx}": float(value) for idx, value in enumerate(arr[class_id].tolist())})
        rows.append(row)
    return rows


def collect_mechanism_diagnostics(
    model: torch.nn.Module,
    x: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    rows: pd.DataFrame,
    episode_groups: Sequence[np.ndarray],
    args: Any,
    device: torch.device,
    output_dir: str | Path,
    k_values: Sequence[int],
    max_episodes: int = 0,
    hard_classes: Sequence[str] | None = None,
) -> dict[str, int | str]:
    """Collect and save diagnostic CSVs for mechanism visualizations."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    label_blend = 0.0 if args.support_write_mode == "pseudo" else (float(args.support_label_blend) if args.support_write_mode == "blend" else 1.0)
    region_name = str(getattr(args, "holdout_region", "") or rows[args.region_column].astype(str).iloc[0])
    groups = list(episode_groups)
    if int(max_episodes or 0) > 0:
        groups = groups[: int(max_episodes)]
    if not groups:
        groups = [np.arange(len(y), dtype=np.int64)]
    hard_set = set(hard_classes or ["e", "\u0259", "o", "\u0254", "i", "y"])

    gate_rows: list[dict[str, float | int | str]] = []
    memory_rows: list[dict[str, float | int | str]] = []
    pcmr_rows: list[dict[str, float | int | str]] = []
    routing_rows: list[dict[str, float | int | str | bool]] = []
    pred_rows: list[dict[str, float | int | str]] = []

    with torch.no_grad():
        for k in [int(v) for v in k_values]:
            support_idx, query_all_idx = split_global_support_query(y, k, seed=int(args.seed) + 700_000 + k * 9_973)
            support_speech, support_aux, support_mask = make_episode_tensors(x, aux, support_idx, device)
            support_labels = torch.tensor(y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
            build = model.build_episode_memory(
                support_speech,
                support_aux,
                support_mask,
                support_labels=support_labels,
                label_blend=label_blend,
            )
            support_counts = np.bincount(y[support_idx], minlength=len(VOWEL_ORDER)) if len(support_idx) else np.zeros(len(VOWEL_ORDER), dtype=int)
            class_gate = build["write_gate"].mean(dim=-1).detach().cpu().numpy()
            for class_id, label in enumerate(VOWEL_ORDER):
                gate_rows.append(
                    {
                        "region": region_name,
                        "k": int(k),
                        "episode": -1,
                        "class_id": int(class_id),
                        "class": label,
                        "gate_global_weight": float(class_gate[class_id]),
                        "support_injection_weight": float(1.0 - class_gate[class_id]),
                        "support_count": int(support_counts[class_id]),
                    }
                )
            memory_rows.extend(_memory_rows(model.global_memory, region_name, k, -1, "global_memory"))
            memory_rows.extend(_memory_rows(build["candidate_memory"], region_name, k, -1, "support_candidate"))
            memory_rows.extend(_memory_rows(build["episode_memory"], region_name, k, -1, "episode_memory"))

            query_set = set(query_all_idx.tolist())
            for ep_pos, episode_indices in enumerate(groups):
                query_idx = np.asarray([int(idx) for idx in episode_indices.tolist() if int(idx) in query_set], dtype=np.int64)
                if len(query_idx) == 0:
                    continue
                query_speech, query_aux, query_mask = make_episode_tensors(x, aux, query_idx, device)
                episode_out = model.predict(query_speech, query_aux, query_mask, build["episode_memory"])
                global_out = model.predict(query_speech, query_aux, query_mask, model.global_memory)
                ep_prob = torch.softmax(episode_out["logits"], dim=-1).detach().cpu().numpy()
                gl_prob = torch.softmax(global_out["logits"], dim=-1).detach().cpu().numpy()
                pred = ep_prob.argmax(axis=-1).astype(np.int64)
                global_pred = gl_prob.argmax(axis=-1).astype(np.int64)

                token_count = query_speech.shape[1]
                acoustic_score = _mean_class_key_tensor(episode_out["acoustic_scores"], token_count, len(VOWEL_ORDER))
                prompt_bias = _mean_class_key_tensor(episode_out["prompt_bias"], token_count, len(VOWEL_ORDER))
                attention_without_prompt = _mean_class_key_tensor(
                    _attention_without_prompt(episode_out["acoustic_scores"]),
                    token_count,
                    len(VOWEL_ORDER),
                )
                attention_with_prompt = _mean_class_key_tensor(episode_out["attention"], token_count, len(VOWEL_ORDER))
                attention_shift = attention_with_prompt - attention_without_prompt
                routes = episode_out["route_weights"].mean(dim=1).detach().cpu().numpy()

                for local_i, row_idx in enumerate(query_idx.tolist()):
                    true_id = int(y[row_idx])
                    true_class = VOWEL_ORDER[true_id]
                    sample_id = str(rows.iloc[row_idx].get("sample_id", f"row_{row_idx:08d}"))
                    pred_rows.append(
                        {
                            "sample_id": sample_id,
                            "region": region_name,
                            "k": int(k),
                            "true_class": true_class,
                            "baseline_pred": VOWEL_ORDER[int(global_pred[local_i])],
                            "ours_pred": VOWEL_ORDER[int(pred[local_i])],
                            "ours_confidence": float(ep_prob[local_i].max()),
                            "baseline_confidence": float(gl_prob[local_i].max()),
                        }
                    )
                    for prompt_id, alpha in enumerate(routes[local_i].tolist()):
                        routing_rows.append(
                            {
                                "sample_id": sample_id,
                                "region": region_name,
                                "k": int(k),
                                "true_class": true_class,
                                "pred_class": VOWEL_ORDER[int(pred[local_i])],
                                "correct": bool(pred[local_i] == true_id),
                                "prompt": int(prompt_id),
                                "alpha": float(alpha),
                            }
                        )
                    if true_class not in hard_set:
                        continue
                    for class_id, label in enumerate(VOWEL_ORDER):
                        for component, values in [
                            ("acoustic_score", acoustic_score),
                            ("prompt_bias", prompt_bias),
                            ("attention_without_prompt", attention_without_prompt),
                            ("attention_with_prompt", attention_with_prompt),
                            ("attention_shift", attention_shift),
                            ("final_attention", attention_with_prompt),
                        ]:
                            pcmr_rows.append(
                                {
                                    "query_id": sample_id,
                                    "region": region_name,
                                    "k": int(k),
                                    "true_class": true_class,
                                    "class_id": int(class_id),
                                    "class": label,
                                    "component": component,
                                    "value": float(values[local_i, class_id]),
                                }
                            )

    files = {
        "gema_gate_diagnostics.csv": gate_rows,
        "memory_trajectory.csv": memory_rows,
        "pcmr_decomposition.csv": pcmr_rows,
        "prompt_routing.csv": routing_rows,
        "predictions.csv": pred_rows,
    }
    summary: dict[str, int | str] = {"output_dir": str(out_dir)}
    for name, file_rows in files.items():
        pd.DataFrame(file_rows).to_csv(out_dir / name, index=False, encoding="utf-8-sig")
        summary[name] = int(len(file_rows))
    return summary
