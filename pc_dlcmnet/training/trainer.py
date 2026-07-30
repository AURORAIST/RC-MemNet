"""Training loop for the dual-memory experiment."""

from __future__ import annotations

import csv
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from pc_dlcmnet.evaluation.downstream import evaluate, route_entropy
from pc_dlcmnet.data.episodes import make_episode_tensors, split_support_query
from pc_dlcmnet.training.supervised import class_loss_weight
from pc_dlcmnet.models.feature_memory import VOWEL_ORDER


def class_key_mean(tensor: torch.Tensor, token_count: int, num_classes: int) -> torch.Tensor:
    """Average [B,H,T,T+C] tensors over heads/tokens and keep class keys."""

    return tensor[:, :, :, token_count : token_count + num_classes].mean(dim=(1, 2))


def append_mean_stats(target: dict[str, float], prefix: str, values: list[np.ndarray]) -> None:
    if not values:
        return
    arr = np.stack(values, axis=0)
    mean = arr.mean(axis=0)
    for idx, value in enumerate(mean.tolist()):
        target[f"{prefix}_{idx}"] = float(value)


def train_batch(
    model: torch.nn.Module,
    batch_episodes: Sequence[np.ndarray],
    train_x: np.ndarray,
    train_aux: np.ndarray,
    train_y: np.ndarray,
    class_weight: torch.Tensor,
    args: Any,
    device: torch.device,
    seed_offset: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = None
    episode_losses = []
    global_losses = []
    accuracies = []
    support_sizes = []
    query_sizes = []
    gate_means = []
    gate_class_rows = []
    absorption_class_rows = []
    adaptation_ratio_rows = []
    feature_weight_rows = []
    route_entropies = []
    prompt_route_rows = []
    true_attention_without_prompt = []
    true_attention_with_prompt = []
    true_attention_shift = []
    used = 0
    for ep_pos, episode_indices in enumerate(batch_episodes):
        support_idx, query_idx = split_support_query(
            episode_indices,
            train_y,
            args.support_shots,
            seed=args.seed + seed_offset + ep_pos,
        )
        if len(query_idx) == 0:
            continue
        support_speech, support_aux_t, support_mask = make_episode_tensors(train_x, train_aux, support_idx, device)
        query_speech, query_aux_t, query_mask = make_episode_tensors(train_x, train_aux, query_idx, device)
        support_labels = torch.tensor(train_y[support_idx], dtype=torch.long, device=device) if len(support_idx) else None
        query_y = torch.tensor(train_y[query_idx], dtype=torch.long, device=device)
        label_blend = 0.0 if args.support_write_mode == "pseudo" else (float(args.support_label_blend) if args.support_write_mode == "blend" else 1.0)
        out = model.forward_episode(
            support_speech=support_speech,
            support_aux=support_aux_t,
            support_mask=support_mask,
            support_labels=support_labels,
            query_speech=query_speech,
            query_aux=query_aux_t,
            query_mask=query_mask,
            label_blend=label_blend,
        )
        loss_episode = F.cross_entropy(
            out["episode_logits"],
            query_y,
            weight=class_weight,
            label_smoothing=float(args.label_smoothing),
        )
        loss_global = F.cross_entropy(
            out["global_logits"],
            query_y,
            weight=class_weight,
            label_smoothing=float(args.label_smoothing),
        )
        loss = loss_episode + float(args.lambda_global) * loss_global
        total_loss = loss if total_loss is None else total_loss + loss
        episode_losses.append(float(loss_episode.detach().cpu()))
        global_losses.append(float(loss_global.detach().cpu()))
        pred = out["episode_logits"].argmax(dim=-1)
        accuracies.append(float((pred == query_y).float().mean().detach().cpu()))
        support_sizes.append(int(len(support_idx)))
        query_sizes.append(int(len(query_idx)))
        gate_means.append(float(out["write_gate"].mean().detach().cpu()))
        class_gate = out["write_gate"].mean(dim=-1).detach().cpu().numpy()
        gate_class_rows.append(class_gate)
        absorption_class_rows.append(1.0 - class_gate)
        global_memory = out["global_memory"].detach()
        candidate_memory = out["candidate_memory"].detach()
        episode_memory = out["episode_memory"].detach()
        denom = (candidate_memory - global_memory).norm(dim=-1).clamp_min(1e-8)
        rho = ((episode_memory - global_memory).norm(dim=-1) / denom).detach().cpu().numpy()
        adaptation_ratio_rows.append(rho)
        if out.get("query_feature_weights") is not None and out["query_feature_weights"].numel() > 0:
            feature_weight_rows.append(out["query_feature_weights"].mean(dim=0).detach().cpu().numpy())
        if out.get("episode_route_weights") is not None and out["episode_route_weights"].numel() > 0:
            route_entropies.append(route_entropy(out["episode_route_weights"]))
            prompt_route_rows.append(out["episode_route_weights"].mean(dim=(0, 1)).detach().cpu().numpy())
        if out.get("episode_attention") is not None and out["episode_attention"].numel() > 0:
            token_count = int(query_speech.shape[1])
            no_prompt = class_key_mean(
                torch.softmax(out["episode_acoustic_scores"], dim=-1),
                token_count,
                len(VOWEL_ORDER),
            )
            with_prompt = class_key_mean(out["episode_attention"], token_count, len(VOWEL_ORDER))
            batch_index = torch.arange(query_y.shape[0], device=device)
            true_no = no_prompt[batch_index, query_y].detach().cpu().numpy()
            true_yes = with_prompt[batch_index, query_y].detach().cpu().numpy()
            true_attention_without_prompt.extend(true_no.tolist())
            true_attention_with_prompt.extend(true_yes.tolist())
            true_attention_shift.extend((true_yes - true_no).tolist())
        used += 1
    if total_loss is None or used == 0:
        raise ValueError("no usable training episodes in current batch")
    total_loss = total_loss / float(used)
    feature_means = (
        np.mean(np.stack(feature_weight_rows, axis=0), axis=0)
        if feature_weight_rows
        else np.zeros((max(1, int(train_aux.shape[-1] > 0))), dtype=np.float32)
    )
    stats = {
        "train_loss": float(total_loss.detach().cpu()),
        "train_episode_loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
        "train_global_loss": float(np.mean(global_losses)) if global_losses else 0.0,
        "train_query_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "train_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
        "train_query_size": float(np.mean(query_sizes)) if query_sizes else 0.0,
        "episode_write_gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
        "prompt_route_entropy": float(np.mean(route_entropies)) if route_entropies else 0.0,
        "true_attention_without_prompt": float(np.mean(true_attention_without_prompt)) if true_attention_without_prompt else 0.0,
        "true_attention_with_prompt": float(np.mean(true_attention_with_prompt)) if true_attention_with_prompt else 0.0,
        "true_attention_shift": float(np.mean(true_attention_shift)) if true_attention_shift else 0.0,
    }
    for idx, value in enumerate(feature_means.tolist()):
        stats[f"feature_weight_{idx}"] = float(value)
    append_mean_stats(stats, "gate_retention_class", gate_class_rows)
    append_mean_stats(stats, "support_absorption_class", absorption_class_rows)
    append_mean_stats(stats, "adaptation_ratio_class", adaptation_ratio_rows)
    append_mean_stats(stats, "prompt_route_mean", prompt_route_rows)
    return total_loss, stats


def write_curve(curve: list[dict[str, float]], path: Path) -> None:
    keys = sorted({key for row in curve for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(curve)


def train_model(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_aux: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_aux: np.ndarray,
    val_y: np.ndarray,
    train_episodes: Sequence[np.ndarray],
    val_episodes: Sequence[np.ndarray],
    args: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]], dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    class_weight = class_loss_weight(train_y, args.class_weight_power, device)
    curve: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_macro = -math.inf
    best_step = 0
    last_improve_step = 0
    stopped_early = False
    stop_step = 0
    early_stop_patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 1e-4) or 0.0)
    early_stop_min_steps = max(0, int(getattr(args, "early_stop_min_steps", 0) or 0))
    configured_max_steps = int(args.max_steps)
    train_until_early_stop = configured_max_steps == 0
    if train_until_early_stop and early_stop_patience <= 0:
        raise ValueError("--max-steps 0 requires --early-stop-patience > 0")
    effective_max_steps = configured_max_steps if configured_max_steps > 0 else 2_147_483_647
    start_time = time.perf_counter()
    episode_order = list(range(len(train_episodes)))
    random.Random(args.seed).shuffle(episode_order)
    cursor = 0
    epoch = 1
    for step in range(1, effective_max_steps + 1):
        if cursor >= len(episode_order):
            epoch += 1
            cursor = 0
            random.Random(args.seed + epoch).shuffle(episode_order)
        batch_ids = episode_order[cursor : cursor + max(1, int(args.episode_batch_size))]
        cursor += max(1, int(args.episode_batch_size))
        batch_episodes = [train_episodes[idx] for idx in batch_ids]
        optimizer.zero_grad(set_to_none=True)
        loss, row = train_batch(
            model=model,
            batch_episodes=batch_episodes,
            train_x=train_x,
            train_aux=train_aux,
            train_y=train_y,
            class_weight=class_weight,
            args=args,
            device=device,
            seed_offset=step * 1000,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        row["step"] = float(step)
        row["epoch"] = float(epoch)
        row["grad_norm"] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        is_final_step = (not train_until_early_stop) and step == effective_max_steps
        if step == 1 or step % int(args.eval_every) == 0 or is_final_step:
            val_metrics, _val_pred, _val_prob, val_diag, val_global_metrics, _gl_prob, _support_flag, _query_flag = evaluate(
                model=model,
                x=val_x,
                aux=val_aux,
                y=val_y,
                episode_groups=val_episodes,
                args=args,
                device=device,
                seed_offset=10_000 + step,
            )
            row.update(
                {
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_global_accuracy": float(val_global_metrics["accuracy"]),
                    "val_global_macro_f1": float(val_global_metrics["macro_f1"]),
                    "val_support_fraction": float(val_diag["support_fraction"]),
                    "val_write_gate_mean": float(val_diag["write_gate_mean"]),
                    "val_prompt_route_entropy": float(val_diag["episode_prompt_route_entropy"]),
                }
            )
            current_macro = float(val_metrics["macro_f1"])
            if best_macro == -math.inf or current_macro > best_macro + early_stop_min_delta:
                best_macro = current_macro
                best_state = deepcopy(model.state_dict())
                best_step = int(step)
                last_improve_step = int(step)
            row["best_val_macro_f1"] = float(best_macro)
            row["best_step"] = float(best_step)
            row["steps_since_improvement"] = float(step - last_improve_step)
            if (
                early_stop_patience > 0
                and step >= early_stop_min_steps
                and (step - last_improve_step) >= early_stop_patience
            ):
                row["early_stop_triggered"] = 1.0
                stopped_early = True
                stop_step = int(step)
        curve.append(row)
        if (not bool(getattr(args, "quiet", False))) and (step == 1 or step % int(args.log_every) == 0 or is_final_step):
            print(
                (
                    f"[train] step={step} epoch={epoch} "
                    f"loss={row['train_loss']:.4f} "
                    f"ep={row['train_episode_loss']:.4f} "
                    f"gl={row['train_global_loss']:.4f} "
                    f"acc={row['train_query_accuracy']:.4f} "
                    f"support={row['train_support_size']:.2f} "
                    f"query={row['train_query_size']:.2f}"
                ),
                flush=True,
            )
            if "val_macro_f1" in row:
                print(
                    (
                        f"[valid] step={step} "
                        f"episode_macro_f1={row['val_macro_f1']:.4f} "
                        f"global_macro_f1={row['val_global_macro_f1']:.4f}"
                    ),
                    flush=True,
                )
                if stopped_early:
                    print(
                        (
                            f"[early-stop] step={step} "
                            f"best_step={best_step} "
                            f"best_val_macro_f1={best_macro:.4f} "
                            f"patience={early_stop_patience}"
                        ),
                        flush=True,
                    )
        if stopped_early:
            break
    model.load_state_dict(best_state)
    summary = {
        "best_val_macro_f1": float(best_macro),
        "best_step": int(best_step),
        "final_step": int(stop_step or (curve[-1]["step"] if curve else 0)),
        "stopped_early": bool(stopped_early),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_min_steps": int(early_stop_min_steps),
        "train_until_early_stop": bool(train_until_early_stop),
        "configured_max_steps": int(configured_max_steps),
        "training_time_sec": float(time.perf_counter() - start_time),
    }
    return model, curve, summary
