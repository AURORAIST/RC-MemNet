"""Paper-ready plotting functions for PC-DLCMNet mechanism diagnostics.

This module only contains plotting and light tabulation logic. It does not
load checkpoints, run the model, or write files unless ``save_figure`` is
called explicitly.

Expected long-table schemas:
- GEMA gate: ``region``, ``class``, ``gate_global_weight`` and optionally ``k``.
- Memory trajectory: ``class``, ``memory_type`` plus embedding columns
  ``f0``, ``f1``, ... or precomputed ``x``/``y`` coordinates.
- PCMR decomposition: ``query_id``, ``true_class``, ``class``, ``component``,
  ``value`` where component is one of ``acoustic_score``, ``prompt_bias``,
  ``final_attention``.
- Prompt routing: ``sample_id``, ``prompt``, ``alpha`` and optionally
  ``region`` and ``correct``.
- Support corruption: ``noise_ratio``, ``method``, ``macro_f1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_VOWEL_ORDER = ["a", "e", "i", "o", "u", "y", "\u0254", "\u0259", "\u025b"]
DEFAULT_REGION_ORDER = ["Qingyang", "Tongling", "Jingxian", "Nanling", "Ningguo", "Lishui"]

COLORS = {
    "global": "#4C78A8",
    "support": "#F58518",
    "episode": "#54A24B",
    "accent": "#E45756",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}

MARKERS = {
    "global_memory": "s",
    "support_candidate": "o",
    "episode_memory": "^",
    "m_g": "s",
    "u": "o",
    "m_e": "^",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def setup_paper_style() -> None:
    """Apply a compact Matplotlib style shared by mechanism figures."""

    plt = _plt()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLORS["light"],
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,
        }
    )


def save_figure(fig, output_dir: str | Path, name: str, formats: Sequence[str] = ("png", "pdf")) -> None:
    """Save a returned figure in one or more formats."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")


def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _ordered_existing(values: Iterable[str], preferred: Sequence[str] | None) -> list[str]:
    seen = [str(v) for v in values if pd.notna(v)]
    unique = list(dict.fromkeys(seen))
    if not preferred:
        return unique
    pref = [str(v) for v in preferred]
    return [v for v in pref if v in unique] + [v for v in unique if v not in pref]


def _normalize_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr * 100.0 if np.nanmax(arr) <= 1.5 else arr


def plot_gema_class_region_gate_heatmap(
    gate_df: pd.DataFrame,
    region_order: Sequence[str] | None = DEFAULT_REGION_ORDER,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    value_col: str = "support_injection_weight",
    title: str = "Support absorption ratio of GEMA",
):
    """Plot GEMA support absorption over target regions and vowel classes.

    By default this plots ``A_c = 1 - mean_j g_{c,j}``. Larger values mean
    stronger absorption of target support evidence.
    """

    _require_columns(gate_df, ["region", "class", value_col], "gate_df")
    plt = _plt()
    setup_paper_style()
    regions = _ordered_existing(gate_df["region"], region_order)
    classes = _ordered_existing(gate_df["class"], class_order)
    pivot = (
        gate_df.groupby(["region", "class"], as_index=False)[value_col]
        .mean()
        .pivot(index="region", columns="class", values=value_col)
        .reindex(index=regions, columns=classes)
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_xlabel("Vowel class")
    ax.set_ylabel("Target region")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Support absorption ratio")
    return fig


def plot_gema_gate_distribution(
    gate_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 5),
    value_col: str = "gate_global_weight",
    k_col: str = "k",
    bins: int = 28,
    title: str = "Distribution of GEMA gate values",
):
    """Plot gate-value histograms for two support-shot settings."""

    _require_columns(gate_df, [k_col, value_col], "gate_df")
    plt = _plt()
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = [COLORS["global"], COLORS["support"], COLORS["episode"]]
    for pos, k in enumerate(k_values):
        vals = gate_df.loc[gate_df[k_col].astype(int) == int(k), value_col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, range=(0, 1), density=False, alpha=0.45, color=colors[pos % len(colors)], label=f"{k}-shot")
        ax.axvline(vals.mean(), color=colors[pos % len(colors)], linestyle="--", linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def plot_gema_gate_lines_by_class(
    gate_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    value_col: str = "gate_global_weight",
    k_col: str = "k",
    title: str = "GEMA retention gate across support shots",
):
    """Plot class-wise mean GEMA retention gate as K changes.

    The plotted value is ``mean_j g_{c,j}``, where higher values mean stronger
    retention of global class memory and lower values mean more support-driven
    adaptation.
    """

    _require_columns(gate_df, ["class", k_col, value_col], "gate_df")
    plt = _plt()
    setup_paper_style()
    d = gate_df.copy()
    d[k_col] = d[k_col].astype(int)
    classes = _ordered_existing(d["class"], class_order)
    summary = d.groupby([k_col, "class"], as_index=False)[value_col].mean()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    cmap = plt.get_cmap("tab10")
    for pos, cls in enumerate(classes):
        group = summary[summary["class"].astype(str) == str(cls)].sort_values(k_col)
        if group.empty:
            continue
        ax.plot(group[k_col], group[value_col], marker="o", color=cmap(pos % 10), label=cls)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Support shots per class")
    ax.set_ylabel("Mean retention gate")
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    ax.legend(frameon=False, ncol=min(5, max(1, len(classes))), loc="best")
    return fig


def plot_gema_gate_lines_by_region(
    gate_df: pd.DataFrame,
    region_order: Sequence[str] | None = DEFAULT_REGION_ORDER,
    value_col: str = "support_injection_weight",
    k_col: str = "k",
    title: str = "GEMA support injection across target regions",
):
    """Plot region-wise support injection ``1 - mean_j g_{c,j}`` as K changes."""

    _require_columns(gate_df, ["region", k_col, value_col], "gate_df")
    plt = _plt()
    setup_paper_style()
    d = gate_df.copy()
    d[k_col] = d[k_col].astype(int)
    regions = _ordered_existing(d["region"], region_order)
    summary = d.groupby([k_col, "region"], as_index=False)[value_col].mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    cmap = plt.get_cmap("tab10")
    for pos, region in enumerate(regions):
        group = summary[summary["region"].astype(str) == str(region)].sort_values(k_col)
        if group.empty:
            continue
        ax.plot(group[k_col], group[value_col], marker="o", color=cmap(pos % 10), label=region)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Support shots per class")
    ax.set_ylabel("Support injection weight (1 - gate)")
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="best")
    return fig


def _project_memory_points(memory_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if {"x", "y"}.issubset(memory_df.columns):
        return memory_df.copy()
    feature_cols = [c for c in memory_df.columns if str(c).startswith("f")]
    if len(feature_cols) < 2:
        raise ValueError("memory_df must contain x/y or at least two embedding columns named f0, f1, ...")
    x = memory_df[feature_cols].to_numpy(dtype=np.float32)
    try:
        from umap import UMAP

        z = UMAP(n_components=2, random_state=seed, n_neighbors=min(15, max(2, len(memory_df) - 1))).fit_transform(x)
    except Exception:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(x)
        perplexity = min(30, max(2, (len(memory_df) - 1) // 3))
        z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(x)
    out = memory_df.copy()
    out["x"] = z[:, 0]
    out["y"] = z[:, 1]
    return out


def plot_memory_adaptation_trajectory(
    memory_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    seed: int = 0,
    title: str = "Global memory, support candidate, and episode memory",
):
    """Visualize ``m_g``, ``u`` and ``m_e`` with arrows toward episode memory."""

    _require_columns(memory_df, ["class", "memory_type"], "memory_df")
    plt = _plt()
    setup_paper_style()
    d = _project_memory_points(memory_df, seed)
    classes = _ordered_existing(d["class"], class_order)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    label_map = {
        "global_memory": "global memory",
        "support_candidate": "support candidate",
        "episode_memory": "episode memory",
        "m_g": "global memory",
        "u": "support candidate",
        "m_e": "episode memory",
    }
    for class_pos, cls in enumerate(classes):
        cdf = d[d["class"].astype(str) == str(cls)]
        color = cmap(class_pos % 10)
        for mem_type, group in cdf.groupby("memory_type"):
            marker = MARKERS.get(str(mem_type), "o")
            ax.scatter(group["x"], group["y"], s=34, marker=marker, color=color, edgecolor="white", linewidth=0.5, alpha=0.88)
        centers = cdf.groupby("memory_type")[["x", "y"]].mean()
        ep_key = "episode_memory" if "episode_memory" in centers.index else "m_e"
        if ep_key in centers.index:
            ep = centers.loc[ep_key]
            for src_key, alpha in [("global_memory", 0.62), ("m_g", 0.62), ("support_candidate", 0.35), ("u", 0.35)]:
                if src_key in centers.index:
                    src = centers.loc[src_key]
                    ax.annotate("", xy=(ep["x"], ep["y"]), xytext=(src["x"], src["y"]), arrowprops={"arrowstyle": "->", "color": color, "alpha": alpha, "linewidth": 1.0})
    for mem_type, marker in [("global_memory", "s"), ("support_candidate", "o"), ("episode_memory", "^")]:
        ax.scatter([], [], marker=marker, color=COLORS["gray"], label=label_map[mem_type])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    return fig


def plot_memory_movement_lines(
    memory_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    k_col: str = "k",
    title: str = "Memory adaptation magnitude",
):
    """Plot how far episode memory moves from global memory and support candidate.

    This line plot is often more interpretable than a 2-D projection because it
    directly measures ``||m_e - m_g||`` and ``||m_e - u||`` per class.
    """

    _require_columns(memory_df, ["class", "memory_type", k_col], "memory_df")
    feature_cols = [c for c in memory_df.columns if str(c).startswith("f")]
    if not feature_cols:
        raise ValueError("memory_df must contain embedding columns named f0, f1, ...")
    classes = _ordered_existing(memory_df["class"], class_order)
    rows = []
    for (k, cls), group in memory_df.groupby([k_col, "class"], sort=True):
        vectors = {str(row["memory_type"]): row[feature_cols].to_numpy(dtype=float) for _, row in group.iterrows()}
        global_vec = vectors.get("global_memory") if "global_memory" in vectors else vectors.get("m_g")
        support_vec = vectors.get("support_candidate") if "support_candidate" in vectors else vectors.get("u")
        episode_vec = vectors.get("episode_memory") if "episode_memory" in vectors else vectors.get("m_e")
        if episode_vec is None:
            continue
        if global_vec is not None:
            rows.append({"k": int(k), "class": str(cls), "distance": float(np.linalg.norm(episode_vec - global_vec)), "metric": "episode-global"})
        if support_vec is not None:
            rows.append({"k": int(k), "class": str(cls), "distance": float(np.linalg.norm(episode_vec - support_vec)), "metric": "episode-support"})
    d = pd.DataFrame(rows)
    if d.empty:
        raise ValueError("memory_df does not contain matching global/support/episode memory rows")
    plt = _plt()
    setup_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, metric in zip(axes, ["episode-global", "episode-support"]):
        sub = d[d["metric"] == metric]
        for pos, cls in enumerate(classes):
            group = sub[sub["class"].astype(str) == str(cls)].sort_values("k")
            if group.empty:
                continue
            ax.plot(group["k"], group["distance"], marker="o", color=cmap(pos % 10), label=cls)
        ax.set_title(metric)
        ax.set_xlabel("Support shots per class")
        ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    axes[0].set_ylabel("L2 distance")
    axes[-1].legend(frameon=False, ncol=3, loc="best")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def compute_gema_adaptation_ratio(
    memory_df: pd.DataFrame,
    k_col: str = "k",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Compute rho = ||m_e - m_g|| / (||u - m_g|| + eps)."""

    _require_columns(memory_df, ["class", "memory_type", k_col], "memory_df")
    feature_cols = [c for c in memory_df.columns if str(c).startswith("f")]
    if not feature_cols:
        raise ValueError("memory_df must contain embedding columns named f0, f1, ...")
    rows = []
    for keys, group in memory_df.groupby([k_col, "region", "episode", "class"], dropna=False):
        k, region, episode, cls = keys
        vectors = {str(row["memory_type"]): row[feature_cols].to_numpy(dtype=float) for _, row in group.iterrows()}
        global_vec = vectors.get("global_memory") if "global_memory" in vectors else vectors.get("m_g")
        support_vec = vectors.get("support_candidate") if "support_candidate" in vectors else vectors.get("u")
        episode_vec = vectors.get("episode_memory") if "episode_memory" in vectors else vectors.get("m_e")
        if global_vec is None or support_vec is None or episode_vec is None:
            continue
        denom = float(np.linalg.norm(support_vec - global_vec)) + float(eps)
        rows.append(
            {
                "k": int(k),
                "region": str(region),
                "episode": int(episode),
                "class": str(cls),
                "adaptation_ratio": float(np.linalg.norm(episode_vec - global_vec) / denom),
            }
        )
    return pd.DataFrame(rows)


def plot_gema_adaptation_ratio_boxplot(
    memory_df: pd.DataFrame,
    k_col: str = "k",
    title: str = "GEMA adaptation ratio",
):
    """Plot adaptation ratio rho by K-shot setting."""

    d = compute_gema_adaptation_ratio(memory_df, k_col=k_col)
    if d.empty:
        raise ValueError("memory_df does not contain matching global/support/episode memory rows")
    plt = _plt()
    setup_paper_style()
    ks = sorted(d["k"].unique().tolist())
    groups = [d.loc[d["k"] == k, "adaptation_ratio"].dropna().to_numpy(dtype=float) for k in ks]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bp = ax.boxplot(groups, labels=[f"{k}-shot" for k in ks], patch_artist=True, widths=0.58, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["episode"])
        patch.set_alpha(0.55)
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax.axhline(1.0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
    ax.set_ylabel("Adaptation ratio rho")
    ax.set_xlabel("Support setting")
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    return fig


def plot_pcmr_decomposition(
    pcmr_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    component_order: Sequence[str] = ("acoustic_score", "prompt_bias", "final_attention"),
    component_titles: Mapping[str, str] | None = None,
    query_label_col: str = "true_class",
    title: str = "Decomposition of prompt-conditioned memory reading",
):
    """Plot acoustic compatibility, prompt bias, and final attention heatmaps."""

    _require_columns(pcmr_df, ["query_id", query_label_col, "class", "component", "value"], "pcmr_df")
    plt = _plt()
    setup_paper_style()
    titles = {
        "acoustic_score": "Acoustic compatibility",
        "prompt_bias": "Prompt-conditioned bias",
        "final_attention": "Final attention",
    }
    if component_titles:
        titles.update(component_titles)
    classes = _ordered_existing(pcmr_df["class"], class_order)
    query_order = (
        pcmr_df[["query_id", query_label_col]]
        .drop_duplicates()
        .sort_values([query_label_col, "query_id"])["query_id"]
        .tolist()
    )
    y_labels = (
        pcmr_df[["query_id", query_label_col]]
        .drop_duplicates()
        .set_index("query_id")
        .reindex(query_order)[query_label_col]
        .astype(str)
        .tolist()
    )

    fig, axes = plt.subplots(1, len(component_order), figsize=(3.2 * len(component_order), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, component in zip(axes, component_order):
        comp = pcmr_df[pcmr_df["component"] == component]
        pivot = comp.pivot_table(index="query_id", columns="class", values="value", aggfunc="mean").reindex(index=query_order, columns=classes)
        values = pivot.to_numpy(dtype=float)
        if component == "final_attention":
            im = ax.imshow(values, cmap="magma", aspect="auto", vmin=0.0, vmax=max(1e-8, np.nanmax(values)))
        else:
            lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
            im = ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim)
        ax.set_title(titles.get(component, component))
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_yticks(np.arange(len(query_order)))
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel("Query samples grouped by true class")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_pcmr_prompt_shift_heatmap(
    pcmr_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    true_col: str = "true_class",
    title: str = "Prompt-induced attention shift in PCMR",
):
    """Plot mean Delta beta = beta_with_prompt - beta_without_prompt."""

    _require_columns(pcmr_df, [true_col, "class", "component", "value"], "pcmr_df")
    d = pcmr_df[pcmr_df["component"] == "attention_shift"]
    if d.empty:
        raise ValueError("pcmr_df must contain component='attention_shift'")
    plt = _plt()
    setup_paper_style()
    classes = _ordered_existing(pd.concat([d[true_col].astype(str), d["class"].astype(str)]), class_order)
    pivot = d.pivot_table(index=true_col, columns="class", values="value", aggfunc="mean").reindex(index=classes, columns=classes)
    values = pivot.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Candidate memory class")
    ax.set_ylabel("True query class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean attention shift")
    return fig


def plot_pcmr_true_class_attention(
    pcmr_df: pd.DataFrame,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    true_col: str = "true_class",
    title: str = "True-class attention before and after prompt",
):
    """Grouped bar plot for ground-truth class attention before/after prompt."""

    _require_columns(pcmr_df, ["query_id", true_col, "class", "component", "value"], "pcmr_df")
    d = pcmr_df[pcmr_df["component"].isin(["attention_without_prompt", "attention_with_prompt"])].copy()
    d = d[d[true_col].astype(str) == d["class"].astype(str)]
    if d.empty:
        raise ValueError("pcmr_df must contain true-class attention rows before and after prompt")
    classes = _ordered_existing(d[true_col], class_order)
    pivot = d.pivot_table(index=true_col, columns="component", values="value", aggfunc="mean").reindex(index=classes).fillna(0.0)
    plt = _plt()
    setup_paper_style()
    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - width / 2, pivot.get("attention_without_prompt", pd.Series(0, index=pivot.index)), width, label="without prompt", color=COLORS["gray"], alpha=0.75)
    ax.bar(x + width / 2, pivot.get("attention_with_prompt", pd.Series(0, index=pivot.index)), width, label="with prompt", color=COLORS["global"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Attention on true class")
    ax.set_xlabel("True vowel class")
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    ax.legend(frameon=False)
    return fig


def plot_prompt_usage_bar(
    routing_df: pd.DataFrame,
    prompt_col: str = "prompt",
    alpha_col: str = "alpha",
    title: str = "Average prompt usage",
):
    """Plot average routing weight for each prompt."""

    _require_columns(routing_df, [prompt_col, alpha_col], "routing_df")
    plt = _plt()
    setup_paper_style()
    usage = routing_df.groupby(prompt_col, as_index=False)[alpha_col].mean().sort_values(prompt_col)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(np.arange(len(usage)), usage[alpha_col], color=COLORS["global"], width=0.66)
    ax.set_xticks(np.arange(len(usage)))
    ax.set_xticklabels([str(v) for v in usage[prompt_col]])
    ax.set_ylim(0, max(0.05, min(1.0, usage[alpha_col].max() * 1.2)))
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("Average routing weight")
    ax.set_title(title)
    return fig


def plot_prompt_usage_region_heatmap(
    routing_df: pd.DataFrame,
    region_order: Sequence[str] | None = DEFAULT_REGION_ORDER,
    prompt_col: str = "prompt",
    alpha_col: str = "alpha",
    title: str = "Region-wise prompt routing distribution",
):
    """Plot mean prompt routing weight by target region."""

    _require_columns(routing_df, ["region", prompt_col, alpha_col], "routing_df")
    plt = _plt()
    setup_paper_style()
    regions = _ordered_existing(routing_df["region"], region_order)
    prompts = sorted(routing_df[prompt_col].dropna().unique().tolist())
    pivot = routing_df.pivot_table(index="region", columns=prompt_col, values=alpha_col, aggfunc="mean").reindex(index=regions, columns=prompts)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=max(1e-8, np.nanmax(pivot.to_numpy(dtype=float))), aspect="auto")
    ax.set_xticks(np.arange(len(prompts)))
    ax.set_xticklabels([str(p) for p in prompts])
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("Target region")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean routing weight")
    return fig


def compute_prompt_entropy(
    routing_df: pd.DataFrame,
    sample_col: str = "sample_id",
    prompt_col: str = "prompt",
    alpha_col: str = "alpha",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Return per-sample prompt entropy from long-form routing weights."""

    _require_columns(routing_df, [sample_col, prompt_col, alpha_col], "routing_df")
    meta_cols = [c for c in ["correct", "region", "true_class", "pred_class", "k"] if c in routing_df.columns]
    alpha = routing_df[[sample_col, prompt_col, alpha_col] + meta_cols].copy()
    alpha[alpha_col] = alpha[alpha_col].clip(lower=eps)
    alpha["_entropy_term"] = -(alpha[alpha_col] * np.log(alpha[alpha_col]))
    ent = alpha.groupby(sample_col, as_index=False)["_entropy_term"].sum().rename(columns={"_entropy_term": "entropy"})
    if meta_cols:
        meta = routing_df[[sample_col] + meta_cols].drop_duplicates(sample_col)
        ent = ent.merge(meta, on=sample_col, how="left")
    return ent


def plot_prompt_entropy_boxplot(
    routing_or_entropy_df: pd.DataFrame,
    entropy_col: str = "entropy",
    correct_col: str = "correct",
    title: str = "Prompt entropy for correct and incorrect samples",
):
    """Plot prompt routing entropy for correct vs. error samples."""

    if entropy_col not in routing_or_entropy_df.columns:
        entropy_df = compute_prompt_entropy(routing_or_entropy_df)
    else:
        entropy_df = routing_or_entropy_df.copy()
    _require_columns(entropy_df, [entropy_col, correct_col], "entropy_df")
    plt = _plt()
    setup_paper_style()
    groups = [
        entropy_df.loc[entropy_df[correct_col].astype(bool), entropy_col].dropna().to_numpy(dtype=float),
        entropy_df.loc[~entropy_df[correct_col].astype(bool), entropy_col].dropna().to_numpy(dtype=float),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bp = ax.boxplot(groups, labels=["Correct", "Error"], patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(bp["boxes"], [COLORS["episode"], COLORS["accent"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("Prompt routing entropy")
    ax.set_title(title)
    return fig


def _confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> np.ndarray:
    label_to_i = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in label_to_i and p in label_to_i:
            cm[label_to_i[t], label_to_i[p]] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.maximum(row_sum, 1.0), out=np.zeros_like(cm), where=row_sum > 0)


def plot_confusion_matrix_comparison(
    y_true: Sequence,
    baseline_pred: Sequence,
    ours_pred: Sequence,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    normalize_percent: bool = True,
    titles: Sequence[str] = ("Baseline", "PC-DLCMNet", "Delta"),
):
    """Plot baseline, PC-DLCMNet, and normalized confusion delta matrices."""

    plt = _plt()
    setup_paper_style()
    labels = _ordered_existing(list(y_true) + list(baseline_pred) + list(ours_pred), class_order)
    base = _confusion_matrix(y_true, baseline_pred, labels)
    ours = _confusion_matrix(y_true, ours_pred, labels)
    delta = ours - base
    mats = [base, ours, delta]
    if normalize_percent:
        mats = [m * 100.0 for m in mats]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharex=True, sharey=True)
    for ax, mat, title in zip(axes, mats, titles):
        if title == titles[2]:
            lim = float(np.max(np.abs(mat))) or 1.0
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            label = "Delta" + (" (%)" if normalize_percent else "")
        else:
            im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=100.0 if normalize_percent else 1.0, aspect="auto")
            label = "Row-normalized confusion" + (" (%)" if normalize_percent else "")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    axes[0].set_ylabel("True class")
    for ax in axes:
        ax.set_xlabel("Predicted class")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_delta(
    y_true: Sequence,
    baseline_pred: Sequence,
    ours_pred: Sequence,
    class_order: Sequence[str] | None = DEFAULT_VOWEL_ORDER,
    normalize_percent: bool = True,
    title: str = "Error reduction over the strongest baseline",
):
    """Plot only Delta CM = CM_ours - CM_baseline."""

    plt = _plt()
    setup_paper_style()
    labels = _ordered_existing(list(y_true) + list(baseline_pred) + list(ours_pred), class_order)
    base = _confusion_matrix(y_true, baseline_pred, labels)
    ours = _confusion_matrix(y_true, ours_pred, labels)
    delta = ours - base
    if normalize_percent:
        delta = delta * 100.0
    lim = float(np.max(np.abs(delta))) or 1.0
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted vowel")
    ax.set_ylabel("True vowel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Delta normalized confusion" + (" (%)" if normalize_percent else ""))
    return fig


def plot_support_corruption_curve(
    corruption_df: pd.DataFrame,
    noise_col: str = "noise_ratio",
    metric_col: str = "macro_f1",
    method_col: str = "method",
    title: str = "Support corruption analysis",
):
    """Plot Macro-F1 under increasing support corruption ratios."""

    _require_columns(corruption_df, [noise_col, metric_col, method_col], "corruption_df")
    plt = _plt()
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    methods = list(dict.fromkeys(corruption_df[method_col].astype(str).tolist()))
    for pos, method in enumerate(methods):
        group = corruption_df[corruption_df[method_col].astype(str) == method].sort_values(noise_col)
        y = _normalize_percent(group[metric_col])
        x = _normalize_percent(group[noise_col])
        ax.plot(x, y, marker="o", label=method, color=list(COLORS.values())[pos % len(COLORS)])
    ax.set_xlabel("Support corruption ratio (%)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    return fig
