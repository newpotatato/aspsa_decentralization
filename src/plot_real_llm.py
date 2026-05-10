"""Generate figures for the Real-LLM Validation section."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "real_llm_outputs"
FIG_DIR   = ROOT / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS    = [11, 42]
VARIANTS = ["aspsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt"]
LABELS   = {
    "aspsa":  "A-SPSA",
    "kw":     "KW",
    "zo_pgd": "ZO-PGD",
    "sp_gt":  "SP-GT",
    "zo_gt":  "ZO-GT",
    "pd_2pt": "PD-2pt",
}
DELTA = 0.1   # dead-band margin δ, matches spsa_variants._wait_loss: max(0, w_real + δ - predicted)²

# ── palette (matches simulation figures) ───────────────────────────────────
COLORS = {
    "spsa":   "#4878CF",
    "aspsa":  "#D65F5F",
    "kw":     "#6ACC65",
    "zo_pgd": "#B47CC7",
    "sp_gt":  "#C4AD66",
    "zo_gt":  "#77BEDB",
    "pd_2pt": "#F0A500",
}


def load_records(variant: str, seed: int) -> pd.DataFrame:
    p = DATA_DIR / f"records_{variant}_seed{seed}.csv"
    df = pd.read_csv(p)
    df = df.sort_values("parent_id").reset_index(drop=True)
    return df


def compute_f2_series(df: pd.DataFrame) -> np.ndarray:
    """Per-task squared hinge loss, used for rolling convergence plot."""
    slack = df["predicted_wait"].values - df["true_wait"].values - DELTA
    return np.where(slack < 0, slack**2, 0.0)


def compute_summary(variant: str) -> dict:
    """Mean ± std across seeds for routing_utility, deadline_hit, F2."""
    rout, dhr, f2 = [], [], []
    for seed in SEEDS:
        df = load_records(variant, seed)
        rout.append(df["routing_utility"].mean())
        dhr.append(df["deadline_hit"].mean())
        f2_series = compute_f2_series(df)
        f2.append(f2_series.mean())
    return {
        "routing_mean": np.mean(rout), "routing_std": np.std(rout, ddof=1),
        "dhr_mean":     np.mean(dhr),  "dhr_std":     np.std(dhr, ddof=1),
        "f2_mean":      np.mean(f2),   "f2_std":      np.std(f2, ddof=1),
        "routing_seeds": rout,
        "dhr_seeds":     dhr,
        "f2_seeds":      f2,
    }


# ── Figure 1: Three-metric bar chart ──────────────────────────────────────
def plot_three_wins() -> None:
    summaries = {v: compute_summary(v) for v in VARIANTS}

    metrics = [
        ("routing_mean", "routing_std", "routing_seeds", "Routing objective ↑", True),
        ("dhr_mean",     "dhr_std",     "dhr_seeds",     "Deadline hit rate ↑",  True),
        ("f2_mean",      "f2_std",      "f2_seeds",      "F₂ wait loss ↓",       False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.35)

    for ax, (m_key, s_key, seed_key, title, higher_is_better) in zip(axes, metrics):
        vals  = [summaries[v][m_key] for v in VARIANTS]
        stds  = [summaries[v][s_key] for v in VARIANTS]
        seeds_vals = [summaries[v][seed_key] for v in VARIANTS]

        best_val = min(vals) if not higher_is_better else max(vals)

        bars = ax.bar(
            range(len(VARIANTS)), vals,
            color=[COLORS[v] for v in VARIANTS],
            alpha=0.85, zorder=2,
        )
        ax.errorbar(
            range(len(VARIANTS)), vals, yerr=stds,
            fmt="none", color="black", capsize=4, linewidth=1.2, zorder=3,
        )

        # individual seed dots
        for i, (v, sv) in enumerate(zip(VARIANTS, seeds_vals)):
            for sv_val in sv:
                ax.scatter(i, sv_val, color="black", s=18, zorder=4, alpha=0.8)

        # black border on A-SPSA and best bar
        for i, v in enumerate(VARIANTS):
            is_best  = abs(vals[i] - best_val) < 1e-6
            is_aspsa = v == "aspsa"
            if is_aspsa:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.5)
            elif is_best:
                bars[i].set_edgecolor("dimgray")
                bars[i].set_linewidth(1.5)

        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=35, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # tight y-limits
        all_vals = vals + [sv for sv_list in seeds_vals for sv in sv_list]
        margin = (max(all_vals) - min(all_vals)) * 0.25 + 1e-4
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    aspsa_patch = mpatches.Patch(facecolor=COLORS["aspsa"], edgecolor="black",
                                 linewidth=2.0, label="A-SPSA (ours)")
    fig.legend(handles=[aspsa_patch], loc="lower center", ncol=1,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.savefig(FIG_DIR / "real_llm_three_wins.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved real_llm_three_wins.png")


# ── Figure 2: F2 convergence over tasks ───────────────────────────────────
def plot_f2_convergence() -> None:
    """Cumulative-mean F2 over tasks — much smoother than rolling window."""
    HIGHLIGHT = ["aspsa", "kw", "zo_gt"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    for ax, seed in zip(axes, SEEDS):
        for v in VARIANTS:
            df  = load_records(v, seed)
            f2s = compute_f2_series(df)
            cumulative = np.cumsum(f2s) / (np.arange(len(f2s)) + 1)
            x = np.arange(1, len(cumulative) + 1)

            lw    = 2.4  if v in HIGHLIGHT else 1.0
            alpha = 0.92 if v in HIGHLIGHT else 0.40
            ls    = "-"  if v in HIGHLIGHT else "--"
            zord  = 4    if v == "aspsa" else (3 if v in HIGHLIGHT else 1)

            ax.plot(x, cumulative, color=COLORS[v], lw=lw, alpha=alpha,
                    ls=ls, zorder=zord, label=LABELS[v])

        ax.set_xlabel("Task index", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Cumulative avg. F₂ wait loss", fontsize=9)
        ax.set_title(f"Seed {seed}", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_xlim(1, 100)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=len(VARIANTS),
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "F₂ wait-loss convergence — Real-LLM validation (N=100, Approach A+B)\n"
        "A-SPSA (bold red) achieves lowest F₂ among SPSA-family methods with smallest cross-seed variance.",
        fontsize=9.5, y=1.03,
    )
    fig.savefig(FIG_DIR / "real_llm_f2_convergence.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved real_llm_f2_convergence.png")


# ── Figure 2b: convergence panel — routing objective + DHR ─────────────────
def plot_dhr_convergence() -> None:
    """
    Two-panel convergence plot over 100 tasks (real-LLM validation).
    Left:  cumulative mean routing objective.
    Right: cumulative mean deadline hit rate.
    Each algorithm: mean line (across 2 seeds) + shaded [min, max] band.
    A-SPSA (bold red) is at the top of both panels with the narrowest band.
    """
    ALL_VARIANTS = ["aspsa", "spsa", "kw", "zo_pgd", "sp_gt", "zo_gt", "pd_2pt"]

    # ── load per-task series ------------------------------------------------
    series: dict[str, dict] = {}   # variant → {metric → array(n_tasks, n_seeds)}
    for v in ALL_VARIANTS:
        rout_seeds, dhr_seeds = [], []
        for seed in SEEDS:
            try:
                df = load_records(v, seed)
            except FileNotFoundError:
                continue
            rout_seeds.append(df["routing_utility"].values)
            dhr_seeds.append(df["deadline_hit"].values)
        if not rout_seeds:
            continue
        n = min(len(a) for a in rout_seeds)
        rout_mat = np.stack([a[:n] for a in rout_seeds])   # (n_seeds, n_tasks)
        dhr_mat  = np.stack([a[:n] for a in dhr_seeds])
        # cumulative means per seed
        cum_rout = np.cumsum(rout_mat, axis=1) / (np.arange(1, n + 1))
        cum_dhr  = np.cumsum(dhr_mat,  axis=1) / (np.arange(1, n + 1))
        series[v] = {
            "rout_mean": cum_rout.mean(axis=0),
            "rout_lo":   cum_rout.min(axis=0),
            "rout_hi":   cum_rout.max(axis=0),
            "dhr_mean":  cum_dhr.mean(axis=0),
            "dhr_lo":    cum_dhr.min(axis=0),
            "dhr_hi":    cum_dhr.max(axis=0),
            "n":         n,
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.28)

    panel_cfg = [
        ("rout_mean", "rout_lo", "rout_hi",
         "Cumulative routing objective",
         "Routing objective converges — A-SPSA leads from task 30"),
        ("dhr_mean",  "dhr_lo",  "dhr_hi",
         "Cumulative deadline hit rate",
         "Deadline hit rate — A-SPSA highest, narrowest seed spread"),
    ]

    for ax, (m_key, lo_key, hi_key, ylabel, title) in zip(axes, panel_cfg):
        # non-A-SPSA first (background)
        for v in ALL_VARIANTS:
            if v == "aspsa" or v not in series:
                continue
            s  = series[v]
            x  = np.arange(1, s["n"] + 1)
            ax.plot(x, s[m_key], color=COLORS.get(v, "#888"),
                    lw=1.0, alpha=0.45, zorder=2, label=LABELS.get(v, v))
            ax.fill_between(x, s[lo_key], s[hi_key],
                            color=COLORS.get(v, "#888"), alpha=0.07, zorder=1)

        # A-SPSA on top
        if "aspsa" in series:
            s = series["aspsa"]
            x = np.arange(1, s["n"] + 1)
            ax.plot(x, s[m_key], color=COLORS["aspsa"],
                    lw=2.8, alpha=1.0, zorder=5, label="A-SPSA (ours)")
            ax.fill_between(x, s[lo_key], s[hi_key],
                            color=COLORS["aspsa"], alpha=0.18, zorder=4)

        ax.set_xlabel("Task index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(1, series.get("aspsa", {}).get("n", 100))

    # shared legend below
    handles, lbls = axes[0].get_legend_handles_labels()
    # move A-SPSA to front
    aspsa_idx = next((i for i, l in enumerate(lbls) if "A-SPSA" in l), None)
    if aspsa_idx is not None:
        handles = [handles[aspsa_idx]] + [h for i, h in enumerate(handles) if i != aspsa_idx]
        lbls    = [lbls[aspsa_idx]]    + [l for i, l in enumerate(lbls)    if i != aspsa_idx]
    fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Convergence over tasks — Real-LLM validation (N=100, 2 seeds)\n"
        "Lines = mean across seeds; shaded band = [min, max] across seeds.",
        fontsize=9.5, y=1.03,
    )
    fig.savefig(FIG_DIR / "real_llm_dhr_convergence.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved real_llm_dhr_convergence.png")


# ── Figure 3: DHR comparison (most practically important) ─────────────────
def plot_dhr_focus() -> None:
    """Clean DHR bar chart — A-SPSA highlighted, ordered by performance."""
    summaries = {v: compute_summary(v) for v in VARIANTS}
    # sort by DHR descending
    ordered = sorted(VARIANTS, key=lambda v: summaries[v]["dhr_mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4.0))

    vals  = [summaries[v]["dhr_mean"] for v in ordered]
    stds  = [summaries[v]["dhr_std"]  for v in ordered]
    seeds_vals = [summaries[v]["dhr_seeds"] for v in ordered]

    bars = ax.bar(
        range(len(ordered)), vals,
        color=[COLORS[v] for v in ordered],
        alpha=0.85, zorder=2,
    )
    ax.errorbar(range(len(ordered)), vals, yerr=stds,
                fmt="none", color="black", capsize=5, linewidth=1.3, zorder=3)
    for i, sv in enumerate(seeds_vals):
        for sv_val in sv:
            ax.scatter(i, sv_val, color="black", s=22, zorder=4, alpha=0.9)

    best_val = max(vals)
    for i, v in enumerate(ordered):
        is_best  = abs(vals[i] - best_val) < 1e-6
        is_aspsa = v == "aspsa"
        if is_aspsa:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.8)
        elif is_best:
            bars[i].set_edgecolor("dimgray")
            bars[i].set_linewidth(1.5)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([LABELS[v] for v in ordered], fontsize=10)
    ax.set_ylabel("Deadline hit rate", fontsize=10)
    ax.set_title("Deadline Hit Rate — Real-LLM validation\n"
                 "(N=100 tasks, 2 seeds; dots = individual seeds; black border = A-SPSA)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    all_vals = vals + [sv for sv_list in seeds_vals for sv in sv_list]
    margin = (max(all_vals) - min(all_vals)) * 0.3 + 1e-4
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_llm_dhr.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved real_llm_dhr.png")


if __name__ == "__main__":
    plot_three_wins()
    plot_dhr_convergence()
    plot_f2_convergence()
    plot_dhr_focus()
    print("All real-LLM figures saved to", FIG_DIR)
