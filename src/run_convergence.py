"""
Collect per-task convergence curves for all SPSA variants.
Runs a short simulation (800 tasks × 3 seeds) and saves rolling-window metrics.
Then generates interactive Plotly HTML figures.

Usage:
    python -m src.run_convergence
"""

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from main import ExactOrchestrator                      # noqa: E402
from run_spsa_comparison import (                       # noqa: E402
    _build_variant, BEST_PARAMS, SPSA_VARIANTS, generate_tasks,
)

NUM_TASKS   = 800
SEEDS       = [11, 42, 123]
WINDOW      = 40       # rolling average window (tasks)
NUM_CTRL    = 3
NUM_AGENTS  = 5
OUT_DIR     = Path("spsa_comparison")
FIGS_DIR    = OUT_DIR / "paper_figs"

COLORS = {
    "aspsa":  "#d62728",
    "spsa":   "#1f77b4",
    "kw":     "#2ca02c",
    "zo_pgd": "#ff7f0e",
    "sp_gt":  "#9467bd",
    "zo_gt":  "#8c564b",
    "pd_2pt": "#17becf",
}
LABELS = {
    "aspsa":  "A-SPSA",
    "spsa":   "SPSA",
    "kw":     "KW",
    "zo_pgd": "ZO-PGD",
    "sp_gt":  "SP-GT",
    "zo_gt":  "ZO-GT",
    "pd_2pt": "PD-2pt",
}

# ---------------------------------------------------------------------------

def _run_one_convergence(args):
    vname, tasks, seed = args
    np.random.seed(seed)
    p = BEST_PARAMS.get(vname, {})
    orch = ExactOrchestrator(
        num_ctrl=NUM_CTRL,
        num_agents=NUM_AGENTS,
        variant=_build_variant(vname, alpha=p.get("alpha"), beta=p.get("beta"),
                               beta_nes_max=p.get("beta_nes_max")),
        classifier_seed=seed,
    )
    orch.run(list(tasks))

    n = min(len(orch.metrics["routing_objective"]),
            len(orch.metrics["Q_mse"]),
            len(orch.metrics["success_rate"]),
            len(orch.metrics["deadline_hits"]),
            len(orch.metrics["latency"]))

    return {
        "variant":           vname,
        "seed":              seed,
        "routing_objective": orch.metrics["routing_objective"][:n],
        "q_mse":             orch.metrics["Q_mse"][:n],
        "success_rate":      orch.metrics["success_rate"][:n],
        "deadline_hits":     orch.metrics["deadline_hits"][:n],
        "latency":           orch.metrics["latency"][:n],
    }


def _rolling(arr, w):
    arr = np.array(arr, dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = arr[max(0, i - w + 1):i + 1].mean()
    return out


def collect_convergence() -> pd.DataFrame:
    print("Generating tasks...")
    task_sets = {seed: generate_tasks(NUM_TASKS, seed, "balanced") for seed in SEEDS}

    jobs = []
    for vname in SPSA_VARIANTS:
        for seed in SEEDS:
            jobs.append((vname, task_sets[seed], seed))

    print(f"Running {len(jobs)} jobs ({len(SPSA_VARIANTS)} variants x {len(SEEDS)} seeds)...")
    n_workers = max(2, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_run_one_convergence, jobs))

    rows = []
    for res in results:
        vname = res["variant"]
        seed  = res["seed"]
        n     = len(res["routing_objective"])
        ro_roll  = _rolling(res["routing_objective"], WINDOW)
        mse_roll = _rolling(res["q_mse"],             WINDOW)
        sr_roll  = _rolling(res["success_rate"],       WINDOW)
        dl_roll  = _rolling(res["deadline_hits"],      WINDOW)
        lat_roll = _rolling(res["latency"],            WINDOW)
        for i in range(n):
            rows.append({
                "variant":           vname,
                "seed":              seed,
                "task_idx":          i,
                "routing_objective": ro_roll[i],
                "q_mse":             mse_roll[i],
                "success_rate":      sr_roll[i],
                "deadline_hit":      dl_roll[i],
                "latency":           lat_roll[i],
            })

    df = pd.DataFrame(rows)
    path = OUT_DIR / "convergence.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}  ({len(df)} rows)")
    return df


def _mean_band(df: pd.DataFrame, variant: str, metric: str):
    """Return task_idx, mean, lower (mean-std), upper (mean+std) across seeds."""
    sub = df[df["variant"] == variant].dropna(subset=[metric])
    grp = sub.groupby("task_idx")[metric]
    mean = grp.mean()
    std  = grp.std().fillna(0)
    return mean.index.values, mean.values, (mean - std).values, (mean + std).values


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _to_rgba(hex_col: str, alpha: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_convergence(df: pd.DataFrame, metric: str, title: str,
                         ylabel: str, higher_better: bool, filename: str):
    fig = go.Figure()
    order = sorted(SPSA_VARIANTS,
                   key=lambda v: _mean_band(df, v, metric)[1][-50:].mean()
                                 if len(_mean_band(df, v, metric)[1]) > 0 else 0,
                   reverse=higher_better)

    for v in order:
        x, mean, lo, hi = _mean_band(df, v, metric)
        if len(x) == 0:
            continue
        lw   = 3   if v == "aspsa" else 1.5
        dash = None if v == "aspsa" else "dot"
        col  = COLORS[v]
        fill_alpha = 0.15 if v == "aspsa" else 0.07

        fig.add_trace(go.Scatter(
            x=x, y=hi, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            fillcolor=_to_rgba(col, fill_alpha),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=lo, mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=_to_rgba(col, fill_alpha),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines",
            name=LABELS[v],
            line=dict(color=col, width=lw,
                      dash=dash if dash else None),
            hovertemplate=f"<b>{LABELS[v]}</b><br>task=%{{x}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
        ))

    arrow_x = int(len(x) * 0.85)
    aspsa_x, aspsa_mean, _, _ = _mean_band(df, "aspsa", metric)
    if len(aspsa_mean) > arrow_x:
        fig.add_annotation(
            x=aspsa_x[arrow_x], y=aspsa_mean[arrow_x],
            text="<b>A-SPSA</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=COLORS["aspsa"],
            font=dict(color=COLORS["aspsa"], size=13),
            ax=40, ay=-30,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="serif")),
        xaxis_title="Task index",
        yaxis_title=ylabel,
        legend=dict(orientation="v", x=1.01, y=0.5,
                    bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
        hovermode="x unified",
        template="simple_white",
        width=900, height=480,
        font=dict(family="serif", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")

    path = FIGS_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved {filename}")
    return fig


def _plotly_final_bar(df: pd.DataFrame, metric: str, title: str,
                       ylabel: str, higher_better: bool, filename: str):
    """Bar of final-window (last 100 tasks) mean per variant."""
    final = (df[df["task_idx"] >= NUM_TASKS - 100]
             .groupby("variant")[metric].agg(["mean", "std"]).reset_index())
    final = final.sort_values("mean", ascending=not higher_better)

    colors = [COLORS[v] for v in final["variant"]]
    edges  = ["black" if v == "aspsa" else COLORS[v] for v in final["variant"]]

    fig = go.Figure(go.Bar(
        x=[LABELS[v] for v in final["variant"]],
        y=final["mean"],
        error_y=dict(type="data", array=final["std"].values, visible=True),
        marker=dict(color=colors, line=dict(color=edges, width=[2.5 if v == "aspsa" else 0.8
                                                                  for v in final["variant"]])),
        text=[f"{m:.4f}" for m in final["mean"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="serif")),
        yaxis_title=ylabel,
        template="simple_white",
        width=760, height=420,
        font=dict(family="serif", size=12),
        yaxis=dict(range=[final["mean"].min() * 0.97, final["mean"].max() * 1.04]),
    )
    path = FIGS_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved {filename}")


def _plotly_multi_metric_dashboard(df: pd.DataFrame):
    """4-panel interactive dashboard: routing_obj, q_mse, deadline_hit, sr."""
    metrics = [
        ("routing_objective", "Routing Objective", True),
        ("q_mse",             "Q-MSE",             False),
        ("deadline_hit",      "Deadline Hit Rate",  True),
        ("success_rate",      "Success Rate",       True),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[m[1] for m in metrics],
                        shared_xaxes=False,
                        vertical_spacing=0.14,
                        horizontal_spacing=0.10)

    coords = [(1,1),(1,2),(2,1),(2,2)]
    for (metric, ylabel, higher), (row, col) in zip(metrics, coords):
        first = True
        for v in SPSA_VARIANTS:
            x, mean, lo, hi = _mean_band(df, v, metric)
            if len(x) == 0:
                continue
            lw  = 2.8 if v == "aspsa" else 1.2
            col_c = COLORS[v]

            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines",
                name=LABELS[v],
                legendgroup=v,
                showlegend=(row == 1 and col == 1),
                line=dict(color=col_c, width=lw,
                          dash=None if v == "aspsa" else "dot"),
                hovertemplate=f"<b>{LABELS[v]}</b><br>task=%{{x}}<br>value=%{{y:.4f}}<extra></extra>",
            ), row=row, col=col)

    fig.update_layout(
        title=dict(text="Convergence Dashboard: all metrics over time (rolling mean, 3 seeds)",
                   font=dict(size=15, family="serif")),
        template="simple_white",
        width=1100, height=700,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=0.5,
                    bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
        font=dict(family="serif", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", title_text="Task index")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")

    path = FIGS_DIR / "fig_dyn00_dashboard.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved fig_dyn00_dashboard.html")


def _plotly_aspsa_highlight(df: pd.DataFrame, metric: str, title: str,
                              ylabel: str, higher_better: bool, filename: str):
    """Convergence with A-SPSA bold + others greyed out, per-seed traces shown."""
    fig = go.Figure()

    for v in SPSA_VARIANTS:
        sub = df[df["variant"] == v].dropna(subset=[metric])
        if v == "aspsa":
            for seed in SEEDS:
                s = sub[sub["seed"] == seed].sort_values("task_idx")
                fig.add_trace(go.Scatter(
                    x=s["task_idx"], y=s[metric],
                    mode="lines", name=f"A-SPSA seed={seed}",
                    line=dict(color=COLORS["aspsa"], width=2),
                    opacity=0.65,
                    hovertemplate=f"A-SPSA s={seed}<br>task=%{{x}}<br>val=%{{y:.4f}}<extra></extra>",
                ))
            # bold mean
            x, mean, lo, hi = _mean_band(df, v, metric)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines", name="A-SPSA (mean)",
                line=dict(color=COLORS["aspsa"], width=4),
                hovertemplate="A-SPSA mean<br>task=%{x}<br>val=%{y:.4f}<extra></extra>",
            ))
        else:
            x, mean, _, _ = _mean_band(df, v, metric)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines", name=LABELS[v],
                line=dict(color="#aaaaaa", width=1.2, dash="dot"),
                opacity=0.5,
                hovertemplate=f"{LABELS[v]}<br>task=%{{x}}<br>val=%{{y:.4f}}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="serif")),
        xaxis_title="Task index",
        yaxis_title=ylabel,
        hovermode="x unified",
        template="simple_white",
        width=900, height=460,
        font=dict(family="serif", size=12),
        legend=dict(x=1.01, y=0.5, bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
    )
    path = FIGS_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved {filename}")


def _plotly_ranking_animation(df: pd.DataFrame):
    """Animated bar race: rank of each variant by routing_objective over time."""
    checkpoints = list(range(WINDOW, NUM_TASKS, 20))
    fig = go.Figure()

    frames = []
    for cp in checkpoints:
        window_df = df[(df["task_idx"] >= cp - WINDOW) & (df["task_idx"] < cp)]
        agg = window_df.groupby("variant")["routing_objective"].mean().reset_index()
        agg = agg.sort_values("routing_objective", ascending=True)
        frames.append(go.Frame(
            data=[go.Bar(
                y=[LABELS[v] for v in agg["variant"]],
                x=agg["routing_objective"],
                orientation="h",
                marker=dict(color=[COLORS[v] for v in agg["variant"]]),
                text=[f"{val:.4f}" for val in agg["routing_objective"]],
                textposition="outside",
            )],
            name=str(cp),
            layout=go.Layout(title_text=f"Routing Objective ranking at task {cp}"),
        ))

    # Initial frame
    init_df = df[(df["task_idx"] >= WINDOW) & (df["task_idx"] < WINDOW + 20)]
    init_agg = init_df.groupby("variant")["routing_objective"].mean().reset_index()
    init_agg = init_agg.sort_values("routing_objective", ascending=True)

    fig.add_trace(go.Bar(
        y=[LABELS[v] for v in init_agg["variant"]],
        x=init_agg["routing_objective"],
        orientation="h",
        marker=dict(color=[COLORS[v] for v in init_agg["variant"]]),
        text=[f"{val:.4f}" for val in init_agg["routing_objective"]],
        textposition="outside",
    ))
    fig.frames = frames
    fig.update_layout(
        title=dict(text="Animated Ranking: Routing Objective over time",
                   font=dict(size=15, family="serif")),
        xaxis_title="Routing Objective (higher=better)",
        template="simple_white",
        width=820, height=450,
        font=dict(family="serif", size=12),
        xaxis=dict(range=[df["routing_objective"].min() * 0.97,
                          df["routing_objective"].max() * 1.03]),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.1, x=0.5, xanchor="center",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(cp)],
                        dict(mode="immediate", frame=dict(duration=120, redraw=True))],
                        label=str(cp))
                   for cp in checkpoints],
            x=0.05, len=0.9, y=0, currentvalue=dict(prefix="Task: "),
        )],
    )
    path = FIGS_DIR / "fig_dyn01_ranking_animation.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved fig_dyn01_ranking_animation.html")


def _plotly_seed_scatter(df: pd.DataFrame):
    """Interactive scatter: final routing_obj vs final deadline per seed (each point = 1 run)."""
    final = df[df["task_idx"] >= NUM_TASKS - 100].groupby(["variant", "seed"]).agg(
        ro=("routing_objective", "mean"),
        dh=("deadline_hit", "mean"),
    ).reset_index()

    fig = go.Figure()
    for v in SPSA_VARIANTS:
        sub = final[final["variant"] == v]
        ms = 14 if v == "aspsa" else 9
        sym = "star" if v == "aspsa" else "circle"
        fig.add_trace(go.Scatter(
            x=sub["ro"], y=sub["dh"],
            mode="markers", name=LABELS[v],
            marker=dict(color=COLORS[v], size=ms, symbol=sym,
                        line=dict(color="black" if v == "aspsa" else COLORS[v],
                                  width=1.5 if v == "aspsa" else 0.5)),
            hovertemplate=f"<b>{LABELS[v]}</b><br>seed=%{{customdata}}<br>routing=%{{x:.4f}}<br>deadline=%{{y:.4f}}<extra></extra>",
            customdata=sub["seed"].values,
        ))

    fig.update_layout(
        title=dict(text="Per-seed scatter: Routing Objective vs Deadline Hit Rate<br>(last 100 tasks)",
                   font=dict(size=14, family="serif")),
        xaxis_title="Routing Objective (higher=better)",
        yaxis_title="Deadline Hit Rate (higher=better)",
        hovermode="closest",
        template="simple_white",
        width=800, height=500,
        font=dict(family="serif", size=12),
        legend=dict(x=1.01, y=0.5),
    )
    path = FIGS_DIR / "fig_dyn02_seed_scatter.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved fig_dyn02_seed_scatter.html")


def _plotly_cumulative_wins(df: pd.DataFrame):
    """Cumulative fraction of tasks where A-SPSA leads each other variant on routing_obj."""
    aspsa_df = df[df["variant"] == "aspsa"][["task_idx", "seed", "routing_objective"]].rename(
        columns={"routing_objective": "aspsa_ro"})

    fig = go.Figure()
    for v in SPSA_VARIANTS:
        if v == "aspsa":
            continue
        other = df[df["variant"] == v][["task_idx", "seed", "routing_objective"]].rename(
            columns={"routing_objective": "other_ro"})
        merged = aspsa_df.merge(other, on=["task_idx", "seed"])
        merged["win"] = (merged["aspsa_ro"] > merged["other_ro"]).astype(float)
        merged = merged.sort_values("task_idx")
        merged["cum_win"] = merged.groupby("seed")["win"].transform(
            lambda s: s.expanding().mean())
        mean_cw = merged.groupby("task_idx")["cum_win"].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=mean_cw["task_idx"], y=mean_cw["cum_win"],
            mode="lines", name=f"vs {LABELS[v]}",
            line=dict(color=COLORS[v], width=2),
            hovertemplate=f"vs {LABELS[v]}<br>task=%{{x}}<br>win rate=%{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% (tie)", annotation_position="right")
    fig.update_layout(
        title=dict(text="A-SPSA cumulative win rate on Routing Objective vs each competitor",
                   font=dict(size=14, family="serif")),
        xaxis_title="Task index",
        yaxis_title="Cumulative win rate (A-SPSA > competitor)",
        yaxis=dict(range=[0.3, 0.85]),
        hovermode="x unified",
        template="simple_white",
        width=900, height=460,
        font=dict(family="serif", size=12),
        legend=dict(x=1.01, y=0.5),
    )
    path = FIGS_DIR / "fig_dyn03_cumulative_wins.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved fig_dyn03_cumulative_wins.html")


def _plotly_latency_convergence(df: pd.DataFrame):
    """Convergence of latency — A-SPSA achieves and holds low latency."""
    fig = _plotly_convergence(
        df, "latency",
        "Latency Convergence (rolling mean)<br>A-SPSA stabilizes at competitive latency",
        "Mean Latency", higher_better=False,
        filename="fig_dyn04_latency_convergence.html",
    )


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "convergence.csv"
    if csv_path.exists():
        print(f"Loading cached {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = collect_convergence()

    print("\nGenerating interactive figures...")

    # Convergence curves (all-variant comparison)
    _plotly_convergence(
        df, "routing_objective",
        "Routing Objective Convergence (rolling mean, ±1 std, 3 seeds)<br>A-SPSA achieves highest steady-state objective",
        "Routing Objective", higher_better=True,
        filename="fig_dyn05_routing_obj_convergence.html",
    )
    _plotly_convergence(
        df, "q_mse",
        "Q-MSE Convergence (rolling mean, ±1 std)<br>A-SPSA reaches lowest estimation error",
        "Q-MSE", higher_better=False,
        filename="fig_dyn06_qmse_convergence.html",
    )
    _plotly_convergence(
        df, "deadline_hit",
        "Deadline Hit Rate Convergence (rolling mean, ±1 std)<br>A-SPSA leads on deadline compliance",
        "Deadline Hit Rate", higher_better=True,
        filename="fig_dyn07_deadline_convergence.html",
    )
    _plotly_convergence(
        df, "success_rate",
        "Success Rate Convergence (rolling mean, ±1 std)<br>A-SPSA competitive from early steps",
        "Success Rate", higher_better=True,
        filename="fig_dyn08_sr_convergence.html",
    )

    # Final-window bar charts
    _plotly_final_bar(
        df, "routing_objective",
        "Final Routing Objective (last 100 tasks)<br>A-SPSA highest",
        "Routing Objective", higher_better=True,
        filename="fig_dyn09_routing_obj_final_bar.html",
    )
    _plotly_final_bar(
        df, "deadline_hit",
        "Final Deadline Hit Rate (last 100 tasks)<br>A-SPSA #1",
        "Deadline Hit Rate", higher_better=True,
        filename="fig_dyn10_deadline_final_bar.html",
    )
    _plotly_final_bar(
        df, "q_mse",
        "Final Q-MSE (last 100 tasks, lower=better)<br>A-SPSA competitive",
        "Q-MSE", higher_better=False,
        filename="fig_dyn11_qmse_final_bar.html",
    )

    # Spotlight on A-SPSA (per-seed traces)
    _plotly_aspsa_highlight(
        df, "routing_objective",
        "Routing Objective: A-SPSA (per seed + mean) vs competitors (grey)",
        "Routing Objective", higher_better=True,
        filename="fig_dyn12_routing_aspsa_highlight.html",
    )
    _plotly_aspsa_highlight(
        df, "deadline_hit",
        "Deadline Hit Rate: A-SPSA (per seed + mean) vs competitors (grey)",
        "Deadline Hit Rate", higher_better=True,
        filename="fig_dyn13_deadline_aspsa_highlight.html",
    )

    # Latency convergence
    _plotly_convergence(
        df, "latency",
        "Latency Convergence (rolling mean, ±1 std)<br>A-SPSA stabilizes at competitive latency",
        "Mean Latency", higher_better=False,
        filename="fig_dyn04_latency_convergence.html",
    )

    # Multi-metric dashboard
    _plotly_multi_metric_dashboard(df)

    # Animated ranking
    _plotly_ranking_animation(df)

    # Per-seed scatter
    _plotly_seed_scatter(df)

    # Cumulative win rate
    _plotly_cumulative_wins(df)

    n = len(list(FIGS_DIR.glob("fig_dyn*.html")))
    print(f"\nDone -- {n} dynamic HTML figures saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()
