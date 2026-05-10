import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .main import ExactOrchestrator, SimulationVariant
except ImportError:
    from main import ExactOrchestrator, SimulationVariant


def build_variants():
    return [
        SimulationVariant(name="current", routing_mode="adaptive"),
        SimulationVariant(name="no_spsa", routing_mode="adaptive", use_spsa=False),
        SimulationVariant(name="no_cost", routing_mode="adaptive", cost_weight=0.0),
        SimulationVariant(name="no_learning", routing_mode="adaptive", use_learning=False),
        SimulationVariant(name="no_judge", routing_mode="adaptive", use_judge=False),
        SimulationVariant(name="best_quality_static", routing_mode="best_quality_static"),
        SimulationVariant(name="cheapest_static", routing_mode="cheapest_static"),
        SimulationVariant(name="fastest_static", routing_mode="fastest_static"),
        SimulationVariant(name="least_queue", routing_mode="least_queue", controller_mode="least_queue"),
        SimulationVariant(name="random", routing_mode="random", controller_mode="random"),
        SimulationVariant(name="centralized", routing_mode="adaptive"),
    ]


def generate_tasks(num_tasks: int, seed: int, num_ctrl: int, num_agents: int):
    np.random.seed(seed)
    orchestrator = ExactOrchestrator(
        num_ctrl=num_ctrl,
        num_agents=num_agents,
        variant=SimulationVariant(name="task_stream_builder", use_spsa=False),
        classifier_seed=seed,
    )
    return orchestrator.generate_realistic_stream(num_tasks)


def create_summary_plots(summary_df: pd.DataFrame, output_dir: Path):
    metric_specs = [
        ("success_rate", "Success Rate"),
        ("mean_q", "Mean Q"),
        ("deadline_hit_rate", "Deadline Hit Rate"),
        ("mean_latency", "Mean Latency"),
        ("total_cost", "Total Cost"),
        ("cost_per_success", "Cost per Success"),
        ("q_mse", "Q MSE"),
        ("assignment_rate", "Assignment Rate"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    axes = axes.flatten()
    labels = summary_df["variant"].tolist()
    for ax, (metric, title) in zip(axes, metric_specs):
        values = summary_df[metric].tolist()
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if "savings_vs_best_quality_static" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(summary_df["variant"], summary_df["savings_vs_best_quality_static"])
        ax.set_title("Savings vs Best Quality Static")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_savings.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _ordered_variant_dirs_for_plots(output_dir: Path) -> list[Path]:
    preferred = [
        "current",
        "no_spsa",
        "no_cost",
        "no_learning",
        "no_judge",
        "centralized",
    ]
    ordered: list[Path] = []
    for name in preferred:
        p = output_dir / name
        if p.is_dir() and (p / "task_records.csv").is_file():
            ordered.append(p)
    for p in sorted(output_dir.iterdir()):
        if not p.is_dir() or p in ordered:
            continue
        if (p / "task_records.csv").is_file():
            ordered.append(p)
    return ordered


def create_error_dynamics_plots(output_dir: Path, window: int = 50) -> None:
    """Линейные графики динамики ошибки предсказания Q (не столбчатые сводки)."""
    window = max(1, int(window))
    variant_paths = _ordered_variant_dirs_for_plots(output_dir)
    if not variant_paths:
        return

    # Rolling MSE по подзадачам
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    n_roll = 0
    for p in variant_paths:
        csv_path = p / "task_records.csv"
        df = pd.read_csv(csv_path)
        if df.empty or not {"q_hat", "q_true"}.issubset(df.columns):
            continue
        sq_err = (df["q_hat"].astype(float) - df["q_true"].astype(float)) ** 2
        rolling = sq_err.rolling(window=window, min_periods=1).mean()
        x = np.arange(1, len(rolling) + 1)
        ax1.plot(x, rolling, linewidth=1.6, label=p.name, alpha=0.9)
        n_roll += 1
    if n_roll:
        ax1.set_xlabel("Индекс записи (подзадача)")
        ax1.set_ylabel("Rolling (q_hat − q_true)²")
        ax1.set_title(f"Динамика ошибки предсказания Q — rolling, окно={window}")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best", fontsize=7, ncol=2)
        fig1.tight_layout()
        fig1.savefig(output_dir / "comparison_error_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Кумулятивное среднее квадрата ошибки
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    n_cum = 0
    for p in variant_paths:
        csv_path = p / "task_records.csv"
        df = pd.read_csv(csv_path)
        if df.empty or not {"q_hat", "q_true"}.issubset(df.columns):
            continue
        sq_err = (df["q_hat"].astype(float) - df["q_true"].astype(float)) ** 2
        cumulative = sq_err.expanding().mean()
        x = np.arange(1, len(cumulative) + 1)
        ax2.plot(x, cumulative, linewidth=1.6, label=p.name, alpha=0.9)
        n_cum += 1
    if n_cum:
        ax2.set_xlabel("Индекс записи (подзадача)")
        ax2.set_ylabel("Кумулятивное среднее (q_hat − q_true)²")
        ax2.set_title("Динамика ошибки предсказания Q — кумулятивное среднее")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=7, ncol=2)
        fig2.tight_layout()
        fig2.savefig(output_dir / "comparison_q_error_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


def run_all_variants(
    num_tasks: int,
    seed: int,
    num_ctrl: int,
    num_agents: int,
    output_dir: str,
    base_tasks: Optional[list]=None,
    variants: Optional[list]=None,
):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    base_tasks = base_tasks if base_tasks is not None else generate_tasks(num_tasks=num_tasks, seed=seed, num_ctrl=num_ctrl, num_agents=num_agents)
    variants = variants if variants is not None else build_variants()
    summaries = []

    for variant in variants:
        variant_dir = root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        run_num_ctrl = 1 if variant.name == "centralized" else num_ctrl
        np.random.seed(seed)
        orchestrator = ExactOrchestrator(
            num_ctrl=run_num_ctrl,
            num_agents=num_agents,
            variant=variant,
            plot_output_path=str(variant_dir / "simulation.png"),
            classifier_seed=seed,
        )
        orchestrator.run(deepcopy(base_tasks))
        summary = orchestrator.collect_summary()
        summaries.append(summary)
        pd.DataFrame(orchestrator.task_records).to_csv(variant_dir / "task_records.csv", index=False)
        pd.DataFrame([summary]).to_csv(variant_dir / "summary.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        best_quality_row = summary_df.loc[summary_df["variant"] == "best_quality_static"]
        if not best_quality_row.empty:
            baseline_cost = float(best_quality_row["total_cost"].iloc[0])
            if baseline_cost > 0:
                summary_df["savings_vs_best_quality_static"] = 1.0 - (summary_df["total_cost"] / baseline_cost)
        no_cost_row = summary_df.loc[summary_df["variant"] == "no_cost"]
        if not no_cost_row.empty:
            no_cost_total = float(no_cost_row["total_cost"].iloc[0])
            if no_cost_total > 0:
                summary_df["savings_vs_no_cost"] = 1.0 - (summary_df["total_cost"] / no_cost_total)
    summary_df.to_csv(root / "summary_metrics.csv", index=False)
    create_summary_plots(summary_df, root)
    create_error_dynamics_plots(root, window=50)
    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run all simulation baselines and collect comparison artifacts.")
    parser.add_argument("--tasks", type=int, default=500, help="Number of tasks in the shared task stream.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--controllers", type=int, default=3, help="Number of controllers for decentralized runs.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents/models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_outputs",
        help="Directory for plots, CSV and per-variant task logs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_all_variants(
        num_tasks=args.tasks,
        seed=args.seed,
        num_ctrl=args.controllers,
        num_agents=args.agents,
        output_dir=args.output_dir,
    )
    print("\n=== COMPARISON SUMMARY ===")
    print(summary.round(4).to_string(index=False))
