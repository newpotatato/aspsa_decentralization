import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = [
    "balanced",
    "coding_heavy",
    "qa_support",
    "tool_ops",
    "urgent_incidents",
    "long_context",
]

VARIANTS = [
    "current",
    "centralized",
    "fastest_static",
    "best_quality_static",
    "cheapest_static",
    "no_spsa",
]

VARIANT_STYLE = {
    "current": {"label": "current", "color": "#1f77b4", "linewidth": 2.6, "zorder": 5},
    "centralized": {"label": "centralized", "color": "#d62728", "linewidth": 2.4, "zorder": 4},
    "fastest_static": {"label": "fastest_static", "color": "#2ca02c", "linewidth": 1.7, "zorder": 3},
    "best_quality_static": {"label": "best_quality_static", "color": "#9467bd", "linewidth": 1.7, "zorder": 3},
    "cheapest_static": {"label": "cheapest_static", "color": "#ff7f0e", "linewidth": 1.7, "zorder": 3},
    "no_spsa": {"label": "no_spsa", "color": "#7f7f7f", "linewidth": 1.7, "zorder": 2},
}

PLOT_METRICS = [
    {
        "key": "success",
        "title": "Success Rate",
        "ylabel": "success",
        "better": "higher",
        "grouped_agg": "mean",
    },
    {
        "key": "deadline_hit",
        "title": "Deadline Hit Rate",
        "ylabel": "deadline hit",
        "better": "higher",
        "grouped_agg": "mean",
    },
    {
        "key": "latency",
        "title": "Latency",
        "ylabel": "seconds",
        "better": "lower",
        "grouped_agg": "mean",
    },
    {
        "key": "p95_latency",
        "title": "P95 Latency",
        "ylabel": "seconds",
        "better": "lower",
        "grouped_agg": "derived",
    },
    {
        "key": "true_wait",
        "title": "True Wait",
        "ylabel": "seconds",
        "better": "lower",
        "grouped_agg": "mean",
    },
    {
        "key": "wait_error",
        "title": "Wait Prediction Error",
        "ylabel": "abs error",
        "better": "lower",
        "grouped_agg": "derived",
    },
    {
        "key": "controller_share",
        "title": "Controller Delay Share",
        "ylabel": "controller wait / latency",
        "better": "lower",
        "grouped_agg": "derived",
    },
    {
        "key": "q_true",
        "title": "Observed Quality",
        "ylabel": "quality",
        "better": "higher",
        "grouped_agg": "mean",
    },
    {
        "key": "cost",
        "title": "Round Cost",
        "ylabel": "USD",
        "better": "lower",
        "grouped_agg": "sum",
    },
    {
        "key": "cost_per_success_cum",
        "title": "Cumulative Cost per Success",
        "ylabel": "USD / success",
        "better": "lower",
        "grouped_agg": "derived",
    },
    {
        "key": "semantic_assignment_error",
        "title": "Semantic Assignment Error",
        "ylabel": "1 - semantic match",
        "better": "lower",
        "grouped_agg": "derived",
    },
]


def rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def mean_series(series_list: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(series) for series in series_list)
    stacked = np.vstack([series[:min_len] for series in series_list])
    return stacked.mean(axis=0)


def load_round_series(
    runs_root: Path,
    metrics: list[dict[str, str]],
    scenarios: list[str],
    variants: list[str],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    base_metric_keys = sorted(
        {
            metric["key"]
            for metric in metrics
            if metric["key"] != "cost_per_success_cum"
        }
    )
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario in scenarios:
        scenario_dir = runs_root / scenario
        if not scenario_dir.exists():
            continue
        data[scenario] = {}
        for variant in variants:
            metric_frames = {metric["key"]: [] for metric in metrics}
            for seed_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
                task_records_path = seed_dir / variant / "task_records.csv"
                if not task_records_path.exists():
                    continue
                df = pd.read_csv(task_records_path)
                grouped = df.groupby("parent_id", as_index=False).agg(
                    {
                        "success": ["mean", "sum"],
                        "deadline_hit": "mean",
                        "latency": "mean",
                        "true_wait": "mean",
                        "q_true": "mean",
                        "cost": "sum",
                        "semantic_match": "mean",
                        "predicted_wait": "mean",
                        "controller_wait": "mean",
                    }
                )
                grouped.columns = [
                    "parent_id",
                    "success_mean",
                    "success_sum",
                    "deadline_hit",
                    "latency",
                    "true_wait",
                    "q_true",
                    "cost",
                    "semantic_match",
                    "predicted_wait",
                    "controller_wait",
                ]
                latency_p95 = (
                    df.groupby("parent_id")["latency"]
                    .quantile(0.95)
                    .reset_index(name="p95_latency")
                )
                grouped = grouped.merge(latency_p95, on="parent_id", how="left")
                grouped["success_mean"] = grouped["success_mean"].astype(float)
                grouped["success_sum"] = grouped["success_sum"].astype(float)
                grouped["cost_per_success_cum"] = (
                    grouped["cost"].cumsum() / grouped["success_sum"].cumsum().clip(lower=1.0)
                )
                grouped["semantic_assignment_error"] = 1.0 - grouped["semantic_match"].astype(float)
                grouped["wait_error"] = (grouped["predicted_wait"] - grouped["true_wait"]).abs()
                grouped["controller_share"] = (
                    grouped["controller_wait"] / grouped["latency"].clip(lower=1e-6)
                )
                for metric in metrics:
                    metric_key = metric["key"]
                    if metric_key in {
                        "deadline_hit",
                        "latency",
                        "p95_latency",
                        "true_wait",
                        "q_true",
                        "wait_error",
                        "controller_share",
                        "semantic_assignment_error",
                    }:
                        values = grouped[metric_key].astype(float).reset_index(drop=True)
                    elif metric_key == "success":
                        values = grouped["success_mean"].astype(float).reset_index(drop=True)
                    else:
                        values = grouped[metric_key].astype(float).reset_index(drop=True)
                    metric_frames[metric_key].append(values)

            series_by_metric: dict[str, np.ndarray] = {}
            for metric, frames in metric_frames.items():
                if not frames:
                    continue
                wide = pd.concat(frames, axis=1)
                series_by_metric[metric] = wide.mean(axis=1).to_numpy(dtype=float)
            if series_by_metric:
                data[scenario][variant] = series_by_metric
    return data


def format_tail_value(value: float, metric_key: str) -> str:
    if metric_key in {"latency", "p95_latency", "true_wait", "wait_error"}:
        return f"{value:.1f}s"
    if metric_key == "controller_share":
        return f"{value:.3f}"
    if metric_key in {"cost", "cost_per_success_cum"}:
        return f"${value:.3f}"
    return f"{value:.3f}"


def best_variant_name(
    scenario_data: dict[str, dict[str, np.ndarray]],
    metric_key: str,
    window: int,
    better: str,
    variants: list[str],
) -> str:
    scored = []
    for variant in variants:
        smoothed = rolling_average(scenario_data[variant][metric_key], window)
        tail_mean = float(smoothed[max(0, len(smoothed) - 30) :].mean())
        scored.append((variant, tail_mean))
    scored.sort(key=lambda item: item[1], reverse=(better == "higher"))
    return scored[0][0]


def add_subplot_annotations(ax: plt.Axes, scenario: str, winner: str, metric_key: str, scenario_data, window: int):
    current_tail = rolling_average(scenario_data["current"][metric_key], window)
    centralized_tail = rolling_average(scenario_data["centralized"][metric_key], window)
    current_value = float(current_tail[max(0, len(current_tail) - 30) :].mean())
    centralized_value = float(centralized_tail[max(0, len(centralized_tail) - 30) :].mean())
    label = (
        f"winner: {winner}\n"
        f"current={format_tail_value(current_value, metric_key)}\n"
        f"centralized={format_tail_value(centralized_value, metric_key)}"
    )
    ax.text(
        0.985,
        0.05,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )
    ax.text(0.01, 0.95, scenario, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def plot_single_chart(
    *,
    data: dict[str, dict[str, dict[str, np.ndarray]]],
    metric: dict[str, str],
    scenario: str | None,
    output_path: Path,
    window: int,
    variants: list[str],
):
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    scenario_label = scenario or "overall mean across scenarios"

    for variant in variants:
        if scenario is None:
            series_list = [
                rolling_average(data[s][variant][metric["key"]], window)
                for s in SCENARIOS
            ]
            series = mean_series(series_list)
        else:
            series = rolling_average(data[scenario][variant][metric["key"]], window)
        rounds = np.arange(1, len(series) + 1)
        style = VARIANT_STYLE[variant]
        ax.plot(
            rounds,
            series,
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=0.98 if variant in {"current", "centralized"} else 0.9,
            zorder=style["zorder"],
        )

    if scenario is None:
        available_variants = [variant for variant in variants if variant in data and metric["key"] in data[variant]]
        winner = best_variant_name(
            {
                variant: {
                    metric["key"]: mean_series(
                        [rolling_average(data[s][variant][metric["key"]], window) for s in data if variant in data[s]]
                    )
                }
                for variant in available_variants
            },
            metric["key"],
            1,
            metric["better"],
            available_variants,
        )
        current_series = mean_series(
            [rolling_average(data[s]["current"][metric["key"]], window) for s in data if "current" in data[s]]
        )
        reference_variant = "centralized" if "centralized" in available_variants else next(v for v in available_variants if v != "current")
        reference_series = mean_series(
            [rolling_average(data[s][reference_variant][metric["key"]], window) for s in data if reference_variant in data[s]]
        )
        current_value = float(current_series[max(0, len(current_series) - 30) :].mean())
        reference_value = float(reference_series[max(0, len(reference_series) - 30) :].mean())
        label = (
            f"winner: {winner}\n"
            f"current={format_tail_value(current_value, metric['key'])}\n"
            f"{reference_variant}={format_tail_value(reference_value, metric['key'])}"
        )
    else:
        available_variants = [variant for variant in variants if variant in data[scenario] and metric["key"] in data[scenario][variant]]
        winner = best_variant_name(data[scenario], metric["key"], window, metric["better"], available_variants)
        current_series = rolling_average(data[scenario]["current"][metric["key"]], window)
        reference_variant = "centralized" if "centralized" in available_variants else next(v for v in available_variants if v != "current")
        reference_series = rolling_average(data[scenario][reference_variant][metric["key"]], window)
        current_value = float(current_series[max(0, len(current_series) - 30) :].mean())
        reference_value = float(reference_series[max(0, len(reference_series) - 30) :].mean())
        label = (
            f"winner: {winner}\n"
            f"current={format_tail_value(current_value, metric['key'])}\n"
            f"{reference_variant}={format_tail_value(reference_value, metric['key'])}"
        )

    ax.text(
        0.985,
        0.03,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95, "boxstyle": "round,pad=0.30"},
    )
    ax.set_title(f"{metric['title']} | {scenario_label}", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel(metric["ylabel"], fontsize=11)
    ax.grid(True, alpha=0.24, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Curves are averaged across seeds and smoothed with rolling window = {window}.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready round-wise PNG plots from benchmark suite logs.")
    parser.add_argument(
        "--suite-dir",
        type=str,
        default="benchmark_suite_results_coordination_reworked",
        help="Benchmark suite directory with runs/<scenario>/seed_<n>/<variant>/task_records.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_suite_results_coordination_reworked/paper_plots_readable",
        help="Directory for generated paper-ready PNG figures.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling mean window for smoothing round curves.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated subset of metric keys to export. Empty means all configured metrics.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated subset of scenarios to export. Empty means all configured scenarios.",
    )
    parser.add_argument(
        "--skip-overall",
        action="store_true",
        help="If set, do not export overall aggregated charts.",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="If set, do not export scenario-specific charts.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated subset of variants to plot. Empty means the default paper set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_metric_keys = [item.strip() for item in args.metrics.split(",") if item.strip()]
    selected_scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    selected_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    metrics = [metric for metric in PLOT_METRICS if not selected_metric_keys or metric["key"] in selected_metric_keys]
    scenarios = [scenario for scenario in SCENARIOS if not selected_scenarios or scenario in selected_scenarios]
    variants = [variant for variant in VARIANTS if not selected_variants or variant in selected_variants]

    data = load_round_series(suite_dir / "runs", metrics, scenarios, variants)

    for metric in metrics:
        metric_dir = output_dir / metric["key"]
        metric_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_overall:
            plot_single_chart(
                data=data,
                metric=metric,
                scenario=None,
                output_path=metric_dir / f"{metric['key']}__overall.png",
                window=args.rolling_window,
                variants=variants,
            )
            print(metric_dir / f"{metric['key']}__overall.png")

        if not args.skip_scenarios:
            for scenario in scenarios:
                plot_single_chart(
                    data=data,
                    metric=metric,
                    scenario=scenario,
                    output_path=metric_dir / f"{metric['key']}__{scenario}.png",
                    window=args.rolling_window,
                    variants=variants,
                )
                print(metric_dir / f"{metric['key']}__{scenario}.png")


if __name__ == "__main__":
    main()
