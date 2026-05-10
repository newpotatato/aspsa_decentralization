import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .main import SemanticFeatureExtractor, Task, TaskType
    from .run_comparisons import build_variants, run_all_variants
except ImportError:
    from main import SemanticFeatureExtractor, Task, TaskType
    from run_comparisons import build_variants, run_all_variants


BASE_PROBS = {
    TaskType.PROGRAMMING: 0.25,
    TaskType.QA: 0.20,
    TaskType.SUMMARIZATION: 0.20,
    TaskType.TRANSLATION: 0.15,
    TaskType.TOOL_USE: 0.20,
}


def get_suite_scenarios():
    return [
        {
            "name": "balanced",
            "description": "Сбалансированный поток задач, близкий к дефолтной симуляции.",
            "probs": BASE_PROBS,
            "arrival_scale": 0.8,
            "complexity_shift": 0.0,
            "urgency_multiplier": 1.0,
            "templates": {
                TaskType.PROGRAMMING: "Build ML pipeline code with feature engineering and model training.",
                TaskType.QA: "Answer factual question and explain reasoning with references.",
                TaskType.SUMMARIZATION: "Summarize the report into concise bullet points.",
                TaskType.TRANSLATION: "Translate customer message from Russian to English.",
                TaskType.TOOL_USE: "Call CRM API tool to create ticket and update status.",
            },
        },
        {
            "name": "coding_heavy",
            "description": "Поток с доминированием сложных engineering и debugging задач.",
            "probs": {
                TaskType.PROGRAMMING: 0.55,
                TaskType.QA: 0.10,
                TaskType.SUMMARIZATION: 0.10,
                TaskType.TRANSLATION: 0.05,
                TaskType.TOOL_USE: 0.20,
            },
            "arrival_scale": 0.7,
            "complexity_shift": 1.0,
            "urgency_multiplier": 1.05,
            "templates": {
                TaskType.PROGRAMMING: "Debug a production ML pipeline, optimize training code, fix data leakage, benchmark latency and prepare deployment notes.",
                TaskType.QA: "Explain benchmark regression root cause and answer engineering postmortem questions with references.",
                TaskType.SUMMARIZATION: "Summarize repository refactor progress and open technical risks for the team.",
                TaskType.TRANSLATION: "Translate code review comments and release notes from Russian to English.",
                TaskType.TOOL_USE: "Use CI and issue tracker APIs to open incidents, update tickets, and attach logs.",
            },
        },
        {
            "name": "qa_support",
            "description": "Поток справочных и support-задач с умеренной стоимостью и высокой объяснимостью.",
            "probs": {
                TaskType.PROGRAMMING: 0.10,
                TaskType.QA: 0.45,
                TaskType.SUMMARIZATION: 0.20,
                TaskType.TRANSLATION: 0.15,
                TaskType.TOOL_USE: 0.10,
            },
            "arrival_scale": 0.9,
            "complexity_shift": -0.2,
            "urgency_multiplier": 1.0,
            "templates": {
                TaskType.PROGRAMMING: "Prepare a small script to validate customer data export format.",
                TaskType.QA: "Answer customer support question, explain billing discrepancy, and cite knowledge base references.",
                TaskType.SUMMARIZATION: "Summarize a support thread and provide concise resolution notes.",
                TaskType.TRANSLATION: "Translate customer email and support answer between Russian and English.",
                TaskType.TOOL_USE: "Create support ticket in CRM, update status, and log troubleshooting steps via API.",
            },
        },
        {
            "name": "tool_ops",
            "description": "Инструментальные и операционные задачи с большим числом tool-use и API действий.",
            "probs": {
                TaskType.PROGRAMMING: 0.10,
                TaskType.QA: 0.10,
                TaskType.SUMMARIZATION: 0.10,
                TaskType.TRANSLATION: 0.05,
                TaskType.TOOL_USE: 0.65,
            },
            "arrival_scale": 0.6,
            "complexity_shift": 0.6,
            "urgency_multiplier": 1.1,
            "templates": {
                TaskType.PROGRAMMING: "Write helper code for API integration and validate payload transformations.",
                TaskType.QA: "Answer operational question about incident handling and API retry policy.",
                TaskType.SUMMARIZATION: "Summarize operations dashboard changes and tool execution results.",
                TaskType.TRANSLATION: "Translate API error report and escalation notes for cross-team handoff.",
                TaskType.TOOL_USE: "Use CRM, SQL and incident APIs to create tickets, execute remediation steps, sync statuses and confirm resolution.",
            },
        },
        {
            "name": "urgent_incidents",
            "description": "Срочные mixed-задачи с усиленным давлением на latency и deadline hit rate.",
            "probs": {
                TaskType.PROGRAMMING: 0.20,
                TaskType.QA: 0.20,
                TaskType.SUMMARIZATION: 0.15,
                TaskType.TRANSLATION: 0.10,
                TaskType.TOOL_USE: 0.35,
            },
            "arrival_scale": 0.45,
            "complexity_shift": 0.4,
            "urgency_multiplier": 1.35,
            "urgency_bias": 0.1,
            "templates": {
                TaskType.PROGRAMMING: "Urgent production incident: debug failing pipeline, patch code, and restore service before deadline.",
                TaskType.QA: "Critical customer question: explain outage status, impact, workaround and ETA immediately.",
                TaskType.SUMMARIZATION: "Summarize incident bridge updates into short executive bullets for the on-call lead.",
                TaskType.TRANSLATION: "Translate incident updates and customer-facing notices between Russian and English ASAP.",
                TaskType.TOOL_USE: "Critical incident: call ticketing, CRM and monitoring APIs to escalate, acknowledge, reroute and close tasks.",
            },
        },
        {
            "name": "long_context",
            "description": "Длинные промпты и более дорогие по токенам задания для оценки cost-aware routing.",
            "probs": {
                TaskType.PROGRAMMING: 0.20,
                TaskType.QA: 0.20,
                TaskType.SUMMARIZATION: 0.25,
                TaskType.TRANSLATION: 0.10,
                TaskType.TOOL_USE: 0.25,
            },
            "arrival_scale": 0.8,
            "complexity_shift": 0.8,
            "urgency_multiplier": 1.0,
            "templates": {
                TaskType.PROGRAMMING: "Review a long architecture proposal, extract requirements, implement a feature-rich ML pipeline, benchmark model variants, write evaluation code, and prepare deployment steps with rollback notes.",
                TaskType.QA: "Read a long product specification and answer detailed factual and procedural questions with grounded explanations and explicit assumptions.",
                TaskType.SUMMARIZATION: "Summarize a long technical report, preserving constraints, risks, milestones, dependencies and performance trade-offs in structured form.",
                TaskType.TRANSLATION: "Translate a long bilingual policy update with operational details, customer impact, technical constraints and release timing notes.",
                TaskType.TOOL_USE: "Process a long operations request, inspect several API payloads, orchestrate tool calls across CRM, monitoring and SQL systems, and document the resulting state transitions.",
            },
        },
    ]


def normalize_probs(prob_dict):
    ordered = [prob_dict[t] for t in TaskType]
    total = float(sum(ordered))
    return [x / total for x in ordered]


def build_phase_probs(base_probs: dict, boosts: dict[TaskType, float]):
    adjusted = {task_type: float(base_probs[task_type]) for task_type in TaskType}
    for task_type, factor in boosts.items():
        adjusted[task_type] = adjusted.get(task_type, 0.0) * float(factor)
    total = float(sum(adjusted.values()))
    return [adjusted[t] / total for t in TaskType]


def get_regime_config(scenario: dict, progress: float):
    base_probs = scenario["probs"]
    ordered_types = sorted(TaskType, key=lambda t: base_probs[t], reverse=True)
    dominant = ordered_types[0]
    secondary = ordered_types[1] if len(ordered_types) > 1 else dominant
    tertiary = TaskType.TOOL_USE if dominant != TaskType.TOOL_USE else TaskType.PROGRAMMING
    base_arrival = float(scenario.get("arrival_scale", 0.8))
    base_complexity = float(scenario.get("complexity_shift", 0.0))
    base_urgency = float(scenario.get("urgency_multiplier", 1.0))

    if progress < 0.30:
        return {
            "phase": "warmup",
            "arrival_scale": base_arrival * 1.25,
            "complexity_shift": base_complexity - 0.15,
            "urgency_multiplier": base_urgency * 0.96,
            "probs": build_phase_probs(base_probs, {dominant: 1.05}),
            "suffix": "Warmup phase: moderate load, stable queueing, routine operating conditions.",
        }
    if progress < 0.70:
        return {
            "phase": "burst",
            "arrival_scale": max(0.12, base_arrival * 0.55),
            "complexity_shift": base_complexity + 0.85,
            "urgency_multiplier": base_urgency * 1.18,
            "probs": build_phase_probs(base_probs, {dominant: 1.35, tertiary: 1.18}),
            "suffix": "Burst phase: queue spike, latency drift, backlog growth, external services unstable.",
        }
    return {
        "phase": "shift",
        "arrival_scale": max(0.12, base_arrival * 0.78),
        "complexity_shift": base_complexity + 0.25,
        "urgency_multiplier": base_urgency * 1.08,
        "probs": build_phase_probs(base_probs, {secondary: 1.30, tertiary: 1.12}),
        "suffix": "Shift phase: workload mix changed, controllers see different recent patterns, recovery incomplete.",
    }


def generate_tasks_for_scenario(num_tasks: int, seed: int, num_ctrl: int, num_agents: int, scenario: dict):
    np.random.seed(seed)
    templates = scenario["templates"]
    tasks = []
    current_arrival = 0.0
    for i in range(num_tasks):
        progress = i / max(num_tasks - 1, 1)
        regime = get_regime_config(scenario, progress)
        task_type = np.random.choice(list(TaskType), p=regime["probs"])
        text = f"{templates[task_type]} {regime['suffix']}"
        phi = SemanticFeatureExtractor.extract(text)
        text_low = text.lower()
        length_f = min(len(text_low) / 800.0, 1.5)
        kw_bonus = sum(0.4 for kw in ("optimize", "pipeline", "debug", "production", "integration", "benchmark") if kw in text_low)
        h_base = float(np.clip(2.0 + 4.0 * float(np.mean(phi)) + 2.0 * length_f + kw_bonus, 1.0, 10.0))
        h = float(np.clip(h_base + float(regime["complexity_shift"]), 1.0, 10.0))
        hot = ("urgent", "asap", "critical", "prod", "incident", "deadline")
        urgency = float(np.clip(0.25 + 0.05 * h + (0.25 if any(w in text_low for w in hot) else 0.0), 0.05, 1.0))
        urgency = float(np.clip(
            urgency * float(regime["urgency_multiplier"]) + float(scenario.get("urgency_bias", 0.0)),
            0.05,
            1.0,
        ))
        gap = float(np.random.exponential(float(regime["arrival_scale"])))
        if regime["phase"] == "burst" and np.random.rand() < 0.35:
            gap *= float(np.random.uniform(0.08, 0.45))
        current_arrival += gap
        tasks.append(Task(i, task_type, float(current_arrival), h, phi, urgency, text))
    return tasks


def save_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    pivot = df.pivot(index="scenario", columns="variant", values=metric)
    scenarios = pivot.index.tolist()
    variants = pivot.columns.tolist()
    values = pivot.values
    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 1.1), max(6, len(scenarios) * 0.8)))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title(title)
    for i in range(len(scenarios)):
        for j in range(len(variants)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_suite_plots(raw_df: pd.DataFrame, scenario_variant_df: pd.DataFrame, overall_df: pd.DataFrame, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric, title in [
        ("success_rate", "Success Rate by Scenario and Variant"),
        ("mean_q", "Mean Q by Scenario and Variant"),
        ("deadline_hit_rate", "Deadline Hit Rate by Scenario and Variant"),
        ("mean_latency", "Mean Latency by Scenario and Variant"),
        ("total_cost", "Total Cost by Scenario and Variant"),
        ("cost_per_success", "Cost per Success by Scenario and Variant"),
    ]:
        save_heatmap(scenario_variant_df, metric, title, plots_dir / f"{metric}_heatmap.png")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(overall_df["total_cost_mean"], overall_df["mean_q_mean"], s=120)
    for _, row in overall_df.iterrows():
        ax.annotate(row["variant"], (row["total_cost_mean"], row["mean_q_mean"]))
    ax.set_xlabel("Total Cost (mean)")
    ax.set_ylabel("Mean Q (mean)")
    ax.set_title("Cost vs Quality Trade-off")
    plt.tight_layout()
    plt.savefig(plots_dir / "cost_vs_quality_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8))
    sorted_df = overall_df.sort_values("overall_rank_score")
    ax.bar(sorted_df["variant"], sorted_df["overall_rank_score"])
    ax.set_title("Overall Rank Score (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_rank_score.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, metric, title in [
        (axes[0, 0], "success_rate_mean", "Success Rate (mean)"),
        (axes[0, 1], "mean_q_mean", "Mean Q (mean)"),
        (axes[1, 0], "deadline_hit_rate_mean", "Deadline Hit Rate (mean)"),
        (axes[1, 1], "savings_vs_best_quality_static_mean", "Savings vs Best Quality Static (mean)"),
    ]:
        ax.bar(overall_df["variant"], overall_df[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_metric_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_ranking_columns(overall_df: pd.DataFrame):
    rank_config = {
        "success_rate_mean": False,
        "mean_q_mean": False,
        "deadline_hit_rate_mean": False,
        "mean_latency_mean": True,
        "total_cost_mean": True,
        "cost_per_success_mean": True,
        "q_mse_mean": True,
    }
    total_rank = np.zeros(len(overall_df))
    for metric, ascending in rank_config.items():
        rank_col = f"rank_{metric}"
        overall_df[rank_col] = overall_df[metric].rank(ascending=ascending, method="average")
        total_rank += overall_df[rank_col].to_numpy()
    overall_df["overall_rank_score"] = total_rank / len(rank_config)
    return overall_df


def write_summary_markdown(overall_df: pd.DataFrame, output_path: Path):
    winners = {
        "best_quality": overall_df.sort_values("mean_q_mean", ascending=False).iloc[0],
        "best_success": overall_df.sort_values("success_rate_mean", ascending=False).iloc[0],
        "best_deadline": overall_df.sort_values("deadline_hit_rate_mean", ascending=False).iloc[0],
        "cheapest": overall_df.sort_values("total_cost_mean", ascending=True).iloc[0],
        "best_cost_per_success": overall_df.sort_values("cost_per_success_mean", ascending=True).iloc[0],
        "best_overall": overall_df.sort_values("overall_rank_score", ascending=True).iloc[0],
    }
    lines = [
        "# Benchmark Suite Summary",
        "",
        "## Победители по ключевым метрикам",
        "",
        f"- Лучшее среднее качество (`Mean Q`): `{winners['best_quality']['variant']}`",
        f"- Лучший `Success Rate`: `{winners['best_success']['variant']}`",
        f"- Лучший `Deadline Hit Rate`: `{winners['best_deadline']['variant']}`",
        f"- Самый дешевый вариант (`Total Cost`): `{winners['cheapest']['variant']}`",
        f"- Лучшая стоимость за успех (`Cost per Success`): `{winners['best_cost_per_success']['variant']}`",
        f"- Лучший общий rank-score: `{winners['best_overall']['variant']}`",
        "",
        "## Интерпретация",
        "",
        "- `overall_rank_score` усредняет ранги по качеству, успеху, deadline, latency, стоимости, cost-per-success и ошибке предсказания качества.",
        "- Чем меньше `overall_rank_score`, тем устойчивее вариант по совокупности метрик.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_seed_list(seed_text: str):
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def run_benchmark_suite(num_tasks: int, num_ctrl: int, num_agents: int, output_dir: str, seeds: list[int], scenario_names: list[str] | None = None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    scenarios = get_suite_scenarios()
    if scenario_names:
        scenarios = [s for s in scenarios if s["name"] in scenario_names]

    config = {
        "num_tasks": num_tasks,
        "num_ctrl": num_ctrl,
        "num_agents": num_agents,
        "seeds": seeds,
        "scenarios": [
            {
                **{k: v for k, v in s.items() if k not in {"templates", "probs"}},
                "probs": {k.name: v for k, v in s["probs"].items()},
            }
            for s in scenarios
        ],
        "variants": [v.name for v in build_variants()],
    }
    (root / "suite_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    raw_frames = []
    for scenario in scenarios:
        for seed in seeds:
            base_tasks = generate_tasks_for_scenario(
                num_tasks=num_tasks,
                seed=seed,
                num_ctrl=num_ctrl,
                num_agents=num_agents,
                scenario=scenario,
            )
            run_dir = runs_root / scenario["name"] / f"seed_{seed}"
            summary = run_all_variants(
                num_tasks=num_tasks,
                seed=seed,
                num_ctrl=num_ctrl,
                num_agents=num_agents,
                output_dir=str(run_dir),
                base_tasks=deepcopy(base_tasks),
            )
            summary["scenario"] = scenario["name"]
            summary["seed"] = seed
            raw_frames.append(summary)

    raw_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    raw_df.to_csv(root / "raw_runs.csv", index=False)

    scenario_variant_df = raw_df.groupby(["scenario", "variant"], as_index=False).agg({
        "success_rate": "mean",
        "mean_q": "mean",
        "mean_latency": "mean",
        "p95_latency": "mean",
        "deadline_hit_rate": "mean",
        "total_cost": "mean",
        "cost_per_success": "mean",
        "q_mse": "mean",
        "assignment_rate": "mean",
        "savings_vs_best_quality_static": "mean",
        "savings_vs_no_cost": "mean",
    })
    scenario_variant_df.to_csv(root / "scenario_variant_aggregates.csv", index=False)

    overall_df = raw_df.groupby("variant", as_index=False).agg({
        "success_rate": ["mean", "std"],
        "mean_q": ["mean", "std"],
        "mean_latency": ["mean", "std"],
        "p95_latency": ["mean", "std"],
        "deadline_hit_rate": ["mean", "std"],
        "total_cost": ["mean", "std"],
        "cost_per_success": ["mean", "std"],
        "q_mse": ["mean", "std"],
        "assignment_rate": ["mean", "std"],
        "savings_vs_best_quality_static": ["mean", "std"],
        "savings_vs_no_cost": ["mean", "std"],
    })
    overall_df.columns = ["_".join([str(x) for x in col if x]).strip("_") for col in overall_df.columns.to_flat_index()]
    overall_df = add_ranking_columns(overall_df)
    overall_df.to_csv(root / "overall_variant_aggregates.csv", index=False)

    create_suite_plots(raw_df, scenario_variant_df, overall_df, root)
    write_summary_markdown(overall_df, root / "SUITE_SUMMARY.md")
    return raw_df, scenario_variant_df, overall_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run a large benchmark suite across scenarios, seeds and routing variants.")
    parser.add_argument("--tasks", type=int, default=300, help="Tasks per scenario/seed run.")
    parser.add_argument("--controllers", type=int, default=3, help="Number of controllers.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents/models.")
    parser.add_argument("--seeds", type=str, default="11,42,123", help="Comma-separated list of seeds.")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated subset of scenarios. Empty means run all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_suite_results",
        help="Directory for raw runs, aggregates and plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario_names = [x.strip() for x in args.scenarios.split(",") if x.strip()] or None
    seeds = parse_seed_list(args.seeds)
    _, _, overall = run_benchmark_suite(
        num_tasks=args.tasks,
        num_ctrl=args.controllers,
        num_agents=args.agents,
        output_dir=args.output_dir,
        seeds=seeds,
        scenario_names=scenario_names,
    )
    print("\n=== BENCHMARK SUITE OVERALL ===")
    print(overall.round(4).to_string(index=False))
