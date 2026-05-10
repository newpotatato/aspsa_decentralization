from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path("comparison_outputs")
    variants = ["current", "no_spsa", "no_cost", "no_learning", "no_judge", "centralized"]

    plt.figure(figsize=(12, 6))

    for variant in variants:
        csv_path = root / variant / "task_records.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty or "success" not in df.columns:
            continue

        # Cumulative success rate is the clearest single-line dynamic view.
        cumulative_success = df["success"].expanding().mean()
        x = range(1, len(cumulative_success) + 1)
        plt.plot(x, cumulative_success, linewidth=2.0, label=variant)

    plt.title("Ablation Dynamics on One Graph (Cumulative Success Rate)")
    plt.xlabel("Task index")
    plt.ylabel("Cumulative success rate")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "ablation_dynamics_one_graph.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
