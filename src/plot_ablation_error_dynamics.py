"""Rolling Q-MSE dynamics for all comparison variants in one figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ordered_variant_dirs(root: Path) -> list[str]:
    preferred = [
        "current",
        "no_spsa",
        "no_cost",
        "no_learning",
        "no_judge",
        "centralized",
    ]
    out: list[str] = []
    for name in preferred:
        if (root / name / "task_records.csv").is_file():
            out.append(name)
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name in out:
            continue
        if (p / "task_records.csv").is_file():
            out.append(name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rolling Q-MSE for all ablation variants on one graph.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation_error_run",
        help="Directory produced by run_comparisons (contains <variant>/task_records.csv).",
    )
    parser.add_argument("--window", type=int, default=50, help="Rolling window size (tasks).")
    parser.add_argument(
        "--out-name",
        type=str,
        default="ablation_error_dynamics_all.png",
        help="Output PNG filename inside output-dir.",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    window = max(1, int(args.window))
    variants = _ordered_variant_dirs(root)

    plt.figure(figsize=(14, 7))
    plotted = 0
    for variant in variants:
        csv_path = root / variant / "task_records.csv"
        df = pd.read_csv(csv_path)
        if df.empty or not {"q_hat", "q_true"}.issubset(df.columns):
            continue
        sq_err = (df["q_hat"].astype(float) - df["q_true"].astype(float)) ** 2
        rolling_mse = sq_err.rolling(window=window, min_periods=1).mean()
        x = range(1, len(rolling_mse) + 1)
        plt.plot(x, rolling_mse, linewidth=1.8, label=variant, alpha=0.9)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No task_records.csv with q_hat/q_true under {root.resolve()}")

    plt.title(f"Динамика ошибки предсказания Q (rolling MSE, окно={window}) — все варианты")
    plt.xlabel("Индекс записи (подзадача)")
    plt.ylabel("Rolling (q_hat − q_true)²")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path = root / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
