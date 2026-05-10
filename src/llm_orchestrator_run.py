# Пример запуска симуляции с реальными LLM API
try:
    from .run_simulation import run_simulation
except ImportError:
    from run_simulation import run_simulation


if __name__ == "__main__":
    run_simulation(
        mode="api",
        num_tasks=20,
        seed=42,
        num_ctrl=3,
        num_agents=5,
        output_dir="single_run_outputs",
    )
