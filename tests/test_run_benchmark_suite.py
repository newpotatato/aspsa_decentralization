import tempfile
import unittest
from pathlib import Path

from src.run_benchmark_suite import generate_tasks_for_scenario, get_suite_scenarios, run_benchmark_suite


class TestRunBenchmarkSuite(unittest.TestCase):
    def test_suite_creates_aggregate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, scenario_variant, overall = run_benchmark_suite(
                num_tasks=20,
                num_ctrl=3,
                num_agents=5,
                output_dir=tmpdir,
                seeds=[42],
                scenario_names=["balanced"],
            )
            root = Path(tmpdir)
            self.assertFalse(scenario_variant.empty)
            self.assertFalse(overall.empty)
            self.assertTrue((root / "raw_runs.csv").exists())
            self.assertTrue((root / "overall_variant_aggregates.csv").exists())
            self.assertTrue((root / "plots" / "success_rate_heatmap.png").exists())
            self.assertTrue((root / "SUITE_SUMMARY.md").exists())

    def test_scenario_generator_is_phased_and_cumulative(self):
        scenario = next(s for s in get_suite_scenarios() if s["name"] == "balanced")
        tasks = generate_tasks_for_scenario(
            num_tasks=30,
            seed=42,
            num_ctrl=3,
            num_agents=5,
            scenario=scenario,
        )
        arrivals = [task.t_arrival for task in tasks]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertGreater(arrivals[-1], arrivals[0])
        texts = [task.text for task in tasks]
        self.assertTrue(any("Warmup phase" in text for text in texts))
        self.assertTrue(any("Burst phase" in text for text in texts))
        self.assertTrue(any("Shift phase" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
