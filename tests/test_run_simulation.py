import tempfile
import unittest
from pathlib import Path

from src.run_simulation import run_simulation


class TestRunSimulation(unittest.TestCase):
    def test_test_mode_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_simulation(
                mode="test",
                num_tasks=20,
                seed=42,
                num_ctrl=3,
                num_agents=5,
                output_dir=tmpdir,
            )
            root = Path(tmpdir)
            self.assertEqual(summary["variant"], "test")
            self.assertTrue((root / "test_simulation.png").exists())
            self.assertTrue((root / "test_summary.csv").exists())
            self.assertTrue((root / "test_task_records.csv").exists())


if __name__ == "__main__":
    unittest.main()
