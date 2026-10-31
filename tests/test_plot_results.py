import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_results import plot_metrics


class TestPlotResults(unittest.TestCase):
    def test_plot_uses_only_actual_metric_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.csv"
            with metrics.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "checkpoint",
                        "preference_accuracy",
                        "response_length_tokens_mean",
                        "implicit_reward_margin_mean",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "checkpoint": "base",
                    "preference_accuracy": 0.5,
                    "response_length_tokens_mean": 12,
                    "implicit_reward_margin_mean": "",
                })
                writer.writerow({
                    "checkpoint": "dpo",
                    "preference_accuracy": 0.6,
                    "response_length_tokens_mean": 14,
                    "implicit_reward_margin_mean": 0.1,
                })

            output = plot_metrics(metrics, root / "plot.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
