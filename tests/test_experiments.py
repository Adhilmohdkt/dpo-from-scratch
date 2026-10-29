import csv
import json
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

from scripts.aggregate_results import aggregate
from src.data import PreferenceDataset, _apply_train_row_ids
from src.data_selection import select_equal_subsets, validate_selection_ids


class TestSelectionAndAggregation(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"_source_row_id": i, "score_chosen": float(i + 3), "score_rejected": 3.0}
            for i in range(10)
        ]

    def test_equal_subsets_and_seeded_random_selection(self):
        margin_ids, random_ids, scored = select_equal_subsets(self.rows, 4, 19)
        again_margin, again_random, _ = select_equal_subsets(self.rows, 4, 19)
        self.assertEqual(len(margin_ids), len(random_ids))
        self.assertEqual(margin_ids, [9, 8, 7, 6])
        self.assertEqual(random_ids, again_random)
        self.assertEqual(margin_ids, again_margin)
        self.assertEqual(len(scored), 10)

    def test_selection_rejects_validation_or_out_of_pool_ids(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_selection_ids([1, 2, 3], [4], [2, 4])
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_selection_ids([1, 2, 3], [4], [1, 8])

    def test_manifest_application_is_train_only_and_equal_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Dataset.from_list([{"_source_row_id": i} for i in (1, 2, 3, 5)])
            valid = Dataset.from_list([{"_source_row_id": 4}])
            payload = {
                "dataset": "org/data", "dataset_revision": "rev", "split": "train_prefs",
                "dataset_fingerprint": "fp", "seed": 42, "validation_fraction": 0.2,
                "max_train_samples": None, "candidate_count": 4, "validation_count": 1,
                "subset_size": 2, "row_ids": [2, 5],
            }
            manifest = Path(tmp, "subset.json")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            cfg = {"name": "org/data", "revision": "rev", "train_preferences_split": "train_prefs", "train_row_ids_file": str(manifest), "seed": 42, "validation_fraction": 0.2, "max_train_samples": None}
            selected = _apply_train_row_ids(train, valid, cfg, "fp")
            self.assertEqual(selected["_source_row_id"], [2, 5])
            payload["row_ids"] = [4]
            payload["subset_size"] = 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation"):
                _apply_train_row_ids(train, valid, cfg, "fp")

    def test_preference_dataset_has_fallback_id_for_heldout_rows(self):
        class Tokenizer:
            eos_token_id = 9
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                prefix = [1, 2]
                return prefix if add_generation_prompt else prefix + [3, 9]

        rows = Dataset.from_list([{
            "prompt_id": "test-id",
            "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
        }])
        item = PreferenceDataset(rows, Tokenizer(), 16)[0]
        self.assertEqual(item["row_id"], 0)

    def test_aggregate_actual_metrics_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "eval")
            for seed, accuracy, length in [(2, 0.5, 10.0), (3, 0.75, 12.0)]:
                run = root / f"seed-{seed}" / "validation"
                run.mkdir(parents=True)
                with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["checkpoint", "seed", "split", "preference_accuracy", "response_length_tokens_mean"])
                    writer.writeheader()
                    writer.writerow({"checkpoint": "dpo", "seed": seed, "split": "validation", "preference_accuracy": accuracy, "response_length_tokens_mean": length})
            output = Path(tmp, "summary.csv")
            rows, result = aggregate(root, output)
            self.assertEqual(rows, 2)
            with result.open(encoding="utf-8", newline="") as stream:
                summary = list(csv.DictReader(stream))
            accuracy = next(row for row in summary if row["metric"] == "preference_accuracy")
            self.assertAlmostEqual(float(accuracy["mean"]), 0.625)
            self.assertAlmostEqual(float(accuracy["std"]), 0.1767766953)
            self.assertEqual(int(accuracy["n_seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
