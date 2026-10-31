import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
import yaml
from transformers import BatchEncoding

from datasets import Dataset

from src.data import load_preference_eval_data
from src.evaluate import _prompt_ids, generation_length_summary, preference_accuracy, wilson_interval


class Transformers5Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return BatchEncoding({"input_ids": [[11, 12, 13]]})


class EvaluationRowsSpy:
    """Rows fixture that fails if validation touches the final-test range."""

    column_names = ["prompt_id", "chosen", "rejected"]
    _fingerprint = "fixture-fingerprint"

    def __init__(self, indices=None, source_size=20, validation_count=5, test_count=7):
        self.indices = list(range(source_size)) if indices is None else list(indices)
        self.source_size = source_size
        self.validation_count = validation_count
        self.test_count = test_count
        self.accessed = []

    def __len__(self):
        return len(self.indices)

    def shuffle(self, seed):
        return self

    def select(self, indices):
        indices = list(indices)
        self.accessed.extend(indices)
        if any(self.validation_count <= index < self.validation_count + self.test_count for index in indices):
            raise AssertionError("validation attempted to select a final-test row")
        return EvaluationRowsSpy(
            [self.indices[index] for index in indices], self.source_size,
            self.validation_count, self.test_count,
        )

    def add_column(self, name, values):
        if len(self) == self.source_size:
            raise AssertionError("validation attempted to add columns to the full split")
        result = EvaluationRowsSpy(self.indices, self.source_size, self.validation_count, self.test_count)
        result.column_names = self.column_names + [name]
        result.source_row_ids = list(values)
        return result

    def __getitem__(self, key):
        if key == "prompt_id":
            raise AssertionError("validation read prompt IDs from an unselected partition")
        if key == "_source_row_id":
            return self.source_row_ids
        if key in {"chosen", "rejected"}:
            return [[] for _ in self.indices]
        raise KeyError(key)


class TestEvaluationPromptEncoding(unittest.TestCase):
    def test_prompt_ids_normalize_transformers5_batch_encoding(self):
        row = {
            "chosen": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        ids = _prompt_ids(row, Transformers5Tokenizer())
        self.assertEqual(ids, [11, 12, 13])


class TestEvaluationMetrics(unittest.TestCase):
    def test_validation_loader_never_selects_or_reads_final_test_rows(self):
        rows = EvaluationRowsSpy()
        cfg = {
            "name": "unused/test-fixture",
            "revision": "a" * 40,
            "eval_preferences_split": "test_prefs",
            "validation_eval_samples": 5,
            "max_eval_samples": 7,
            "max_length": 32,
        }
        with patch("src.data.load_dataset", return_value=rows):
            validation, identity = load_preference_eval_data(cfg, tokenizer=None, split="validation", return_identity=True)
        self.assertEqual(len(validation), 5)
        self.assertEqual(rows.accessed, list(range(5)))
        self.assertEqual(identity["revision"], "a" * 40)
        self.assertEqual(identity["fingerprint"], rows._fingerprint)

    def test_evaluation_resume_skips_completed_checkpoint_and_saves_each_result(self):
        from src.evaluate import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "eval.yaml"
            config = {
                "seed": 42,
                "model": {"name_or_path": "fixture/model", "revision": "a" * 40},
                "training": {"device": "cpu", "output_dir": str(root / "missing-sft")},
                "data": {"name": "fixture/data", "revision": "b" * 40},
                "dpo": {"sft_checkpoint": str(root / "missing-sft")},
                "evaluation": {
                    "split": "validation", "output_dir": str(root / "evaluation"),
                    "max_new_tokens": 8, "batch_size": 1,
                    "checkpoints": {"base_a": "base", "base_b": "base"},
                    "beta_by_checkpoint": {},
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            data_identity = {
                "dataset": "fixture/data", "revision": "b" * 40, "split": "test_prefs",
                "fingerprint": "heldout-fp", "selected_rows": 2,
                "selected_row_ids_sha256": "row-hash",
                "validation_rows": None, "validation_row_ids_sha256": None,
            }
            class TinyDataset:
                rows = []
                def __len__(self):
                    return 2

            dataset = TinyDataset()
            evaluated = []

            def evaluate_one(label, *args):
                evaluated.append(label)
                if label == "base_b" and evaluated.count("base_b") == 1:
                    raise RuntimeError("simulated interruption after first checkpoint")
                output_dir = args[5]
                safe = label.replace("/", "_")
                (output_dir / f"{safe}_predictions.jsonl").write_text("{}\n", encoding="utf-8")
                (output_dir / f"{safe}_response_lengths.csv").write_text("index\n0\n", encoding="utf-8")
                return {"checkpoint": label, "preference_accuracy": 0.5}

            common_patches = (
                patch("src.evaluate.load_tokenizer", return_value=object()),
                patch("src.evaluate.load_preference_eval_data", return_value=(dataset, data_identity)),
            )
            with common_patches[0], common_patches[1], patch("src.evaluate.evaluate_checkpoint", side_effect=evaluate_one):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    evaluate(str(config_path))
                metrics_path = root / "evaluation" / "validation" / "metrics.json"
                self.assertEqual([row["checkpoint"] for row in json.loads(metrics_path.read_text(encoding="utf-8"))], ["base_a"])
                evaluate(str(config_path))

            self.assertEqual(evaluated, ["base_a", "base_b", "base_b"])
            state = json.loads((root / "evaluation" / "validation" / "evaluation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(set(state["completed"]), {"base_a", "base_b"})

    def test_identical_policy_and_reference_ties_report_chance_accuracy(self):
        metrics = preference_accuracy(torch.zeros(500))
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["ties"], 500)
        self.assertAlmostEqual(metrics["ci_95_low"], 0.4565, places=3)
        self.assertAlmostEqual(metrics["ci_95_high"], 0.5435, places=3)

    def test_accuracy_interval_and_fractional_tie_accounting(self):
        metrics = preference_accuracy(torch.tensor([1.0, -1.0, 0.0, 2.0]))
        self.assertEqual(metrics["accuracy"], 0.625)
        self.assertEqual(metrics["ties"], 1)
        low, high = wilson_interval(18, 32)
        self.assertAlmostEqual(low, 0.3932559113685198)
        self.assertAlmostEqual(high, 0.7183466888495963)

    def test_generation_length_and_cap_fraction(self):
        summary = generation_length_summary([5, 10, 10, 2], max_new_tokens=10)
        self.assertEqual(summary, {"mean": 6.75, "median": 7.5, "cap_fraction": 0.5, "examples": 4})

    def test_heldout_validation_and_test_are_disjoint_reproducible_and_exact_size(self):
        rows = Dataset.from_list([
            {"prompt_id": f"prompt-{i}", "chosen": [], "rejected": []}
            for i in range(20)
        ])
        cfg = {
            "name": "unused/test-fixture",
            "eval_preferences_split": "test_prefs",
            "seed": 11,
            "validation_eval_samples": 5,
            "max_eval_samples": 7,
            "max_length": 32,
        }
        with patch("src.data.load_dataset", return_value=rows):
            validation = load_preference_eval_data(cfg, tokenizer=None, split="validation")
        with patch("src.data.load_dataset", return_value=rows):
            final_test = load_preference_eval_data(cfg, tokenizer=None, split="test")
        with patch("src.data.load_dataset", return_value=rows):
            validation_again = load_preference_eval_data(cfg, tokenizer=None, split="validation")
        validation_ids = set(validation.rows["prompt_id"])
        test_ids = set(final_test.rows["prompt_id"])
        self.assertEqual(len(validation), 5)
        self.assertEqual(len(final_test), 7)
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual(validation_ids, set(validation_again.rows["prompt_id"]))
        self.assertTrue(set(validation.rows["_source_row_id"]).isdisjoint(final_test.rows["_source_row_id"]))

    def test_heldout_partition_fails_if_it_cannot_supply_both_sets(self):
        rows = Dataset.from_list([
            {"prompt_id": f"prompt-{i}", "chosen": [], "rejected": []}
            for i in range(5)
        ])
        cfg = {"name": "unused/test-fixture", "validation_eval_samples": 3, "max_eval_samples": 3}
        with patch("src.data.load_dataset", return_value=rows):
            with self.assertRaisesRegex(ValueError, "6 are required"):
                load_preference_eval_data(cfg, tokenizer=None, split="validation")


if __name__ == "__main__":
    unittest.main()
