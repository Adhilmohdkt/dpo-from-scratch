import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.hub import (hub_run_path, latest_checkpoint, push_run_snapshot,
                     resolve_resume_checkpoint, restore_run_snapshot)
from src.training import append_metrics


class TestRunPersistence(unittest.TestCase):
    def test_metrics_are_written_to_jsonl_and_csv_on_each_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "train_metrics.jsonl")
            append_metrics(path, {"step": 1, "loss": 0.8})
            append_metrics(path, {"step": 2, "loss": 0.6, "step_seconds": 6.2})
            with Path(tmp, "train_metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], "1")
            self.assertEqual(rows[0]["step_seconds"], "")
            self.assertEqual(rows[1]["step_seconds"], "6.2")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_hub_run_path_is_unique_by_seed_stage_and_output(self):
        self.assertEqual(
            hub_run_path({"run_prefix": "dpo-study"}, 19, "dpo", "results/dpo-beta-0.1"),
            "dpo-study/seed-19/dpo/dpo-beta-0.1",
        )

    def test_enabled_hub_push_requires_environment_token_without_echoing_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is not set"):
                push_run_snapshot("results/run", {"enabled": True, "repo_id": "owner/private"}, 42, "sft")

    def test_hub_resume_reference_restores_checkpoint_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote_path = "runs/seed-42/sft/sft/checkpoint-10"
            expected = Path(tmp, remote_path)
            expected.mkdir(parents=True)
            with patch.dict(os.environ, {"HF_TOKEN": "unit-test-token"}):
                with patch("huggingface_hub.snapshot_download", return_value=tmp) as download:
                    result = resolve_resume_checkpoint(
                        f"hf://owner/repo/{remote_path}",
                        {"token_env": "HF_TOKEN", "restore_dir": tmp},
                    )
            self.assertEqual(Path(result), expected)
            self.assertEqual(download.call_args.kwargs["allow_patterns"], [f"{remote_path}/**"])

    def test_latest_checkpoint_uses_numeric_order_and_requires_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint-8", "checkpoint-12", "checkpoint-100"):
                (root / name).mkdir()
            (root / "checkpoint-8" / "training_state.pt").touch()
            (root / "checkpoint-100" / "training_state.pt").touch()
            self.assertEqual(latest_checkpoint(root), root / "checkpoint-100")

    def test_restore_run_snapshot_copies_only_target_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "restore" / "prefix" / "seed-42" / "dpo" / "arm"
            snapshot.mkdir(parents=True)
            (snapshot / "run_complete.json").write_text("{}", encoding="utf-8")
            output = root / "results" / "arm"
            settings = {"enabled": True, "repo_id": "owner/repo", "run_prefix": "prefix", "restore_dir": str(root / "restore")}
            with patch.dict(os.environ, {"HF_TOKEN": "unit-test-token"}):
                with patch("huggingface_hub.snapshot_download", return_value=str(root / "restore")) as download:
                    restored = restore_run_snapshot(output, settings, 42, "dpo")
            self.assertTrue(restored)
            self.assertTrue((output / "run_complete.json").is_file())
            self.assertEqual(download.call_args.kwargs["allow_patterns"], ["prefix/seed-42/dpo/arm/**"])


if __name__ == "__main__":
    unittest.main()
