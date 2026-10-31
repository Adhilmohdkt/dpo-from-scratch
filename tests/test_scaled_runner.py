import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_scaled_experiments import (_completed_stage_uploaded,
                                            _config_digest, _stage_complete)


class TestScaledRunnerResume(unittest.TestCase):
    def test_resume_path_does_not_change_run_identity(self):
        config = {"training": {"resume_from_checkpoint": None}, "data": {"max_train_samples": 512}}
        digest = _config_digest(config)
        config["training"]["resume_from_checkpoint"] = "/tmp/checkpoint-10"
        self.assertEqual(digest, _config_digest(config))

    def test_completed_run_requires_matching_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"training": {}, "data": {"max_train_samples": 512}}
            identity = {"dataset": "fixture/data", "revision": "a" * 40, "split": "train_prefs", "fingerprint": "fp-v1"}
            (root / "dataset_identity.json").write_text(json.dumps(identity), encoding="utf-8")
            (root / "run_complete.json").write_text(json.dumps({"config_sha256": _config_digest(config), "dataset_identity": identity}), encoding="utf-8")
            self.assertTrue(_stage_complete(root, config))
            (root / "dataset_identity.json").write_text(json.dumps({**identity, "fingerprint": "fp-v2"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset fingerprint identity"):
                _stage_complete(root, config)
            changed = {"training": {}, "data": {"max_train_samples": 256}}
            with self.assertRaisesRegex(ValueError, "different settings"):
                _stage_complete(root, changed)

    def test_failed_completed_stage_upload_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"training": {}, "data": {"max_train_samples": 512}}
            identity = {"dataset": "fixture/data", "revision": "a" * 40, "split": "train_prefs", "fingerprint": "fp-v1"}
            (root / "dataset_identity.json").write_text(json.dumps(identity), encoding="utf-8")
            (root / "run_complete.json").write_text(json.dumps({"config_sha256": _config_digest(config), "dataset_identity": identity}), encoding="utf-8")
            settings = {"enabled": True}
            with patch("scripts.run_scaled_experiments.push_run_snapshot", side_effect=[OSError("offline"), "uploaded"]) as upload:
                with self.assertRaisesRegex(OSError, "offline"):
                    _completed_stage_uploaded(root, config, settings, 42, "dpo")
                self.assertTrue(_completed_stage_uploaded(root, config, settings, 42, "dpo"))
            self.assertEqual(upload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
