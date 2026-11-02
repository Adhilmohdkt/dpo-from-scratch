import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml
from datasets import Dataset

from scripts import run_evaluation_only as workflow
from src.data import PreferenceDataset
from src.evaluate import _select_validation_subset


REVISION = workflow.DATASET_REVISION
DATASET = "HuggingFaceH4/ultrafeedback_binarized"


def fixture_config(root: Path) -> dict:
    return {
        "seed": 42,
        "model": {"name_or_path": workflow.MODEL_ID, "revision": "main", "use_lora": True},
        "data": {
            "name": DATASET, "revision": REVISION, "train_preferences_split": "train_prefs",
            "eval_preferences_split": "test_prefs", "sft_split": "train_sft",
            "max_train_samples": 512, "max_validation_samples": 32,
            "validation_eval_samples": 500, "max_eval_samples": 500,
            "eval_split_seed_offset": 2, "max_length": 512, "train_row_ids_file": None,
        },
        "training": {
            "output_dir": str(root / "seed-42" / "sft"), "batch_size": 1,
            "gradient_accumulation_steps": 8, "epochs": 1, "resume_from_checkpoint": None,
            "device": "cuda",
        },
        "dpo": {"beta": 0.1, "sft_checkpoint": str(root / "seed-42" / "sft"),
                "output_dir": str(root / "seed-42" / "dpo-beta-0.1"),
                "reference_logps_path": str(root / "seed-42" / "reference" / "full.pt")},
        "evaluation": {"split": "validation", "output_dir": str(root / "evaluation"),
                       "max_new_tokens": 256, "batch_size": 8, "checkpoints": {}, "beta_by_checkpoint": {}},
        "experiment": {"seeds": [42], "output_dir": str(root), "beta_values": [0.1, 0.5],
                        "selection_subset_size": 128, "selection_beta": 0.1},
        "hub": {"enabled": True, "private": True, "repo_id": workflow.HUB_REPO_ID,
                "token_env": "HF_TOKEN", "run_prefix": "scaled-t4-512"},
    }


def write_selection(root: Path, config: dict) -> tuple[list[int], dict[str, list[int]], str]:
    selection = root / "seed-42" / "selection"
    selection.mkdir(parents=True)
    full_ids = list(range(512))
    subsets = {"margin": full_ids[:128], "random": full_ids[128:256]}
    fingerprint = "train-fingerprint-v1"
    common = {
        "dataset": DATASET, "dataset_revision": REVISION, "split": "train_prefs",
        "dataset_fingerprint": fingerprint, "seed": 42, "validation_fraction": 0.02,
        "max_train_samples": 512, "subset_size": 128, "candidate_count": 512,
        "validation_count": 32,
    }
    (selection / "full_candidate_row_ids.json").write_text(json.dumps(full_ids), encoding="utf-8")
    (selection / "selection_experiment.json").write_text(json.dumps(common), encoding="utf-8")
    for arm, row_ids in subsets.items():
        payload = {**common, "method": "top_external_reward_margin" if arm == "margin" else "uniform_random_without_replacement", "row_ids": row_ids}
        (selection / f"{arm}_subset.json").write_text(json.dumps(payload), encoding="utf-8")
    return full_ids, subsets, fingerprint


def write_stage(path: Path, config: dict, stage: str, split: str, rows: int,
                step: int, fingerprint: str, row_hash: str) -> dict:
    path.mkdir(parents=True)
    identity = {
        "dataset": DATASET, "revision": REVISION, "split": split,
        "fingerprint": fingerprint, "selected_rows": rows,
        "selected_row_ids_sha256": row_hash,
        "validation_rows": 32, "validation_row_ids_sha256": "e" * 64,
    }
    (path / "dataset_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    if stage == "sft":
        (path / "sft_manifest.json").write_text("{}", encoding="utf-8")
    checkpoint = path / f"checkpoint-{step}"
    checkpoint.mkdir()
    for filename in ("training_state.pt", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / filename).write_bytes(b"checkpoint")
    marker = {"stage": stage, "seed": 42, "config_sha256": workflow.config_digest(config), "dataset_identity": identity}
    (path / "run_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    return identity


def populate_training(root: Path, config: dict) -> dict[str, Path]:
    full_ids, subsets, fingerprint = write_selection(root, config)
    seed_cfg, specs = workflow.make_stage_configs(config, root)
    full_hash = hashlib.sha256(json.dumps(full_ids, separators=(",", ":")).encode()).hexdigest()
    paths = {}
    for label, spec in specs.items():
        if label == "sft":
            row_hash = "d" * 64
            split, rows, fp = "train_sft", 512, "sft-fingerprint"
        else:
            split, rows, fp = "train_prefs", (128 if label in {"dpo-margin", "dpo-random"} else 512), fingerprint
            ids = full_ids if rows == 512 else subsets[label.removeprefix("dpo-")]
            selected = set(ids)
            ordered = [value for value in full_ids if value in selected]
            row_hash = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()
        write_stage(spec["path"], spec["config"], spec["stage"], split, rows, spec["step"], fp, row_hash)
        paths[label] = spec["path"]
    return paths


def evaluation_identity() -> dict:
    rows = Dataset.from_list([
        {"prompt_id": f"heldout-{i}", "chosen": [], "rejected": [], "_source_row_id": i}
        for i in range(500)
    ])
    source_ids_hash = hashlib.sha256(json.dumps(list(range(500)), separators=(",", ":")).encode()).hexdigest()
    original = {
        "dataset": DATASET, "revision": REVISION, "split": "test_prefs",
        "fingerprint": "validation-fingerprint", "selected_rows": 500,
        "selected_row_ids_sha256": source_ids_hash,
        "validation_rows": None, "validation_row_ids_sha256": None,
    }
    _, identity, _ = _select_validation_subset(
        PreferenceDataset(rows, tokenizer=object(), max_length=512), original, 200, 45
    )
    return identity


def fixture_plan(root: Path) -> dict:
    return {
        "source_validation_size": 500,
        "validation_subset_size": 200,
        "validation_subset_seed": 45,
        "selection_method": workflow.SUBSET_METHOD,
        "output_dir": str(root / "evaluation-200"),
        "hub_run_prefix": "scaled-t4-512-validation-200",
    }


def write_plan(root: Path, plan: dict) -> Path:
    path = root / "evaluation-plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    return path


def fake_eval_identity(eval_cfg: dict, paths: dict[str, Path]):
    identity = evaluation_identity()
    _, _, manifest = _select_validation_subset(
        PreferenceDataset(Dataset.from_list([
            {"prompt_id": f"heldout-{i}", "chosen": [], "rejected": [], "_source_row_id": i}
            for i in range(500)
        ]), tokenizer=object(), max_length=512),
        {"dataset": DATASET, "revision": REVISION, "split": "test_prefs",
         "fingerprint": "validation-fingerprint",
         "selected_row_ids_sha256": hashlib.sha256(json.dumps(list(range(500)), separators=(",", ":")).encode()).hexdigest()},
        200, 45,
    )
    labels = eval_cfg["evaluation"]["checkpoints"]
    ev = {
        "seed": 42, "split": "validation", "max_new_tokens": 256, "do_sample": False,
        "batch_size": 8, "checkpoints": labels,
        "checkpoint_sha256": {k: f"hash-{k}" for k in labels},
        "beta_by_checkpoint": {"dpo_beta_0.1": 0.1, "dpo_beta_0.5": 0.5, "dpo_margin": 0.1, "dpo_random": 0.1},
        "validation_eval_samples": 500, "final_test_samples": 500,
        "dataset_identity": identity,
        "validation_subset": {"size": 200, "seed": 45, "selection_method": workflow.SUBSET_METHOD,
                              "manifest_sha256": manifest["manifest_sha256"]},
    }
    digest = hashlib.sha256(json.dumps(ev, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return {"evaluator_config": ev, "dataset_identity": ev["dataset_identity"], "dataset_rows": 200,
            "subset_manifest": manifest}, digest


def write_completed_eval(eval_output: Path, eval_cfg: dict, evaluator_config: dict,
                         evaluator_digest: str, subset_manifest: dict):
    eval_output.mkdir(parents=True, exist_ok=True)
    identity = evaluator_config["dataset_identity"]
    (eval_output / "evaluation_config.json").write_text(json.dumps(evaluator_config), encoding="utf-8")
    (eval_output / "validation_subset.json").write_text(json.dumps(subset_manifest), encoding="utf-8")
    state = {"evaluation_config_sha256": evaluator_digest, "dataset_identity": identity, "completed": {}}
    rows = []
    for label in workflow.EXPECTED_LABELS:
        state["completed"][label] = {"identity_sha256": evaluator_digest, "metrics": {"checkpoint": label}}
        rows.append({"checkpoint": label, "seed": "42", "split": "validation", "beta": "", "preference_accuracy": "0.5",
                     "preference_accuracy_ci_95_low": "0.45", "preference_accuracy_ci_95_high": "0.55",
                     "preference_logp_margin_mean": "0.0", "implicit_reward_margin_mean": "0.0",
                     "response_length_tokens_mean": "40", "response_fraction_hitting_cap": "0.0",
                     "preference_examples": "200", "generation_examples": "200"})
        safe = label.replace("/", "_")
        (eval_output / f"{safe}_predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (eval_output / f"{safe}_response_lengths.csv").write_text("index\n0\n", encoding="utf-8")
    (eval_output / "evaluation_state.json").write_text(json.dumps(state), encoding="utf-8")
    with (eval_output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (eval_output / "dataset_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    marker = {"stage": "evaluation", "seed": 42, "config_sha256": workflow.config_digest(eval_cfg), "dataset_identity": identity}
    (eval_output / "run_complete.json").write_text(json.dumps(marker), encoding="utf-8")


class TestEvaluationOnly(unittest.TestCase):
    def test_exactly_200_validation_examples_are_deterministic_and_manifested(self):
        identity = evaluation_identity()
        rows = Dataset.from_list([
            {"prompt_id": f"heldout-{i}", "chosen": [], "rejected": [], "_source_row_id": i}
            for i in range(500)
        ])
        source = PreferenceDataset(rows, tokenizer=object(), max_length=512)
        first, first_identity, first_manifest = _select_validation_subset(source, identity, 200, 45)
        second, second_identity, second_manifest = _select_validation_subset(source, identity, 200, 45)
        self.assertEqual(len(first), 200)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(first.rows["_source_row_id"], second.rows["_source_row_id"])
        self.assertEqual(first_identity["selected_rows"], 200)
        self.assertEqual(first_identity["validation_subset_seed"], 45)
        self.assertEqual(len(first_manifest["selected_source_row_ids"]), 200)
        self.assertNotEqual(first_manifest["selected_source_row_ids"], list(range(200)))

    def test_stage_digests_match_original_runner_identity(self):
        from scripts.run_scaled_experiments import _config_digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            cfg = fixture_config(root)
            write_selection(root, cfg)
            _, specs = workflow.make_stage_configs(cfg, root)
            for spec in specs.values():
                self.assertEqual(workflow.config_digest(spec["config"]), _config_digest(spec["config"]))

    def test_script_has_no_training_entrypoint_or_subprocess_invocation(self):
        source = Path(workflow.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        self.assertFalse(any(name in {"scripts.run_scaled_experiments", "src.train_sft", "src.train_dpo"} for name in imports))
        self.assertNotIn("subprocess", imports)
        self.assertFalse({"train", "train_sft", "train_dpo", "run_experiments"} & set(calls))

    def test_all_five_training_stages_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            cfg = fixture_config(root)
            seen = []
            restored = []
            original = workflow.validate_training_stage

            def restore(path, hub, seed, stage):
                restored.append((Path(path).name, stage))
                if stage == "selection":
                    write_selection(root, cfg)
                    return True
                _, specs = workflow.make_stage_configs(cfg, root)
                spec = next(value for value in specs.values() if value["path"].name == Path(path).name)
                label = next(name for name, value in specs.items() if value is spec)
                ids = list(range(512))
                if label == "sft":
                    split, rows, fp, row_hash = "train_sft", 512, "sft-fingerprint", "d" * 64
                else:
                    split = "train_prefs"
                    rows = 128 if label in {"dpo-margin", "dpo-random"} else 512
                    fp = "train-fingerprint-v1"
                    selected = ids[:128] if label == "dpo-margin" else (ids[128:256] if label == "dpo-random" else ids)
                    row_hash = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
                write_stage(Path(path), spec["config"], spec["stage"], split, rows, spec["step"], fp, row_hash)
                return True

            def recording(*args, **kwargs):
                seen.append(Path(args[0]).name)
                return original(*args, **kwargs)

            with patch.object(workflow, "validate_training_stage", side_effect=recording):
                _, validated = workflow.restore_and_validate_training(cfg, root, cfg["hub"], restore)
            self.assertEqual(set(validated), {"sft", "dpo-beta-0.1", "dpo-beta-0.5", "dpo-margin", "dpo-random"})
            self.assertEqual(len(seen), 5)
            self.assertEqual(len(restored), 6)  # selection manifest plus all five trained stages
            self.assertEqual(set(seen), {path.name for path in validated.values()})

    def test_invalid_dataset_identity_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            cfg = fixture_config(root)
            populate_training(root, cfg)
            target = root / "seed-42" / "dpo-beta-0.1"
            identity_path = target / "dataset_identity.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["revision"] = "0" * 40
            identity_path.write_text(json.dumps(identity), encoding="utf-8")
            marker_path = target / "run_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["dataset_identity"] = identity
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            _, specs = workflow.make_stage_configs(cfg, root)
            with self.assertRaisesRegex(ValueError, "revision"):
                workflow.validate_training_stage(target, specs["dpo-beta-0.1"]["config"], "dpo", 64,
                                                 "train_prefs", 512, "train-fingerprint-v1")

    def test_completed_evaluation_is_skipped_and_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            root = work / "results"
            cfg = fixture_config(root)
            paths = populate_training(root, cfg)
            seed_cfg, _ = workflow.make_stage_configs(cfg, root)
            plan = fixture_plan(root)
            eval_cfg = workflow.make_evaluation_config(seed_cfg, paths, root, plan)
            eval_output = Path(eval_cfg["evaluation"]["output_dir"]) / "validation"
            info, digest = fake_eval_identity(eval_cfg, paths)
            write_completed_eval(eval_output, eval_cfg, info["evaluator_config"], digest, info["subset_manifest"])
            old_eval = root / "evaluation" / "seed-42" / "validation"
            old_eval.mkdir(parents=True)
            sentinel = old_eval / "metrics.csv"
            sentinel.write_text("preserve original 500-example evaluation\n", encoding="utf-8")
            training_hashes_before = {
                label: hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
                for label, path in paths.items()
            }
            fake_hub = types.ModuleType("huggingface_hub")

            class Api:
                def __init__(self, token):
                    self.token = token
                def whoami(self):
                    return {"name": "fixture"}
                def repo_info(self, **kwargs):
                    return types.SimpleNamespace(private=True)

            fake_hub.HfApi = Api
            uploads = []
            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                output = workflow.run(
                    _write_config(work, cfg), workflow.HUB_REPO_ID, work / "work", write_plan(work, plan),
                    restore_fn=lambda *args: False,
                    upload_fn=lambda *args: uploads.append(args),
                    evaluate_fn=lambda *args, **kwargs: self.fail("complete evaluation must be skipped"),
                    preflight_fn=lambda *args: (info, digest),
                )
            self.assertEqual(output, eval_output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve original 500-example evaluation\n")
            self.assertEqual(training_hashes_before, {
                label: hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
                for label, path in paths.items()
            })
            self.assertEqual(len(uploads), 1)
            self.assertEqual(uploads[0][1]["run_prefix"], plan["hub_run_prefix"])
            summary = output / "checkpoint_summary.csv"
            self.assertTrue(summary.is_file())
            with summary.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            aggregate_file = output / "validation_metrics.csv"
            self.assertTrue(aggregate_file.is_file())

    def test_incomplete_evaluation_is_sent_to_existing_resumable_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            root = work / "results"
            cfg = fixture_config(root)
            paths = populate_training(root, cfg)
            seed_cfg, _ = workflow.make_stage_configs(cfg, root)
            plan = fixture_plan(root)
            eval_cfg = workflow.make_evaluation_config(seed_cfg, paths, root, plan)
            eval_output = Path(eval_cfg["evaluation"]["output_dir"]) / "validation"
            info, digest = fake_eval_identity(eval_cfg, paths)
            eval_cfg_payload = info["evaluator_config"]
            restored_state = {}

            def restore(path, hub, seed, stage):
                if stage == "evaluation":
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "evaluation_config.json").write_text(json.dumps(eval_cfg_payload), encoding="utf-8")
                    partial = {
                        "evaluation_config_sha256": digest,
                        "dataset_identity": info["dataset_identity"],
                        "completed": {"base": {"identity_sha256": digest, "metrics": {"checkpoint": "base"}}},
                    }
                    (path / "evaluation_state.json").write_text(json.dumps(partial), encoding="utf-8")
                    restored_state.update(partial)
                    return True
                return False

            fake_hub = types.ModuleType("huggingface_hub")

            class Api:
                def __init__(self, token):
                    self.token = token
                def whoami(self):
                    return {"name": "fixture"}
                def repo_info(self, **kwargs):
                    return types.SimpleNamespace(private=True)

            fake_hub.HfApi = Api
            eval_calls = []

            def fake_evaluate(config_path, split=None):
                self.assertEqual(split, "validation")
                self.assertIn("base", json.loads((eval_output / "evaluation_state.json").read_text(encoding="utf-8"))["completed"])
                eval_calls.append(config_path)
                write_completed_eval(eval_output, eval_cfg, eval_cfg_payload, digest, info["subset_manifest"])

            with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                workflow.run(
                    _write_config(work, cfg), workflow.HUB_REPO_ID, work / "work", write_plan(work, plan),
                    restore_fn=restore, upload_fn=lambda *args: None,
                    evaluate_fn=fake_evaluate, preflight_fn=lambda *args: (info, digest),
                )
            self.assertEqual(len(eval_calls), 1)
            self.assertEqual(set(restored_state["completed"]), {"base"})
            self.assertTrue((eval_output / "run_complete.json").is_file())


def _write_config(work: Path, config: dict) -> Path:
    path = work / "runtime.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
