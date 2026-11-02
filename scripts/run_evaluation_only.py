"""Restore, validate, evaluate, and aggregate the completed scaled DPO runs.

This entry point intentionally has no training imports or training subprocesses.
It fails closed if any saved run does not match the configured experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable

import yaml

from scripts.aggregate_results import aggregate
from src.hub import hub_run_path, push_run_snapshot, restore_run_snapshot


HUB_REPO_ID = "adhill/dpo-experiment-checkpoint"
DATASET_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
EXPECTED_LABELS = ("base", "sft", "dpo_beta_0.1", "dpo_beta_0.5", "dpo_margin", "dpo_random")
EVALUATION_PLAN_PATH = Path("configs/scaled_t4_eval_200.yaml")
SUBSET_METHOD = "sha256_ranked_without_replacement_v1"


def config_digest(config: dict) -> str:
    """Match the training runner's digest, including selected-row manifest bytes."""
    stable = copy.deepcopy(config)
    stable.get("training", {}).pop("resume_from_checkpoint", None)
    row_ids_path = stable.get("data", {}).get("train_row_ids_file")
    if row_ids_path:
        row_ids_file = Path(row_ids_path)
        if row_ids_file.is_file():
            stable["data"]["train_row_ids_sha256"] = hashlib.sha256(row_ids_file.read_bytes()).hexdigest()
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_experiment_config(config: dict) -> None:
    """Reject configs that do not describe the already completed experiment."""
    data = config.get("data", {})
    model = config.get("model", {})
    experiment = config.get("experiment", {})
    evaluation = config.get("evaluation", {})
    expected = {
        "seed": config.get("seed") == SEED and experiment.get("seeds") == [SEED],
        "model": model.get("name_or_path") == MODEL_ID,
        "dataset": data.get("name") == "HuggingFaceH4/ultrafeedback_binarized",
        "dataset revision": data.get("revision") == DATASET_REVISION,
        "sequence length": data.get("max_length") == 512,
        "train size": data.get("max_train_samples") == 512,
        "beta values": experiment.get("beta_values") == [0.1, 0.5],
        "selection size": experiment.get("selection_subset_size") == 128,
        "selection beta": experiment.get("selection_beta") == 0.1,
        "validation size": data.get("validation_eval_samples") == 500,
        "final test size": data.get("max_eval_samples") == 500,
        "generation cap": evaluation.get("max_new_tokens") == 256,
        "evaluation batch size": evaluation.get("batch_size") == 8,
        "validation split": evaluation.get("split", "validation") == "validation",
    }
    failed = [name for name, valid in expected.items() if not valid]
    if failed:
        raise ValueError("Config differs from the completed scaled experiment: " + ", ".join(failed))
    if not re.fullmatch(r"[0-9a-f]{40}", str(data.get("revision", ""))):
        raise ValueError("Dataset revision must be an immutable 40-character commit SHA")


def validate_evaluation_plan(plan: dict, config: dict) -> None:
    """Validate the separate 200-row plan without changing the source config."""
    source_size = int(config["data"].get("validation_eval_samples", 0))
    expected_seed = int(config.get("seed", SEED)) + int(config["data"].get("eval_split_seed_offset", 2)) + 1
    if plan.get("source_validation_size") != source_size or source_size != 500:
        raise ValueError("The 200-row plan must derive from the original 500-row validation partition")
    if plan.get("validation_subset_size") != 200:
        raise ValueError("The evaluation-only plan must select exactly 200 validation examples")
    if plan.get("validation_subset_seed") != expected_seed:
        raise ValueError(f"Expected validation subset seed {expected_seed} from the experiment seed and split offset")
    if plan.get("selection_method") != SUBSET_METHOD:
        raise ValueError(f"Unsupported validation subset method: {plan.get('selection_method')!r}")
    if not plan.get("output_dir") or Path(plan["output_dir"]) == Path(config["evaluation"]["output_dir"]):
        raise ValueError("The 200-row evaluation requires a separate local output directory")
    if not plan.get("hub_run_prefix") or plan["hub_run_prefix"] == config["hub"].get("run_prefix"):
        raise ValueError("The 200-row evaluation requires a separate Hub run prefix")


def make_stage_configs(config: dict, result_root: Path) -> tuple[dict, dict[str, dict]]:
    """Construct the same effective per-stage configs used by the training runner."""
    seed_root = result_root / f"seed-{SEED}"
    sft_dir = seed_root / "sft"
    seed_cfg = copy.deepcopy(config)
    seed_cfg["seed"] = SEED
    seed_cfg["training"]["output_dir"] = str(sft_dir)
    seed_cfg["training"]["resume_from_checkpoint"] = None
    seed_cfg["dpo"]["sft_checkpoint"] = str(sft_dir)
    seed_cfg["dpo"]["output_dir"] = str(seed_root / "dpo-beta-0.1")
    seed_cfg["dpo"]["reference_logps_path"] = str(seed_root / "reference" / "full.pt")

    specs: dict[str, dict] = {"sft": {"path": sft_dir, "stage": "sft", "config": seed_cfg, "step": 64}}
    for beta in (0.1, 0.5):
        label = f"dpo-beta-{beta:g}"
        output = seed_root / label
        stage_cfg = copy.deepcopy(seed_cfg)
        stage_cfg["training"]["resume_from_checkpoint"] = None
        stage_cfg["dpo"].update({
            "beta": beta,
            "output_dir": str(output),
            "reference_logps_path": str(seed_root / "reference" / "full.pt"),
        })
        specs[label] = {"path": output, "stage": "dpo", "config": stage_cfg, "step": 64}

    selection_dir = seed_root / "selection"
    for arm in ("margin", "random"):
        label = f"dpo-{arm}"
        output = seed_root / label
        stage_cfg = copy.deepcopy(seed_cfg)
        stage_cfg["training"]["resume_from_checkpoint"] = None
        stage_cfg["data"]["train_row_ids_file"] = str(selection_dir / f"{arm}_subset.json")
        stage_cfg["dpo"].update({
            "beta": 0.1,
            "output_dir": str(output),
            "reference_logps_path": str(seed_root / "reference" / f"{arm}.pt"),
        })
        specs[label] = {"path": output, "stage": "dpo", "config": stage_cfg, "step": 16}
    return seed_cfg, specs


def _restore_without_overwriting_training(path: Path, hub: dict, stage: str, restore_fn: Callable) -> None:
    if path.exists():
        if not path.is_dir() or not any(path.iterdir()):
            raise ValueError(f"Existing training path is not an empty/new or complete run directory: {path}")
        return
    restored = restore_fn(path, hub, SEED, stage)
    if not restored:
        raise FileNotFoundError(f"Required Hub snapshot is missing: {hub_run_path(hub, SEED, stage, path)}")


def _dataset_identity(path: Path, cfg: dict, expected_split: str, expected_rows: int,
                      expected_fingerprint: str | None = None,
                      expected_row_ids_sha256: str | None = None) -> dict:
    identity_path = path / "dataset_identity.json"
    if not identity_path.is_file():
        raise ValueError(f"Missing dataset identity: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    required = {
        "dataset": cfg["data"]["name"],
        "revision": cfg["data"]["revision"],
        "split": expected_split,
        "selected_rows": expected_rows,
    }
    mismatches = [key for key, value in required.items() if identity.get(key) != value]
    if not isinstance(identity.get("fingerprint"), str) or not identity["fingerprint"]:
        mismatches.append("fingerprint")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("selected_row_ids_sha256", ""))):
        mismatches.append("selected row ID hash")
    if expected_fingerprint is not None and identity.get("fingerprint") != expected_fingerprint:
        mismatches.append("fingerprint vs selection manifest")
    if expected_row_ids_sha256 is not None and identity.get("selected_row_ids_sha256") != expected_row_ids_sha256:
        mismatches.append("selected row IDs")
    if mismatches:
        raise ValueError(f"Dataset identity mismatch at {identity_path}: {', '.join(mismatches)}")
    return identity


def _validate_selection(selection_dir: Path, config: dict) -> tuple[list[int], dict[str, list[int]], str]:
    metadata_path = selection_dir / "selection_experiment.json"
    full_ids_path = selection_dir / "full_candidate_row_ids.json"
    if not metadata_path.is_file() or not full_ids_path.is_file():
        raise ValueError(f"Selection snapshot is incomplete: {selection_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    full_ids = json.loads(full_ids_path.read_text(encoding="utf-8"))
    data = config["data"]
    expected = {
        "dataset": data["name"], "dataset_revision": data["revision"],
        "split": data.get("train_preferences_split", "train_prefs"),
        "seed": SEED, "subset_size": 128,
        "candidate_count": data["max_train_samples"],
        "validation_count": data.get("max_validation_samples"),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Selection metadata does not match the configured dataset revision, split, seed, or sizes")
    fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Selection metadata has no dataset fingerprint")
    if (not isinstance(full_ids, list) or len(full_ids) != 512
            or any(not isinstance(value, int) or isinstance(value, bool) for value in full_ids)
            or len(set(full_ids)) != len(full_ids)):
        raise ValueError("Selection candidate row IDs are missing, duplicated, or not 512 rows")
    subsets: dict[str, list[int]] = {}
    for arm in ("margin", "random"):
        subset_path = selection_dir / f"{arm}_subset.json"
        if not subset_path.is_file():
            raise ValueError(f"Missing {arm} subset manifest: {subset_path}")
        payload = json.loads(subset_path.read_text(encoding="utf-8"))
        row_ids = payload.get("row_ids")
        expected_method = "top_external_reward_margin" if arm == "margin" else "uniform_random_without_replacement"
        if (payload.get("method") != expected_method or payload.get("subset_size") != 128
                or payload.get("dataset_fingerprint") != fingerprint or payload.get("seed") != SEED
                or not isinstance(row_ids, list) or len(row_ids) != 128
                or any(not isinstance(value, int) or isinstance(value, bool) for value in row_ids)
                or len(set(row_ids)) != 128 or not set(row_ids).issubset(set(full_ids))):
            raise ValueError(f"Invalid {arm} selection manifest or row IDs")
        subsets[arm] = row_ids
    return full_ids, subsets, fingerprint


def validate_training_stage(path: Path, config: dict, stage: str, step: int,
                            expected_split: str, expected_rows: int,
                            expected_fingerprint: str | None = None,
                            expected_row_ids_sha256: str | None = None) -> dict:
    """Check stage marker, identity, final adapter, and final optimizer checkpoint."""
    marker_path = path / "run_complete.json"
    if not marker_path.is_file():
        raise ValueError(f"Missing completion marker for {stage}: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("stage") != stage:
        raise ValueError(f"Completion marker stage mismatch at {marker_path}: expected {stage!r}")
    if marker.get("seed") != SEED:
        raise ValueError(f"Completion marker seed mismatch at {marker_path}: expected {SEED}")
    expected_digest = config_digest(config)
    if marker.get("config_sha256") != expected_digest:
        raise ValueError(f"Configuration digest mismatch for {stage} at {path}; refusing evaluation")
    identity = _dataset_identity(path, config, expected_split, expected_rows,
                                 expected_fingerprint, expected_row_ids_sha256)
    if marker.get("dataset_identity") != identity:
        raise ValueError(f"Completion marker dataset identity differs from {path / 'dataset_identity.json'}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Required trained adapter file is missing: {path / filename}")
    checkpoint = path / f"checkpoint-{step}"
    for filename in ("training_state.pt", "adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"Final {step}-step checkpoint is incomplete: {checkpoint / filename}")
    if stage == "sft" and not (path / "sft_manifest.json").is_file():
        raise FileNotFoundError(f"SFT model manifest is missing: {path / 'sft_manifest.json'}")
    return identity


def restore_and_validate_training(config: dict, result_root: Path, hub: dict,
                                  restore_fn: Callable = restore_run_snapshot) -> tuple[dict, dict[str, Path]]:
    seed_cfg, specs = make_stage_configs(config, result_root)
    selection_dir = result_root / f"seed-{SEED}" / "selection"
    _restore_without_overwriting_training(selection_dir, hub, "selection", restore_fn)
    full_ids, subset_ids, fingerprint = _validate_selection(selection_dir, seed_cfg)
    stage_paths: dict[str, Path] = {}
    for label, spec in specs.items():
        path = spec["path"]
        _restore_without_overwriting_training(path, hub, spec["stage"], restore_fn)
        is_sft = label == "sft"
        is_selected = label in {"dpo-margin", "dpo-random"}
        if is_sft:
            split = seed_cfg["data"].get("sft_split", "train_sft")
            rows = 512
            expected_fingerprint = None
            expected_hash = None
        else:
            split = seed_cfg["data"].get("train_preferences_split", "train_prefs")
            rows = 128 if is_selected else 512
            expected_fingerprint = fingerprint
            selected = subset_ids[label.removeprefix("dpo-")] if is_selected else full_ids
            selected_set = set(selected)
            ordered_ids = [int(value) for value in full_ids if value in selected_set]
            expected_hash = hashlib.sha256(json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        identity = validate_training_stage(
            path, spec["config"], spec["stage"], spec["step"], split, rows,
            expected_fingerprint, expected_hash,
        )
        if not is_sft and identity["fingerprint"] != fingerprint:
            raise ValueError(f"{label} dataset fingerprint differs from the validated selection snapshot")
        stage_paths[label] = path
    return seed_cfg, stage_paths


def make_evaluation_config(seed_config: dict, stage_paths: dict[str, Path], result_root: Path,
                           evaluation_plan: dict) -> dict:
    eval_cfg = copy.deepcopy(seed_config)
    eval_cfg["training"]["resume_from_checkpoint"] = None
    eval_dir = Path(evaluation_plan["output_dir"]) / f"seed-{SEED}"
    eval_cfg["evaluation"].update({"split": "validation", "output_dir": str(eval_dir)})
    # Keep data.validation_eval_samples=500: it defines the historical
    # validation/test boundary. The evaluator samples 200 inside that slice.
    eval_cfg["evaluation"]["validation_subset_size"] = int(evaluation_plan["validation_subset_size"])
    eval_cfg["evaluation"]["validation_subset_seed"] = int(evaluation_plan["validation_subset_seed"])
    eval_cfg["hub"]["run_prefix"] = str(evaluation_plan["hub_run_prefix"])
    eval_cfg["evaluation"]["checkpoints"] = {"base": "base", "sft": str(stage_paths["sft"])}
    eval_cfg["evaluation"]["beta_by_checkpoint"] = {}
    for source_label, beta in (("dpo-beta-0.1", 0.1), ("dpo-beta-0.5", 0.5),
                               ("dpo-margin", 0.1), ("dpo-random", 0.1)):
        # Preserve the original runner's label conversion exactly: hyphens
        # become underscores while beta decimal points remain in the label.
        eval_label = source_label.replace("-", "_")
        eval_cfg["evaluation"]["checkpoints"][eval_label] = str(stage_paths[source_label])
        eval_cfg["evaluation"]["beta_by_checkpoint"][eval_label] = beta
    return eval_cfg


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def preflight_evaluation(eval_cfg: dict, eval_output: Path) -> tuple[dict, str]:
    """Compute the exact evaluator identity using tokenizer/data on CPU only."""
    from src.data import load_preference_eval_data
    from src.evaluate import _checkpoint_sha256, _select_validation_subset
    from src.modeling import (load_sft_manifest, load_tokenizer, model_config_for_sft_manifest,
                              validate_sft_tokenizer)

    cfg = copy.deepcopy(eval_cfg)
    sft_path = Path(cfg["dpo"].get("sft_checkpoint", cfg["training"]["output_dir"]))
    manifest = None
    if (sft_path / "adapter_config.json").exists():
        manifest = load_sft_manifest(sft_path)
        cfg["model"] = model_config_for_sft_manifest(cfg["model"], manifest)
    tokenizer = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"].get("trust_remote_code", False), cfg["model"].get("revision"))
    if manifest is not None:
        validate_sft_tokenizer(tokenizer, manifest)
    data_cfg = {**cfg["data"], "seed": int(cfg.get("seed", SEED))}
    dataset, source_identity = load_preference_eval_data(data_cfg, tokenizer, split="validation", return_identity=True)
    source_size = int(cfg["data"].get("validation_eval_samples", 500))
    if len(dataset) != source_size:
        raise ValueError(f"Expected the original {source_size}-row validation partition, got {len(dataset)}")
    dataset, data_identity, subset_manifest = _select_validation_subset(
        dataset, source_identity,
        int(cfg["evaluation"]["validation_subset_size"]),
        int(cfg["evaluation"]["validation_subset_seed"]),
    )
    if len(dataset) != 200:
        raise ValueError(f"Expected exactly 200 validation examples, got {len(dataset)}")
    selected = cfg["evaluation"]["checkpoints"]
    beta_by_checkpoint = dict(cfg["evaluation"].get("beta_by_checkpoint", {}))
    checkpoint_sha = {label: _checkpoint_sha256(path if path != "base" else None, cfg["model"])
                      for label, path in selected.items()}
    evaluator_config = {
        "seed": int(cfg.get("seed", SEED)),
        "split": "validation",
        "max_new_tokens": int(cfg["evaluation"].get("max_new_tokens", 256)),
        "do_sample": False,
        "batch_size": int(cfg["evaluation"].get("batch_size", 2)),
        "checkpoints": {label: (str(path) if path is not None else "base") for label, path in selected.items()},
        "checkpoint_sha256": checkpoint_sha,
        "beta_by_checkpoint": beta_by_checkpoint,
        "validation_eval_samples": int(data_cfg.get("validation_eval_samples", 500)),
        "final_test_samples": int(data_cfg.get("max_eval_samples", 500)),
        "dataset_identity": data_identity,
        "validation_subset": {
            "size": subset_manifest["subset_size"],
            "seed": subset_manifest["subset_seed"],
            "selection_method": SUBSET_METHOD,
            "manifest_sha256": subset_manifest["manifest_sha256"],
        },
    }
    digest = hashlib.sha256(json.dumps(evaluator_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return {"evaluator_config": evaluator_config, "dataset_identity": data_identity,
            "dataset_rows": len(dataset), "subset_manifest": subset_manifest}, digest


def _validate_existing_evaluation(eval_cfg: dict, eval_output: Path, identity: dict,
                                  evaluator_config: dict, evaluator_digest: str) -> bool:
    marker_path = eval_output / "run_complete.json"
    complete = marker_path.is_file()
    if complete:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("stage") != "evaluation" or marker.get("seed") != SEED:
            raise ValueError("Existing evaluation completion marker has the wrong stage or seed")
        if marker.get("config_sha256") != config_digest(eval_cfg):
            raise ValueError("Existing evaluation completion marker has a different config digest")
        if marker.get("dataset_identity") != identity:
            raise ValueError("Existing evaluation marker has a different validation dataset identity")

    config_path = eval_output / "evaluation_config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != evaluator_config:
            raise ValueError("Existing evaluation_config.json differs from this run; refusing to reset recovery state")
    state_path = eval_output / "evaluation_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("evaluation_config_sha256") != evaluator_digest:
            raise ValueError("Existing evaluation state has a different identity; refusing to discard completed work")
        if state.get("dataset_identity") != identity:
            raise ValueError("Existing evaluation state has a different validation dataset identity")
        if not isinstance(state.get("completed"), dict):
            raise ValueError("Existing evaluation state has an invalid completed map")
        unknown = set(state["completed"]) - set(EXPECTED_LABELS)
        if unknown:
            raise ValueError(f"Evaluation state has unexpected checkpoint labels: {sorted(unknown)}")
        for label, entry in state["completed"].items():
            if (not isinstance(entry, dict) or entry.get("identity_sha256") != evaluator_digest
                    or not isinstance(entry.get("metrics"), dict)):
                raise ValueError(f"Existing evaluation state for {label} has an incompatible identity or metrics record")
        if not config_path.is_file():
            raise ValueError("evaluation_state.json exists without its evaluation_config.json")
    elif complete:
        raise ValueError("Evaluation completion marker exists but evaluation_state.json is missing")

    if complete:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for label in EXPECTED_LABELS:
            entry = state["completed"].get(label, {})
            if entry.get("identity_sha256") != evaluator_digest or not isinstance(entry.get("metrics"), dict):
                raise ValueError(f"Evaluation is marked complete but state for {label} is invalid")
        metrics_path = eval_output / "metrics.csv"
        if not metrics_path.is_file():
            raise ValueError("Evaluation is marked complete but metrics.csv is missing")
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            labels = {row.get("checkpoint") for row in csv.DictReader(stream)}
        if labels != set(EXPECTED_LABELS) or set(state["completed"]) != set(EXPECTED_LABELS):
            raise ValueError("Evaluation is marked complete but does not contain all six checkpoint results")
        for label in EXPECTED_LABELS:
            safe = label.replace("/", "_")
            if not (eval_output / f"{safe}_predictions.jsonl").is_file() or not (eval_output / f"{safe}_response_lengths.csv").is_file():
                raise ValueError(f"Evaluation is marked complete but outputs for {label} are missing")
    return complete


def _write_checkpoint_summary(eval_output: Path) -> Path:
    metrics_path = eval_output / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluator did not produce {metrics_path}")
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = {row["checkpoint"]: row for row in csv.DictReader(stream)}
    if set(rows) != set(EXPECTED_LABELS):
        raise ValueError(f"Expected results for all six checkpoints; found {sorted(rows)}")
    fields = ["checkpoint", "beta", "preference_accuracy", "preference_accuracy_ci_95_low",
              "preference_accuracy_ci_95_high", "preference_logp_margin_mean",
              "implicit_reward_margin_mean", "response_length_tokens_mean",
              "response_fraction_hitting_cap", "preference_examples", "generation_examples"]
    output = eval_output / "checkpoint_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label in EXPECTED_LABELS:
            writer.writerow({key: rows[label].get(key) for key in fields})
    return output


def run(config_path: str | Path, repo_id: str = HUB_REPO_ID, work_dir: str | Path = "/content/dpo-evaluation-only",
        evaluation_plan_path: str | Path = EVALUATION_PLAN_PATH,
        *, restore_fn: Callable | None = None, upload_fn: Callable | None = None,
        evaluate_fn: Callable | None = None, aggregate_fn: Callable | None = None,
        preflight_fn: Callable | None = None) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_experiment_config(config)
    evaluation_plan = yaml.safe_load(Path(evaluation_plan_path).read_text(encoding="utf-8"))
    validate_evaluation_plan(evaluation_plan, config)
    hub = copy.deepcopy(config.get("hub", {}))
    if hub.get("enabled") is not True or hub.get("private") is not True:
        raise ValueError("Private Hugging Face Hub persistence must remain enabled")
    if repo_id != HUB_REPO_ID:
        raise ValueError(f"This recovery workflow is pinned to private repository {HUB_REPO_ID}")
    hub["repo_id"] = repo_id
    # The repo ID was part of the original runner's config digest. Apply the
    # runtime override in memory only; never rewrite the training config file.
    config.setdefault("hub", {})["repo_id"] = repo_id
    if not os.environ.get(str(hub.get("token_env", "HF_TOKEN"))):
        raise RuntimeError("HF_TOKEN is missing from the environment; retrieve it from Colab Secrets without printing it")

    if restore_fn is None:
        restore_fn = restore_run_snapshot
    if upload_fn is None:
        upload_fn = push_run_snapshot
    if evaluate_fn is None:
        from src.evaluate import evaluate as evaluate_fn
    if aggregate_fn is None:
        aggregate_fn = aggregate
    if preflight_fn is None:
        preflight_fn = preflight_evaluation

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ[str(hub.get("token_env", "HF_TOKEN"))])
    api.whoami()
    repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
    if getattr(repo_info, "private", None) is not True:
        raise RuntimeError(f"Expected private Hugging Face repository {repo_id}")

    result_root = Path(config["experiment"]["output_dir"])
    seed_cfg, stage_paths = restore_and_validate_training(config, result_root, hub, restore_fn)
    eval_cfg = make_evaluation_config(seed_cfg, stage_paths, result_root, evaluation_plan)
    eval_hub = copy.deepcopy(hub)
    eval_hub["run_prefix"] = str(evaluation_plan["hub_run_prefix"])
    eval_output = Path(eval_cfg["evaluation"]["output_dir"]) / "validation"
    if eval_output.exists():
        # Only the isolated 200-row namespace is restored or updated.
        restore_fn(eval_output, eval_hub, SEED, "evaluation")
    else:
        restored = restore_fn(eval_output, eval_hub, SEED, "evaluation")
        if not restored:
            eval_output.mkdir(parents=True, exist_ok=True)

    identity_info, evaluator_digest = preflight_fn(eval_cfg, eval_output)
    identity = identity_info["dataset_identity"]
    from src.evaluate import _write_validation_subset_manifest
    _write_validation_subset_manifest(eval_output, identity_info["subset_manifest"])
    complete = _validate_existing_evaluation(
        eval_cfg, eval_output, identity, identity_info["evaluator_config"], evaluator_digest
    )
    eval_config_file = Path(work_dir) / "evaluate.yaml"
    eval_config_file.parent.mkdir(parents=True, exist_ok=True)
    eval_config_file.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")

    if not complete:
        evaluate_fn(str(eval_config_file), split="validation")
        metrics = eval_output / "metrics.csv"
        if not metrics.is_file():
            raise RuntimeError("src.evaluate returned without writing metrics.csv")
        with metrics.open(encoding="utf-8", newline="") as stream:
            evaluated_labels = {row.get("checkpoint") for row in csv.DictReader(stream)}
        if evaluated_labels != set(EXPECTED_LABELS):
            raise RuntimeError(f"Evaluation did not produce all six checkpoints: {sorted(evaluated_labels)}")
        # Evaluation identity is written by src.evaluate from the loaded validation data.
        actual_identity = json.loads((eval_output / "dataset_identity.json").read_text(encoding="utf-8"))
        if actual_identity != identity:
            raise ValueError("src.evaluate produced a validation identity different from CPU preflight")
        _write_json_atomic(eval_output / "run_complete.json", {
            "config_sha256": config_digest(eval_cfg), "seed": SEED,
            "stage": "evaluation", "dataset_identity": identity,
        })

    aggregate_output = eval_output / "validation_metrics.csv"
    aggregate_fn(Path(evaluation_plan["output_dir"]), aggregate_output)
    summary_path = _write_checkpoint_summary(eval_output)
    # Upload is required even for an already complete local evaluation so a prior
    # interrupted upload is retried instead of being mistaken for durable success.
    upload_fn(eval_output, eval_hub, SEED, "evaluation")

    with (eval_output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    print(json.dumps({
        "status": "complete",
        "training": "restored and validated; no training invoked",
        "evaluation_split": "validation",
        "checkpoint_count": len(summary_rows),
        "summary_csv": str(summary_path),
        "metrics": summary_rows,
    }, indent=2))
    return eval_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaled_t4.yaml", help="Runtime config with private Hub repo configured")
    parser.add_argument("--repo-id", default=HUB_REPO_ID)
    parser.add_argument("--work-dir", default="/content/dpo-evaluation-only")
    parser.add_argument("--evaluation-plan", default=str(EVALUATION_PLAN_PATH))
    args = parser.parse_args()
    run(args.config, args.repo_id, args.work_dir, args.evaluation_plan)


if __name__ == "__main__":
    main()
