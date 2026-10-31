"""Run the configured time-boxed SFT/DPO comparison from one SFT start per seed."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml

from src.hub import (ensure_hub_repo, latest_checkpoint, push_run_snapshot,
                     resolve_resume_checkpoint, restore_run_snapshot)


def write_config(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def _config_digest(config: dict) -> str:
    stable = copy.deepcopy(config)
    stable.get("training", {}).pop("resume_from_checkpoint", None)
    row_ids_path = stable.get("data", {}).get("train_row_ids_file")
    if row_ids_path:
        row_ids_file = Path(row_ids_path)
        if row_ids_file.is_file():
            stable["data"]["train_row_ids_sha256"] = hashlib.sha256(row_ids_file.read_bytes()).hexdigest()
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_complete(output_dir: Path, config: dict) -> bool:
    marker = output_dir / "run_complete.json"
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    expected = _config_digest(config)
    if saved.get("config_sha256") != expected:
        raise ValueError(
            f"{output_dir} contains a completed run with different settings. "
            "Choose a new output_dir/run_prefix to preserve both runs."
        )
    identity_path = output_dir / "dataset_identity.json"
    if not identity_path.is_file():
        raise ValueError(f"Completed stage {output_dir} has no dataset_identity.json")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if saved.get("dataset_identity") != identity:
        raise ValueError(f"Completed stage {output_dir} has a different dataset fingerprint identity")
    return True


def _mark_complete(output_dir: Path, config: dict, settings: dict, seed: int, stage: str) -> None:
    identity_path = output_dir / "dataset_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"Cannot mark {stage} complete without {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    (output_dir / "run_complete.json").write_text(
        json.dumps({"config_sha256": _config_digest(config), "seed": seed, "stage": stage, "dataset_identity": identity}, indent=2),
        encoding="utf-8",
    )
    # A failure is fatal. A later invocation re-uploads this locally complete
    # folder before it treats the stage as durably complete.
    push_run_snapshot(output_dir, settings, seed, stage)


def _record_experiment_dataset_identity(result_root: Path, identity: dict, run_key: str) -> None:
    path = result_root / "experiment_identity.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "dataset": identity["dataset"], "revision": identity["revision"], "split_fingerprints": {}, "run_identities": {}
    }
    if existing["dataset"] != identity["dataset"] or existing["revision"] != identity["revision"]:
        raise ValueError("Dataset name or immutable revision changed within this experiment")
    split = identity["split"]
    prior_fingerprint = existing["split_fingerprints"].get(split)
    if prior_fingerprint is not None and prior_fingerprint != identity["fingerprint"]:
        raise ValueError(f"Dataset fingerprint changed for split {split!r}")
    prior_identity = existing["run_identities"].get(run_key)
    if prior_identity is not None and prior_identity != identity:
        raise ValueError(f"Dataset or row identity changed for experiment run {run_key!r}")
    existing["split_fingerprints"][split] = identity["fingerprint"]
    existing["run_identities"][run_key] = identity
    payload = json.dumps(existing, indent=2)
    for target in (path, result_root / "summary" / "experiment_identity.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)


def _require_uploaded_snapshot(output_dir: Path, settings: dict, seed: int, stage: str) -> None:
    # `push_run_snapshot` raises for any auth/network failure. Never convert it
    # into a successful skip; the same runner invocation can be retried later.
    push_run_snapshot(output_dir, settings, seed, stage)


def _completed_stage_uploaded(output_dir: Path, config: dict, settings: dict, seed: int, stage: str) -> bool:
    if not _stage_complete(output_dir, config):
        return False
    _require_uploaded_snapshot(output_dir, settings, seed, stage)
    return True


def _record_run(summary_root: Path, settings: dict, seed: int, stage: str, arm: str, output_dir: Path, config: dict) -> None:
    """Persist a compact run ledger after each arm and mirror it to the Hub."""
    summary_root.mkdir(parents=True, exist_ok=True)
    path = summary_root / "run_status.csv"
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    rows = [row for row in rows if (row["seed"], row["stage"], row["arm"]) != (str(seed), stage, arm)]
    identity_path = output_dir / "dataset_identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    elif (output_dir / "selection_experiment.json").is_file():
        selection = json.loads((output_dir / "selection_experiment.json").read_text(encoding="utf-8"))
        identity = {"revision": selection.get("dataset_revision"), "fingerprint": selection.get("dataset_fingerprint")}
    else:
        identity = {}
    rows.append({
        "seed": seed,
        "stage": stage,
        "arm": arm,
        "status": "complete",
        "output_dir": str(output_dir),
        "config_sha256": _config_digest(config),
        "dataset_revision": identity.get("revision"),
        "dataset_fingerprint": identity.get("fingerprint"),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "stage", "arm", "status", "output_dir", "config_sha256", "dataset_revision", "dataset_fingerprint", "recorded_utc"])
        writer.writeheader()
        writer.writerows(rows)
    push_run_snapshot(summary_root, settings, seed, "summary")


def run_experiments(config_path: str | Path) -> Path:
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    experiment = base.get("experiment", {})
    seeds = [int(value) for value in experiment.get("seeds", [int(base.get("seed", 42))])]
    subset_size = int(experiment.get("selection_subset_size", 0))
    beta_values = [float(value) for value in experiment.get("beta_values", [0.1, 0.5])]
    selection_beta = float(experiment.get("selection_beta", 0.1))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("experiment.seeds must be a non-empty list of unique seeds")
    if subset_size < 1 or len(beta_values) < 2:
        raise ValueError("Configure a positive selection subset size and beta values for 0.1 and 0.5")
    hub_settings = base.get("hub", {})
    if not hub_settings.get("enabled", False):
        raise ValueError("Hub persistence must be enabled for the scaled Colab experiment")
    revision = str(base.get("data", {}).get("revision", ""))
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        raise ValueError("Pin data.revision to the immutable 40-character UltraFeedback commit SHA")
    ensure_hub_repo(hub_settings)

    project_root = Path(__file__).resolve().parents[1]
    result_root = Path(base.get("experiment", {}).get("output_dir", "results/scaled-t4-512"))
    config_root = result_root / "configs"
    log_root = result_root / "logs"
    summary_root = result_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    all_evaluation_roots = []

    for seed in seeds:
        seed_root = result_root / f"seed-{seed}"
        sft_dir = seed_root / "sft"
        seed_config = copy.deepcopy(base)
        seed_config["seed"] = seed
        seed_config["training"]["output_dir"] = str(sft_dir)
        seed_config["training"]["resume_from_checkpoint"] = None
        seed_config["dpo"]["sft_checkpoint"] = str(sft_dir)
        seed_config["dpo"]["output_dir"] = str(seed_root / "dpo-beta-0.1")
        seed_config["dpo"]["reference_logps_path"] = str(seed_root / "reference" / "full.pt")
        sft_config_path = write_config(seed_config, config_root / f"seed-{seed}" / "sft.yaml")
        sft_log = log_root / f"seed-{seed}" / "sft.log"
        restore_run_snapshot(sft_dir, hub_settings, seed, "sft")
        if (sft_dir / "dataset_identity.json").is_file():
            _record_experiment_dataset_identity(result_root, json.loads((sft_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/sft")
        sft_complete = _completed_stage_uploaded(sft_dir, seed_config, hub_settings, seed, "sft")
        if not sft_complete:
            resume_sft = latest_checkpoint(sft_dir)
            if resume_sft is None:
                configured_resume = (experiment.get("resume_sft") or {}).get(str(seed))
                resume_sft = resolve_resume_checkpoint(configured_resume, hub_settings)
            seed_config["training"]["resume_from_checkpoint"] = str(resume_sft) if resume_sft else None
            sft_config_path = write_config(seed_config, config_root / f"seed-{seed}" / "sft.yaml")
            run_logged([sys.executable, "-m", "src.train_sft", "--config", str(sft_config_path)], sft_log)
            (sft_dir / "console.log").write_text(sft_log.read_text(encoding="utf-8"), encoding="utf-8")
            _mark_complete(sft_dir, seed_config, hub_settings, seed, "sft")
        _record_experiment_dataset_identity(result_root, json.loads((sft_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/sft")
        _record_run(summary_root, hub_settings, seed, "sft", "sft", sft_dir, seed_config)

        selection_dir = seed_root / "selection"
        selection_log = log_root / f"seed-{seed}" / "selection.log"
        restore_run_snapshot(selection_dir, hub_settings, seed, "selection")
        selection_metadata_path = selection_dir / "selection_experiment.json"
        selection_valid = False
        if selection_metadata_path.is_file():
            selection_metadata = json.loads(selection_metadata_path.read_text(encoding="utf-8"))
            selection_valid = (
                int(selection_metadata.get("seed", -1)) == seed
                and int(selection_metadata.get("subset_size", -1)) == subset_size
                and selection_metadata.get("dataset") == seed_config["data"]["name"]
                and selection_metadata.get("dataset_revision") == seed_config["data"].get("revision", "main")
                and selection_metadata.get("max_train_samples") == seed_config["data"].get("max_train_samples")
            )
        if not selection_valid or not (selection_dir / "margin_subset.json").is_file() or not (selection_dir / "random_subset.json").is_file():
            run_logged([
                sys.executable, "-m", "scripts.select_preference_data",
                "--config", str(sft_config_path),
                "--subset-size", str(subset_size),
                "--output-dir", str(selection_dir),
            ], selection_log)
            (selection_dir / "console.log").write_text(selection_log.read_text(encoding="utf-8"), encoding="utf-8")
        # Upload even when the local selection artifacts already exist. This
        # retries a previously failed upload instead of recording a false skip.
        push_run_snapshot(selection_dir, hub_settings, seed, "selection")
        selection_identity = json.loads(selection_metadata_path.read_text(encoding="utf-8"))
        identity = {
            "dataset": selection_identity["dataset"],
            "revision": selection_identity["dataset_revision"],
            "split": selection_identity["split"],
            "fingerprint": selection_identity["dataset_fingerprint"],
            "selected_rows": selection_identity["candidate_count"],
            "selected_row_ids_sha256": hashlib.sha256(
                json.dumps(json.loads((selection_dir / "full_candidate_row_ids.json").read_text(encoding="utf-8")), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "validation_rows": selection_identity["validation_count"],
            "validation_row_ids_sha256": None,
        }
        _record_experiment_dataset_identity(result_root, identity, f"seed-{seed}/selection-candidates")
        _record_run(summary_root, hub_settings, seed, "selection", "margin_random_full", selection_dir, seed_config)

        run_descriptors = []
        for beta in beta_values:
            label = f"dpo-beta-{beta:g}"
            output_dir = seed_root / label
            cfg = copy.deepcopy(seed_config)
            cfg["training"]["resume_from_checkpoint"] = None
            cfg["dpo"].update({
                "beta": beta,
                "output_dir": str(output_dir),
                "reference_logps_path": str(seed_root / "reference" / "full.pt"),
            })
            restore_run_snapshot(output_dir, hub_settings, seed, "dpo")
            if (output_dir / "dataset_identity.json").is_file():
                _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
            cfg_path = write_config(cfg, config_root / f"seed-{seed}" / f"{label}.yaml")
            log_path = log_root / f"seed-{seed}" / f"{label}.log"
            command = [sys.executable, "-m", "src.train_dpo", "--config", str(cfg_path)]
            if _completed_stage_uploaded(output_dir, cfg, hub_settings, seed, "dpo"):
                _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
                run_descriptors.append((label.replace("-", "_"), output_dir, beta, None))
                _record_run(summary_root, hub_settings, seed, "dpo", label, output_dir, cfg)
                continue
            resume = latest_checkpoint(output_dir)
            if resume is None:
                resume_uri = (experiment.get("resume_checkpoints") or {}).get(f"seed-{seed}/{label}")
                resume = resolve_resume_checkpoint(resume_uri, hub_settings)
            if resume:
                command.extend(["--resume", str(resume)])
            run_logged(command, log_path)
            (output_dir / "console.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
            _mark_complete(output_dir, cfg, hub_settings, seed, "dpo")
            _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
            _record_run(summary_root, hub_settings, seed, "dpo", label, output_dir, cfg)
            run_descriptors.append((label.replace("-", "_"), output_dir, beta, None))

        for arm in ("margin", "random"):
            label = f"dpo-{arm}"
            output_dir = seed_root / label
            cfg = copy.deepcopy(seed_config)
            cfg["training"]["resume_from_checkpoint"] = None
            cfg["data"]["train_row_ids_file"] = str(selection_dir / f"{arm}_subset.json")
            cfg["dpo"].update({
                "beta": selection_beta,
                "output_dir": str(output_dir),
                "reference_logps_path": str(seed_root / "reference" / f"{arm}.pt"),
            })
            restore_run_snapshot(output_dir, hub_settings, seed, "dpo")
            if (output_dir / "dataset_identity.json").is_file():
                _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
            cfg_path = write_config(cfg, config_root / f"seed-{seed}" / f"{label}.yaml")
            log_path = log_root / f"seed-{seed}" / f"{label}.log"
            command = [sys.executable, "-m", "src.train_dpo", "--config", str(cfg_path)]
            if _completed_stage_uploaded(output_dir, cfg, hub_settings, seed, "dpo"):
                _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
                run_descriptors.append((label.replace("-", "_"), output_dir, selection_beta, None))
                _record_run(summary_root, hub_settings, seed, "dpo", label, output_dir, cfg)
                continue
            resume = latest_checkpoint(output_dir)
            if resume is None:
                resume_uri = (experiment.get("resume_checkpoints") or {}).get(f"seed-{seed}/{label}")
                resume = resolve_resume_checkpoint(resume_uri, hub_settings)
            if resume:
                command.extend(["--resume", str(resume)])
            run_logged(command, log_path)
            (output_dir / "console.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
            _mark_complete(output_dir, cfg, hub_settings, seed, "dpo")
            _record_experiment_dataset_identity(result_root, json.loads((output_dir / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/{label}")
            _record_run(summary_root, hub_settings, seed, "dpo", label, output_dir, cfg)
            run_descriptors.append((label.replace("-", "_"), output_dir, selection_beta, None))

        eval_config = copy.deepcopy(seed_config)
        eval_config["training"]["resume_from_checkpoint"] = None
        eval_dir = result_root / "evaluation" / f"seed-{seed}"
        eval_config["evaluation"].update({"split": "validation", "output_dir": str(eval_dir)})
        eval_config["evaluation"]["checkpoints"] = {"base": "base", "sft": str(sft_dir)}
        eval_config["evaluation"]["beta_by_checkpoint"] = {}
        for label, output_dir, beta, _ in run_descriptors:
            eval_config["evaluation"]["checkpoints"][label] = str(output_dir)
            eval_config["evaluation"]["beta_by_checkpoint"][label] = beta
        eval_path = write_config(eval_config, config_root / f"seed-{seed}" / "evaluate.yaml")
        eval_log = log_root / f"seed-{seed}" / "evaluation-validation.log"
        eval_output = eval_dir / "validation"
        restore_run_snapshot(eval_output, hub_settings, seed, "evaluation")
        if (eval_output / "dataset_identity.json").is_file():
            _record_experiment_dataset_identity(result_root, json.loads((eval_output / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/evaluation-validation")
        if not _completed_stage_uploaded(eval_output, eval_config, hub_settings, seed, "evaluation"):
            run_logged([sys.executable, "-m", "src.evaluate", "--config", str(eval_path), "--split", "validation"], eval_log)
            (eval_output / "console.log").write_text(eval_log.read_text(encoding="utf-8"), encoding="utf-8")
            _mark_complete(eval_output, eval_config, hub_settings, seed, "evaluation")
        _record_experiment_dataset_identity(result_root, json.loads((eval_output / "dataset_identity.json").read_text(encoding="utf-8")), f"seed-{seed}/evaluation-validation")
        _record_run(summary_root, hub_settings, seed, "evaluation", "validation_all_checkpoints", eval_output, eval_config)
        all_evaluation_roots.append(eval_dir)

        run_logged([
            sys.executable, "-m", "scripts.aggregate_results",
            "--root", str(result_root / "evaluation"),
            "--output", str(summary_root / "validation_metrics.csv"),
        ], log_root / "aggregate.log")
        push_run_snapshot(summary_root, hub_settings, seed, "summary")

    record = {
        "config": str(Path(config_path)),
        "seeds": seeds,
        "training_pairs_per_full_arm": int(base["data"].get("max_train_samples") or 0),
        "selection_subset_size": subset_size,
        "beta_values": beta_values,
        "evaluation_roots": [str(path) for path in all_evaluation_roots],
        "dataset_identity": json.loads((result_root / "experiment_identity.json").read_text(encoding="utf-8")),
    }
    (result_root / "experiment_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    push_run_snapshot(result_root, hub_settings, int(seeds[0]), "experiment")
    return result_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaled_t4.yaml")
    args = parser.parse_args()
    print(f"Experiment outputs: {run_experiments(args.config)}")


if __name__ == "__main__":
    main()
