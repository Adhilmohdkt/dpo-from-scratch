from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import (SOURCE_ROW_ID, PreferenceDataset, _flat_token_ids,
                      collate_preferences, load_preference_eval_data)
from src.dpo_loss import sequence_logps
from src.hub import ensure_hub_repo, push_run_snapshot
from src.modeling import (load_model, load_sft_manifest, load_sft_policy, load_tokenizer,
                          model_config_for_sft_manifest, resolve_device, validate_sft_tokenizer)
from src.training import set_seed


def wilson_interval(successes: float, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval; fractional successes support half-credit ties."""
    if trials < 1 or not 0.0 <= successes <= trials:
        raise ValueError("Wilson interval needs 0 <= successes <= trials and trials > 0")
    n = float(trials)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def preference_accuracy(margins: torch.Tensor, tie_tolerance: float = 1e-8) -> dict[str, float | int]:
    """Return chosen-vs-rejected accuracy, assigning exact ties half credit."""
    flat = margins.detach().float().cpu().reshape(-1)
    if flat.numel() == 0 or not torch.isfinite(flat).all():
        raise ValueError("Preference accuracy requires finite, non-empty margins")
    wins = int((flat > tie_tolerance).sum())
    ties = int((flat.abs() <= tie_tolerance).sum())
    trials = int(flat.numel())
    effective_successes = wins + 0.5 * ties
    low, high = wilson_interval(effective_successes, trials)
    return {
        "accuracy": effective_successes / trials,
        "ci_95_low": low,
        "ci_95_high": high,
        "ties": ties,
        "effective_successes": effective_successes,
    }


def generation_length_summary(lengths: list[int], max_new_tokens: int) -> dict[str, float | int]:
    """Summarize generated token counts and the fraction that reached the cap."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "cap_fraction": 0.0, "examples": 0}
    if any(length < 0 for length in lengths):
        raise ValueError("Generated response lengths cannot be negative")
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "cap_fraction": float(np.mean([length >= max_new_tokens for length in lengths])),
        "examples": len(lengths),
    }


@torch.no_grad()
def score_pairs(model, dataset, tokenizer, device, batch_size: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda xs: collate_preferences(xs, tokenizer.pad_token_id))
    scores = torch.empty((len(dataset), 2), dtype=torch.float32)
    for batch in loader:
        chosen_ids = batch["chosen_input_ids"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        chosen = sequence_logps(
            model(input_ids=chosen_ids, attention_mask=batch["chosen_attention_mask"].to(device)).logits,
            chosen_ids,
            batch["chosen_response_mask"].to(device),
        )
        rejected = sequence_logps(
            model(input_ids=rejected_ids, attention_mask=batch["rejected_attention_mask"].to(device)).logits,
            rejected_ids,
            batch["rejected_response_mask"].to(device),
        )
        idx = batch["indices"]
        scores[idx, 0] = chosen.float().cpu()
        scores[idx, 1] = rejected.float().cpu()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite held-out sequence log-probabilities")
    return scores


def _prompt_ids(row, tokenizer):
    messages = row["chosen"]
    if isinstance(messages, str):
        prompt_messages = [{"role": "user", "content": row.get("prompt", "")}]
    else:
        first_answer = next((j for j, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
        prompt_messages = [{"role": m["role"], "content": m["content"]} for m in messages[:first_answer]]
    return _flat_token_ids(tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True))


@torch.no_grad()
def evaluate_checkpoint(label, path, cfg, tokenizer, dataset, device, output_dir, reference_logps, beta):
    model_cfg = cfg["model"]
    if path is None or str(path).lower() in {"base", "__base__"}:
        model = load_model(model_cfg, adapter=False)
    else:
        model = load_sft_policy(model_cfg, str(path), trainable=False)
    model.to(device).eval()
    batch_size = int(cfg["evaluation"].get("batch_size", 2))
    scores = score_pairs(model, dataset, tokenizer, device, batch_size)
    margins = scores[:, 0] - scores[:, 1]
    accuracy = preference_accuracy(margins)
    metrics = {
        "checkpoint": label,
        "seed": int(cfg.get("seed", 42)),
        "preference_accuracy": accuracy["accuracy"],
        "preference_accuracy_ci_95_low": accuracy["ci_95_low"],
        "preference_accuracy_ci_95_high": accuracy["ci_95_high"],
        "preference_accuracy_tie_count": accuracy["ties"],
        "preference_accuracy_interval": "Wilson score; exact ties contribute half a success",
        "preference_logp_margin_mean": float(margins.mean()),
        "preference_logp_margin_median": float(margins.median()),
        "preference_examples": len(dataset),
    }
    if reference_logps is not None and beta is not None:
        chosen_rewards = beta * (scores[:, 0] - reference_logps[:, 0])
        rejected_rewards = beta * (scores[:, 1] - reference_logps[:, 1])
        reward_margins = chosen_rewards - rejected_rewards
        metrics.update({
            "beta": beta,
            "chosen_reward_mean": float(chosen_rewards.mean()),
            "rejected_reward_mean": float(rejected_rewards.mean()),
            "implicit_reward_margin_mean": float(reward_margins.mean()),
        })
    else:
        metrics.update({"beta": beta, "chosen_reward_mean": None, "rejected_reward_mean": None, "implicit_reward_margin_mean": None})

    tokenizer.padding_side = "left"
    prompts = [_prompt_ids(dataset.rows[i], tokenizer) for i in range(len(dataset))]
    predictions = []
    lengths = []
    max_new_tokens = int(cfg["evaluation"].get("max_new_tokens", 256))
    generation_batch = batch_size
    for start in range(0, len(prompts), generation_batch):
        group = prompts[start : start + generation_batch]
        inp = tokenizer.pad({"input_ids": group}, padding=True, return_tensors="pt").to(device)
        generated = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = generated[:, inp["input_ids"].shape[1] :]
        for offset, token_row in enumerate(continuation):
            ids = token_row.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            response = tokenizer.decode(ids, skip_special_tokens=True)
            lengths.append(len(ids))
            hit_cap = len(ids) >= max_new_tokens
            predictions.append({
                "index": start + offset,
                "prompt_id": dataset.rows[start + offset].get("prompt_id"),
                "prompt": dataset.rows[start + offset].get("prompt", ""),
                "response": response,
                "response_tokens_including_eos": len(ids),
                "hit_generation_cap": hit_cap,
            })
    safe_label = label.replace("/", "_")
    with (output_dir / f"{safe_label}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    length_summary = generation_length_summary(lengths, max_new_tokens)
    metrics.update({
        "response_length_tokens_mean": length_summary["mean"],
        "response_length_tokens_median": length_summary["median"],
        "response_fraction_hitting_cap": length_summary["cap_fraction"],
        "generation_examples": length_summary["examples"],
        "generation_max_new_tokens": max_new_tokens,
        "generation_do_sample": False,
    })
    with (output_dir / f"{safe_label}_response_lengths.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "prompt_id", "response_tokens_including_eos", "hit_generation_cap"])
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _checkpoint_sha256(path, model_cfg: dict) -> str:
    digest = hashlib.sha256()
    if path is None or str(path).lower() in {"base", "__base__"}:
        digest.update(json.dumps(model_cfg, sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()
    checkpoint = Path(path)
    adapter_files = sorted(checkpoint.glob("adapter*"))
    if not adapter_files:
        raise FileNotFoundError(f"No adapter files found for evaluation identity at {checkpoint}")
    for file in adapter_files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_metrics(output: Path, results: list[dict]) -> None:
    _atomic_json(output / "metrics.json", results)
    if results:
        fields = list(dict.fromkeys(key for result in results for key in result))
        temporary = output / "metrics.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        os.replace(temporary, output / "metrics.csv")


def _select_validation_subset(dataset, identity: dict, size: int | None, seed: int):
    """Select a stable subset from an already isolated validation partition.

    A SHA-256 keyed ranking avoids dependence on Python/NumPy RNG implementation
    details. The input dataset must already be the validation slice; this helper
    never loads or selects from the held-out test range.
    """
    if size is None:
        return dataset, identity, None
    size = int(size)
    source_size = len(dataset)
    if size < 1 or size > source_size:
        raise ValueError(f"validation_subset_size must be between 1 and {source_size}; got {size}")
    if identity.get("split") != "test_prefs":
        raise ValueError("Validation subsampling requires the configured held-out preference split")
    if SOURCE_ROW_ID not in dataset.rows.column_names:
        raise ValueError("Validation dataset has no source row IDs for reproducible subsampling")

    source_ids = [int(value) for value in dataset.rows[SOURCE_ROW_ID]]
    if len(source_ids) != source_size or len(set(source_ids)) != source_size:
        raise ValueError("Validation source row IDs must be unique and match the loaded validation partition")
    if size == source_size:
        selected_positions = list(range(source_size))
    else:
        def rank(source_id: int) -> bytes:
            key = f"{identity['dataset']}|{identity['revision']}|{identity['fingerprint']}|{seed}|{source_id}"
            return hashlib.sha256(key.encode("utf-8")).digest()

        chosen_ids = set(sorted(source_ids, key=lambda row_id: (rank(row_id), row_id))[:size])
        selected_positions = [i for i, row_id in enumerate(source_ids) if row_id in chosen_ids]

    selected_rows = dataset.rows.select(selected_positions)
    selected_ids = [source_ids[i] for i in selected_positions]
    prompt_ids = list(selected_rows["prompt_id"]) if "prompt_id" in selected_rows.column_names else []
    prompt_ids_payload = json.dumps(prompt_ids, ensure_ascii=False, separators=(",", ":"), default=str)
    selected_ids_payload = json.dumps(selected_ids, separators=(",", ":"))
    manifest = {
        "dataset": identity["dataset"],
        "revision": identity["revision"],
        "split": identity["split"],
        "source_dataset_fingerprint": identity["fingerprint"],
        "source_validation_rows": source_size,
        "source_validation_row_ids_sha256": identity["selected_row_ids_sha256"],
        "subset_size": size,
        "subset_seed": int(seed),
        "selection_method": "sha256-ranked sampling without replacement; retained source-validation order",
        "selected_indices_in_validation_partition": selected_positions,
        "selected_source_row_ids": selected_ids,
        "selected_prompt_ids": prompt_ids,
        "selected_source_row_ids_sha256": hashlib.sha256(selected_ids_payload.encode("utf-8")).hexdigest(),
        "selected_prompt_ids_sha256": hashlib.sha256(prompt_ids_payload.encode("utf-8")).hexdigest(),
        "selected_dataset_fingerprint": getattr(selected_rows, "_fingerprint", None),
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()

    subset_identity = dict(identity)
    subset_identity.update({
        "source_validation_rows": source_size,
        "source_validation_row_ids_sha256": identity["selected_row_ids_sha256"],
        "selected_rows": size,
        "selected_row_ids_sha256": manifest["selected_source_row_ids_sha256"],
        "validation_subset_seed": int(seed),
        "validation_subset_manifest_sha256": manifest["manifest_sha256"],
    })
    subset = PreferenceDataset(selected_rows, dataset.tokenizer, dataset.max_length)
    return subset, subset_identity, manifest


def _write_validation_subset_manifest(output: Path, manifest: dict | None) -> None:
    if manifest is None:
        return
    path = output / "validation_subset.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("Existing validation subset manifest differs from the requested subset")
        return
    _atomic_json(path, manifest)


def evaluate(config_path: str, checkpoints: list[str] | None = None, output_dir: str | None = None, split: str | None = None, betas: list[str] | None = None):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg["training"].get("device", "auto"))
    sft_path = Path(cfg["dpo"].get("sft_checkpoint", cfg["training"]["output_dir"]))
    if (sft_path / "adapter_config.json").exists():
        manifest = load_sft_manifest(sft_path)
        cfg["model"] = model_config_for_sft_manifest(cfg["model"], manifest)
    tokenizer = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"].get("trust_remote_code", False), cfg["model"].get("revision"))
    if (sft_path / "adapter_config.json").exists():
        validate_sft_tokenizer(tokenizer, manifest)
    data_cfg = {**cfg["data"], "seed": int(cfg.get("seed", 42))}
    selected_split = split or cfg.get("evaluation", {}).get("split", "validation")
    if selected_split not in {"validation", "test"}:
        raise ValueError("evaluation split must be 'validation' or 'test'")
    dataset, data_identity = load_preference_eval_data(data_cfg, tokenizer, split=selected_split, return_identity=True)
    subset_manifest = None
    subset_size = cfg.get("evaluation", {}).get("validation_subset_size")
    if subset_size is not None:
        if selected_split != "validation":
            raise ValueError("validation_subset_size is supported only for the validation split")
        subset_seed = int(cfg["evaluation"].get("validation_subset_seed", int(cfg.get("seed", 42)) + int(data_cfg.get("eval_split_seed_offset", 2)) + 1))
        dataset, data_identity, subset_manifest = _select_validation_subset(dataset, data_identity, int(subset_size), subset_seed)

    selected = {}
    if checkpoints:
        for item in checkpoints:
            if "=" not in item:
                raise ValueError("Each --checkpoint must be LABEL=PATH; use LABEL=base for the base model")
            label, path = item.split("=", 1)
            selected[label] = None if path.lower() == "base" else path
    else:
        selected = cfg.get("evaluation", {}).get("checkpoints", {"base": None})
    beta_by_checkpoint = dict(cfg.get("evaluation", {}).get("beta_by_checkpoint", {}))
    for item in betas or []:
        if "=" not in item:
            raise ValueError("Each --beta must be LABEL=VALUE")
        label, value = item.split("=", 1)
        beta_by_checkpoint[label] = float(value)
    if selected_split == "test":
        dpo_models = [label for label in selected if beta_by_checkpoint.get(label) is not None]
        if len(dpo_models) > 1:
            raise ValueError("Use validation for beta selection; final test evaluation accepts at most one beta-tagged DPO checkpoint")

    output_root = Path(output_dir or cfg.get("evaluation", {}).get("output_dir", "outputs/evaluation"))
    output = output_root / selected_split
    output.mkdir(parents=True, exist_ok=True)
    _write_validation_subset_manifest(output, subset_manifest)
    hub_settings = cfg.get("hub", {})
    if hub_settings.get("enabled", False):
        ensure_hub_repo(hub_settings)
    (output / "dataset_identity.json").write_text(json.dumps(data_identity, indent=2), encoding="utf-8")
    evaluation_config = {
        "seed": int(cfg.get("seed", 42)),
        "split": selected_split,
        "max_new_tokens": int(cfg["evaluation"].get("max_new_tokens", 256)),
        "do_sample": False,
        "batch_size": int(cfg["evaluation"].get("batch_size", 2)),
        "checkpoints": {label: (str(path) if path is not None else "base") for label, path in selected.items()},
        "checkpoint_sha256": {label: _checkpoint_sha256(path, cfg["model"]) for label, path in selected.items()},
        "beta_by_checkpoint": beta_by_checkpoint,
        "validation_eval_samples": int(data_cfg.get("validation_eval_samples", 500)),
        "final_test_samples": int(data_cfg.get("max_eval_samples", 500)),
        "dataset_identity": data_identity,
    }
    if subset_manifest is not None:
        evaluation_config["validation_subset"] = {
            "size": subset_manifest["subset_size"],
            "seed": subset_manifest["subset_seed"],
            "selection_method": "sha256_ranked_without_replacement_v1",
            "manifest_sha256": subset_manifest["manifest_sha256"],
        }
    evaluation_config_sha256 = hashlib.sha256(json.dumps(evaluation_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    _atomic_json(output / "evaluation_config.json", evaluation_config)
    state_path = output / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("evaluation_config_sha256") != evaluation_config_sha256:
        state = {"evaluation_config_sha256": evaluation_config_sha256, "dataset_identity": data_identity, "completed": {}}
    completed = state.setdefault("completed", {})
    results_by_label = {}
    for label, entry in completed.items():
        safe_label = label.replace("/", "_")
        files_exist = (
            (output / f"{safe_label}_predictions.jsonl").is_file()
            and (output / f"{safe_label}_response_lengths.csv").is_file()
        )
        if entry.get("identity_sha256") == evaluation_config_sha256 and isinstance(entry.get("metrics"), dict) and files_exist:
            results_by_label[label] = entry["metrics"]
    if not results_by_label:
        (output / "metrics.csv").unlink(missing_ok=True)
    _write_metrics(output, [results_by_label[label] for label in selected if label in results_by_label])
    # Compute the frozen SFT reference scores once, on the chosen evaluation
    # split, for reward-margin diagnostics of each DPO checkpoint.
    reference_logps = None
    if (sft_path / "adapter_config.json").exists():
        reference_model = load_sft_policy(cfg["model"], str(sft_path), trainable=False).to(device).eval()
        reference_logps = score_pairs(reference_model, dataset, tokenizer, device, int(cfg["evaluation"].get("batch_size", 2)))
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = []
    for label, path in selected.items():
        expected_files = (
            output / f"{label.replace('/', '_')}_predictions.jsonl",
            output / f"{label.replace('/', '_')}_response_lengths.csv",
        )
        saved = completed.get(label, {})
        if (
            saved.get("identity_sha256") == evaluation_config_sha256
            and isinstance(saved.get("metrics"), dict)
            and all(file.is_file() for file in expected_files)
        ):
            results.append(saved["metrics"])
            continue
        beta = beta_by_checkpoint.get(label)
        if beta is not None and reference_logps is None:
            raise FileNotFoundError("DPO reward margins need the saved SFT reference adapter")
        result = evaluate_checkpoint(label, path, cfg, tokenizer, dataset, device, output, reference_logps, beta)
        result["split"] = selected_split
        results.append(result)
        completed[label] = {"identity_sha256": evaluation_config_sha256, "metrics": result}
        state["completed"] = completed
        _atomic_json(state_path, state)
        results_by_label[label] = result
        _write_metrics(output, [results_by_label[key] for key in selected if key in results_by_label])
        push_run_snapshot(output, hub_settings, int(cfg.get("seed", 42)), "evaluation")
    push_run_snapshot(output, hub_settings, int(cfg.get("seed", 42)), "evaluation")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", action="append", help="Repeat as LABEL=PATH; PATH=base evaluates the base model")
    parser.add_argument("--beta", action="append", help="Repeat as LABEL=VALUE for implicit reward margin calculations")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir, args.split, args.beta)
