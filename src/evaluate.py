from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import _flat_token_ids, collate_preferences, load_preference_data, load_preference_test_data
from src.dpo_loss import sequence_logps
from src.modeling import (load_model, load_sft_manifest, load_sft_policy, load_tokenizer,
                          model_config_for_sft_manifest, resolve_device, validate_sft_tokenizer)
from src.training import set_seed


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
    metrics = {
        "checkpoint": label,
        "seed": int(cfg.get("seed", 42)),
        "preference_accuracy": float((margins > 0).float().mean()),
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
    generation_batch = batch_size
    for start in range(0, len(prompts), generation_batch):
        group = prompts[start : start + generation_batch]
        inp = tokenizer.pad({"input_ids": group}, padding=True, return_tensors="pt").to(device)
        generated = model.generate(
            **inp,
            max_new_tokens=int(cfg["evaluation"].get("max_new_tokens", 128)),
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
            predictions.append({
                "index": start + offset,
                "prompt_id": dataset.rows[start + offset].get("prompt_id"),
                "prompt": dataset.rows[start + offset].get("prompt", ""),
                "response": response,
                "response_tokens_including_eos": len(ids),
            })
    safe_label = label.replace("/", "_")
    with (output_dir / f"{safe_label}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics.update({
        "response_length_tokens_mean": float(np.mean(lengths)) if lengths else 0.0,
        "response_length_tokens_median": float(np.median(lengths)) if lengths else 0.0,
        "generation_examples": len(lengths),
        "generation_max_new_tokens": int(cfg["evaluation"].get("max_new_tokens", 128)),
        "generation_do_sample": False,
    })
    with (output_dir / f"{safe_label}_response_lengths.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "prompt_id", "response_tokens_including_eos"])
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


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
    if selected_split == "validation":
        _, dataset = load_preference_data(data_cfg, tokenizer)
    elif selected_split == "test":
        dataset = load_preference_test_data(data_cfg, tokenizer)
    else:
        raise ValueError("evaluation split must be 'validation' or 'test'")

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
        beta = beta_by_checkpoint.get(label)
        if beta is not None and reference_logps is None:
            raise FileNotFoundError("DPO reward margins need the saved SFT reference adapter")
        result = evaluate_checkpoint(label, path, cfg, tokenizer, dataset, device, output, reference_logps, beta)
        result["split"] = selected_split
        results.append(result)
    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (output / "evaluation_config.json").write_text(json.dumps({
        "seed": int(cfg.get("seed", 42)),
        "split": selected_split,
        "max_new_tokens": int(cfg["evaluation"].get("max_new_tokens", 128)),
        "do_sample": False,
        "batch_size": int(cfg["evaluation"].get("batch_size", 2)),
        "checkpoints": {label: (str(path) if path is not None else "base") for label, path in selected.items()},
        "beta_by_checkpoint": beta_by_checkpoint,
    }, indent=2), encoding="utf-8")
    if results:
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
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
