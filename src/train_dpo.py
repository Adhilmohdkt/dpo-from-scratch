from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import datasets
import tokenizers
import transformers
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import load_config
from src.data import PREPROCESSING_VERSION, collate_preferences, load_preference_data
from src.dpo_loss import dpo_loss, sequence_logps
from src.modeling import (base_model_identity, load_sft_manifest, load_sft_policy, load_tokenizer,
                          model_config_for_sft_manifest, resolve_device, validate_sft_tokenizer)
from src.training import append_metrics, checkpoint_metadata, restore_training_state, save_checkpoint, set_seed


def _score(model, ids, masks, attention, device):
    ids, masks, attention = ids.to(device), masks.to(device), attention.to(device)
    return sequence_logps(model(input_ids=ids, attention_mask=attention).logits, ids, masks)


@torch.no_grad()
def precompute_reference(model, dataset, loader, device, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    result = torch.empty((len(dataset), 2), dtype=torch.float32)
    for batch in loader:
        indices = batch["indices"]
        chosen = _score(model, batch["chosen_input_ids"], batch["chosen_response_mask"], batch["chosen_attention_mask"], device)
        rejected = _score(model, batch["rejected_input_ids"], batch["rejected_response_mask"], batch["rejected_attention_mask"], device)
        result[indices, 0] = chosen.float().cpu()
        result[indices, 1] = rejected.float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite reference log-probabilities")
    torch.save(result, cache_path)
    return result


def reference_cache_matches(cache_path: Path, metadata_path: Path, expected: dict, row_count: int) -> bool:
    try:
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            return False
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return isinstance(cached, torch.Tensor) and cached.shape == (row_count, 2) and bool(torch.isfinite(cached).all())
    except Exception:
        return False


def build_reference_cache_metadata(cfg: dict, sft_path: Path, train_dataset, tokenizer, model_identity: dict) -> dict:
    adapter_files = sorted(sft_path.glob("adapter*"))
    if not adapter_files:
        raise FileNotFoundError(f"No saved LoRA adapter files found in {sft_path}")
    adapter_hash = hashlib.sha256()
    for file in adapter_files:
        adapter_hash.update(file.name.encode("utf-8"))
        adapter_hash.update(file.read_bytes())
    from src.modeling import tokenizer_identity
    data_cfg = cfg["data"]
    return {
        "sft_checkpoint": str(sft_path.resolve()),
        "sft_adapter_sha256": adapter_hash.hexdigest(),
        "dataset": data_cfg["name"],
        "dataset_revision": data_cfg.get("revision", "main"),
        "dataset_fingerprint": getattr(train_dataset.rows, "_fingerprint", None),
        "split": data_cfg.get("train_preferences_split", "train_prefs"),
        "rows": len(train_dataset),
        "max_train_samples": data_cfg.get("max_train_samples"),
        "max_validation_samples": data_cfg.get("max_validation_samples"),
        "validation_fraction": data_cfg.get("validation_fraction", 0.02),
        "seed": int(cfg.get("seed", 42)),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_libraries": {
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "datasets": datasets.__version__,
        },
        "max_length": data_cfg.get("max_length", 1024),
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity(tokenizer),
    }


def train(config_path: str, sft_checkpoint: str | None = None, resume: str | None = None, force_reference: bool = False, beta_override: float | None = None):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg["training"].get("device", "auto"))
    if not cfg["model"].get("use_lora", False):
        raise ValueError("Current DPO trainer expects a saved LoRA SFT checkpoint")
    sft_path = Path(sft_checkpoint or cfg["dpo"].get("sft_checkpoint", cfg["training"]["output_dir"]))
    if not (sft_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Expected the saved SFT LoRA adapter at {sft_path}")
    manifest = load_sft_manifest(sft_path)
    cfg["model"] = model_config_for_sft_manifest(cfg["model"], manifest)
    tok = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"].get("trust_remote_code", False), cfg["model"].get("revision"))
    validate_sft_tokenizer(tok, manifest)
    data_cfg = {**cfg["data"], "seed": int(cfg.get("seed", 42))}
    train_ds, valid_ds = load_preference_data(data_cfg, tok)
    batch_size = int(cfg["training"]["batch_size"])
    collate = lambda xs: collate_preferences(xs, tok.pad_token_id)
    loader_generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    valid_generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + 1)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, generator=loader_generator)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, generator=valid_generator)
    cache_path = Path(cfg["dpo"].get("reference_logps_path", "checkpoints/reference_logps.pt"))
    # Validate cache identity as well as shape so stale scores cannot silently
    # be reused after changing the SFT checkpoint or data/tokenization setup.
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    model_identity = base_model_identity(cfg["model"])
    expected = build_reference_cache_metadata(cfg, sft_path, train_ds, tok, model_identity)
    can_use = cache_path.exists() and meta_path.exists() and not force_reference and reference_cache_matches(cache_path, meta_path, expected, len(train_ds))
    if not can_use:
        reference_model = load_sft_policy(cfg["model"], str(sft_path), trainable=False).to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference = precompute_reference(reference_model, train_ds, train_dl, device, cache_path)
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    else:
        reference = torch.load(cache_path, map_location="cpu", weights_only=True)
        if reference.shape != (len(train_ds), 2):
            raise ValueError("Reference-logprob cache has the wrong shape")
    beta = float(beta_override if beta_override is not None else cfg["dpo"]["beta"])
    run_metadata = checkpoint_metadata(cfg, "dpo")
    run_metadata["effective_beta"] = beta
    out = Path(cfg["dpo"].get("output_dir", str(Path(cfg["training"]["output_dir"]) / "dpo")))
    configured_beta = float(cfg["dpo"]["beta"])
    if beta_override is not None and beta != configured_beta:
        out = Path(cfg["training"]["output_dir"]).parent / f"dpo-beta-{beta:g}"
    resume = resume or cfg["training"].get("resume_from_checkpoint")
    policy = load_sft_policy(cfg["model"], str(resume or sft_path), trainable=True, gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False)).to(device)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 0)))
    accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    total_steps = max(1, math.ceil(len(train_dl) / accum) * int(cfg["training"]["epochs"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    step, first_epoch, first_batch = 0, 0, 0
    if resume:
        step, first_epoch, extra = restore_training_state(optimizer, resume, scheduler, run_metadata)
        first_batch = int(extra.get("batch_in_epoch", 0))
    for epoch in range(first_epoch, int(cfg["training"]["epochs"])):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            if epoch == first_epoch and i < first_batch:
                continue
            pc = _score(policy, batch["chosen_input_ids"], batch["chosen_response_mask"], batch["chosen_attention_mask"], device)
            pr = _score(policy, batch["rejected_input_ids"], batch["rejected_response_mask"], batch["rejected_attention_mask"], device)
            refs = reference[batch["indices"]]
            loss, metrics = dpo_loss(pc, pr, refs[:, 0].to(device), refs[:, 1].to(device), beta)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DPO loss")
            group_start = (i // accum) * accum
            group_size = min(accum, len(train_dl) - group_start)
            (loss / group_size).backward()
            if (i + 1) % accum == 0 or i + 1 == len(train_dl):
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["training"].get("max_grad_norm", 1)))
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient norm")
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % int(cfg["training"].get("logging_steps", 10)) == 0:
                    append_metrics(out / "train_metrics.jsonl", {"step": step, "epoch": epoch, "loss": float(loss.detach().cpu()), "reward_accuracy": float(metrics["reward_accuracy"].mean().cpu()), "reward_margin": float(metrics["reward_margin"].mean().cpu()), "beta": beta})
                every = int(cfg["training"].get("save_steps", 100))
                if every > 0 and step % every == 0:
                    save_checkpoint(policy, optimizer, step, epoch, out, {"batch_in_epoch": i + 1}, scheduler, run_metadata)
        if len(valid_ds):
            policy.eval(); eval_correct, eval_seen = 0.0, 0
            with torch.no_grad():
                for vb in valid_dl:
                    pc = _score(policy, vb["chosen_input_ids"], vb["chosen_response_mask"], vb["chosen_attention_mask"], device)
                    pr = _score(policy, vb["rejected_input_ids"], vb["rejected_response_mask"], vb["rejected_attention_mask"], device)
                    eval_correct += float((pc - pr > 0).sum().cpu()); eval_seen += len(pc)
            append_metrics(out / "eval_metrics.jsonl", {"step": step, "epoch": epoch, "preference_accuracy": eval_correct / eval_seen, "beta": beta})
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out); tok.save_pretrained(out)
    save_checkpoint(policy, optimizer, step, int(cfg["training"]["epochs"]), out, {"batch_in_epoch": 0}, scheduler, run_metadata)
    print(f"DPO complete: {out} (step={step}, beta={beta})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--force-reference-recompute", action="store_true")
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()
    train(args.config, args.sft_checkpoint, args.resume, args.force_reference_recompute, args.beta)
