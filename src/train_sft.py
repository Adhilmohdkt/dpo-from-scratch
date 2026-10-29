from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import load_config
from src.data import collate_sft, load_sft_data
from src.dpo_loss import sequence_logps
from src.modeling import (base_model_identity, load_model, load_sft_policy, load_tokenizer,
                          make_sft_manifest, model_config_for_sft_manifest, resolve_device)
from src.training import append_metrics, checkpoint_metadata, restore_training_state, save_checkpoint, set_seed


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg["training"].get("device", "auto"))
    resume = cfg["training"].get("resume_from_checkpoint")
    if resume:
        from src.modeling import load_sft_manifest, validate_sft_tokenizer
        manifest = load_sft_manifest(resume)
        cfg["model"] = model_config_for_sft_manifest(cfg["model"], manifest)
        tok = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"].get("trust_remote_code", False), cfg["model"].get("revision"))
        validate_sft_tokenizer(tok, manifest)
        model_identity = manifest["model_identity"]
    else:
        model_identity = base_model_identity(cfg["model"])
        if model_identity.get("resolved_commit"):
            cfg["model"] = {**cfg["model"], "revision": model_identity["resolved_commit"]}
        tok = load_tokenizer(cfg["model"]["name_or_path"], cfg["model"].get("trust_remote_code", False), cfg["model"].get("revision"))
        manifest = make_sft_manifest(cfg["model"], tok, model_identity)
    data_cfg = {**cfg["data"], "seed": int(cfg.get("seed", 42))}
    train_ds, valid_ds = load_sft_data(data_cfg, tok)
    batch = int(cfg["training"]["batch_size"])
    collate = lambda xs: collate_sft(xs, tok.pad_token_id)
    # Sequential batches make a mid-epoch checkpoint resumable without storing
    # a potentially large sampler permutation in the checkpoint.
    loader_generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    valid_generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + 1)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=False, collate_fn=collate, generator=loader_generator)
    valid_dl = DataLoader(valid_ds, batch_size=batch, shuffle=False, collate_fn=collate, generator=valid_generator)
    out = Path(cfg["training"]["output_dir"])
    run_metadata = checkpoint_metadata(cfg, "sft")
    if resume:
        model = load_sft_policy(cfg["model"], resume, trainable=True, gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False))
    else:
        model = load_model(cfg["model"], adapter=True, gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False))
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 0)))
    accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    total_steps = max(1, math.ceil(len(train_dl) / accum) * int(cfg["training"]["epochs"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    step, first_epoch, first_batch = 0, 0, 0
    if resume:
        step, first_epoch, extra = restore_training_state(optimizer, resume, scheduler, run_metadata)
        first_batch = int(extra.get("batch_in_epoch", 0))
    for epoch in range(first_epoch, int(cfg["training"]["epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for i, batch_data in enumerate(train_dl):
            if epoch == first_epoch and i < first_batch:
                continue
            ids, mask = batch_data["input_ids"].to(device), batch_data["response_mask"].to(device)
            attn = batch_data["attention_mask"].to(device)
            loss = -sequence_logps(model(input_ids=ids, attention_mask=attn).logits, ids, mask).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SFT loss")
            group_start = (i // accum) * accum
            group_size = min(accum, len(train_dl) - group_start)
            (loss / group_size).backward()
            if (i + 1) % accum == 0 or i + 1 == len(train_dl):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("max_grad_norm", 1)))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite SFT gradient norm")
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % int(cfg["training"].get("logging_steps", 10)) == 0:
                    append_metrics(out / "train_metrics.jsonl", {"step": step, "epoch": epoch, "loss": float(loss.detach().cpu())})
                every = int(cfg["training"].get("save_steps", 100))
                if every > 0 and step % every == 0:
                    checkpoint = save_checkpoint(model, optimizer, step, epoch, out, {"batch_in_epoch": i + 1}, scheduler, run_metadata)
                    (checkpoint / "sft_manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
        if len(valid_ds):
            model.eval(); vals = []
            with torch.no_grad():
                for vb in valid_dl:
                    ids, mask = vb["input_ids"].to(device), vb["response_mask"].to(device)
                    attn = vb["attention_mask"].to(device)
                    vals.append(float((-sequence_logps(model(input_ids=ids, attention_mask=attn).logits, ids, mask).mean()).cpu()))
            append_metrics(out / "eval_metrics.jsonl", {"step": step, "epoch": epoch, "loss": sum(vals) / len(vals)})
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "sft_manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    # A full optimizer checkpoint is saved separately for a subsequent resume.
    checkpoint = save_checkpoint(model, optimizer, step, int(cfg["training"]["epochs"]), out, {"batch_in_epoch": 0}, scheduler, run_metadata)
    (checkpoint / "sft_manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    print(f"SFT complete: {out} (step={step})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    train(parser.parse_args().config)
