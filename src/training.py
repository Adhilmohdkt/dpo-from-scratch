from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def checkpoint_metadata(cfg: dict, stage: str) -> dict:
    training = {k: v for k, v in cfg.get("training", {}).items() if k != "resume_from_checkpoint"}
    result = {"stage": stage, "seed": cfg.get("seed", 42), "model": cfg["model"], "data": cfg["data"], "training": training}
    if stage == "dpo":
        result["dpo"] = cfg["dpo"]
    return result


def _metadata_digest(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, step: int, epoch: int, output_dir: str | Path, extra=None, scheduler=None, metadata=None):
    path = Path(output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    state = {
        "step": step,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "run_metadata": metadata,
        "run_metadata_sha256": _metadata_digest(metadata) if metadata is not None else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    torch.save(state, path / "training_state.pt")
    return path


def restore_training_state(optimizer, checkpoint: str | Path, scheduler=None, expected_metadata=None):
    state_path = Path(checkpoint) / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No training_state.pt in resume checkpoint: {checkpoint}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if expected_metadata is not None and state.get("run_metadata_sha256") != _metadata_digest(expected_metadata):
        raise ValueError("Checkpoint was created with a different model/data/training configuration")
    if "optimizer" not in state:
        raise ValueError("Checkpoint is missing optimizer state")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state.get("scheduler") is None:
            raise ValueError("Checkpoint is missing scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]), int(state["epoch"]), state.get("extra", {})


def append_metrics(path: str | Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
