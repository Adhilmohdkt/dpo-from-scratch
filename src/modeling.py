from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no CUDA support")
    return device


def load_tokenizer(name: str, trust_remote_code: bool = False, revision: str | None = None):
    kwargs = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        # Tiny/base checkpoints without a published template use a simple
        # fallback; instruction models keep their own tokenizer template.
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(cfg: dict[str, Any], *, adapter: bool, gradient_checkpointing: bool = False):
    dtype = getattr(torch, cfg.get("torch_dtype", "float32"))
    kwargs: dict[str, Any] = {"trust_remote_code": cfg.get("trust_remote_code", False)}
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    if dtype != torch.float32:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(cfg["name_or_path"], **kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if adapter and cfg.get("use_lora", False):
        lora = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=list(lora["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def load_sft_policy(cfg: dict[str, Any], adapter_path: str, *, trainable: bool, gradient_checkpointing: bool = False):
    """Restore the exact saved SFT policy (base weights plus its LoRA adapter)."""
    dtype = getattr(torch, cfg.get("torch_dtype", "float32"))
    kwargs: dict[str, Any] = {"trust_remote_code": cfg.get("trust_remote_code", False)}
    if cfg.get("revision"):
        kwargs["revision"] = cfg["revision"]
    if dtype != torch.float32:
        kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(cfg["name_or_path"], **kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
    if gradient_checkpointing and trainable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def base_model_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a Hub commit without downloading weights; hash local config/weight files."""
    source = str(cfg["name_or_path"])
    local_path = Path(source)
    if local_path.exists():
        files = sorted(
            file for file in local_path.rglob("*")
            if file.is_file() and (file.name == "config.json" or file.suffix in {".safetensors", ".bin"})
        )
        return {
            "source": source,
            "local_files": [
                {"path": str(p.relative_to(local_path)), "size": p.stat().st_size, "sha256": _file_sha256(p)}
                for p in files
            ],
        }
    config = AutoConfig.from_pretrained(
        source,
        revision=cfg.get("revision"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    commit = getattr(config, "_commit_hash", None)
    if not commit:
        raise RuntimeError(f"Could not resolve an immutable Hub commit for {source}; refusing an unpinned SFT reference")
    return {"source": source, "resolved_commit": commit, "requested_revision": cfg.get("revision")}


def tokenizer_identity(tokenizer) -> dict[str, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else json.dumps(sorted(tokenizer.get_vocab().items()), separators=(",", ":"))
    init = getattr(tokenizer, "init_kwargs", {})
    return {
        "name": tokenizer.name_or_path,
        "resolved_commit": getattr(tokenizer, "_commit_hash", None) or init.get("_commit_hash"),
        "serialized_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "special_tokens": json.loads(json.dumps(tokenizer.special_tokens_map, default=str)),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest(),
    }


def make_sft_manifest(model_cfg: dict[str, Any], tokenizer, model_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": str(model_cfg["name_or_path"]),
        "model_identity": model_identity,
        # Only a resolved Hub commit is a stable revision pin. Local models
        # are instead compared against their content hashes at reload time.
        "model_revision": model_identity.get("resolved_commit"),
        "tokenizer_identity": tokenizer_identity(tokenizer),
    }


def load_sft_manifest(adapter_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(adapter_path) / "sft_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing. This SFT adapter does not record its exact base-model revision; retrain SFT with this version before DPO."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def model_config_for_sft_manifest(model_cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Pin adapter reloads to the exact base weights used to create SFT."""
    if str(model_cfg["name_or_path"]) != manifest["model_name"]:
        raise ValueError("Configured base model does not match the model recorded in the SFT manifest")
    configured = dict(model_cfg)
    resolved = manifest.get("model_revision")
    requested = configured.get("revision")
    if resolved:
        if requested not in (None, "main", resolved):
            raise ValueError("Configured model revision differs from the SFT checkpoint's base revision")
        configured["revision"] = resolved
    elif base_model_identity(configured) != manifest["model_identity"]:
        raise ValueError("Local base model files changed since SFT; the saved adapter no longer identifies its original policy")
    return configured


def validate_sft_tokenizer(tokenizer, manifest: dict[str, Any]) -> None:
    if tokenizer_identity(tokenizer) != manifest["tokenizer_identity"]:
        raise ValueError("Tokenizer files/settings differ from those used for the SFT checkpoint")


def unwrap_for_generation(model):
    return model


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()
