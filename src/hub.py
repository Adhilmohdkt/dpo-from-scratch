"""Private Hugging Face Hub persistence for resumable training runs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _token_name(settings: dict) -> str:
    return str(settings.get("token_env", "HF_TOKEN"))


def _required_token(settings: dict) -> str:
    name = _token_name(settings)
    token = os.environ.get(name)
    if not token:
        raise RuntimeError(f"Hub persistence is enabled, but environment variable {name} is not set")
    return token


def ensure_hub_repo(settings: dict) -> None:
    """Validate credentials and create the configured private model repo."""
    if not settings.get("enabled", False):
        return
    repo_id = settings.get("repo_id")
    if not repo_id:
        raise ValueError("hub.repo_id is required when Hub persistence is enabled")
    from huggingface_hub import HfApi

    HfApi(token=_required_token(settings)).create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=bool(settings.get("private", True)),
        exist_ok=True,
    )


def hub_run_path(settings: dict, seed: int, stage: str, output_dir: str | Path) -> str:
    """Make a stable, non-colliding path for a seed/stage/output directory."""
    prefix = str(settings.get("run_prefix", "runs")).strip("/")
    output_name = Path(output_dir).name
    parts = [part for part in (prefix, f"seed-{seed}", stage, output_name) if part]
    return "/".join(parts)


def push_run_snapshot(output_dir: str | Path, settings: dict, seed: int, stage: str) -> str | None:
    """Upload current logs, config, adapters, and resumable checkpoints."""
    if not settings.get("enabled", False):
        return None
    from huggingface_hub import HfApi

    repo_id = settings["repo_id"]
    path_in_repo = hub_run_path(settings, seed, stage, output_dir)
    api = HfApi(token=_required_token(settings))
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(Path(output_dir)),
        path_in_repo=path_in_repo,
        commit_message=f"Update {stage} run for seed {seed}",
    )
    return f"{repo_id}/{path_in_repo}"


def resolve_resume_checkpoint(reference: str | None, settings: dict) -> str | None:
    """Restore ``hf://owner/repo/path/to/checkpoint-N`` into local storage."""
    if reference is None or not reference.startswith("hf://"):
        return reference
    from huggingface_hub import snapshot_download

    parts = reference[5:].split("/", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError("Hub resume path must look like hf://owner/repo/path/to/checkpoint-N")
    repo_id = f"{parts[0]}/{parts[1]}"
    remote_path = parts[2].strip("/")
    root = Path(settings.get("restore_dir", "checkpoints/hub_restore"))
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{remote_path}/**"],
        local_dir=str(root),
        token=_required_token(settings),
    )
    local_path = Path(snapshot) / remote_path
    if not local_path.is_dir():
        raise FileNotFoundError(f"Hub resume checkpoint was not restored to {local_path}")
    return str(local_path)


def restore_run_snapshot(output_dir: str | Path, settings: dict, seed: int, stage: str) -> bool:
    """Restore a previously uploaded run folder into the current Colab filesystem."""
    if not settings.get("enabled", False):
        return False
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    remote_path = hub_run_path(settings, seed, stage, output_dir)
    restore_root = Path(settings.get("restore_dir", "checkpoints/hub_restore"))
    try:
        snapshot_root = Path(snapshot_download(
            repo_id=settings["repo_id"],
            repo_type="model",
            allow_patterns=[f"{remote_path}/**"],
            local_dir=str(restore_root),
            token=_required_token(settings),
        ))
    except Exception as exc:
        # A brand-new run has no remote files. Do not hide auth or network errors.
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
        if isinstance(exc, EntryNotFoundError):
            return False
        if isinstance(exc, HfHubHTTPError) and getattr(exc.response, "status_code", None) == 404:
            return False
        raise
    source = snapshot_root / remote_path
    if not source.is_dir():
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output_dir, dirs_exist_ok=True)
    return True


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the numerically latest complete local optimizer checkpoint."""
    candidates = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-", 1)[1])
        except ValueError:
            continue
        if (path / "training_state.pt").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
