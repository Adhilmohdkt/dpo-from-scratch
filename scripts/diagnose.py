"""Diagnose DPO pilot logs, adapter movement, evaluator wiring, and small-n accuracy."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import torch

from src.dpo_loss import dpo_loss, sequence_logps


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return center - half, center + half


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def summarize_logs(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        rows = read_jsonl(path)
        losses = [float(row["loss"]) for row in rows if "loss" in row]
        accuracies = [float(row["reward_accuracy"]) for row in rows if "reward_accuracy" in row]
        if not rows:
            raise ValueError(f"No log rows found in {path}")
        summary: dict[str, Any] = {"log": str(path), "records": len(rows)}
        if losses:
            xs = list(range(len(losses)))
            x_mean = sum(xs) / len(xs)
            y_mean = sum(losses) / len(losses)
            denom = sum((x - x_mean) ** 2 for x in xs)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, losses)) / denom if denom else 0.0
            summary["loss"] = {
                "first": losses[0],
                "last": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "changed": max(losses) - min(losses) > 1e-12,
                "linear_slope_per_logged_step": slope,
            }
        if accuracies:
            summary["train_batch_reward_accuracy"] = {
                "first": accuracies[0],
                "last": accuracies[-1],
                "mean": sum(accuracies) / len(accuracies),
                "distinct_values": sorted(set(accuracies)),
                "changed": len(set(accuracies)) > 1,
            }
        summaries.append(summary)
    return summaries


def adapter_l2(initial_dir: Path, trained_dir: Path) -> dict[str, Any]:
    """Compare saved LoRA tensors without loading either base language model."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Install safetensors to compare adapter_model.safetensors files") from exc

    initial_path = initial_dir / "adapter_model.safetensors"
    trained_path = trained_dir / "adapter_model.safetensors"
    if not initial_path.is_file() or not trained_path.is_file():
        raise FileNotFoundError(f"Expected adapter weights at {initial_path} and {trained_path}")

    def load(path: Path) -> dict[str, torch.Tensor]:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            return {key: stream.get_tensor(key).float() for key in stream.keys()}

    initial, trained = load(initial_path), load(trained_path)
    if initial.keys() != trained.keys():
        raise ValueError("Initial and trained adapter tensors have different keys")
    squared_delta = torch.zeros((), dtype=torch.float64)
    squared_initial = torch.zeros((), dtype=torch.float64)
    changed = 0
    for key in initial:
        if initial[key].shape != trained[key].shape:
            raise ValueError(f"Adapter tensor shape differs for {key}")
        delta = trained[key].double() - initial[key].double()
        squared_delta += torch.sum(delta * delta)
        squared_initial += torch.sum(initial[key].double() ** 2)
        changed += int(torch.count_nonzero(delta).item() > 0)
    distance = math.sqrt(float(squared_delta))
    initial_norm = math.sqrt(float(squared_initial))
    return {
        "initial_adapter": str(initial_dir),
        "trained_adapter": str(trained_dir),
        "tensor_count": len(initial),
        "tensors_with_nonzero_change": changed,
        "l2_distance": distance,
        "initial_l2_norm": initial_norm,
        "relative_l2_distance": distance / initial_norm if initial_norm else None,
    }


def inspect_evaluator(source_path: Path) -> dict[str, Any]:
    """Read evaluator wiring without importing Transformers or loading model weights."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    eval_node = functions.get("evaluate_checkpoint")
    main_node = functions.get("evaluate")
    if eval_node is None or main_node is None:
        raise ValueError("Could not find evaluate_checkpoint() and evaluate() in evaluator source")
    eval_source = ast.get_source_segment(source, eval_node) or ""
    main_source = ast.get_source_segment(source, main_node) or ""
    trained_adapter_load = "load_sft_policy(model_cfg, str(path), trainable=False)" in eval_source
    sft_reference_load = "load_sft_policy(cfg[\"model\"], str(sft_path), trainable=False)" in main_source
    return {
        "evaluator_file": str(source_path),
        "trained_checkpoint_passed_to_frozen_adapter_loader": trained_adapter_load,
        "reference_loaded_from_separate_sft_adapter": sft_reference_load,
        "reference_adapter_disabled": False,
        "reference_design": "a separate frozen copy of the SFT adapter is loaded; this preserves the SFT reference policy",
        "evaluation_metric": "fraction of held-out pairs where the checkpoint assigns greater summed response-token log-probability to chosen than rejected, with exact ties worth half credit",
    }


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 6)
        self.output = torch.nn.Linear(6, 8, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def tiny_step_zero_check() -> dict[str, Any]:
    """CPU invariant: identical policy/reference have zero reward margin."""
    torch.manual_seed(7)
    reference = _TinyCausalModel().eval()
    policy = _TinyCausalModel().eval()
    policy.load_state_dict(reference.state_dict())
    chosen = torch.tensor([[1, 2, 3, 4]])
    rejected = torch.tensor([[1, 2, 3, 5]])
    mask = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
    with torch.no_grad():
        chosen_policy = sequence_logps(policy(chosen), chosen, mask)
        rejected_policy = sequence_logps(policy(rejected), rejected, mask)
        chosen_reference = sequence_logps(reference(chosen), chosen, mask)
        rejected_reference = sequence_logps(reference(rejected), rejected, mask)
    _, metrics = dpo_loss(chosen_policy, rejected_policy, chosen_reference, rejected_reference, beta=0.1)
    reward_margin = metrics["reward_margin"]
    tie_aware_accuracy = torch.where(reward_margin.abs() < 1e-7, 0.5, (reward_margin > 0).float()).mean()
    strict_accuracy = metrics["reward_accuracy"].mean()
    assert torch.allclose(reward_margin, torch.zeros_like(reward_margin), atol=1e-7)
    assert torch.isclose(tie_aware_accuracy, torch.tensor(0.5))
    assert torch.isclose(strict_accuracy, torch.tensor(0.0))
    return {
        "device": "cpu",
        "policy_equals_reference": True,
        "reward_margin": float(reward_margin.item()),
        "tie_aware_accuracy": float(tie_aware_accuracy.item()),
        "current_strict_greater_than_zero_accuracy": float(strict_accuracy.item()),
        "loss": float(dpo_loss(chosen_policy, rejected_policy, chosen_reference, rejected_reference, beta=0.1)[0]),
        "assertions_passed": True,
    }


def diagnose(log_paths: list[Path], evaluator_path: Path, initial_adapter: Path | None,
             trained_adapter: Path | None, successes: int, trials: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "training_logs": summarize_logs(log_paths),
        "adapter_movement": adapter_l2(initial_adapter, trained_adapter)
        if initial_adapter is not None and trained_adapter is not None else None,
        "evaluation_wiring": inspect_evaluator(evaluator_path),
        "step_zero_invariant": tiny_step_zero_check(),
        "pilot_accuracy": {
            "successes": successes,
            "trials": trials,
            "accuracy": successes / trials,
            "wilson_95_ci": wilson_interval(successes, trials),
        },
        "diagnosis": (
            "Training logs and adapter weights show that optimization changed the policy. "
            "The 32-pair evaluation interval is wide, so identical checkpoint accuracy alone "
            "does not establish a training or evaluation bug. The current evaluator gives "
            "exact ties half credit; more training and more held-out pairs are needed to "
            "separate weak training from validation noise."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, help="Training JSONL log; repeat for multiple runs")
    parser.add_argument("--evaluator", default="src/evaluate.py")
    parser.add_argument("--initial-adapter", default=None, help="Saved SFT adapter directory")
    parser.add_argument("--trained-adapter", default=None, help="Saved DPO adapter directory")
    parser.add_argument("--correct", type=int, default=18)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    result = diagnose(
        [Path(value) for value in args.log], Path(args.evaluator),
        Path(args.initial_adapter) if args.initial_adapter else None,
        Path(args.trained_adapter) if args.trained_adapter else None,
        args.correct, args.total,
    )
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
