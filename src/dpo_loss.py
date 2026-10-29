"""Numerically stable DPO objective and causal response log-probabilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logps(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum token log-probabilities for response targets only; mask includes token positions."""
    if logits.ndim != 3 or input_ids.ndim != 2 or response_mask.shape != input_ids.shape or logits.shape[:2] != input_ids.shape:
        raise ValueError("Expected logits [B,T,V], IDs/mask [B,T]")
    if input_ids.shape[1] < 2:
        raise ValueError("At least two tokens are needed to score a causal response token")
    shifted_logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    target_mask = response_mask[:, 1:].to(dtype=torch.bool)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * target_mask.to(token_logps.dtype)).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    *,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return sigmoid DPO loss, using summed sequence log-probabilities."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    tensors = (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    if any(t.shape != tensors[0].shape for t in tensors[1:]):
        raise ValueError("All four sequence-log-probability tensors must have identical shapes")
    policy_margin = policy_chosen_logps.float() - policy_rejected_logps.float()
    # Stop gradients at the reference boundary even if a caller accidentally
    # supplies log-probabilities produced by a trainable reference model.
    reference_margin = reference_chosen_logps.detach().float() - reference_rejected_logps.detach().float()
    logits = beta * (policy_margin - reference_margin)
    losses = -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Non-finite DPO loss")
    metrics = {
        "logits": logits.detach(),
        "rewards_chosen": (beta * (policy_chosen_logps.float() - reference_chosen_logps.detach().float())).detach(),
        "rewards_rejected": (beta * (policy_rejected_logps.float() - reference_rejected_logps.detach().float())).detach(),
        "reward_accuracy": (logits > 0).float().detach(),
        "reward_margin": logits.detach(),
    }
    return losses.mean(), metrics
