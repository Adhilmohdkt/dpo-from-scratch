"""Reproducible external reward-score margin subset selection utilities."""

from __future__ import annotations

import math
import random
from typing import Iterable


def reward_score_margin(row: dict) -> float:
    """Return dataset reward margin: score_chosen minus score_rejected."""
    try:
        chosen = float(row["score_chosen"])
        rejected = float(row["score_rejected"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every candidate needs numeric score_chosen and score_rejected") from exc
    margin = chosen - rejected
    if not math.isfinite(margin):
        raise ValueError("Preference reward scores and margins must be finite")
    return margin


def select_equal_subsets(rows: Iterable[dict], subset_size: int, seed: int):
    """Choose top external-score margins and a seeded random subset of equal size."""
    rows = list(rows)
    ids = [int(row["_source_row_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Training candidate row IDs must be unique")
    if not 1 <= int(subset_size) <= len(rows):
        raise ValueError(f"subset_size must be between 1 and {len(rows)}")
    scored = [(int(row["_source_row_id"]), reward_score_margin(row)) for row in rows]
    selected = [row_id for row_id, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:subset_size]]
    random_ids = random.Random(int(seed)).sample(ids, int(subset_size))
    return selected, random_ids, scored


def validate_selection_ids(train_ids: Iterable[int], validation_ids: Iterable[int], selected_ids: Iterable[int]):
    """Fail closed if selection includes validation or unavailable row IDs."""
    train, validation, selected = set(map(int, train_ids)), set(map(int, validation_ids)), list(map(int, selected_ids))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected row IDs must be a non-empty unique sequence")
    if set(selected) & validation:
        raise ValueError("Selection includes validation row IDs")
    if not set(selected) <= train:
        raise ValueError("Selection includes rows outside the training candidate pool")
    return selected
