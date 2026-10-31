"""Plot metrics produced by ``src.evaluate`` without synthesizing missing runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_metrics(metrics_csv: str | Path, output_path: str | Path) -> Path:
    """Plot validation preference accuracy, response length, and DPO reward margin."""
    metrics_path = Path(metrics_csv)
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"checkpoint", "preference_accuracy", "response_length_tokens_mean"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("Metrics file has no experiment rows to plot")

    labels = [row["checkpoint"] for row in rows]
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].bar(positions, [float(row["preference_accuracy"]) for row in rows], color="#4776a8")
    axes[0].set_title("Validation preference accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)

    axes[1].bar(positions, [float(row["response_length_tokens_mean"]) for row in rows], color="#5b9a73")
    axes[1].set_title("Generated response length")
    axes[1].set_ylabel("Mean tokens (max 64)")

    if "implicit_reward_margin_mean" in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row["implicit_reward_margin_mean"]))
            except (TypeError, ValueError):
                values.append(math.nan)
        valid = [not math.isnan(value) for value in values]
        axes[2].bar(
            [p for p, keep in zip(positions, valid) if keep],
            [value for value, keep in zip(values, valid) if keep],
            color="#c58b42",
        )
        axes[2].set_title("DPO implicit reward margin")
        axes[2].set_ylabel("Mean reward difference")
    else:
        axes[2].set_visible(False)

    for axis in axes:
        if axis.get_visible():
            axis.set_xticks(positions, labels, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Validation pilot (single seed; response length is descriptive, not a quality score)",
        fontsize=12,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="src.evaluate metrics.csv")
    parser.add_argument("--output", required=True, help="PNG output path")
    args = parser.parse_args()
    print(plot_metrics(args.metrics, args.output))


if __name__ == "__main__":
    main()
