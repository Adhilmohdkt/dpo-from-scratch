from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


IDENTITY_FIELDS = {"checkpoint", "seed", "split", "beta", "preference_examples", "generation_examples", "generation_max_new_tokens", "generation_do_sample"}


def aggregate(root: str | Path, output: str | Path):
    root, output = Path(root), Path(output)
    files = sorted(root.rglob("metrics.csv"))
    records = []
    for file in files:
        with file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                row.setdefault("split", file.parent.name)
                try:
                    row["seed"] = int(row["seed"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"{file} is missing an integer seed; cannot aggregate reproducibly")
                records.append(row)
    metric_names = sorted({key for row in records for key, value in row.items() if key not in IDENTITY_FIELDS and value not in (None, "") and _numeric(value)})
    groups = defaultdict(list)
    for row in records:
        groups[(row.get("split", ""), row.get("checkpoint", ""), row.get("beta", ""))].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "checkpoint", "beta", "metric", "n_seeds", "mean", "std"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (split, checkpoint, beta), rows in sorted(groups.items()):
            seeds = [row["seed"] for row in rows]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"Duplicate seed in aggregation group {(split, checkpoint, beta)}")
            for metric in metric_names:
                values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "") and _numeric(row[metric])]
                if not values:
                    continue
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
                writer.writerow({"split": split, "checkpoint": checkpoint, "beta": beta, "metric": metric, "n_seeds": len(values), "mean": mean, "std": math.sqrt(variance) if len(values) > 1 else ""})
    return len(records), output


def _numeric(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate actual evaluation metrics across seeds")
    parser.add_argument("--root", default="outputs/evaluation")
    parser.add_argument("--output", default="outputs/summary_metrics.csv")
    args = parser.parse_args()
    count, path = aggregate(args.root, args.output)
    print(f"Aggregated {count} actual metric rows into {path}")
