from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import load_config
from src.data import SOURCE_ROW_ID, load_preference_training_rows
from src.data_selection import select_equal_subsets, validate_selection_ids


def run(config_path: str, subset_size: int, output_dir: str):
    cfg = load_config(config_path)
    data_cfg = {**cfg["data"], "seed": int(cfg.get("seed", 42))}
    train_rows, validation_rows, fingerprint = load_preference_training_rows(data_cfg)
    train_ids = [int(value) for value in train_rows[SOURCE_ROW_ID]]
    validation_ids = [int(value) for value in validation_rows[SOURCE_ROW_ID]]
    margin_ids, random_ids, scored = select_equal_subsets(train_rows, subset_size, int(cfg.get("seed", 42)))
    validate_selection_ids(train_ids, validation_ids, margin_ids)
    validate_selection_ids(train_ids, validation_ids, random_ids)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": data_cfg["name"],
        "dataset_revision": data_cfg.get("revision", "main"),
        "split": data_cfg.get("train_preferences_split", "train_prefs"),
        "dataset_fingerprint": fingerprint,
        "seed": int(cfg.get("seed", 42)),
        "validation_fraction": float(data_cfg.get("validation_fraction", 0.02)),
        "max_train_samples": data_cfg.get("max_train_samples"),
        "subset_size": int(subset_size),
        "candidate_count": len(train_rows),
        "validation_count": len(validation_rows),
        "margin_definition": "score_chosen - score_rejected, from the dataset's external scalar scores; measured before any DPO update",
        "selection_method": "descending external reward-score margin; ties broken by source row ID",
        "paper_relationship": "simplified inspired experiment; not a reproduction",
    }
    for filename, method, ids in (
        ("margin_subset.json", "top_external_reward_margin", margin_ids),
        ("random_subset.json", "uniform_random_without_replacement", random_ids),
    ):
        payload = {**common, "method": method, "row_ids": ids}
        (out / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "full_candidate_row_ids.json").write_text(json.dumps(train_ids, indent=2), encoding="utf-8")
    with (out / "candidate_reward_margins.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "score_chosen", "score_rejected", "reward_margin", "selected_by_margin", "selected_random"])
        margin_set, random_set = set(margin_ids), set(random_ids)
        row_by_id = {int(row[SOURCE_ROW_ID]): row for row in train_rows}
        for row_id, margin in scored:
            row = row_by_id[row_id]
            writer.writerow([row_id, row["score_chosen"], row["score_rejected"], margin, row_id in margin_set, row_id in random_set])
    (out / "selection_experiment.json").write_text(
        json.dumps({**common, "arms": {"margin": "margin_subset.json", "random": "random_subset.json", "full": "full_candidate_row_ids.json"}, "training_config": cfg}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote equal-size margin/random subsets (n={subset_size}) and full candidate IDs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train-only, equal-size preference selection manifests")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.config, args.subset_size, args.output_dir)
