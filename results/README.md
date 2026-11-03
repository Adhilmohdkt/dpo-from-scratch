# Published result artifacts

This directory contains the project's lightweight, committed experiment summaries and provenance. The tracked pilot evaluation has reusable CSV/JSON metrics. Model adapters, optimizer checkpoints, and full datasets are not required to read those summaries.

## Reproducible pilot summary

The beta sweep and margin-selected versus random-selected results are recorded in:

- [`pilot-128/evaluation/validation/metrics.csv`](pilot-128/evaluation/validation/metrics.csv) — aggregate metrics for Base, SFT, three beta values, and both subset arms. Each row records seed, validation example count, beta, raw preference accuracy, implicit reward margin when applicable, response length, and generation cap.
- [`pilot-128/evaluation/validation/metrics.json`](pilot-128/evaluation/validation/metrics.json) — same metrics in JSON format.
- [`pilot-128/evaluation/validation/evaluation_config.json`](pilot-128/evaluation/validation/evaluation_config.json) — evaluation seed, split, generation settings, checkpoint labels, and beta mapping.
- [`pilot-128/selection/selection_experiment.json`](pilot-128/selection/selection_experiment.json) — selection seed, candidate/validation/subset sizes, score-gap definition, dataset fingerprint, and training configuration snapshot.
- [`pilot-128/selection/margin_subset.json`](pilot-128/selection/margin_subset.json) and [`pilot-128/selection/random_subset.json`](pilot-128/selection/random_subset.json) — selected training row IDs for the two subset arms.

These artifacts support only the 128-candidate, one-seed pilot and its 32-example validation comparison. The pilot generation cap was 64 tokens. The metric `preference_accuracy` is raw chosen-versus-rejected response sequence likelihood accuracy; `implicit_reward_margin_mean` is the mean policy-versus-SFT-reference log-ratio difference, scaled by beta. They are distinct quantities. The table in the root README is derived from the committed `metrics.csv`; do not combine this pilot with another run.

## 200-example validation run

A later six-checkpoint Colab evaluation was reported with a 200-example validation subset and a 256-token generation cap. Its per-checkpoint CSV/JSON results and completion summary are **not available in this checkout or Git history**, and were not restored from the private Hugging Face repository during this finalization. No numeric 200-example result is published here. Recover the original artifacts from `scaled-t4-512-validation-200/seed-42/evaluation/validation` in the private Hub repository before making those results publicly verifiable. Prior prose, screenshots, or summary figures are not a substitute for the machine-readable run outputs.

The summary and provenance files linked above are lightweight. Existing raw predictions, training logs, and timing evidence are retained as historical repository files; this final documentation change does not add or alter them. This commit adds no model checkpoints, credentials, or private files.
