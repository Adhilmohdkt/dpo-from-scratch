# Evaluation only on Google Colab

This workflow restores the completed seed-42 training artifacts from the private Hugging Face repository, validates them, and evaluates Base, SFT, and four DPO checkpoints. It does not call the scaled training runner, SFT, or DPO training. It evaluates validation only; the final-test partition remains reserved.

## Evaluation sample and isolation

The original `configs/scaled_t4.yaml` remains unchanged. It defines 500 validation pairs and 500 final-test pairs in UltraFeedback's held-out preference split. The loader deterministically shuffles that split using seed `42 + 2 = 44`, selects the first 500 shuffled rows as validation, and reserves the following 500 as final test.

The new evaluation-only plan, `configs/scaled_t4_eval_200.yaml`, samples 200 rows without replacement **from the existing 500-row validation slice**. It ranks each validation row ID by SHA-256 of the pinned dataset identity, subset seed `45`, and row ID; then takes the first 200 ranked rows and retains their validation order. This is deterministic across Python and NumPy RNG versions. It does not change the validation/test boundary or select any final-test row.

`validation_subset.json` records the selected validation indices, source row IDs, prompt IDs, dataset revision and fingerprint, subset seed, and manifest hash. One subset is loaded once and shared by all six checkpoints. The subset identity is included in the evaluator's config hash, so it cannot reuse completion records for another subset.

The 500-row evaluation, if present, stays at its original local and Hub paths. The 200-row run uses a separate local directory (`results/scaled-t4-512/evaluation-200`) and Hub prefix (`scaled-t4-512-validation-200`). The 200-row validation is computationally constrained and has wider uncertainty than the 500-row evaluation; it is not presented as equivalent evidence.

## Before running

1. Push the project changes to GitHub `main` and open the repository in Colab.
2. Select a Tesla T4 runtime and install the pinned dependencies using the notebook setup instructions.
3. Add a write-enabled `HF_TOKEN` to Colab Secrets and enable notebook access. Do not paste the token into a cell or print it.
4. Run the scaled notebook's setup cells 1–3 to check out the project, install dependencies, and write `/content/scaled_t4_hub_run.yaml`. Do **not** run its training cell. The temporary Hub persistence test cell is unnecessary if it has already passed; the evaluation-only script checks authentication and private-repository access.

## Exact Colab commands

Read the token from Colab Secrets without displaying it:

```python
import os
from google.colab import userdata

_token = userdata.get("HF_TOKEN")
if not _token:
    raise RuntimeError("HF_TOKEN is missing; configure it in Colab Secrets and enable notebook access")
os.environ["HF_TOKEN"] = _token
del _token
```

From the checked-out project directory, run:

```bash
!python -m scripts.run_evaluation_only \
  --config /content/scaled_t4_hub_run.yaml \
  --repo-id adhill/dpo-experiment-checkpoint \
  --work-dir /content/dpo-evaluation-only \
  --evaluation-plan configs/scaled_t4_eval_200.yaml
```

Do not run `scripts.run_scaled_experiments`, `src.train_sft`, or `src.train_dpo`. If validation fails, stop and inspect the error; do not remove or bypass a check.

## What the script validates and writes

It validates the original training config and restores the selection snapshot plus all five completed training stages from Hub. Each training stage must match its completion marker, seed, original configuration digest, immutable UltraFeedback revision, dataset identity, selection IDs where applicable, adapter files, and final optimizer checkpoint. Existing nonempty local training directories are validated without being overwritten. Mismatches stop before evaluation.

The script preflights the exact 200 selected rows on CPU, checks any existing subset manifest and evaluation state, and restores only the isolated 200-example evaluation namespace. It uses the existing `src.evaluate` implementation with the existing 256-token greedy generation setting and metrics. Completed checkpoint evaluations are skipped; interrupted/incomplete checkpoint evaluations resume through `evaluation_state.json`. A changed dataset fingerprint, subset manifest, checkpoint adapter, or evaluation config fails closed rather than reusing incompatible state.

Expected local output:

- `results/scaled-t4-512/evaluation-200/seed-42/validation/metrics.csv` and `metrics.json`
- `evaluation_config.json`, `evaluation_state.json`, `validation_subset.json`, and `run_complete.json`
- Per-checkpoint `*_predictions.jsonl` and `*_response_lengths.csv`
- `checkpoint_summary.csv` with Base, SFT, DPO beta 0.1, DPO beta 0.5, margin-selected DPO, and random-selected DPO
- `validation_metrics.csv` from the existing aggregation function

The corresponding Hub path is `scaled-t4-512-validation-200/seed-42/evaluation/validation`. Evaluation results and completion markers are uploaded there. Upload failure is fatal; rerunning retries the upload. Training snapshots and the existing 500-example evaluation namespace are not uploaded to or overwritten by this run.

The summary contains actual run metrics only. No results are supplied in advance. The 200-example confidence intervals will be wider than those from 500 pairs; report this as a smaller validation comparison and keep final-test evaluation for a future independent report.

## Runtime estimate

The earlier T4 pilot took 851 seconds for seven checkpoints, 32 pairs, and a 64-token cap. A cap-linear extrapolation for six checkpoints, 200 validation pairs, and a 256-token cap is about 5.1 hours. Generation batches of eight may reduce this, but actual sequence lengths, batching throughput, and model loading are unmeasured. Budget several hours; per-checkpoint state is uploaded so an interrupted run can resume.
