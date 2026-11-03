# DPO from scratch: PyTorch implementation and small-scale study

- **Built:** A from-scratch DPO implementation in plain PyTorch, including SFT, LoRA, and scoring against a frozen reference model.
- **Explored:** DPO beta settings and margin-selected versus random preference-data subsets.
- **Finding:** These small experiments did not demonstrate a meaningful preference-accuracy improvement over the base model.
- **Why it matters:** The project examines preference-optimization implementation and evaluation, including raw policy likelihood versus DPO implicit reward.

This repository implements supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) with PyTorch and PEFT/LoRA. It explores a practical question: can a small instruction model be preference-tuned, and what can a small, compute-limited evaluation actually tell us? Training uses the project's own PyTorch loops and loss; it does not use TRL's `DPOTrainer`.

## What is implemented

- Chat preference and SFT data loading, template-aware tokenization, response-only scoring masks, padding masks, and EOS handling.
- LoRA SFT and DPO trainers, finite-value checks, periodic checkpoints, and optimizer/scheduler/RNG/cursor resumption.
- An SFT-derived frozen reference policy with precomputed, identity-checked reference log-probabilities.
- Held-out preference scoring, deterministic generation, response-length exports, checkpoint-level evaluation recovery, and result aggregation.
- A simplified train-only data-selection experiment using UltraFeedback's `score_chosen - score_rejected` margin against an equal-size seeded random subset.
- CPU unit tests and a separate Colab workflow for the completed training/evaluation artifacts.

The selection experiment is inspired by [Less is More](https://arxiv.org/abs/2502.14560); it is not a reproduction. It uses dataset-provided score gaps, not the paper's full method.

## Pipeline

```mermaid
flowchart TD
    A[UltraFeedback pinned revision] --> B[Template and tokenize]
    B --> C[SFT: response-token cross-entropy]
    C --> D[Save SFT adapter]
    D --> E[Score reference pairs with frozen SFT]
    E --> F[Cache reference log-probabilities with identity]
    D --> G[Reload same SFT adapter as trainable DPO policy]
    F --> G
    G --> H[Preference optimization and resumable checkpoints]
    H --> I[Held-out scoring and generation]
    I --> J[CSV/JSONL metrics and aggregation]
```

## DPO objective

For prompt `x`, preferred response `y_w`, rejected response `y_l`, policy `πθ`, frozen reference `πref`, and coefficient `β`, the per-pair sigmoid loss is

```text
-log σ( β ((log πθ(y_w|x) - log πθ(y_l|x))
       - (log πref(y_w|x) - log πref(y_l|x))) )
```

Sequence log-probability is the sum of next-token log-probabilities over response tokens only. Prompt tokens and padding do not contribute. EOS is included when present in the formatted response. The implicit reward represented by the objective is `β * (log πθ(y|x) - log πref(y|x))`. `β` controls the scale/constraint relative to the reference; it is not an accuracy target. The reference is the exact saved SFT policy. Disabling its adapter would instead score the base model, so this implementation scores and caches the SFT reference separately, then loads the same SFT adapter as the trainable policy.

See [docs/EXPLAINER.md](docs/EXPLAINER.md) for implementation details, data handling, recovery, and caveats.

## Stack and setup

The implementation uses Python, PyTorch, Transformers, PEFT, Hugging Face Datasets, Accelerate, PyYAML, and unittest. TRL is pinned for reproducible environment comparisons, but project training does not invoke `DPOTrainer`. Local pins target Python 3.11/3.12; Colab pins are separate and preserve Colab-managed CUDA PyTorch. See [requirements.txt](requirements.txt) and [requirements-colab.txt](requirements-colab.txt).

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The default training config is `configs/default.yaml`. It loads data from Hugging Face, so first use may download model and dataset artifacts. The reproducible scaled experiment pins UltraFeedback to revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` in `configs/scaled_t4.yaml`.

Run the test suite with `python -m unittest discover -s tests -v`. In this final pass, all 50 tests passed under the pinned Python 3.12 environment, and the optional TRL cross-check ran successfully locally. In the latest recorded Colab audit, 50 tests passed and one optional test was skipped because that environment's installed TRL no longer exposes the legacy comparison helper. The cross-check was not reported as passing in Colab. This documentation pass did not train or evaluate models.

### Train and evaluate (new run)

These commands start training and are provided for reproduction; they are not needed to inspect the completed experiment:

```bash
python -m src.train_sft --config configs/default.yaml
python -m src.train_dpo --config configs/default.yaml --sft-checkpoint checkpoints/qwen25-05b/sft
python -m src.evaluate --config configs/default.yaml
```

Train separate beta runs with `--beta 0.05`, `--beta 0.1`, or `--beta 0.5`, using the same SFT adapter and distinct output/cache paths. Do not use the final test split for model or beta selection. The data-selection command and constraints are documented in [docs/EXPLAINER.md](docs/EXPLAINER.md).

### Colab reproduction

- [Training notebook](notebooks/colab_scaled_run.ipynb): runs the configured SFT and DPO stages. The five reported stages have already completed; do not rerun them just to reproduce evaluation.
- [Evaluation-only notebook](notebooks/colab_evaluation_only.ipynb): restores and validates the five existing stages from the private Hub, then runs or resumes the six-checkpoint 200-example validation workflow. **It does not train.**
- [Evaluation-only guide](docs/EVALUATION_ONLY.md): setup, exact command, outputs, recovery behavior, and runtime estimate.

For the current completed evaluation, launch a T4 Colab runtime, set the `HF_TOKEN` secret, follow the evaluation-only notebook, and execute only its evaluation workflow. Never paste or print the token. Do not execute the training notebook's runner cell for evaluation recovery.

## Experimental Results and Observations

The reproducible result artifacts currently committed are for the earlier one-seed pilot: 128 training candidates, 32 validation pairs, and greedy generation capped at 64 tokens. The table below is copied from [`metrics.csv`](results/pilot-128/evaluation/validation/metrics.csv); configuration and selection provenance are in [`evaluation_config.json`](results/pilot-128/evaluation/validation/evaluation_config.json) and [`selection_experiment.json`](results/pilot-128/selection/selection_experiment.json). Values are shown to the precision in the CSV.

| Pilot checkpoint (seed 42, 32 validation pairs) | Raw preference accuracy | Mean implicit reward margin | Mean generated tokens |
|---|---:|---:|---:|
| Base | 56.25% | — | 53.875 |
| SFT | 56.25% | — | 50.6875 |
| DPO β = 0.05 | 56.25% | 0.093127 | 54.65625 |
| DPO β = 0.1 | 56.25% | 0.160635 | 54.625 |
| DPO β = 0.5 | 56.25% | 0.447438 | 51.5625 |
| Margin-selected DPO, 32 rows | 56.25% | 0.097014 | 51.0 |
| Random-selected DPO, 32 rows | 56.25% | 0.074296 | 48.59375 |

All seven pilot checkpoints tied on raw preference accuracy for this 32-pair validation sample. The beta and selection runs therefore did not show an accuracy gain in this pilot. The implicit reward margins differ from raw accuracy and do not by themselves establish improved human preference alignment.

**200-example evaluation:** A later Colab run was reported with six-checkpoint accuracy, implicit reward margin, and response-length summaries, but the corresponding evaluation CSV/JSON files are not present in the checkout or in the committed Git history. This environment has no Hugging Face token to restore the private run artifacts. The numeric 200-example results are therefore marked **unavailable for independent verification** and are not reproduced here. No values have been reconstructed from the prior documentation or prompt text. The 200-example sample size and 256-token generation setting describe the planned/completed workflow configuration, not verified published metric rows. See [`results/README.md`](results/README.md) for the artifact inventory and recovery note.

The evaluator's `preference_accuracy` is **raw policy likelihood accuracy**: it asks whether the model assigns greater response-token sequence log-likelihood to the dataset's chosen answer than to the rejected answer (exact ties receive half credit). The implicit reward margin instead subtracts the reference policy's chosen/rejected log-likelihood gap and multiplies by β. They are different measures. The evaluator does not report **implicit-reward pairwise accuracy**; that proposed 500-example metric audit was not implemented or run and remains future work.

Many responses reached the 256-token cap. Lengths are descriptive and do not mean a response is better. The supervision is a dataset preference signal, not a new human study. The 200 examples are a computationally constrained subsample of the original 500-example validation slice, not equivalent evidence to the 500-example evaluation and not a final-test result. Keep the reserved final-test partition untouched for a future independent report.

## Evaluation audit and limitations

The DPO loss was checked against the original [DPO paper](https://arxiv.org/abs/2305.18290); no confirmed loss implementation bug was identified. Audits covered SFT-reference loading/freezing, reference-log-probability caching, prompt and padding masks, and EOS handling. The metric distinction above is an evaluation-interpretation issue, not a discovered loss bug.

The experiment is limited by the 0.5B model, a 512-example full-arm training cap, 128-example selection arms, one seed, and a free T4 runtime. It does not reproduce the full UltraFeedback scale or the reference selection paper. The 128-example pilot has committed metric artifacts. The later 200-example run is not numerically reported here because its output artifact could not be recovered for verification.

### Future work

- Add implicit-reward pairwise accuracy and inspect per-example reward behavior.
- Evaluate a larger validation set; perform multiple seeds and paired statistical comparisons with uncertainty/power analysis.
- Study the small training subset, beta values, and response-token weighting in SFT.
- Test other preference data and DPO variants, and compare against other preference-optimization methods.
- Evaluate generated responses with an independent, appropriate human or model-based protocol.
- Repeat with stronger models and more compute.

No further training or the proposed 500-example implicit-reward metric audit is claimed here. Results must be generated by actual runs; this repository does not contain fabricated plots or benchmark gains.

## AI assistance

AI assistance was used during implementation, debugging, review, and documentation. The project owners are responsible for checking code, experimental records, and claims.
