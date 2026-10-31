# DPO from scratch

A research-oriented implementation of supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) using PyTorch and PEFT/LoRA. Training does not use TRL's `DPOTrainer`; TRL is pinned only to keep the research environment reproducible and available for later extensions.

## Current status

The SFT/DPO trainers, cache validation, checkpoint resumption, held-out scoring, generation export, train-only subset selection, and result aggregation are implemented. All 21 local CPU unit tests pass. A complete run on UltraFeedback or Qwen, GPU training, and Colab execution have not been performed. No experiment results are checked in or claimed. Result tables must come from actual runs.

## Setup

Use Python 3.11 or 3.12 and install the pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency set was installed and checked together in a clean Python 3.12 environment (`pip check` passed): PyTorch 2.5.1, Transformers 4.46.3, PEFT 0.13.2, Datasets 3.1.0, TRL 0.12.2, Accelerate 1.1.1, and PyYAML 6.0.2. For Colab, use `notebooks/colab_run.ipynb` and keep Colab’s preinstalled CUDA-enabled PyTorch. The notebook installs the other pinned packages from `requirements-colab.txt`; that file intentionally omits `torch` so it cannot replace the runtime build. PyTorch 2.5/torchvision 0.20 supports Python only through 3.12, while current Colab runtimes use Python 3.13; the [Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) and [PyTorch/torchvision compatibility matrix](https://github.com/pytorch/vision) are the references for runtime changes. The local `requirements.txt` targets Python 3.11/3.12 and retains the tested PyTorch 2.5.1 pin. The default Qwen config uses fp16 and LoRA. Use `configs/tiny.yaml` for a CPU run; it downloads model and dataset artifacts from Hugging Face, so it is not an offline test.

## Training

```bash
python -m src.train_sft --config configs/default.yaml
python -m src.train_dpo --config configs/default.yaml --sft-checkpoint checkpoints/qwen25-05b/sft
```

SFT saves a PEFT adapter, tokenizer, and `sft_manifest.json` with the resolved base-model revision and tokenizer identity. DPO requires this manifest, pins both reference and trainable policy to the exact base revision, and validates the tokenizer. Before the policy is updated, the frozen SFT adapter is loaded once to compute reference log-probabilities for each training preference pair. Those scores are cached on disk, allowing the training process to keep only the trainable policy in accelerator memory. Cache identity includes adapter contents, base commit/content identity, dataset fingerprint and revision, split, subset/sampling settings, preprocessing version, tokenization identity, and sequence limit. Use `--force-reference-recompute` to rebuild it.

Train the beta sweep as separate runs:

```bash
python -m src.train_dpo --config configs/default.yaml --beta 0.05
python -m src.train_dpo --config configs/default.yaml --beta 0.1
python -m src.train_dpo --config configs/default.yaml --beta 0.5
```

Each beta override writes to `checkpoints/qwen25-05b/dpo-beta-{beta}`. Each run uses the same configured seed, SFT checkpoint, and data partition. For seed replications, use a separate config/output directory per seed.

## Preference-data selection experiment

The selector uses only the training preference split after its deterministic validation partition and `max_train_samples` cap. It defines the external score margin as `score_chosen - score_rejected`, measured before DPO updates, and selects the top `--subset-size` margins (ties broken by source row ID). A seeded uniform random subset of exactly the same size is also produced; the full arm is the entire configured training candidate pool. UltraFeedback's dataset card describes its chosen response as the highest GPT-4 overall-score completion and rejected as a randomly selected alternative; this experiment uses the exposed scalar `score_chosen` / `score_rejected` fields. It does not use DPO implicit margins, a pilot DPO model, validation, or test rows to select data. This is inspired by [Less is More](https://arxiv.org/abs/2502.14560), not a full reproduction.

Create selection artifacts and train the margin/random arms:

```bash
python -m scripts.select_preference_data --config configs/default.yaml --subset-size 6000 --output-dir outputs/selection/seed-42
python -m src.train_dpo --config configs/default.yaml --train-row-ids-file outputs/selection/seed-42/margin_subset.json --output-dir outputs/dpo/seed-42/margin --reference-logps-path outputs/reference/seed-42/margin.pt
python -m src.train_dpo --config configs/default.yaml --train-row-ids-file outputs/selection/seed-42/random_subset.json --output-dir outputs/dpo/seed-42/random --reference-logps-path outputs/reference/seed-42/random.pt
python -m src.train_dpo --config configs/default.yaml --output-dir outputs/dpo/seed-42/full --reference-logps-path outputs/reference/seed-42/full.pt
```

The three DPO runs must use the same saved SFT checkpoint and training settings, but distinct output directories and reference-cache paths as shown. The selector writes both ID manifests, all candidate scores/margins, the complete candidate ID list, seed, source dataset fingerprint, and the training config snapshot. Data loading rejects manifests from another dataset revision/fingerprint and any IDs in validation or outside the configured candidate pool. Set `data.train_row_ids_file` to a manifest in YAML instead of passing the CLI option when preferred.

Compare the three trained arms on validation with one shared generation/evaluation configuration:

```bash
python -m src.evaluate --config configs/default.yaml --split validation \
  --checkpoint base=base \
  --checkpoint sft=checkpoints/qwen25-05b/sft \
  --checkpoint margin=outputs/dpo/seed-42/margin \
  --checkpoint random=outputs/dpo/seed-42/random \
  --checkpoint full=outputs/dpo/seed-42/full \
  --beta margin=0.1 --beta random=0.1 --beta full=0.1 \
  --output-dir outputs/selection-evaluation/seed-42
```

The Colab notebook builds this validation comparison when the selection experiment toggle is enabled.

## Colab

Open [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb). Commit and push local changes before cloning from GitHub. The notebook checks CUDA, preserves Colab’s managed PyTorch, installs the remaining pinned dependencies together from `requirements-colab.txt`, imports the project modules, mounts Drive, and runs a model/tokenizer finite-logit forward smoke check before training. It stores checkpoints/reference caches/logs/results on Drive, starts at 128 train examples and 512 tokens, and offers resume variables. Increase the sample cap only after SFT and DPO smoke runs complete. A real T4 run has not been performed.

Resume from a periodic `checkpoint-N` folder (SFT and DPO checkpoints include adapter, optimizer, scheduler, RNG, run metadata, and batch cursor):

```bash
python -m src.train_dpo --config configs/default.yaml --sft-checkpoint checkpoints/qwen25-05b/sft --resume checkpoints/qwen25-05b/dpo-beta-0.1/checkpoint-100
```

Set `training.resume_from_checkpoint` in YAML for SFT resumption. The resume path must be a checkpoint produced by this implementation; older adapters without `sft_manifest.json` must be retrained to guarantee the exact DPO reference.

## Evaluation

```bash
python -m src.evaluate --config configs/default.yaml
```

Evaluation defaults to a validation partition made only from training data. Explicitly select `--split test` for final reporting after model/beta choices are fixed. It scores chosen-vs-rejected response log-probabilities and exports deterministic generations with token-length summaries; generated text is not automatically judged for quality. It writes one prediction JSONL and per-example `*_response_lengths.csv` per checkpoint plus `metrics.json` and `metrics.csv`. Generation settings are common to Base/SFT/DPO in one invocation. Length is descriptive, not a quality target. Final test data is not used by either trainer or validation selection.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the DPO equation (including equivalence with the installed TRL sigmoid loss), prompt masking, EOS inclusion, padding exclusion, identical chosen/rejected responses, reference detachment, SFT adapter identity/freezing, cache invalidation, manifest revision/tokenizer validation, checkpoint state/resumption, equal-size/reproducible data selection, split isolation, and aggregation. They do not establish full-dataset training performance or GPU compatibility.

Aggregate actual evaluations across seeds (each evaluation run must be placed under a separate seed-specific directory):

```bash
python -m scripts.aggregate_results --root outputs/evaluation --output outputs/summary_metrics.csv
```

The output contains one row per metric/checkpoint/split/beta, with mean, sample standard deviation (`n-1`; blank for one seed), and seed count. With no runs, there are no result rows to report.

## Colab T4 checklist

1. Select a T4 runtime and run the notebook from a fresh session. It verifies Python, the preinstalled CUDA PyTorch build, the GPU name, and CUDA; do not install `requirements.txt` in Colab because its `torch==2.5.1` pin is for local Python 3.11/3.12 only.
2. Verify CUDA and GPU name, then mount Drive and place `training.output_dir`, `dpo.reference_logps_path`, and evaluation outputs on Drive so runtime resets do not discard state.
3. Start with a short run using a small `max_train_samples` (for example 128) and `max_length: 512`; verify SFT, DPO, checkpoint resume, and evaluation before scaling.
4. The default 0.5B LoRA workload should be a more suitable starting point for 16GB T4 memory than a larger base model, but memory/runtime have not been validated on Colab. If OOM occurs, reduce sequence length first and keep per-device batch size at 1.
5. Train SFT once, then run each beta from that same SFT checkpoint. Tune/select on validation; evaluate the selected comparison on `test` only once.

## Design and limitations

See [docs/EXPLAINER.md](docs/EXPLAINER.md) for the objective, log-probability masking, reference-policy design, checkpoint/cache behavior, and research limitations.

The data-selection idea is a simplified experiment inspired by [Less is More](https://arxiv.org/abs/2502.14560), not a reproduction. The proposed later comparison is top reward-score margin, equal-size seeded random subset, and full training data. It omits the paper's implicit DPO margins and Bayesian multi-source aggregation.
