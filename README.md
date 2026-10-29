# DPO from scratch

A research-oriented implementation of supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) using PyTorch and PEFT/LoRA. Training does not use TRL's `DPOTrainer`; TRL is pinned only to keep the research environment reproducible and available for later extensions.

## Current status

The core SFT/DPO trainers, cache validation, checkpoint resumption, held-out scoring, generation export, and a configurable data interface are implemented. The 16 unit tests pass in the local CPU environment, including a tiny local LoRA/reference-policy update and comparison of the loss to the installed TRL implementation. A complete run on UltraFeedback or Qwen, GPU training, and Colab execution have not been performed. No experiment results are checked in or claimed. Result tables and plots must come from actual runs.

## Setup

Use Python 3.11 or 3.12 and install the pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency set was installed and checked together in a clean Python 3.12 environment (`pip check` passed): PyTorch 2.5.1, Transformers 4.46.3, PEFT 0.13.2, Datasets 3.1.0, TRL 0.12.2, Accelerate 1.1.1, and PyYAML 6.0.2. For Colab, first install the PyTorch 2.5.1 CUDA 12.4 wheel, then install the pinned requirements and verify `torch.cuda.is_available()` and the selected GPU. The default Qwen config uses fp16 and LoRA. Use `configs/tiny.yaml` for a CPU run; it downloads model and dataset artifacts from Hugging Face, so it is not an offline test.

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

Each beta override writes to `checkpoints/qwen25-05b/dpo-beta-{beta}`. The data-selection experiment is intentionally not implemented in this core phase. `PreferenceDataset` keeps the dataset score fields and stable row indices so a later selector can supply a chosen subset without changing the DPO objective or trainer.

Resume from a periodic `checkpoint-N` folder (SFT and DPO checkpoints include adapter, optimizer, scheduler, RNG, run metadata, and batch cursor):

```bash
python -m src.train_dpo --config configs/default.yaml --sft-checkpoint checkpoints/qwen25-05b/sft --resume checkpoints/qwen25-05b/dpo-beta-0.1/checkpoint-100
```

Set `training.resume_from_checkpoint` in YAML for SFT resumption. The resume path must be a checkpoint produced by this implementation; older adapters without `sft_manifest.json` must be retrained to guarantee the exact DPO reference.

## Evaluation

```bash
python -m src.evaluate --config configs/default.yaml
```

Evaluation defaults to a validation partition made only from training data. Explicitly select `--split test` for final reporting after model/beta choices are fixed. It scores chosen-vs-rejected response log-probabilities and exports deterministic generations with token-length summaries; generated text is not automatically judged for quality. It writes one prediction JSONL per checkpoint plus `metrics.json` and `metrics.csv`. Configured checkpoint paths include Base, SFT, and each beta output. Missing trained checkpoints must be produced before running the comparison. Final test data is not used by either trainer or validation selection.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the DPO equation (including equivalence with the installed TRL sigmoid loss), prompt masking, EOS inclusion, padding exclusion, identical chosen/rejected responses, reference detachment, SFT adapter identity/freezing, cache invalidation, manifest revision/tokenizer validation, and checkpoint state/resumption. They do not establish full-dataset training performance or GPU compatibility.

## Colab T4 checklist

1. Select a T4 runtime and inspect Python/CUDA; install the matching PyTorch wheel first, then `requirements.txt`.
2. Verify CUDA and GPU name, then mount Drive and place `training.output_dir`, `dpo.reference_logps_path`, and evaluation outputs on Drive so runtime resets do not discard state.
3. Start with a short run using a small `max_train_samples` (for example 128) and `max_length: 512`; verify SFT, DPO, checkpoint resume, and evaluation before scaling.
4. The default 0.5B LoRA workload should be a more suitable starting point for 16GB T4 memory than a larger base model, but memory/runtime have not been validated on Colab. If OOM occurs, reduce sequence length first and keep per-device batch size at 1.
5. Train SFT once, then run each beta from that same SFT checkpoint. Tune/select on validation; evaluate the selected comparison on `test` only once.

## Design and limitations

See [docs/EXPLAINER.md](docs/EXPLAINER.md) for the objective, log-probability masking, reference-policy design, checkpoint/cache behavior, and research limitations.

The data-selection idea is a simplified experiment inspired by [Less is More](https://arxiv.org/abs/2502.14560), not a reproduction. The proposed later comparison is top reward-score margin, equal-size seeded random subset, and full training data. It omits the paper's implicit DPO margins and Bayesian multi-source aggregation.
