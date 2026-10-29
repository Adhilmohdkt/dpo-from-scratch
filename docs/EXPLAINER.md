# DPO implementation notes

## Objective

For a prompt `x`, preferred response `y+`, rejected response `y-`, trainable policy `πθ`, frozen reference `πref`, and temperature `β`, the standard sigmoid DPO loss for one pair is:

`-log σ( β [ (log πθ(y+|x) - log πθ(y-|x)) - (log πref(y+|x) - log πref(y-|x)) ] )`.

The code sums token log-probabilities over each response. It computes the DPO arithmetic in float32 even when model weights/activations use fp16. `logsigmoid` is used for stability; non-finite loss and gradient norms stop training.

## Prompt and response boundaries

The tokenizer's `apply_chat_template` formats the prompt with a generation prefix and formats the complete conversation separately. Token IDs are checked to ensure the prompt rendering is an exact prefix of the full conversation. Loss masks cover assistant response targets and EOS, excluding the prompt. Right padding is represented by a separate attention mask and is excluded from response masks. Overlong prompts are left-truncated to preserve the response; overlong responses are truncated and explicitly terminated by EOS.

Instruction models use their own published chat template. A simple fallback exists only for base/tiny models without a template.

## Why the SFT adapter is the reference

At the beginning of DPO, the reference policy must be exactly the policy produced by SFT. Disabling a LoRA adapter would instead evaluate the underlying base model. This implementation saves the SFT adapter, reloads it as a frozen reference, computes chosen/rejected reference log-probabilities over the fixed training examples, writes those values and identity metadata to disk, and releases the reference model. The DPO policy is separately reloaded from the same SFT adapter and then updated. The cache is reusable only when checkpoint digest, model, data split, row count, and max sequence length match.

This trades one precomputation pass and local disk for lower accelerator memory use. SFT writes a manifest with the resolved Hub commit (or content hashes for local base-model files) and tokenizer/template identity. DPO refuses adapters without this manifest, pins model loads to the recorded base revision, and validates tokenizer identity before scoring. The reference-cache key also includes the SFT adapter digest, resolved base identity, dataset fingerprint/revision, train split and subset settings, deterministic seed, preprocessing version, tokenizer identity, and maximum sequence length. Corrupt, non-finite, wrong-shape, or metadata-mismatched caches are recomputed. Changing any data selection later requires a correspondingly selected reference cache or a cache regeneration.

## Dataset assumptions and selection experiment

UltraFeedback Binarized provides `chosen` and `rejected` as two-message lists and SFT `messages`; the preference split includes `prompt_id` and scalar `score_chosen` / `score_rejected` values. The [dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) describes GPT-4 scoring the candidate completions and chosen as the highest overall score. The loader assigns stable source row IDs before split/capping so experiment manifests can target an exact candidate subset.

The simplified research add-on defines an **external reward-score margin** as `score_chosen - score_rejected` using those dataset scalar fields, before any DPO updates. It selects the largest margins, with row ID as a deterministic tie-break. Its comparator samples uniformly without replacement from the same post-validation, post-cap train candidates, using the configured seed, and has exactly the same subset size. The full-data arm is all rows in that same candidate pool (the train split minus validation and, if configured, the sample cap). Validation IDs are explicitly rejected by the data loader; final test rows are never loaded by the selector. The command `python -m scripts.select_preference_data --config ... --subset-size N --output-dir ...` writes selected/random IDs, full candidate IDs, candidate score/margin CSV, seed, dataset revision/fingerprint, selection metadata, and the training config snapshot.

Train each arm from the same SFT output with its own DPO output/cache path (`--output-dir` and `--reference-logps-path`); this keeps checkpoints and reference caches from overwriting one another. The Colab notebook optionally runs all three arms and evaluates them together on validation. For independent seeds, create a new seed-specific config/output directory and rerun selection with that seed.

This is a simplified experiment inspired by [Deng et al., *Less is More*](https://arxiv.org/abs/2502.14560), not a reproduction. The referenced paper proposes margin maximization with a Bayesian aggregation of external and implicit margin sources; this implementation uses a single dataset-provided scalar-score difference and does not use implicit DPO margins, pilot training, or the paper's aggregation procedure. No improvement is assumed.

## Checkpoint resumption

Periodic checkpoints contain the PEFT adapter plus optimizer/scheduler state, random number generator states, global step, epoch, run-configuration digest, sequential batch cursor, and (for SFT checkpoints) the model/tokenizer manifest. Sequential batches use a fixed deterministic data order to make the next batch resumable. A resume request with different run metadata is rejected. Training has finite-loss and finite-gradient-norm guards. Checkpoint recovery has unit-test coverage but interruption/restart has not been tested during a full dataset run.

## Comparison and research caveats

The trainer validates on a deterministic partition of the training splits, grouped by prompt identity. Final preference test data is only loaded by the evaluator when `--split test` is explicit; evaluation defaults to validation. The evaluator reports chosen-vs-rejected response sequence-log-probability preference accuracy and margins. For DPO checkpoints it can also report implicit rewards relative to the exact SFT reference. It exports deterministic generated responses and their token lengths; the generations are not automatically quality-scored. These diagnostics are not a human preference benchmark. The test split should be evaluated only after beta/model selection is complete. No benchmark performance or improvement is implied by the code.

Base, SFT, and DPO are generated in the same evaluation invocation with common `max_new_tokens` and deterministic decoding. Evaluation writes per-example response-length CSVs plus aggregate metrics CSV/JSON. Response length is descriptive and is not interpreted as better alignment. `scripts/aggregate_results.py` pools actual `metrics.csv` files by split/checkpoint/beta and seed, reporting a mean and sample standard deviation only from available runs; one-seed standard deviation is blank. No output is fabricated when runs are absent.

## Colab workflow

`notebooks/colab_run.ipynb` clones the selected GitHub branch, checks and prints the GPU, installs the pinned dependencies, verifies CUDA, mounts Drive, and routes model/dataset cache, checkpoints, reference caches, logs, and outputs to Drive. Commit and push local changes before running its clone cell. The editable smoke defaults are 128 training examples and 512 tokens, with batch size 1 and the Qwen 0.5B LoRA config. It runs SFT before DPO, supports setting periodic checkpoint paths to resume, and supports a beta sweep that reuses the same SFT checkpoint and split. After smoke succeeds, increase the sample cap and rerun SFT then DPO. Colab/T4 throughput and memory remain unvalidated.

## Validation status

The local CPU suite checks the DPO equation against both a closed form and installed TRL 0.12.2, masking/EOS/padding behavior, cache identity, manifest validation, exact frozen SFT adapter restoration, a tiny LoRA optimizer update, and checkpoint state restoration. Dependency versions were installed together and `pip check` passed. There has been no complete UltraFeedback training run, Qwen fine-tuning run, GPU run, or Colab/T4 validation. Colab runtime capacity, throughput, and VRAM usage therefore remain to be measured.
