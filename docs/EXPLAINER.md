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

## Dataset assumptions

UltraFeedback Binarized provides `chosen` and `rejected` as message lists and provides SFT `messages`; examples are expected to have a user/context followed by one assistant response. `score_chosen` and `score_rejected` are retained as metadata for later selection analysis. The current trainer does not select/filter rows.

## Checkpoint resumption

Periodic checkpoints contain the PEFT adapter plus optimizer/scheduler state, random number generator states, global step, epoch, run-configuration digest, sequential batch cursor, and (for SFT checkpoints) the model/tokenizer manifest. Sequential batches use a fixed deterministic data order to make the next batch resumable. A resume request with different run metadata is rejected. Training has finite-loss and finite-gradient-norm guards. Checkpoint recovery has unit-test coverage but interruption/restart has not been tested during a full dataset run.

## Comparison and research caveats

The trainer validates on a deterministic partition of the training splits, grouped by prompt identity. Final preference test data is only loaded by the evaluator when `--split test` is explicit; evaluation defaults to validation. The evaluator reports chosen-vs-rejected response sequence-log-probability preference accuracy and margins. For DPO checkpoints it can also report implicit rewards relative to the exact SFT reference. It exports deterministic generated responses and their token lengths; the generations are not automatically quality-scored. These diagnostics are not a human preference benchmark. The test split should be evaluated only after beta/model selection is complete. No benchmark performance or improvement is implied by the code.

## Validation status

The local CPU suite checks the DPO equation against both a closed form and installed TRL 0.12.2, masking/EOS/padding behavior, cache identity, manifest validation, exact frozen SFT adapter restoration, a tiny LoRA optimizer update, and checkpoint state restoration. Dependency versions were installed together and `pip check` passed. There has been no complete UltraFeedback training run, Qwen fine-tuning run, GPU run, or Colab/T4 validation. Colab runtime capacity, throughput, and VRAM usage therefore remain to be measured.

The planned data-selection extension uses the difference between dataset scores as an external margin, compares its top-ranked subset to a seeded random subset of equal size, and includes full-data DPO. This is a deliberately simplified, inspired experiment. The referenced paper additionally uses implicit DPO margins and Bayesian aggregation, and reports results on larger models and benchmarks; reproducing those claims is out of scope.
