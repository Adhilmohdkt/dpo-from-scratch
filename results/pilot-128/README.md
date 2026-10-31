# UltraFeedback T4 pilot

This is a one-seed, 128-example pilot on a Tesla T4, not a full benchmark. The run used `configs/pilot_128.yaml`, Qwen2.5-0.5B-Instruct, UltraFeedback `train_sft` and `train_prefs`, seed 42, max sequence length 512, 128 capped training rows, and a 32-example validation partition. No final-test examples were used.

## Runs

- SFT: one epoch, 16 optimizer steps.
- DPO beta sweep: beta 0.05, 0.1, and 0.5; each run began from the same saved SFT adapter and used the same 128 preference rows.
- Selection pilot: top 32 external score margins versus a seeded random subset of 32 from the same 128 training candidates. Margin is `score_chosen - score_rejected`, read from the dataset before DPO. The beta-0.1, 128-row run is the full-candidate-pool comparator.
- Evaluation: Base, SFT, three beta checkpoints, and both subset checkpoints on the same 32 validation examples, greedy decoding, and 64-token generation cap.

## Results and files

- `evaluation/validation/metrics.csv` and `metrics.json`: actual aggregate evaluation metrics.
- `evaluation/validation/*_predictions.jsonl`: generated outputs; `*_response_lengths.csv`: per-example lengths.
- `plots/validation_comparison.png`: plot generated from the CSV metrics.
- `selection/`: candidate margins and row-ID manifests for the selected, random, and full-candidate arms.
- `configs/`: beta and subset run snapshots.
- `training_logs/`: step-level SFT/DPO training and evaluation logs.
- `summary/training_loss_summary.csv`: actual mean and final logged step losses. SFT uses response-token-summed cross-entropy; DPO uses pairwise logistic loss, so these scales are not comparable.
- `summary/run_resources.csv`: elapsed time and measured GPU memory.
- `summary/aggregated_metrics.csv`: aggregation output. Every group has one seed, so standard deviations are blank.
- `checkpoints/`: local ignored model artifacts. The six final LoRA adapters were copied from Colab; step-8 SFT and beta-0.1 DPO checkpoints include optimizer/resume state. Base model weights are not copied.

Google Drive authorization timed out in this run. Artifacts were transferred from Colab's `/content` into the local workspace before the runtime was released. Drive-backed persistence and recovery have not been validated. Checkpoint binaries are ignored by Git and must be backed up separately before any cleanup of the local workspace.

## Reproduce

In a configured Colab T4 session, run SFT from the pilot config, then the beta-0.1 DPO config. Generate the other beta configs with the notebook's beta-sweep cell. For the selection arms, first run `python -m scripts.select_preference_data --config configs/pilot_128.yaml --subset-size 32 --output-dir results/pilot-128/selection`; then run `results/pilot-128/configs/dpo-margin-n32.yaml` and `dpo-random-n32.yaml`. Evaluate all checkpoints in one call with `python -m src.evaluate --config configs/pilot_128.yaml --split validation --output-dir results/pilot-128/evaluation --checkpoint base=base --checkpoint sft=results/pilot-128/checkpoints/sft --checkpoint dpo_beta_0.05=results/pilot-128/checkpoints/dpo-beta-0.05 --checkpoint dpo_beta_0.1=results/pilot-128/checkpoints/dpo-beta-0.1 --checkpoint dpo_beta_0.5=results/pilot-128/checkpoints/dpo-beta-0.5 --checkpoint margin=results/pilot-128/checkpoints/dpo-margin-n32 --checkpoint random=results/pilot-128/checkpoints/dpo-random-n32`. Then run `python -m scripts.plot_results --metrics results/pilot-128/evaluation/validation/metrics.csv --output results/pilot-128/plots/validation_comparison.png`.

The 61,135-row training splits, independent seeds, full-scale selection arms, and final-test evaluation remain outstanding. The equal 0.5625 preference accuracy across these 32 validation examples does not establish a model or selection improvement. Response length is descriptive; all seven medians reached the 64-token cap.
