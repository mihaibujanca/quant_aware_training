# Cycle 054 — Item 70: Training-distribution probing with OpenWebText — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle uses OpenWebText. Cycles 044/045 used Shakespeare and WikiText-2 only.
- P2 (dependencies): None. Existing pipeline + HuggingFace `datasets` library.
- P3 (concrete): Single hypothesis with explicit +15% threshold on training-distribution eval.
- P4 (feasible): ~30 min. Same pipeline as cycle 045; only the corpus changes.
- P5 (null baseline): Shakespeare-probed +8.5%, WikiText-2-probed +44% (cycle 045). OpenWebText fills the training-distribution gap.
- P6 (alignment): Same model (GPT-2), blocks [7,5,4,1], 6x ratio, correction at layer 8, 3 seeds [42,123,7].
- P7 (power): 3 seeds; threshold band (+15% vs +30%) is wide enough to distinguish from prior results.

## Hypothesis
OpenWebText-probed, OpenWebText-evaluated k=4@6x corrected PPL degradation: **<=+15%**.

## Method
1. Download OpenWebText via `datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)`. Take first 500 documents (stream to avoid full download). Save extracted text to `data/openwebtext_sample.txt` with double-newline separators (same format as shakespeare.txt).
2. Load GPT-2 teacher via `load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))`.
3. For each seed in [42, 123, 7]:
   a. Build probe corpus from `data/openwebtext_sample.txt` via `ProbeCorpusSpec(text_path=..., probe_train_sequences=200, probe_holdout_sequences=50, eval_val_sequences=200, sequence_length=128, dataset_name="openwebtext", shuffle_seed=seed)`.
   b. Evaluate baseline PPL on OpenWebText eval split.
   c. Probe FFN geometry on blocks [7,5,4,1] with `probe_ffn_geometry(max_tokens_per_layer=2048, batch_size=4, jacobian_samples=8)`.
   d. Derive compression at `hidden_fraction=1/6, selection_mode="topk"`.
   e. Swap compressed FFNs, evaluate uncorrected PPL.
   f. Fit affine correction at layer 8 via lstsq (same as cycle 045 `fit_correction`).
   g. Apply correction hook, evaluate corrected PPL.
4. Compute per-seed relative degradation: `(corrected - baseline) / baseline * 100`.
5. Report mean and std across 3 seeds.

## Key parameters
- `target_blocks = [7, 5, 4, 1]`, `hidden_fraction = 1/6`, `selection_mode = "topk"`
- `probe_train_sequences = 200`, `eval_val_sequences = 200`, `sequence_length = 128`
- `correction_layer = 8`, seeds = [42, 123, 7]
- OpenWebText: 500 streaming documents (sufficient for 200+200 sequences at seqlen 128)

## Implementation
- New file: `experiments/openwebtext_training_dist_probe.py`
- Data prep: streaming download at script start, save to `data/openwebtext_sample.txt`
- Reuse `run_recipe` pattern from `experiments/compression_recipe_validation.py`
- Output: `autoresearch/results/cycle_054_openwebtext_probe.json`

## Outputs
- Per seed: baseline_ppl, uncorrected_ppl, corrected_ppl, rel_uncorrected%, rel_corrected%
- Mean +/- std of corrected degradation across seeds
- **CONFIRMED** if mean corrected degradation <= +15%
- **REFUTED** if mean corrected degradation >= +30%
- **INCONCLUSIVE** if 15-30%
