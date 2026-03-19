# Cycle 053 — Item 69: Per-block corpus sensitivity isolation — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle tests single-block corpus sensitivity. Cycles 044/045 compressed all 4 blocks jointly.
- P2 (dependencies): None. Uses existing pipeline.
- P3 (concrete): Per-block sensitivity delta with explicit 50% concentration threshold.
- P4 (feasible): 8 single-block compressions + 16 evals (~15 min single GPU). No training.
- P5 (null baseline): Cycle 044 established the all-blocks gap. Single-block decomposition tests additivity.
- P6 (alignment): Same model, blocks, ratio, probe counts as cycles 044/045.
- P7 (power): 4 blocks x 2 probe corpora x 2 eval corpora = 16 conditions; threshold is one block >= 50%.

## Hypothesis
The corpus sensitivity gap (Shakespeare +6% vs WikiText +120% PPL) is dominated by one or two blocks. Single-block swaps will reveal which block(s) are the bottleneck and whether the sensitivity is in channel selection or lstsq refit.

## Method
1. Load nanochat teacher via `load_nanochat_teacher`.
2. Build two probe corpora via `build_probe_corpus`: Shakespeare (200 train, 200 eval seqs) and WikiText-2 (200 train, 200 eval seqs). Seed 42.
3. For each block b in [7, 5, 4, 1], independently:
   a. Compress block b only (`layer_ids=(b,)`) using Shakespeare probes at 6x (`hidden_fraction=1/6, selection_mode="topk"`).
   b. Swap that single block into the teacher model via `swap_compressed_ffns`.
   c. Evaluate PPL on Shakespeare eval and WikiText-2 eval via `_evaluate_model`.
   d. Record: `shk_ppl_b`, `wt2_ppl_b`.
4. Repeat step 3 using WikiText-2 probes for channel selection + refit.
5. Compute teacher baseline PPL on both eval sets (no compression).
6. Per-block sensitivity delta = `(wt2_ppl_b / wt2_baseline - 1) - (shk_ppl_b / shk_baseline - 1)`.
7. Rank blocks by sensitivity delta. Check if max delta >= 50% of sum of all deltas.

## Key parameters
- `target_blocks = [7, 5, 4, 1]`, tested one at a time
- `hidden_fraction = 1/6`, `selection_mode = "topk"`
- `probe_train_sequences = 200`, `probe_eval_sequences = 200`, `sequence_length = 128`, seed = 42

## Implementation
- New file: `experiments/per_block_corpus_sensitivity.py`
- Reuse from `aleph/lm_geometric_compression.py`: `load_nanochat_teacher`, `build_probe_corpus`, `probe_ffn_geometry`, `derive_ffn_compression`, `swap_compressed_ffns`, `_evaluate_model`, `ProbeCorpusSpec`, `CompressionBudgetSpec`, `TeacherLoadSpec`
- Output: `autoresearch/results/cycle_053_per_block_sensitivity.json`
- Plot: bar chart of per-block sensitivity delta (Shakespeare vs WikiText probes side by side)

## Outputs
- Per block, per probe corpus: shk_ppl, wt2_ppl, sensitivity_delta
- CONFIRMED if one block accounts for >= 50% of total sensitivity delta sum
- REFUTED if all blocks within +/-25% of mean delta
- INCONCLUSIVE if concentration switches between channel-selection and refit across blocks
