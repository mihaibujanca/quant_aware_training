# Cycle 056 Plan: Compression Ratio Sweep on OpenWebText — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle sweeps ratios on OpenWebText. Cycles 034-036 swept on Shakespeare only; cycle 054 tested a single ratio (6x) on OWT.
- P2 (dependencies): Cycle 054 ran and was REFUTED (+48% PPL at 6x). OWT sample cached at `data/openwebtext_sample.txt`. Pipeline unchanged.
- P3 (concrete): Explicit threshold (<10% PPL) and quorum (>=2/3 blocks).
- P4 (feasible): 36 configs (4 ratios x 3 blocks x 3 seeds). Each config ~1 min. ~45 min total.
- P5 (null baseline): Cycle 054 6x@OWT = +48%. Shakespeare 6x = +8.5%. Expect monotonic degradation curve; question is where 10% threshold falls.
- P6 (alignment): Same model (GPT-2), same pipeline, subset of cycle 054 blocks.
- P7 (power): 4 ratio points gives enough resolution to locate the frontier; 3 seeds per point.

## Hypothesis
At least one ratio in {2x, 3x, 4x} achieves <10% PPL degradation on OpenWebText for >=2 of 3 blocks [7, 5, 4].

## Method
Fork `experiments/openwebtext_training_dist_probe.py` into `experiments/owt_ratio_sweep.py`. Changes:

1. Replace `HIDDEN_FRACTION = 1/6` with `RATIOS = {2: 1/2, 3: 1/3, 4: 1/4, 6: 1/6}`.
2. Replace `TARGET_BLOCKS = [7, 5, 4, 1]` with `TARGET_BLOCKS = [7, 5, 4]` (drop block 1).
3. Outer loop: for each seed, build probe corpus once. Inner loops: for each block, for each ratio, compress that single block only, eval PPL (no correction — raw degradation).
4. No affine correction in this sweep — measures raw compression viability per ratio.
5. Reuse cached `data/openwebtext_sample.txt` (skip download if exists — already handled).

## Key Parameters
- `target_blocks = [7, 5, 4]`, ratios = {2x, 3x, 4x, 6x}
- `probe_train_sequences = 200`, `eval_val_sequences = 200`, `sequence_length = 128`
- Seeds = [42, 123, 7]. No correction layer. Single-block swaps only.

## Output Files
- `experiments/owt_ratio_sweep.py`
- `autoresearch/results/cycle_056_owt_ratio_sweep.json` — per-block, per-ratio, per-seed PPL
- `autoresearch/results/cycle_056_owt_ratio_sweep.png` — heatmap: block x ratio, color = mean rel PPL%

## Success Criteria
- **Confirmed**: >=1 ratio <=4x achieves <10% PPL degradation for >=2/3 blocks
- **Refuted**: even 2x exceeds 10% PPL for majority of blocks (>=2/3)
- **Inconclusive**: 2x works but 3x already fails (narrow window)

## Risks
- Single-block swap without correction is a harder bar than cycle 054 (which used correction). But correction added negligible benefit in cycle 054 (+48% corrected vs +56% uncorrected), so omitting it is fair for a sweep.
- 6x data point should replicate cycle 054 uncorrected numbers — if it diverges >5%, flag a data issue.
