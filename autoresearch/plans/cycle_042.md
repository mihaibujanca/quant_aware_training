# Cycle 042 Plan — Multi-block Stacking at 10x per-block

## Goal
Test whether 10x single-block headroom (cycle 040) composes at k=4 multi-block, with and without residual correction (cycle 039). Compare against k=4@6x baselines.

## Script
New file: `experiments/multiblock_10x_stacking_gpt2.py` (adapt from `multiblock_accumulation_6x.py` + `ffn_residual_correction_k5.py`).

## Procedure

### Step 1 — Setup
Load GPT-2, standard pipeline. Constants:
- `TARGET_BLOCKS = [7, 5, 4, 1]` (easiest-first, same as cycle 036/037)
- `HIDDEN_FRACTION = 0.1` (10x compression)
- `SEEDS = [42, 123, 7]`
- `PROBE_TRAIN_SEQ = 200`, `PROBE_EVAL_SEQ = 200`

### Step 2 — Per-seed accumulation (no correction)
For each seed:
1. Build probe corpus with `shuffle_seed=seed`, 200 train / 200 eval.
2. Compute baseline PPL on eval.
3. Probe geometry for blocks [7, 5, 4, 1].
4. Accumulate k=1..4 at 10x. Record PPL per k.

### Step 3 — Residual correction at k=4
For each seed, on the k=4 compressed model:
1. Register forward pre-hook on block `max(compressed_blocks) + 1` (= block 8).
2. Collect `h_float` and `h_compressed` on probe_train (200 seqs).
3. Fit affine correction via `torch.linalg.lstsq` (augmented [N, 769] -> [N, 768]).
4. Apply correction hook, evaluate PPL.

### Step 4 — Cross-checks
- k=1 (block 7 only) at 10x must match cycle 040 single-block result within 1pp.
- If k=1 diverges >2pp, flag pipeline discrepancy before interpreting k>1.
- k=4 uncorrected must be >= k=4@6x uncorrected (+9.59%) since 10x is more aggressive.

### Step 5 — Aggregate and plot
Per-k: mean and std of relative PPL across 3 seeds.
Bar chart: k on x-axis, bars for 10x-uncorrected / 10x-corrected / 6x-reference (from cycle 036).
Horizontal lines at 12% (confirm) and 20% (refute) thresholds.

## Verdict Criteria
- **CONFIRMED**: k=4@10x corrected mean PPL <=12% (all 3 seeds below 15%).
- **REFUTED**: k=4@10x corrected mean PPL >20%.
- **INCONCLUSIVE**: mean 12-20%, or cross-seed std >3pp.

## Outputs
- `autoresearch/results/cycle_042_multiblock_10x.json`
- `autoresearch/results/cycle_042_multiblock_10x.png`

## Estimated cost
~15 min: load + 3 seeds x (probe + 4 accumulation evals + hidden collection + lstsq + corrected eval).
