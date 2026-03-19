# Cycle 040 Plan — Compression Ceiling Scan: 8x and 10x Single-Block

## Goal
Find the per-block compression wall for blocks 7, 5, 4 by sweeping {6x, 8x, 10x} with 3 seeds and 200 eval sequences. 6x serves as cross-check against cycle 035.

## Script
New file: `experiments/compression_ceiling_scan_gpt2.py` (adapt from `multiseed_6x_confirmation_gpt2.py`).

## Procedure

### Step 1 — Setup
Load GPT-2, same pipeline as cycle 035. Constants:
- `TARGET_BLOCKS = [7, 5, 4]`
- `RATIOS = [6, 8, 10]`
- `SEEDS = [42, 123, 7]`
- `PROBE_TRAIN_SEQ = 200`, `PROBE_HOLDOUT_SEQ = 200`
- `shuffle_seed = seed` for each seed (matching cycle 035)

### Step 2 — Per-seed loop
For each seed:
1. Build probe corpus with `shuffle_seed=seed`, 200 train / 200 eval sequences.
2. Compute baseline PPL on eval sequences.
3. Probe geometry for blocks [7, 5, 4].
4. For each ratio in [6, 8, 10]:
   - Derive topk compression at `hidden_fraction = 1/ratio`.
   - Swap single block, measure PPL.
   - Record: seed, block, ratio, baseline_ppl, compressed_ppl, relative_change.

### Step 3 — Aggregate
Per block per ratio: compute mean and std of relative PPL change across 3 seeds.

### Step 4 — Plot
Bar chart: blocks on x-axis, grouped bars for 6x/8x/10x, error bars = cross-seed std. Horizontal line at +10% threshold.

## Verdict Criteria
- **CONFIRMED**: >=1 block has mean relative PPL <=10% at 8x.
- **REFUTED**: all 3 blocks exceed 10% at 8x.
- **INCONCLUSIVE**: block-dependent (1 passes, pattern unclear).

## Outputs
- `autoresearch/results/cycle_040_ceiling_scan.json`
- `autoresearch/results/cycle_040_ceiling_scan.png`

## Cross-checks
- 6x results must match cycle 035 within 1pp (same seeds, same shuffle_seed, same eval count).
- If 6x diverges from cycle 035 by >1pp on any block, flag pipeline discrepancy before interpreting 8x/10x.

## Estimated cost
~12 min: load + 3 seeds * (probe + 3 blocks * 3 ratios * eval) = 27 single-block swaps + 3 baselines.
