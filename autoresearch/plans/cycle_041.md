# Cycle 041 Plan -- GPT-2-medium Single-Block 4x Generalization Probe

## Pre-flights
- P1 PASS: Item 59 is P1, eligible per STATUS.md.
- P2 PASS: Depends on item 52 (DONE, cycle 035).
- P3 PASS: No skip conditions (item 59 has no "skip if refuted" clause).
- P4 PASS: Not a moonshot.

## Goal
Determine whether the topk + lstsq refit pipeline generalizes from GPT-2-small to GPT-2-medium at 4x single-block compression. Test 6 evenly spaced blocks with 3 seeds.

## Script
New file: `experiments/gpt2_medium_generalization_probe.py` (adapt from `compression_ceiling_scan_gpt2.py`).

## Procedure

### Step 1 -- Setup
Load GPT-2-medium via `TeacherLoadSpec(hf_model_name="gpt2-medium")`. Constants:
- `TARGET_BLOCKS = [0, 4, 8, 12, 16, 20]`
- `RATIO = 4` (hidden_fraction = 0.25, keeping 1024 of 4096 channels)
- `SEEDS = [42, 123, 7]`
- `PROBE_TRAIN_SEQ = 200`, `PROBE_HOLDOUT_SEQ = 200`
- `EVAL_VAL_SEQ = 200`, `SEQUENCE_LENGTH = 128`

### Step 2 -- Baseline
For each seed: build probe corpus with `shuffle_seed=seed`, compute baseline PPL on 200 eval sequences. Record GPT-2-medium baseline PPL (expect ~20-25, lower than GPT-2-small's ~120).

### Step 3 -- Single-block sweeps
For each seed, for each block in TARGET_BLOCKS:
1. Probe FFN geometry for the block.
2. Compress at 4x via topk channel selection + lstsq refit.
3. Swap single block, measure PPL on eval sequences.
4. Record: seed, block, baseline_ppl, compressed_ppl, relative_change_pct.

### Step 4 -- Aggregate
Per block: mean and std of relative PPL change across 3 seeds.

### Step 5 -- Plot
Bar chart: 6 blocks on x-axis, bars showing mean relative PPL change, error bars = cross-seed std. Horizontal line at +10%.

## Verdict Criteria
- **CONFIRMED**: >=4/6 blocks have mean relative PPL <=10%.
- **REFUTED**: <=1/6 blocks pass.
- **INCONCLUSIVE**: 2-3/6 blocks pass.

## Outputs
- `autoresearch/results/cycle_041_medium_generalization.json`
- `autoresearch/results/cycle_041_medium_generalization.png`

## Risk: lstsq sample count
At 4x: keeping 1024 channels. Lstsq fits 1024 inputs to 1024 outputs with ~25K tokens (200 seqs * 128). Ratio ~25:1, safe (well above MP3 threshold of 2:1).

## Risk: memory
GPT-2-medium is ~1.4GB float32. Probe activations for 200 seqs at d=4096: ~200*128*4096*4B = ~400MB per block. Total fits in 16GB RAM. If tight, reduce to 128 train / 128 eval sequences and note in results.

## Estimated cost
~20 min: model load (~30s) + 3 seeds * (probe 6 blocks + 6 single-block swaps + eval) = 18 swap+eval cycles.
