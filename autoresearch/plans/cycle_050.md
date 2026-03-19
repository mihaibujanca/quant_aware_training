# Cycle 050 — Item 65: Oracle-guided adaptive compression ratio — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle tested adaptive per-block ratios. Cycle 036 (6x uniform), 042 (10x uniform) are both fixed-ratio. Novel.
- P2 (dependencies): None. Prior results (cycle 048 P1, cycle 036/042 baselines) are inputs, not dependencies.
- P3 (concrete): Yes — single experiment with clear formula and comparisons.
- P5 (null baseline): Uniform k=4@6x = +9.59% (cycle 036), k=4@10x corrected = +8.45% mean (cycle 042). Oracle must beat both.
- P6 (alignment): Same pipeline (topk + lstsq + accumulation), only ratio varies per block.
- P7 (power): 3 seeds (matching cycle 042), reporting mean +/- std.

## Hypothesis
P1 Frobenius std (rho=0.84 with damage) predicts compressibility. Allocating per-block ratios inversely proportional to P1 should yield lower PPL than uniform ratios at equal or greater total FFN param reduction.

## Method
1. Load P1 values from `cycle_048_local_complexity_gpt2.json` for blocks [7, 5, 4, 1] (the k=4 set).
2. Compute per-block ratio: `ratio_i = 10 * (P1_min / P1_i)`, clamp to [4, 10]. P1 values for k=4 set: block 1 = 5.89, block 4 = 5.79, block 5 = 6.46, block 7 = 7.34. P1_min = 5.79 (block 4).
   - Block 4: ratio = 10.0 (easiest, P1=5.79)
   - Block 1: ratio = 9.8 (P1=5.89)
   - Block 5: ratio = 8.97 (P1=6.46)
   - Block 7: ratio = 7.89 (P1=7.34)
3. For each block, `k_channels = int(3072 / ratio_i)` — variable hidden dim per block.
4. Run k=4 accumulation (order: 7, 5, 4, 1) with these oracle-assigned ratios. Apply lstsq correction.
5. Evaluate PPL on Shakespeare holdout. 3 seeds, 200 eval sequences each.
6. Compute total FFN param reduction: weighted average of per-block ratios.

## Baselines (from prior cycles)
| Config | k=4 PPL delta | Source |
|--------|--------------|--------|
| Uniform 6x, no correction | +9.59% | cycle 036 |
| Uniform 6x, corrected | +8.35% | cycle 039 |
| Uniform 10x, corrected | +8.45% mean | cycle 042 |

## Key parameters
- `target_blocks = [7, 5, 4, 1]` (easiest-first, same as cycles 036/042)
- Per-block ratios: computed from P1, clamped [4, 10]
- `selection_mode = "topk"`, lstsq correction on each block
- `probe_train_sequences = 200`, `probe_holdout_sequences = 50`
- 3 seeds: [42, 123, 7]

## Implementation
- New file: `experiments/oracle_adaptive_compression_gpt2.py`
- Reuse `derive_ffn_compression`, `swap_compressed_ffns`, `_evaluate_model` from `aleph/lm_geometric_compression.py`
- Key change: pass variable `CompressionBudgetSpec.hidden_fraction` per block instead of a fixed fraction.
- Also run uniform-ratio at the oracle's mean ratio as a control (mean ratio ~ 9.2x).

## Outputs
- `autoresearch/results/cycle_050_oracle_adaptive.json`: per-block ratios, per-seed PPL, mean/std, total param reduction
- CONFIRMED if oracle k=4 corrected <= +8% PPL AND total param reduction >= 6x equivalent.
- REFUTED if oracle k=4 corrected > +8.45% (worse than uniform 10x corrected).
- INCONCLUSIVE if oracle matches but doesn't beat uniform baselines (ratio allocation doesn't matter).
