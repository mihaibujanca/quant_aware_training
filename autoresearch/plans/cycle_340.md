# Cycle 340 Plan

## Goal
Recompute eff_rank using exp(entropy) formula on existing cycle 332 checkpoints. Compare to participation ratio values.

## Step 1: Write script
Path: `experiments/eff_rank_entropy_cycle340.py`

Script logic:
1. Import TinyGPT from cycle 331 script
2. Load each checkpoint (3 activations × 3 seeds = 9 models)
3. Hook `block.mlp.c_proj` output for each block
4. Run 200 calib sequences (same as cycle 332)
5. Compute output covariance per block
6. Eigendecompose, compute exp(entropy) where p_i = λ_i / Σλ, H = -Σ p_i ln p_i
7. Print and save JSON to `autoresearch/results/cycle_340_eff_rank_entropy.json`

## Step 2: Execute and compare
- Run script, compare exp(entropy) values to cycle 332 participation ratio values
- Check verdict criteria: confirmed if squared > GELU by ≥10% relative at ≥2 blocks

## Verdict thresholds
- Confirmed: exp(entropy) eff_rank for ReLU² or LeakyReLU² exceeds GELU by ≥10% relative at ≥2 blocks
- Refuted: differences < 5% relative or ordering reversed
- Inconclusive: mixed ordering across blocks
