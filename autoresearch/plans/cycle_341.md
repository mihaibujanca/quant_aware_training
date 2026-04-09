# Cycle 341 Plan

## Goal
Full output-covariance spectrum census on GPT-2 small (all 12 blocks). Compare to weight SVD. Identify LASER-viable blocks.

## Step 1: Write script
Path: `experiments/output_cov_census_cycle341.py`

Script logic:
1. Load GPT-2 small from HuggingFace
2. Hook `model.transformer.h[i].mlp.c_proj` for each block i=0..11
3. Run 200 wikitext-2 val sequences, 128 tokens each, seed 42 (same calib set as cycles 278/338)
4. Collect all MLP output activations Y_i per block, shape (N_total, 768)
5. Compute output covariance C_i = Y_i^T Y_i / N_total
6. Eigendecompose with torch.linalg.eigh
7. Compute: top-1/5/10/13 energy fractions, exp(entropy) eff_rank
8. Load cycle 338 weight SVD data for comparison
9. Print comparison table and save JSON to `autoresearch/results/cycle_341_output_cov_census.json`

## Step 2: Compare to cycle 278
- Sanity check: eff_rank values should match cycle 278 (same calib set, same hooks)
- If discrepancy > 1%, flag as error

## Verdict thresholds (from item 350)
- CONFIRMED: >= 3 blocks have output-covariance top-5 energy >= 50%
- REFUTED: only b1 has top-5 >= 50%, all others below 30%
- INCONCLUSIVE: 1-2 blocks besides b1 in 30-50% range

## LASER-viability labels
- Strong: output-covariance top-5 >= 50%
- Moderate: output-covariance top-5 in 30-50%
- Weak: output-covariance top-5 < 30%
