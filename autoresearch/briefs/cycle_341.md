# Cycle 341 Brief

## Item
Item 350: Output-covariance spectrum census on GPT-2 — which blocks are LASER-viable?

## Context
- Cycle 278 measured output-covariance eff_rank for all 12 blocks: b1=1.94, b2=1.05, b3=24.3, b10=16.7, b11=22.5 (concentrated); b5-b9 are diffuse (179-298)
- Cycle 338 measured weight SVD for all blocks: all flat (top-5 ≈ 5-12%). LASER works despite flat weight SVD.
- Key insight: LASER viability depends on output-covariance concentration (data-dependent), not weight SVD concentration.
- This cycle produces the definitive comparison table: output-covariance vs weight SVD, with LASER-viability labels.

## Key Question
Which GPT-2 blocks have concentrated output covariance (top-5 energy >= 50%), making them LASER candidates?

## Key Files
- `autoresearch/results/cycle_278_all_blocks_eff_rank.json` — prior output-covariance data (sanity check)
- `autoresearch/results/cycle_338_spectrum_census.json` — weight SVD data
