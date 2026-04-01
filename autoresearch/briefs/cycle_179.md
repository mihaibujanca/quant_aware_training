# Cycle 179 — Brief

**Item:** 189 — PruneNet spectral channel selection — GPT-2 FFN k=4
**Type:** empirical
**Depends on:** none (weight-only criterion, no calibration data needed for selection)

## Context

Our standard channel selection uses activation-magnitude topk: score each channel by std(h) * ||w_out_col|| * (|mean(h)| + |mean(h')|), then pick the top-k. This requires calibration data to compute h. Spectral selection instead preserves the singular value distribution of W_down (768×3072): pick channels that minimize KS-distance between the spectrum of W_down[:,S] and the full W_down. This is calibration-free for selection (lstsq fitting still uses data).

Baseline from recent cycles: topk stacked PPL ~87.7 (no lstsq), ~70.7 (with lstsq+polish) at k=4, 4 blocks [1,4,7,11].

## What this cycle measures

1. **Spectral greedy selection**: for each block's W_down, greedily build channel set S of size 512 that minimizes KS-distance of singular value distributions.
2. **Stacked PPL** with spectral selection + lstsq + polish vs topk + lstsq + polish.
3. Both conditions at k=4 (512 channels), blocks [1,4,7,11], 3 seeds.

## Criteria

- **Confirmed:** spectral selection stacked PPL ≤ topk - 2 PPL (either raw or with lstsq)
- **Refuted:** spectral selection ≥ topk + 2 PPL
- **Inconclusive:** within ±2 PPL
