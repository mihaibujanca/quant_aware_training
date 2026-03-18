# Cycle 024 Brief — CKA Representational Similarity Matrix (GPT-2)

## Item 31 Summary

Compute centered kernel alignment (CKA) between all pairs of GPT-2 layer
representations (13 boundaries: embedding + 12 blocks). CKA ≈ 1 means same
geometry up to rotation/scale; CKA ≈ 0 means unrelated. Plot 13×13 heatmap.

## Hypotheses

1. **Plateau hypothesis**: transformer residual stream causes consecutive layers
   to share geometry (CKA ≥ 0.80 in contiguous groups of ≥3).
2. **Damage correlation**: layers doing novel work (large CKA drop to next
   layer) should be harder to remove — Spearman(CKA_drop, damage) ≥ 0.40.

## Known anchor

Cycle 005 per-block damage (8x geometric): layer 0 = 17.7%, layer 11 = 6.7%,
layers 1–10 mostly 0.1–3.1%.

## Criteria (from agenda)

- Confirmed: ≥1 group of ≥3 layers with pairwise CKA ≥ 0.80 AND Spearman ≥ 0.40
- Refuted: CKA < 0.5 for all non-adjacent pairs
- Inconclusive: plateaus exist but removing plateau layers still causes >20% ppl increase

## Plan

Linear CKA on Shakespeare probe → 13×13 matrix → heatmap → plateau detection →
Spearman vs cycle 005 damage.
