# Cycle 085 — Item 97: GELU activation pattern diversity — GPT-2 FFN

## Goal
Measure how many distinct linear behaviors (activation patterns) GPT-2 FFN has. For each block, compute GELU' on 2048 mixed-corpus tokens × 3072 channels → 3072D soft activation patterns. Binarize at threshold 0.5, count unique patterns. Cluster continuous GELU' vectors with k-means (k ∈ {8,16,32,64,128,256}). Find K* = minimum k such that centroid reconstruction error < 5% of block output norm.

## Related Work
- **Cycle 083** (item 94): GELU near-linearity census refuted — found 0.04% dead, 0.01% near-linear across 12 blocks. GELU nonlinearity fully utilized per-channel. Item 97 complements with per-token pattern analysis instead of per-channel.
- **Cycle 084** (item 95): Jacobian clustering plan rejected (design review) due to affine linearization inconsistency. Item 97 uses GELU' clustering directly (no Jacobian).
- **Cycle 072**: Stacking error = retained-spectrum corruption (analytical, no per-pattern analysis).
- **jacobian_kmeans_toy.py**: Toy model k-means infrastructure exists but for Jacobian, not GELU patterns.

## Infrastructure
- **Cycle 083 script** (`experiments/gelu_nearlinearity_census_v2.py`): Loads GPT-2, builds 4096-token mixed-corpus probe, hooks pre-GELU activations z per block, computes GELU'. Can adapt for k-means clustering of GELU' vectors and pattern diversity counting.
- **Mixed-corpus data**: shakespeare.txt, wikitext2_valid.txt, openwebtext_sample.txt, c4_sample.txt in `data/`.
- **Output**: Unique pattern counts, k vs reconstruction error curves (per block), K* values.

## Success Criteria
- **Confirmed**: K* ≤ 64 for ≥ 6/12 blocks (manageable piecewise-linear complexity).
- **Refuted**: K* > 256 for majority (GELU too diverse to cluster).
- **Inconclusive**: K* varies wildly.

## Dependencies
None. Item 97 is independent.

## Notes
- Binarize GELU'(z_j) at 0.5 threshold (not continuous clustering initially).
- Measure Hamming distances between binary patterns for diversity insight.
- Report centroid reconstruction error as % of block output norm (not per-channel).
- If K* ≤ 64 found, item 98 (centroid affine map compression) is a follow-up.
