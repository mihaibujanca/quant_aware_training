# Cycle 086: FFN Output Manifold Geometry — Item 99

## Item
Investigate whether random+lstsq matching SVD+lstsq (cycles 071, 077) is explained by near-isotropic output activations, or if there's structure beyond Johnson-Lindenstrauss lemma.

## Metrics
For blocks [7,5,4,1] on 4096 mixed-corpus tokens:
1. **Isotropy ratio** = λ₅₁₂ / λ₁ at k=512 (6× compression)
2. **Compare to Marchenko-Pastur null** (random matrix spectrum)
3. **lstsq condition numbers** for topk, SVD, random bases
4. **JL bound tightness**: empirical distance distortion vs O(√(log(n)/k)) prediction

## Confirmation
- **Confirmed**: isotropy ratio > 0.1 for ≥3/4 blocks → JL explains random~SVD parity
- **Refuted**: isotropy ratio < 0.01 + random still works → structure beyond JL
- **Inconclusive**: moderate isotropy (0.01–0.1)

## Context
- **Cycles 071, 077**: SVD+lstsq ≈ random+lstsq within 3% at k/n∈{0.67, 0.3, 0.2, 0.1}
- **Puzzle**: Why don't "good" directions (top-k singular vectors) outperform random?
- **Literature**: Razzhigaev et al. (EACL 2024) measured anisotropy in transformer representations; bell-shaped profile, ID compression dynamics. Validates anisotropy as a proxy for low effective rank
- **Status.md**: Item 99 is P0 (next priority after cycle 085 refutes item 98)

## Implementation
- Compute cov(h) where h = hidden activations (n×3072)
- Eigendecomposition, measure spectrum tail-to-head ratio
- Collect empirical distances under random, SVD, topk projections
- Test JL predictionε = sqrt(log(n)/k) vs measured distortion

## No dependencies
