# Item 135: Data-Conditioned Local Affine Map Extraction — GPT-2 FFN

## Item
Extract per-token Jacobians of GPT-2 FFN blocks to count distinct local affine behaviors. For blocks {4, 7}, compute J_l(x) = d(FFN_l(x))/dx via `torch.func.jacrev` over 128 OWT tokens. Cluster by pairwise cosine similarity of flattened 768x768 Jacobians, report effective number of distinct groups (spectral gap), and verify functional interchangeability within high-similarity pairs.
- **Confirmed**: <=50 distinct affine groups cover >=90% of tokens in both blocks.
- **Refuted**: >200 distinct groups needed for 90% coverage.
- **Inconclusive**: between 50-200 groups, or groups exist but functional interchangeability fails.

## Context
- Walker et al. 2025 centroid affinity: GELU FFNs act as piecewise-affine maps (MASO/spline viewpoint). If few distinct behaviors exist, centroid-affine compression is viable.
- Cycle 129: isotropy ratio ~0.015 in blocks 4-7 (learned, not architectural). High isotropy may mean Jacobians are similar.
- Cycle 130: Procrustes residuals >84% across blocks — blocks are functionally dissimilar. But WITHIN-block token similarity is untested.
- Item 23 (OBSOLETE) was a similar idea but used k-means on full Jacobians. This item uses cosine similarity + spectral gap, which is more informative.

## Scope Reduction
Original item: 1024 tokens x 4 blocks = 4096 Jacobians. Each 768x768 Jacobian via `torch.func.jacrev` takes ~0.5-1s on CPU. Total: 30-60 min compute + memory-intensive pairwise similarity.
**Scoped to**: 128 tokens x 2 blocks (4, 7) = 256 Jacobians. ~5-10 min compute. Sufficient to test whether distinct affine groups exist. Centroid-affine compression test deferred.

## Dependencies
- None hard. GPT-2 pretrained weights only.

## Key Files
- Item 23 (obsolete, similar idea): agenda.md line 31
- Isotropy results: `autoresearch/results/cycle_129_isotropy_architectural.json`
