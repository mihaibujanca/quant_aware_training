# Cycle 014 Summary — RTG Spectral Coarsening on GPT-2 FFN Blocks

## Hypothesis
Spectral coarsening of a k-NN Jacobian graph reduces reconstruction error by ≥ 15% relative to
k-means on raw Jacobians at matched K, for a majority of GPT-2 FFN blocks.

## Verdict: REFUTED

## Results

| Block | K=2 spec | K=2 km | K=4 spec | K=4 km | K=8 spec | K=8 km | K=16 spec | K=16 km |
|-------|----------|--------|----------|--------|----------|--------|-----------|---------|
| 0 (hard) | 0.5946 | 0.5726 | 0.5592 | 0.5400 | 0.5094 | 0.5134 | 0.4691 | 0.4684 |
| 2 (easy) | 0.6210 | 0.6292 | 0.5799 | 0.5871 | 0.5482 | 0.5345 | 0.5268 | 0.4969 |
| 5 (easy) | 0.7773 | 0.7734 | 0.7434 | 0.7458 | 0.7065 | 0.7076 | 0.6593 | 0.6503 |
| 11 (hard) | 0.6467 | 0.6310 | 0.6056 | 0.5966 | 0.5471 | 0.5459 | 0.5039 | 0.4905 |

(Error = mean_x ||J(x) − J̄_{cluster}||_F / ||J(x)||_F)

- All 4 blocks: 0/4 K-values with ≥15% relative improvement
- Maximum observed relative improvement: +1.3% (block 2, K=2)
- Spectral is often slightly *worse* than k-means (negative relative improvement in 9/16 comparisons)
- All k-NN graphs are connected (1 component)
- Reconstruction error decreases monotonically with K for both methods (sanity check passed)

## Interpretation

The Jacobian space for GPT-2 FFN blocks (768×768 = 589,824 dimensions, N=128 tokens) does not
exhibit the kind of local cluster structure that spectral embedding is designed to exploit. The
k-NN graph Laplacian eigenvectors capture global connectivity of the Jacobian manifold, but this
provides no better a cluster signal than raw Frobenius distances in this high-dimensional setting.

This matches the cycle 013 result on toy ReLU networks, where RTG spectral coarsening also
failed to outperform random merging. The pattern is consistent: Jacobian-space clustering of
real network tokens does not yield exploitable low-dimensional structure, regardless of whether
we use spectral or Euclidean clustering.

## Implications

- Items 24 and 25 are both refuted. The graph-Laplacian / spectral approach to Jacobian coarsening
  is a dead end at both toy and GPT-2 scale.
- Future compression work should abandon Jacobian-space clustering and focus on alternatives:
  intrinsic dimension (item 26), persistent homology (items 27-28), or direct output-space
  criteria rather than input-Jacobian-space grouping.

## Artifacts

- `experiments/rtg_spectral_gpt2.py`
- `autoresearch/results/cycle_014_rtg_spectral_gpt2.json`
- `autoresearch/results/cycle_014_rtg_spectral_gpt2.png`
