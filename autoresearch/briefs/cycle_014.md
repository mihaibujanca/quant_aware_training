# Cycle 014 Handoff — Item 25: RTG Spectral Coarsening on GPT-2 FFN

## Item
**25. RTG spectral coarsening — GPT-2 FFN analog**
- Build k-NN graph on per-token Jacobians for each GPT-2 FFN block (k=10, distance=Frobenius norm).
- Compute Laplacian, extract K eigenvectors, cluster tokens into K groups.
- Fit per-cluster affine approximation, measure output reconstruction error.
- Compare vs k-means on raw Jacobians (exp 23 — skipped, but run k-means baseline inline).

## Context
- **Cycle 013**: RTG spectral on toy ReLU refuted fragmentation problem. Hamming-1 graph = 79 disconnected components → spectral clustering averaged across unrelated regions. Accuracy plateaued at 55-60% vs float 95.5%.
- **Gap filled by item 25**: Replace Hamming-1 (binary patterns) with k-NN graph in Jacobian space → dense neighborhood structure avoids fragmentation.
- **Dependency check**: Item 9 ✓ (cycle 004), Item 24 ✓ (cycle 013, inconclusive not refuted, so item 25 remains eligible).
- **Skip dependency**: Item 23 was contingent on item 22. Item 22 refuted (K*/N=1.0). Skip item 23 formal experiment, but item 25 runs k-means as baseline for comparison anyway.

## Dependencies
- **Item 9** (benchmark lock-in): proxy gpt2 on Shakespeare corpus. Confirmed cycle 004.
- **Item 24** (toy RTG): inconclusive. Spectral signal exists but toy graph is fragmented. No refutation → item 25 still eligible.
- **No hard blockers**: Item 22 refuted (k-means on toy MASO maps). Item 23 skipped. Item 25 is independent experiment with different data (GPT-2 Jacobians).

## Key Files
- **Input**: `experiments/jacobian_variance_atlas.py` (Jacobian computation patterns for GPT-2 FFN).
- **Output**: `experiments/rtg_spectral_gpt2.py` (new), `autoresearch/results/cycle_014_rtg_spectral_gpt2.json`.
- **Baselines**: Inline k-means on flattened Jacobians per block (replace item 23 experiment).

## Constraints
- **k-NN density**: Hamming-1 = 282 edges / 330 nodes → sparse. k-NN Jacobian graph expected to be dense (all tokens have k=10 neighbors in 768×768 Jacobian space) → well-connected Laplacian → stable spectral clustering.
- **Comparison criterion**: Spectral clustering should reduce reconstruction error **≥15% relative to k-means** at matched K for majority of blocks (confirmation). Parity or worse = refutation.
- **Metric**: Use Frobenius norm on per-token Jacobians J(x_i) ∈ ℝ^{768×768}.

## Handoff
Run spectral clustering on GPT-2 FFN Jacobians. Compare vs k-means baseline (item 23 inline). Test K ∈ {2, 4, 8, 16}. Report per-block reconstruction error and total accuracy/perplexity impact. Confirm spectral outperforms k-means by ≥15% relative or refute.
