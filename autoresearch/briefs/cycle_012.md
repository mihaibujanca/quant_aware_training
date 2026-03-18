# Item
22. Effective number of affine behaviors via Jacobian k-means — toy ReLU

# Context
- Cycle 010 (item 20): toy ReLU net on spirals CONFIRMED — deep layers near-linear (r < 0.06), but **do not assume all deep layers use single affine map globally.**
- This item tests: many activation patterns exist in practice (due to ReLU branching), but how many are **functionally distinct** after clustering by their A_ω affine maps?
- If K* << N_patterns, compression can collapse many patterns into fewer representatives with minimal accuracy loss.

# Dependencies
- None (no prior items required).
- Builds on cycle 010 methodology (single-affine residual r) but extends to multi-region analysis.
- Feeds into item 23 (GPT-2 Jacobian k-means), items 24-25 (RTG spectral merging), item 41 (partition synthesis).

# Key Files
- `aleph/relu_geometric_compression.py` — MASO affine extraction (A_ω = ∏ diag(q^i) W^i) logic likely present or needed
- `experiments/` — candidate for experiment script (e.g., `experiments/jacobian_kmeans_toy.py`)
- `autoresearch/results/` — store K* sweep curve + cluster diagnostics

# Constraints
- **Use spirals toy (2D→64×4→1) from cycle 010** — trained network exists or reproducible.
- **Extraction method**: compute all observed activation patterns on probe data (2nd half, n=1000 from cycle 010); extract A_ω for each via MASO formula `A_ω = W_L·diag(q^L)·…·W_1·diag(q^1)`.
- **Clustering**: k-means on flattened A_ω vectors; sweep K ∈ {2, 4, 8, 16, 32, …} up to K = N_patterns.
- **Metric**: reconstruction error when replacing each A_ω with nearest cluster-centroid A_ω̄_k on held-out test set. Find K* where accuracy drop ≤ 1%.
- **Success criterion**: K* / N_patterns ≤ 0.2 (≥80% pattern redundancy).
- **Failure**: K* / N_patterns > 0.7 (most patterns functionally distinct).
- **Plot**: K vs. accuracy loss curve; report K*, N_patterns, ratio.

# Handoff
Implement activation-pattern extraction and A_ω clustering pipeline. Use cycle 010 toy network and probe data. Extract exact affine map for each observed pattern; k-means on flattened matrices. Measure reconstruction accuracy as K grows. Report K* and redundancy ratio to determine if functional redundancy justifies multi-region compression for toy ReLU.
