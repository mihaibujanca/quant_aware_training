# Item
**106: GELU output clustering (corrected) — GPT-2 FFN**

# Context
- Cycle 085 (item 97) refuted centroid compression using Taylor linearization around cluster centroids. But wrong methodology: tested `y ≈ gelu(z_c) + gelu'(z_c)*(z - z_c)`, which trivially fails for smooth nonlinearities.
- Correct question: do tokens in same GELU'(z) cluster produce similar **actual FFN outputs**? (Apply GELU per-token, compare outputs.)
- Cycles 099–100: compression error diffuse (rank_95≈500/768, 85% of residual dims). Projection corrections ruled out. Jacobian factorization (c100) also refuted.
- Item 98 (centroid affine compression) blocked pending this correction.

# Dependencies
- None. Supersedes item 97 (corrects methodology).
- Unblocks item 98 if K* ≤ 64.

# Key Files
- Cycle 085 code (if available): item 97 refutation baseline.
- Existing FFN capture utilities in `experiments/` (data loading, GELU derivatives).

# Constraints
- **Confirmed**: K* ≤ 64 for ≥6/12 blocks (tokens in same GELU' cluster produce similar outputs).
- **Refuted**: K* > 256 for majority of blocks (outputs too diverse even with correct reconstruction).
- **Inconclusive**: K* varies wildly across blocks.
- Metric: within-cluster output variance (actual FFN outputs, not linearized).
- Test k ∈ {8, 16, 32, 64, 128, 256} on all 12 blocks.

# Handoff
Cluster GELU'(z) vectors (3072-dim) via k-means across 2048 tokens. For each block and each k: compute mean FFN output per cluster, measure within-cluster reconstruction error as fraction of output norm. Report K* = min k where error < 5%. Also compute within-cluster vs between-cluster output variance ratio. If K* low, item 98 returns as viable (cluster-mean outputs as centroid reps).
