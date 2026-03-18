# Audit — Cycle 014

## Item
25 — RTG spectral coarsening — GPT-2 FFN analog

## Phase 1: Verdict Check

Claimed outcome: **refuted** — spectral coarsening performs at parity with or worse than k-means on raw Jacobians.

Agenda refuted criterion: "spectral coarsening performs at parity with or worse than k-means on raw Jacobians."

Evidence: Across 4 blocks × 4 K values (16 comparisons), spectral is worse in 9/16, best improvement is 1.3% (far below 15% threshold). Max spectral error at K=8 ranges 0.51-0.71.

**Verdict is correct.** The refuted criterion is met cleanly.

## Phase 2: Quality Checks

### G1: Methodology
- 4 blocks tested (0, 2, 5, 11) — representative of easy to hard. Acceptable scope reduction.
- 128 tokens per block. For 768×768 Jacobians, this gives k-NN with k=10 on 128 points — quite sparse (10/128 ≈ 8% connectivity). The k-NN parameter might affect results. However, increasing k would make the graph denser and the Laplacian more like a kernel matrix, converging toward standard k-means behavior.
- Analytical Jacobian J = W_down @ diag(σ'(z)) @ W_up — correct computation.
- Per-cluster affine fit + reconstruction error metric — standard approach.

### G2: Reproducibility
Single seed. Acceptable for exploratory work. The result is consistent across all 4 blocks.

### G3: Tautology Check
Not tautological. The equivalence between spectral and k-means is an empirical finding.

### G4: Scope and Connection
Within scope. Answers whether graph-Laplacian structure in Jacobian space provides better merging than flat k-means → answer is no.

### G5: Counterintuitive
Not counterintuitive. The Jacobian space is already Euclidean, so there's no reason the graph structure would add information beyond what k-means captures. The k-NN graph Laplacian converges to the data density kernel as k→∞, which means spectral clustering on a dense k-NN graph approximates kernel k-means — similar to standard k-means in practice.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: none
reason: Clear refutation across all blocks and K values. Spectral clustering on Jacobian k-NN graphs provides no advantage over k-means. The result is consistent and interpretable.
recommendations: Both Euclidean clustering (cycle 012) and graph-based clustering (cycles 013-014) fail to find exploitable redundancy. The problem may not be the clustering method but the representation space. Item 34 (region volume Pareto) tests a fundamentally different question: are there dominant computational modes regardless of clustering? Item 36 (Procrustes) tests whole-layer linearity. These should be prioritized.
