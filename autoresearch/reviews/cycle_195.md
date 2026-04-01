# Cycle 195 Review: Jacobian Effective Rank Census (Item 206)

## Verdict: REFUTED

FFN block Jacobians are high-rank (near full-rank) for all 12 blocks. No low-rank active
subspace exists. Geometric compression via Jacobian-informed subspace is not viable.

## Key Results

| Block | mean_rank_90 | std | stacked_rank |
|-------|-------------|-----|--------------|
| 0     | 166.8       | 46.5 | 359         |
| 1     | 387.6       | 42.0 | 549         |
| 2     | 441.4       | 51.6 | 588         |
| 3     | 500.8       | 41.5 | 617         |
| 4     | 519.9       | 36.7 | 618         |
| 5     | 523.1       | 37.0 | 623         |
| 6     | 524.6       | 37.3 | 625         |
| 7     | 531.6       | 36.2 | 633         |
| 8     | 536.8       | 37.1 | 638         |
| 9     | 543.5       | 33.2 | 642         |
| 10    | 533.9       | 33.6 | 633         |
| 11    | 459.3       | 49.3 | 591         |

Overall mean rank: 472.4/768 (61% of dimensions).
Low-rank blocks (<100 mean): 0/12.

## Interpretation

- Jacobians are data-active across 60-70% of the embedding dimension for blocks 2-11
- Block 0 is lower (166.8/768 = 22%) but still not low-rank by the 100/768 criterion
- Stacked Jacobians span 359-642 dimensions — nearly half to nearly all of the 768-dim space
- Rank gradient: low early (block 0: 166) → high middle (block 9: 543) → slight drop (block 11: 459)
  This mirrors block 11 anomaly from cycle 184/189
- No block has mean_rank < 100 (CONFIRMED threshold) → REFUTED

## Consistency with Prior Results

- Cycle 184: residual stream divergence has rank_90=300-434. Jacobians now show 167-543. Consistent.
- Cycle 189: cascade errors diffuse in content (rank_90=320-553). Consistent.
- R3 hypothesis (MASO/spline theory: blocks operate in low-rank active subspaces) is REFUTED for GPT-2.
- GPT-2 FFN blocks function in near-full-rank subspaces. MASO-style exact geometric compression
  cannot exploit a low-rank active subspace because no such subspace exists.

## Methodological Notes

- jacfwd + vmap on CPU. CHUNK=8 to manage memory under concurrent training process.
- N_SEQS=5 (160 tokens) due to memory pressure. Block-0 cross-check: 30-seq run (960 tokens)
  gave mean_rank=169.9 — consistent with 5-seq run (166.8).
- Stacked rank computed via XtX eigendecomposition to avoid O(N_tokens × 768^2) memory.
