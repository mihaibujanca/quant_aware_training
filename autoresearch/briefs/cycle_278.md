# Item
288 (P1): All-blocks c_proj output eff_rank census — extend cycle 267 to blocks {0, 2, 3, 4, 6, 7, 8, 9, 10}.

# Context
- Cycle 267 measured blocks {1, 5, 11}: eff_rank = 1.94, 179.19, 22.49.
- Pure analysis, no training. ~5 min.

# Dependencies
- Cycle 267 SVD machinery.

# Key Files
- experiments/block11_cproj_decomp_cycle267.py — SVD template.

# Constraints
- For each block l ∈ {0, 2, 3, 4, 6, 7, 8, 9, 10}: collect c_proj output on calib, SVD covariance, compute eff_rank, top-{1, 5, 20} energy.
- Also compute per-block c_proj.weight SVD eff_rank.
- Include cycle 267 values for blocks {1, 5, 11} as sanity check.
- Verdict: confirmed if block 4 or 7 has eff_rank ≤ 50 AND mid-blocks {2, 3, 6, 8, 9, 10} all ≥ 100.
- Refuted if blocks 4 AND 7 both ≥ 100.
- Inconclusive otherwise.
- Trainable params: 0.

# Handoff
Plan agent: reuse cycle 267 template. New script `experiments/all_blocks_eff_rank_cycle278.py`.
