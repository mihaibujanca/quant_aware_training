# Item
80 — FFN Jacobian effective rank census — GPT-2 FFN

# Context
- **Hypothesis**: Each FFN block implements a globally low-rank function (output directions shared across tokens)
- **Goal**: Measure per-token Jacobian J(x) = W_out @ diag(σ'(z)) @ W_in for each block to validate this claim
- **Output**: Effective rank at 95% and 99% energy thresholds, mean rank, cross-token variance per block
- **Validation criteria**: mean_rank(99%) ≤ 500 for ≥10/12 blocks AND Spearman(rank, P1_frob_std) ≥ 0.5

# Dependencies
None — item 80 is first in P0 spectral analysis tier. Items 81–84 depend on its results.

# Key Files
- `aleph/differential.py` — jacobian() function exists; computes df/dx via torch.func
- `aleph/lm_geometric_compression.py` — GPT-2 loader, ProbeCorpus infrastructure
- Reference data: cycle 048 (P1 Frobenius std per block, rho=0.839 vs 8x damage), cycle 005 (single-block swap damage at 8x/16x)
- Corpus: OpenWebText 512 tokens (per item spec)

# Constraints
- Jacobian computation is O(N × 768²) per block; batch chunks by 256 tokens max
- SVD energy thresholds: compute exact cumsum(s²_i / sum(s²)) for 95% and 99%
- Must report mean and variance of effective_rank across N=512 tokens per block
- Correlation: Spearman rank test vs P1 Frobenius std from cycle 048 JSON (rho ≥ 0.5 to confirm)

# Handoff
Implement notebook that: (1) loads GPT-2, probes OpenWebText ×512 tokens; (2) for each of 12 blocks, computes J(x) using differential.jacobian() or custom FFN Jacobian (W_out @ diag(σ') @ W_in); (3) SVDs each J, reports rank@95%/@99% energy per token; (4) aggregates stats (mean, variance, cross-token alignment); (5) correlates with cycle_048 P1 data. Expected result: early/late layers hard, middle layers ranked 200–500.
