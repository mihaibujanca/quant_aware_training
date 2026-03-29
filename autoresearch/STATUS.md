# Research Status — 100 Cycles (001–100)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 30 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 100** (item 103): Low-rank Jacobian factorization — REFUTED
  - Mean-Jacobian SVD median reconstruction error = 0.809 at r=128 (threshold: 0.5)
  - Per-token Jacobians nearly full-rank (rank_95 = 270–470/768)
  - Even oracle S(x) gives +14–74% PPL degradation
  - Random U,V outperforms SVD U,V (JL effect — high effective dimensionality)
  - Two conflated failures: rank truncation AND linearization error (GELU non-homogeneity)

## Key Structural Findings (consolidated)

### Cascade Rates
- **FFN**: c≈0.41/block, ratio/allocation/order-independent
- **Attention**: c≈0.07/block, spatially dependent (proximity penalty)
- **Joint FFN+attention**: SUPER-ADDITIVE (1.93× additive prediction)
- **Viable joint boundary**: k=2 FFN + k=2 attn ≈ +19% PPL

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 blocks at 6× → ~+56% unpolished, ~+5% polished (k=1)
2. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 individual viable
3. **Joint**: k=2+k=2 is the practical limit (~+19%)
4. Stacking limit is structural cascade amplification

### Error Structure (cycles 099–100)
- FFN compression error spans ~85% of residual-stream dimensions (cycle 099)
- Error and signal subspaces overlap near-totally — cheap subspace-projection corrections ruled out
- Per-token Jacobians nearly full-rank, mean-Jacobian SVD bases unrepresentative (cycle 100)
- J(x)@x ≠ FFN(x) for GELU: linearization error fundamental limitation
- Random bases ≥ SVD bases (JL effect at high effective dim)

## Remaining NEXT Items
- **P0**: Items 106, 107, 108, 109
- **P1**: Items 100, 101
- **P2**: Items 87, 88
- 8 items total
