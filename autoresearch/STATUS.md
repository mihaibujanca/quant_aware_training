# Research Status — 101 Cycles (001–101)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 31 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 101** (item 106): GELU output clustering (corrected) — REFUTED
  - K* > 256 for all 12/12 blocks (threshold: K* <= 64 for >=6 blocks)
  - k=256 reconstruction error: 0.67–1.02 (far above 5% threshold)
  - Error increases with k in blocks 3–10 (GELU' clustering actively misleads)
  - Corrected methodology confirms cycle 085 conclusion, closes centroid-output path
  - Item 98 (centroid affine compression) now obsolete

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

### Error/Factorization Structure (cycles 099–101)
- FFN compression error spans ~85% of residual-stream dims (cycle 099)
- Per-token Jacobians nearly full-rank, J(x)@x ≠ FFN(x) for GELU (cycle 100)
- GELU' clustering: K* > 256 all blocks, error increases with k (cycle 101)
- Random bases ≥ SVD bases (JL effect at high effective dim)
- Subspace-projection corrections, Jacobian factorization, AND centroid clustering all ruled out

## Remaining NEXT Items
- **P0**: Items 107, 108, 109
- **P1**: Items 100, 101
- **P2**: Items 87, 88
- 7 items total
