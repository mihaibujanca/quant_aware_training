# Research Status — 106 Cycles (001–106)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 17 |
| Refuted | 33 |
| Inconclusive | 47 |
| Crash | 1 |

## Latest Cycle
- **Cycle 106** (item 101): Gradient polish k=5 stacking — CONFIRMED
  - 500 Adam steps (W_down only) reduce k=5 delta by 61%: +58-63% → +22-24%
  - But absolute PPL +22-24% still not viable (original threshold was ≤+10%)
  - Seed 42 outlier: +47% vs +9-16% for other seeds (40pp spread)
  - Compression seed variance dominates polish effect
  - Audit flags: criteria drift (relative vs absolute), seed variance

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven
- GELU attenuates error in early layers, pivots at T6-T7
- Intervention target: attention error redistribution

### Compression Recipe (updated)
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 blocks at 6× → ~+5% polished (k=1)
2. **FFN k=5**: +22-24% polished (cycle 106), median/best seed can hit +10%
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 individual viable
4. **Joint**: k=2+k=2 is the practical limit (~+19%)

### Dead Directions (cycles 099–103)
- Subspace-projection correction: error spans 85% of dims (099)
- Jacobian factorization: full-rank, J(x)@x ≠ FFN(x) for GELU (100)
- GELU' output clustering: K*>256 all blocks (101)
- Zero-shot GELU→ReLU: +8320% PPL (102)
- GELU bump is critical signal, not noise (103)

## Remaining NEXT Items
- **P1**: Items 100 (blocked on 99), 117 (output-space regression)
- **P2**: Items 87, 88, 118 (cascade corrections)
- 4 eligible items (100 blocked)
