# Research Status — 105 Cycles (001–105)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 16 |
| Refuted | 33 |
| Inconclusive | 47 |
| Crash | 1 |

## Latest Cycle
- **Cycle 105** (item 116): GELU negative-leak aware compression — INCONCLUSIVE (design review reject round 2)
  - lstsq target ordering issue: b_leak post-hoc creates systematic overshoot
  - Item remains valid, needs corrected plan

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven
- Attention redistributes FFN error δ across sequence positions, amplifying norm
- GELU attenuates error in early layers, pivots at T6-T7
- Intervention target: attention error redistribution

### Cascade Rates
- **FFN**: c≈0.41/block, ratio/allocation/order-independent
- **Attention**: c≈0.07/block (own compression), spatially dependent
- **Joint FFN+attention**: SUPER-ADDITIVE (1.93× additive prediction)

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 blocks at 6× → ~+56% unpolished, ~+5% polished (k=1)
2. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 individual viable
3. **Joint**: k=2+k=2 is the practical limit (~+19%)

### Dead Directions (cycles 099–103)
- Subspace-projection correction: error spans 85% of dims (099)
- Jacobian factorization: full-rank, J(x)@x ≠ FFN(x) for GELU (100)
- GELU' output clustering: K*>256 all blocks (101)
- Zero-shot GELU→ReLU: +8320% PPL (102)
- GELU bump is critical signal, not noise (103)

## Remaining NEXT Items
- **P1**: Items 100, 101, 117 (output-space regression)
- **P2**: Items 87, 88, 118 (cascade corrections)
- 6 eligible items
