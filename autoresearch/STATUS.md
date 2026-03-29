# Research Status — 98 Cycles (001–098)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 28 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 098** (item 113): Joint FFN+attention compression — INCONCLUSIVE
  - Joint k=4 FFN + k=3 attn: +134% PPL (super-additive, 1.93× additive prediction)
  - Joint k=2+k=2: +19% (viable boundary)
  - Cross-modality cascade: compressed attention cannot correct FFN errors in residual stream

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

## Remaining NEXT Items
- **P0**: Items 103, 112
- **P2**: Items 87-88
- Only 4 items
