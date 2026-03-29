# Research Status — 99 Cycles (001–099)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 29 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 099** (item 112): Residual-stream projection correction — REFUTED
  - Error is diffuse: rank_95 ≈ 500–636/768 (~85% of ambient dim)
  - Projecting out top-r error directions causes catastrophic signal destruction (PPL 70 → 33k at r=4)
  - Error-signal entanglement is near-total: even 4/768 projected dims cause 500× PPL increase
  - Inter-block projection correction is fundamentally flawed for this architecture

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

### Error Structure (NEW — cycle 099)
- FFN compression error spans ~85% of residual-stream dimensions
- Error and signal subspaces overlap near-totally
- Cheap subspace-projection corrections ruled out

## Remaining NEXT Items
- **P0**: Items 103, 106, 107, 108, 109
- **P2**: Items 87-88
- 7 items total (agenda refresh may be needed soon)
