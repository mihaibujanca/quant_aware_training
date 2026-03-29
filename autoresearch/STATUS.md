# Research Status — 103 Cycles (001–103)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 33 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 103** (item 108): GELU-ReLU disagreement anatomy — REFUTED
  - Bump zone accounts for 31.5% mean activation energy (threshold: 20%)
  - Max per-block prediction change from masking: 53.8% (block 0; threshold: 10%)
  - 94.6% of activations fall in |z| < 2.5 (bump zone IS the operating regime)
  - 84-88% of disagreement energy from z < 0 (negative leak dominates)
  - All-blocks masking: +1318 PPL, 91.2% prediction change

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

### GELU/ReLU Structure (cycles 102–103)
- Zero-shot GELU→ReLU: +8320% PPL, block 0 catastrophic (cycle 102)
- Bump zone: 31.5% energy, 53.8% prediction changes (cycle 103)
- 94.6% of activations in bump zone — GELU operates predominantly in transition
- Negative leak (z<0) dominates: 84-88% of disagreement energy
- Piecewise-linear theory CANNOT be applied directly to GELU GPT-2

### Dead Directions (cycles 099–103)
- Subspace-projection correction: error spans 85% of dims (099)
- Jacobian factorization: full-rank, J(x)@x ≠ FFN(x) for GELU (100)
- GELU' output clustering: K*>256 all blocks (101)
- Zero-shot GELU→ReLU: +8320% PPL (102)
- GELU bump is critical signal, not noise (103)

## Remaining NEXT Items
- **P0**: Item 109 (BLOCKED — depends on 107 refuted, swap gap >>15%)
- **P1**: Items 100, 101
- **P2**: Items 87, 88
- 4 eligible items — at threshold for agenda refresh
