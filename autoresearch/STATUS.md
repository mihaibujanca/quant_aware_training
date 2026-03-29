# Research Status — 102 Cycles (001–102)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 32 |
| Inconclusive | 46 |
| Crash | 1 |

## Latest Cycle
- **Cycle 102** (item 107): Zero-shot GELU→ReLU swap — REFUTED
  - Full-model swap: +8320% PPL (catastrophic)
  - Block 0: +2664% alone (embedding-to-representation transition most sensitive)
  - Late blocks (9-10): +5-7% (locally tolerable)
  - Post-ReLU sparsity: 56-98% zeros
  - Zero-shot swap is dead; piecewise-linear theory requires finetuning after swap

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

### Dead Directions (cycles 099–102)
- Subspace-projection correction: error spans 85% of dims (099)
- Jacobian factorization: J(x) nearly full-rank, J(x)@x ≠ FFN(x) for GELU (100)
- GELU' output clustering: K*>256 all blocks, error increases with k (101)
- Zero-shot GELU→ReLU: +8320% PPL, block 0 catastrophic (102)

## Remaining NEXT Items
- **P0**: Items 108, 109 (109 depends on 107 — now blocked/skipped since 107 refuted)
- **P1**: Items 100, 101
- **P2**: Items 87, 88
- 6 items (minus 109 blocked) = 5 eligible items — may need agenda refresh soon
