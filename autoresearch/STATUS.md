# Research Status — 108 Cycles (001–108)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 17 |
| Refuted | 33 |
| Inconclusive | 49 |
| Crash | 1 |

## Latest Cycle
- **Cycle 108** (item 119): Pre-activation distribution anatomy — INCONCLUSIVE
  - Early blocks (1-3): 21-29% excess concentration in GELU knee (active, not variance-driven)
  - Late blocks (7-11): match Gaussian within 5% (concentration = variance effect)
  - Bump-dominated channels: 44% at block 1, ~0% by block 7
  - Negative ρ(σ, importance): bump channels matter for compression

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven
- GELU attenuates error in early layers, pivots at T6-T7

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished (k=1)
2. **FFN k=5**: +22-24% polished (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 viable
4. **Joint**: k=2+k=2 practical limit (~+19%)

### Block Structure Gradient
- Middle blocks (4, 7): nearly linear (cycle 107)
- Early (1-3): active GELU concentration, bump-dominated channels (cycle 108)
- Late (7-11): variance-driven, Gaussian pre-activations (cycle 108)

### Dead Directions (cycles 099–103)
- Subspace projection, Jacobian factorization, GELU' clustering, zero-shot GELU→ReLU, bump masking

## Remaining NEXT Items
- **P0**: Items 120, 121, 122, 123
- **P1**: Items 100 (blocked), 124, 125
- **P2**: Items 87, 88, 118, 126, 127
- 10 eligible items
