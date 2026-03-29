# Research Status — 110 Cycles (001–110)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 33 |
| Inconclusive | 50 |
| Crash | 1 |

## Latest Cycle
- **Cycle 110** (item 121): Channel bump anatomy — INCONCLUSIVE
  - Weight-only Spearman ρ_B: sign inverts with depth (+0.03 block 0 → -0.60 block 9)
  - Bump channels have LOW weight norms in deep layers (compression-unimportant)
  - Full-formula ρ_A = +0.79 is tautological artifact (GELU' term drives it)
  - Bump-critical channels do compute token-varying signals (var > 0.01)

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven

### GELU Structure (cycles 102-103, 108-110)
- Negative leak = 87-99% constant bias (cycle 109)
- Early blocks: active concentration in GELU knee (cycle 108)
- Bump channels: low weight norms in deep layers, compression-unimportant (cycle 110)
- Full-formula importance correlation was tautological artifact (cycle 110)

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished
2. **FFN k=5**: +22-24% polished (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

### Block Structure Gradient
- Middle blocks (4, 7): nearly linear (cycle 107)
- Early (1-3): active GELU concentration (cycle 108)
- Late (7-11): Gaussian pre-activations (cycle 108)

## Remaining NEXT Items
- **P0**: Items 122, 123
- **P1**: Items 100 (blocked), 124, 125
- **P2**: Items 87, 88, 118, 126, 127
- 9 eligible items
