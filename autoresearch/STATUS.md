# Research Status — 111 Cycles (001–111)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 34 |
| Inconclusive | 50 |
| Crash | 1 |

## Latest Cycle
- **Cycle 111** (item 122): GELU depth gradient — REFUTED
  - GELU operating regime is stationary with depth (~85% gated at all layers)
  - 6/11 monotonicity violations, slope 0.3%/block, R²=0.083
  - Block 1 anomalously linear (73% neg_frac, E[GELU']=0.238)
  - No need for depth-adapted compression strategies based on GELU regime

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven

### GELU Structure (cycles 102-103, 108-111)
- Negative leak = 87-99% constant bias (cycle 109)
- GELU regime is depth-invariant ~85% gated (cycle 111)
- Bump channels: low weight norms in deep layers (cycle 110)
- Block 1: anomalously linear (cycle 111)

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished
2. **FFN k=5**: +22-24% polished (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

## Remaining NEXT Items
- **P0**: Item 123 (corrected leak-aware compression)
- **P1**: Items 100 (blocked), 124, 125
- **P2**: Items 87, 88, 118, 126, 127
- 8 eligible items
