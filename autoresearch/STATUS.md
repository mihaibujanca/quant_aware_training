# Research Status — 112 Cycles (001–112)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 34 |
| Inconclusive | 51 |
| Crash | 1 |

## Latest Cycle
- **Cycle 112** (item 123): GELU leak-aware compression corrected — INCONCLUSIVE (design failure)
  - Positive-channel partition (neg_frac < 0.5) is near-vacuous for blocks 4/7/11: N_pos = 53–71 (< 3%)
  - Fair comparison forces k_eff = 66 → ~46× compression, catastrophically degrading all conditions
  - B and C identical in PPL (b_leak correction irrelevant when channel count is bottleneck)
  - Only block 1 eligible (N_pos=546); wins≥2 impossible → INCONCLUSIVE by construction
  - Root cause: cycle 111 neg_frac measured per-token-average, not per-channel

## Prior Latest Cycle
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
- **P0**: (Item 123 done — inconclusive)
- **P1**: Items 100 (blocked), 124, 125
- **P2**: Items 87, 88, 118, 126, 127
- 7 eligible items
