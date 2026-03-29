# Research Status — 113 Cycles (001–113)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 35 |
| Inconclusive | 51 |
| Crash | 1 |

## Latest Cycle
- **Cycle 113** (item 124): Per-head attention amplification census — REFUTED
  - Amplification uniform across heads: max ≤15% in 8/11 transitions
  - Top-3 share = 29-39% (far below 70% threshold)
  - Exception: block 10→11 (top-3 = 72%, heads 0+8 dominate)
  - No surgical head targets for cascade suppression — whole-layer effect

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycles 104, 113)
- **c≈0.41 is attention-driven** (cycle 104), but **uniform across heads** (cycle 113)
- Error diffuses equally through all 12 heads via QKV projections
- No head-level surgical targets — cascade suppression requires whole-layer intervention
- GELU attenuates error in early layers (cycle 104)

### GELU Structure (cycles 102-103, 108-111)
- Negative leak = 87-99% constant bias (cycle 109)
- GELU regime depth-invariant ~85% gated (cycle 111)
- Block 1 anomalously linear (cycle 111)

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished
2. **FFN k=5**: +22-24% polished (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)

## Remaining NEXT Items
- **P1**: Items 100 (blocked), 125
- **P2**: Items 87, 88, 118, 126, 127
- 6 eligible items
