# Research Status — 109 Cycles (001–109)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 18 |
| Refuted | 33 |
| Inconclusive | 49 |
| Crash | 1 |

## Latest Cycle
- **Cycle 109** (item 120): Negative leak decomposition — CONFIRMED
  - GELU negative leak is 87-99% constant per-channel bias (b_energy 0.871-0.985)
  - Leak is large (b_ratio up to 2.4× block output) but structurally predictable
  - Only 1-13% is token-dependent signal
  - Could fold negative-regime mean into c_proj.bias for compression

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven
- GELU attenuates error in early layers, pivots at T6-T7

### GELU Structure (cycles 102-103, 108-109)
- Zero-shot GELU→ReLU: catastrophic +8320% (cycle 102)
- Bump zone: 31.5% energy, 94.6% of activations (cycle 103)
- Early blocks: active concentration in GELU knee (cycle 108)
- **Negative leak = constant bias (87-99%), not computation (cycle 109)**

### Compression Recipe
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 at 6× → ~+5% polished
2. **FFN k=5**: +22-24% polished (cycle 106)
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0)
4. **Joint**: k=2+k=2 practical limit (~+19%)

### Block Structure Gradient
- Middle blocks (4, 7): nearly linear (cycle 107)
- Early (1-3): active GELU concentration, bump-dominated (cycle 108)
- Late (7-11): Gaussian pre-activations (cycle 108)

## Remaining NEXT Items
- **P0**: Items 121, 122, 123
- **P1**: Items 100 (blocked), 124, 125
- **P2**: Items 87, 88, 118, 126, 127
- 10 eligible items
