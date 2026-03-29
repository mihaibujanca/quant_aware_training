# Research Status — 97 Cycles (001–097)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 28 |
| Inconclusive | 45 |
| Crash | 1 |

## Latest Cycle
- **Cycle 097** (item 114): Block-0-skip attention stacking — REFUTED
  - 5-block skip-0: +43% PPL (above +20% threshold)
  - BUT attention cascade rate c≈0.07/block — 6× weaker than FFN c≈0.41
  - Block proximity drives 2× penalty: [1,3,4]=+25% vs spread [1,4,9]=+13%
  - Attention errors are spatial; FFN errors are position-independent

## Key Structural Findings

### Cascade Rates
- **FFN**: c≈0.41/block — ratio, allocation, order independent (089, 093, 094)
- **Attention**: c≈0.07/block — much weaker but spatially dependent (097)
- Single-block ≤+5% threshold does NOT predict stackability

### FFN Compression
- Topk + lstsq + 500 Adam → ~+5% PPL at k≤4 blocks 6×
- All optimization directions exhausted (basis, channels, allocation, Newton, piecewise-linear)

### Attention Compression
- 5/12 blocks viable at 5.4× individually (095)
- Stacking at 5 blocks: +43% — too much, but mechanism is different from FFN
- Spatial interaction: adjacent compressed blocks amplify each other

## Remaining NEXT Items
- **P0**: Items 103 (low-rank Jacobian), 112 (inter-block residual correction)
- **P1**: Item 113 (joint FFN+attention compression)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
