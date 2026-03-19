# Research Status — 32 Cycles (009-039)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 6 (010, 019, 026, 027, 033, 039) |
| Refuted | 13 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Latest (cycle 039) — BREAKTHROUGH
Item 58: Residual-stream lstsq correction CONFIRMED. k=5 recovers 3.6pp (+16.12% → +12.53%). k=4 also improves (+9.59% → +8.35%). Zero extra inference parameters. This partially breaks the 4-block wall.

## Practical Compression Summary (GPT-2 FFN, proxy)
| Config | Blocks | Ratio | FFN reduction | PPL impact | Notes |
|--------|--------|-------|-------------:|------------|-------|
| **Best w/ correction** | 5 (7,5,1,3,6) | 6× | ~34% | +12.53% | lstsq correction at block 8 |
| **Best w/o correction** | 4 (7,5,4,1) | 6× | 27.8% | +9.77% | |
| **Best w/ correction** | 4 (7,5,4,1) | 6× | 27.8% | +8.35% | lstsq correction at block 8 |
| Conservative | 4 (7,5,4,1) | 4× | ~18.5% | +8.84% | |

## Key Methods
1. **Topk channel selection** with centroid similarity merge signal (026-027)
2. **Lstsq output refit** per block — acts as regularization at high compression (035)
3. **Residual-stream affine correction** — recovers 3.6pp at k=5, zero inference cost (039)

## Eligible Next Items
- **P1 Item 59**: Generalization to GPT-2-medium
- **P1 Item 54**: Compression ceiling 8×/10× single-block
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor

## Open Questions
- Can multi-point corrections (one per block cluster) push k=5 below 10%?
- Does the method generalize beyond GPT-2-small?
- What is the single-block ceiling (8×? 10×?)?
