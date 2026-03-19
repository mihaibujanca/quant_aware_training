# Research Status — 31 Cycles (009-038)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 13 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Latest (cycle 038)
Item 57: 5-block combo sweep. All 6 combos >15% PPL. Best: drop block 4 (+16.12%). Spread 4.76pp. REFUTED. The k=5 wall is robust — neither per-block ratio (037) nor block combo (038) can break it.

## Practical Compression Summary (GPT-2 FFN, proxy)
| Config | Blocks | Ratio | FFN reduction | PPL impact |
|--------|--------|-------|-------------:|------------|
| **Best viable** | 4 (7,5,4,1) | 6× | **27.8%** | +9.77% |
| Conservative | 4 (7,5,4,1) | 4× | ~18.5% | +8.84% |
| 5 blocks (all combos fail) | any 5 | 6× | ~34% | >+16% |

## The Block Count Wall
Cycles 036-038 establish that the 4-block ceiling is hard:
- Per-block ratio doesn't matter (036: 4×→6× costs +0.75pp)
- 5th block at any ratio fails (037: all k=5 hetero >15%)
- 5-block combo selection fails (038: all combos >15%, spread only 4.76pp)
- **Root cause**: compounding error through residual stream, not individual block quality

## Eligible Next Items
- **P1 Item 58**: Inter-block residual-stream correction — last hope to break the wall
- **P1 Item 59**: Generalization to GPT-2-medium
- **P1 Item 54**: Compression ceiling 8×/10× single-block
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor
