# Research Status — 29 Cycles (009-036)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 11 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Latest (cycle 036)
Item 53: 6× multi-block accumulation. k=4 at +9.59% PPL (6× avg compression) vs cycle 033's +8.84% (4×). Only +0.75pp extra for 50% more compression. INCONCLUSIVE (8-12% range). Block count dominates ratio for degradation.

## Key Compression Results (GPT-2 FFN, proxy)
- **Single-block 6× is safe** for all 6 easiest blocks (035, multi-seed)
- **4-block stacking at 4×**: +8.84% PPL (033, CONFIRMED)
- **4-block stacking at 6×**: +9.59% PPL (036, INCONCLUSIVE — marginal)
- **Per-block ratio barely matters**: 4×→6× costs only +0.75pp extra at k=4
- **Block count is the binding constraint**: k=4→k=5 jumps +8pp regardless of ratio
- Sub-baseline PPL at single-block 6× = channel selection regularizes (035)

## Eligible Next Items
P0 empty — need agenda refresh.
- **P1 Item 54**: Compression ceiling scan 8×/10× (depends on 52✓)
- **P1 Item 55**: Lstsq refit conditioning analysis (depends on 54)
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor
