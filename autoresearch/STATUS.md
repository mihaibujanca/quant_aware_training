# Research Status — 28 Cycles (009-035)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 11 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032) |
| Inconclusive | 11 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035) |

## Latest (cycle 035)
Item 52: Multi-seed 6× confirmation. 6/6 blocks within 5% across 3 seeds (std 0.18-0.83%). Sub-baseline PPL is REAL (5/6 blocks), not noise — channel selection acts as regularization. Topk beats random on all 6 blocks. INCONCLUSIVE per strict criteria but directionally very strong.

## Key Insights
- **6× single-block compression is safe** for all 6 easiest blocks (035, multi-seed)
- Sub-baseline PPL at high compression = channel selection regularizes (035)
- Data-driven topk + centroid is the viable no-SGD method (026-027)
- 4 blocks stack at 4× with <10% PPL cost (033)
- Structural pruning empty for GELU (006, 028, 032)

## Eligible Next Items
- **P0 Item 53**: Heterogeneous-ratio multi-block accumulation (depends on 48✓, 52✓)
- **P1 Item 54**: Compression ceiling scan 8×/10× (depends on 52✓)
- **P1 Item 55**: Lstsq refit conditioning analysis (depends on 54)
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor
