# Research Status — 27 Cycles (009-034)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 11 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032) |
| Inconclusive | 10 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034) |

## Latest (cycle 034)
Item 49: Block-specific compression ratio sweep. 6/6 blocks tolerate 6× within 5% PPL (29/30 data points). Non-monotonic PPL (sub-baseline at high compression). Audit downgraded to INCONCLUSIVE — single seed + 50 eval sequences. Directional finding is strong but needs multi-seed confirmation.

## Key Insights
- Data-driven channel selection (topk + centroid) is the viable no-SGD method (026-027)
- 4 blocks can be compressed at 4× with <10% PPL cost (033, CONFIRMED)
- Per-block compression likely extends to 6× (034, needs multi-seed confirmation)
- Structural pruning is empty for GELU (006, 028, 032)
- MASO/SplineCam theory inapplicable to GELU (025)

## Eligible Next Items
P0 queue empty — agenda refresh needed.
- **P1 Item 50**: Centroid-merge quality vs probe corpus size
- **P1 Item 51**: Per-block learned compressor
- **P1 Item 30**: Local complexity proxy (GPT-2)
- **P2 Moonshots**: Items 42, 43, 45

## Blocked
- Items 23, 35, 41, 44: depend on refuted item 22
- Items 27, 28: depend on ripser (not installed)

## Action Needed
Agenda refresh — P0 queue is empty. Next P0 items should address:
1. Multi-seed confirmation of cycle 034 (compression tolerance at 6×)
2. Multi-block stacking with per-block optimal ratios from cycle 034
3. Tighter compression (8×, 10×) to find actual ceiling
