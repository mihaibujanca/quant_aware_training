# Research Status — 26 Cycles (009-033)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 11 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032) |
| Inconclusive | 9 (013, 015, 016, 017, 018, 022, 023, 024, 030) |

## Latest (cycle 033)
Item 48: Easiest-first multi-block accumulation CONFIRMED. k=4 at +8.84% PPL (threshold ≤10%). Sub-additive through k=4, super-additive at k=5 (+17.5%).

## Key Insights
- Data-driven channel selection (topk + centroid) is the viable no-SGD method (026-027)
- 4 blocks can be compressed at 4× with <10% PPL cost (033)
- Structural pruning is empty for GELU (006, 028, 032)
- MASO/SplineCam theory is inapplicable to GELU at useful granularity (025)
- Blocks don't share reusable compression structure (030)
- Grokking does not improve distillability (021, 031)

## Eligible Next Items
- **P0 Item 49**: Block-specific compression ratio sweep — natural follow-up to cycle 033
- **P1 Item 50**: Centroid-merge quality vs probe corpus size
- **P1 Item 51**: Per-block learned compressor
- **P1 Item 30**: Local complexity proxy as compression predictor (GPT-2)
- **P2 Moonshots**: Items 42, 43, 45 (no blocked deps)

## Blocked
- Items 23, 35, 41, 44: depend on refuted item 22
- Items 27, 28: depend on ripser (not installed)

## Dead Ends
- KAN architectures, tropical geometry, raw direction clustering
- Single-affine collapse for GPT-2, GELU→PWL approximation
- Shared cross-block compressors, grokking as compression catalyst
