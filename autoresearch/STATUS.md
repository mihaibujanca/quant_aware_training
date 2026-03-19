# Research Status — 37 Cycles (009-044)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 9 (010, 019, 026, 027, 033, 039, 040, 041, 042) |
| Refuted | 13 |
| Inconclusive | 14 (+ 044) |

## CRITICAL FINDING (Cycle 044)
**The method is highly corpus-specific.** Prior results (033-042) showing +6-9% PPL on Shakespeare are valid only for narrow-distribution text. On WikiText-2:
- Corpus-specific compression: **+31.86% PPL** (unacceptable)
- Zero-shot transfer from Shakespeare: **+120.57% PPL** (catastrophic)

This fundamentally recontextualizes all prior "good" results. The no-SGD geometric compression method works when the probe distribution is narrow and matched to eval, but fails on broad-distribution text even with corpus-specific fitting.

## Revised Practical Assessment
| Condition | PPL cost | Viable? |
|-----------|---------|---------|
| Narrow-distribution text (Shakespeare-like) | +6-9% | YES |
| Broad-distribution text (WikiText-2) | +32% | NO |
| Cross-domain zero-shot transfer | +120% | CATASTROPHIC |

## What Works (narrow-distribution only)
1. Topk channel selection + lstsq refit at up to 10× per block
2. 4-block stacking with single end-point correction
3. Method generalizes across model sizes (GPT-2-small → GPT-2-medium)

## What Doesn't Work
1. Domain transfer — Shakespeare-fit compression destroys WikiText quality
2. Broad-distribution compression — even corpus-specific WikiText is +32%
3. Multi-point corrections — cascading corrections interfere

## Remaining Items
- **P0 Item 63**: Practical recipe consolidation (should now include domain caveat)
- **P1 Items 50, 51, 55**: Secondary experiments
