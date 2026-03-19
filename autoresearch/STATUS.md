# Research Status — 33 Cycles (009-040)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 7 (010, 019, 026, 027, 033, 039, 040) |
| Refuted | 13 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Latest (cycle 040)
Item 54: Ceiling scan. ALL 3 blocks tolerate 10× compression (sub-baseline PPL). CONFIRMED. No ceiling found. Audit flags: underdetermined lstsq at 10× (samples < params), Shakespeare self-similarity may mask overfitting.

## Practical Compression Summary (GPT-2 FFN, proxy)
| Config | Blocks | Ratio | FFN reduction | PPL impact | Notes |
|--------|--------|-------|-------------:|------------|-------|
| Single-block ceiling | 1 | 10× | ~8.3% | ≈0% | No ceiling found (040) |
| **Best k=5 + correction** | 5 | 6× | ~34% | +12.53% | With lstsq correction (039) |
| **Best k=4 + correction** | 4 | 6× | 27.8% | +8.35% | With lstsq correction (039) |
| Best k=4 uncorrected | 4 | 6× | 27.8% | +9.77% | (037) |

## Key Methods (confirmed working)
1. **Topk channel selection** + centroid similarity (026-027)
2. **Lstsq output refit** — regularization effect at high compression (035, 040)
3. **Residual-stream affine correction** — 3.6pp recovery at k=5, zero cost (039)

## Open Questions
- Does 10× single-block survive non-Shakespeare eval? (audit concern)
- What's the multi-block ceiling with 10× per-block ratios?
- Does the method generalize to GPT-2-medium?

## Eligible Next Items
- **P1 Item 59**: GPT-2-medium generalization
- **P1 Item 55**: Lstsq refit conditioning analysis
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor
