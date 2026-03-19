# Research Status — 30 Cycles (009-037)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 5 (010, 019, 026, 027, 033) |
| Refuted | 12 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Latest (cycle 037)
Item 56: FFN param budget. k=4@6×: **27.8% total FFN param reduction at +9.77% PPL** (2 seeds). All k=5 configs exceed +15% — the block count wall is hard, reducing 5th block ratio to 2× doesn't help. REFUTED for k=5.

## Practical Compression Summary (GPT-2 FFN, proxy benchmark)
| Config | Blocks | Per-block ratio | FFN param reduction | PPL impact |
|--------|--------|----------------|--------------------:|------------|
| Best viable | 4 (7,5,4,1) | 6× | **27.8%** | +9.77% |
| Conservative | 4 (7,5,4,1) | 4× | ~18.5% | +8.84% |
| Aggressive (fails) | 5 blocks | mixed | ~34% | >+15% |

## Key Insights
- **Block count is the hard wall**, not per-block compression ratio
- 4× → 6× per-block costs only ~1pp extra; going 4→5 blocks costs >8pp
- Channel selection regularizes (sub-baseline PPL at high single-block compression)
- Topk + centroid similarity is the viable no-SGD method

## Eligible Next Items
- **P0 Item 57**: Block ordering optimization for 5-block — can ordering break the wall?
- **P1 Item 58**: Inter-block residual-stream correction — can lstsq correction break the wall?
- **P1 Item 59**: Generalization to GPT-2-medium
- **P1 Item 54**: Compression ceiling 8×/10×
- **P1 Item 50**: Centroid-merge quality vs probe size
