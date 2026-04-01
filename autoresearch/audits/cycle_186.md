# Audit: Cycle 186 — Item 195

**Status**: PASS-WITH-FLAGS
**Verdict accepted**: yes

## Flags

- **G4 (Cross-seed variance)**: Seed 42 exhibits higher baseline PPL (57.7) than seeds 137/7 (48.8, 50.4), driving overall variance. Its 4-block baseline (70.82) is 18–34 PPL higher than other seeds. This reflects legitimate heterogeneity in model initialization, not a bug. Mean-across-seeds verdict is correct, but the best-pair standard deviation (±5.26 PPL) represents ~10% uncertainty band around 50.44 PPL. This is acceptable given 3 seeds, but worth noting for threshold interpretation.

- **G5 (Confounders)**: The experiment conflates two factors: (i) reducing cascade depth and (ii) reducing compression scope (2 vs 4 blocks). The plan acknowledges this is exploratory, not mechanistic. The test correctly answers "does halving blocks help?" but cannot isolate whether the −9 PPL improvement comes from fewer cascades or smaller compression budget. Design is sound for the stated research question.

## Interesting observations

1. **Pair (1,4) is robust best across all seeds**: (1,4) wins in all 3 seeds (PPLs: 57.75, 45.61, 47.96). No other pair is consistently top-2 across all seeds. Suggests blocks 1 and 4 are better-suited to topk compression geometry than other combinations.

2. **Cascade improvement is real but insufficient**: 2-block mean PPL (50.44) beats 4-block (59.42) by 9 PPL, confirming the hypothesis that cascade depth is a cost driver. However, this improvement alone cannot overcome the threshold gap (50.44 > 48). The compression floor persists even with maximally reduced cascade.

3. **Seed heterogeneity is genuine**: Seed 42's high baseline PPL (57.7 vs float ~29–58 range) indicates that initialization luck matters significantly. This is not an experimental error; it reflects real training instability. All seeds follow the same trend (2-block < 4-block), so the verdict is robust.

## Recommendation

Verdict REFUTED is correctly applied. The best 2-block pair (1,4) achieves 50.44 mean PPL, which exceeds the refutation threshold of 48 PPL. The experimental design is sound, data collection is complete (3 seeds × 7 runs per seed), and aggregation logic matches the plan verbatim.

**Follow-up**: Cycle 187 should test Item 196 (the final P2 item). After that, trigger agenda refresh to prioritize new research directions that directly target the 48 PPL floor.
