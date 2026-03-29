verdict: APPROVE — 0 blocking, 3 advisory
round: 1

## R1-R5 checklist

**R1 (meaningful question):** Pass. Natural follow-up to cycle 089 (per-block damage varies) and cycle 093 (c is ratio-independent). Asks whether non-uniform allocation exploits per-block damage variance. Confirmation threshold (10% relative PPL) is concrete and practically meaningful. Note: cycle 093's finding that c is ratio-independent under *uniform* allocation does not preempt this experiment — non-uniform allocation is a different intervention (redistributing budget across blocks, not changing global ratio).

**R2 (null baseline):** Pass. Uniform 6x is included as a built-in replication of cycle 089 k=4 (expected mean_ppl ~80.7, delta ~56%). Any deviation flags methodology before interpreting non-uniform results.

**R3 (3 seeds):** Marginal pass. At k=4, cycle 089 shows seed std ~13.3 PPL around mean 80.7. The 10% confirmation threshold translates to ~8 PPL improvement. With n=3, SE ~7.7 PPL — so a true 10% effect is ~1 SE, giving very low statistical power. This is acceptable for a directional probe (CONFIRMED vs REFUTED) but the INCONCLUSIVE zone (3-10%) will be noise-dominated. If the result lands in the inconclusive zone, 3 seeds cannot distinguish signal from noise. Not blocking because the plan correctly defines the zone; just note that "inconclusive" here means "underpowered," not "ambiguous signal."

**R4 (cost estimate):** Pass. 45 passes at ~8-12 min is reasonable. Atlas reuse across allocations within a seed is correct and efficient.

**R5 (known pitfalls):** Pass. MP3 not triggered (no fitting). MP5 not triggered (compression pipeline works at 6x from cycle 089). MP6: thresholds are concrete, no risk of uninformative metric. MP7/MP8 not applicable (PPL-based).

## Advisory

**A1 (damage ordering is inherited, not measured).** The allocation strategy gives more channels to "high-damage" blocks, using the ordering from cycles 005/007 (block 7 = least damage, block 1 = most damage). Cycle 089 did not measure individual block damage — it measured cumulative stacking in this fixed order. The allocation strategy's validity depends on this ordering being correct for the k=4 subset {7,5,4,1}. If block 1 is not actually the highest-damage block among these four, the aggressive allocation sends extra budget to the wrong block. Diagnostic: at k=1, measure each of the 4 blocks individually (4 single-block evals per seed, ~2 min extra) and report per-block delta_pct. If the rank order differs from the assumed [7<5<4<1], flag the result.

**A2 (polish convergence asymmetry across allocations).** In the aggressive allocation, block 1 has 768 hidden channels (W_down: 768x768 = 590K params) while block 7 has 308 channels (W_down: 308x768 = 237K params). Same 500 Adam steps for 2.5x different parameter counts creates a convergence asymmetry. This is a conservative bias (under-polished large blocks hurt the non-uniform condition). Log final polish loss per block per allocation for post-hoc convergence comparison.

**A3 (cycle 093 suggests structural futility).** Cycle 093 showed c is ratio-independent: improving per-block compression quality (3x vs 6x) had zero effect on stacking dynamics. Non-uniform allocation also improves per-block compression quality (for the high-damage blocks). If the bottleneck is cascade dynamics (as cycle 093 suggests), non-uniform allocation cannot help. The plan's null hypothesis P1 captures this. The experiment is still worth running because non-uniform allocation is qualitatively different from uniform-lower-ratio (it could preferentially reduce error at the cascade bottleneck). But the prior from cycle 093 is strongly against confirmation. Worth noting in the interpretation.
