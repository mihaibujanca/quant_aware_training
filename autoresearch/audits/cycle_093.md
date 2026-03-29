# Audit — Cycle 093

- **Item**: 108 — Lower compression ratio stacking — GPT-2 FFN
- **Cycle**: 093
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=c_3x=0.406, c_4x=0.404, c_6x=0.408 — all within 0.004 of each other. Refutation threshold was c_3x >= 0.30; clearly met. 6x sanity check passed (c=0.408 vs prior c~0.41). Implementation is clean: atlas probed once per seed, ratio only changes budget, polish and accumulation correctly separated, block order identical across ratios. The ratio-invariance of c is striking but internally consistent: single-block PPL improves at lower ratios (k=1: -5.5%, -4.5%, -3.0% for 3x/4x/6x) but the improvement washes out by k=4 (all ~+58%), confirming the cascade amplification factor is a network property, not a function of per-block error magnitude.
guardrail_flags=G4
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Test whether polish duration interacts with ratio (run 3x at 2000 steps — if c_3x drops, 500 steps underpolishes the larger W_down). Consider testing whether block ordering interacts with ratio (run 1 alt order at 3x).
interesting=true
interesting_reason=The stacking coefficient c is a property of the network topology, not the compression quality. This separates the stacking problem into two independent factors: per-block approximation quality (ratio-dependent) and inter-block error amplification (ratio-independent, c~0.41). The bottleneck for multi-block compression is entirely the cascade dynamics.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Compression ratios correctly map to hidden_fraction: 3x=1/3 (k=1024), 4x=1/4 (k=768), 6x=1/6 (k=512). Atlas probed once per seed (ratio-independent), budget varies only hidden_fraction. Polish is per-block per-ratio with polished weights injected into the stacked model. Block order [7,5,4,1,3,6,8,9] is identical across all ratios. The accumulation sweep correctly builds from atlas with the per-ratio budget. No issues found.

**G2 (Seeds)**: PASS. 3 seeds (42, 123, 7). Per-seed raw values stored in JSON for all 24 cells (3 ratios x 8 k-values). Standard deviations are consistent (e.g., k=8: std=185, 175, 155 for 3x/4x/6x — roughly proportional to means). Seed 42 consistently produces higher PPL (likely harder eval split), but this is consistent across ratios and cancels in the c fit.

**G3 (Single-block PPL vs stacking)**: The data directly addresses this. At k=1:
- 3x: 48.80 (-5.5%), 4x: 49.35 (-4.5%), 6x: 50.10 (-3.0%)

The per-block error is clearly ratio-dependent: 3x is better. But at k=4, the ratios converge: 81.8, 81.9, 80.7 (all within 1.5%). By k=8 they're indistinguishable: 525, 515, 524. The stacking mechanism overwhelms the per-block signal. This is consistent with c being ratio-independent: the cascade amplification factor (~41%/block) is identical, so whatever advantage 3x has at k=1 gets multiplied by the same factor and becomes negligible relative to the compounding total.

**G4 (Polish sufficiency)**: MINOR FLAG. Polish final losses show 6x converging to lower loss than 3x (block 7: 3x=4.07, 4x=4.10, 6x=3.97). Since 3x has a larger W_down (768x1024 vs 768x512), 500 Adam steps may underpolish 3x relative to 6x. However, this would push c_3x *higher* (worse), meaning the true c_3x could be even lower than 0.406. Since c_3x is already approximately equal to c_6x, underpolishing cannot explain the ratio-invariance — it would only strengthen it. Not a critical issue, but worth noting: a 2000-step run at 3x would confirm.

**G5 (Interpretation)**: The conclusion that the stacking coefficient is intrinsic to the network's inter-block coupling is well-supported. The multiplicative model PPL(k) = a*(1+c)^k fits well (R^2 > 0.92 for all ratios), and c is constant to within 0.004. The fitted intercepts (a) are also nearly identical (26.8, 27.1, 27.0), which makes sense: the multiplicative fit absorbs the single-block advantage into the curve shape, and the exponential dominates. This is a clean decomposition: per-block error affects the linear term, but the exponential rate is a structural constant.

**G6 (Block order confound)**: PASS. The fixed block order could in principle interact with ratio if certain blocks are harder to compress at certain ratios. But single-block deltas show the same rank ordering across ratios (block 4 is consistently one of the best, block 9 consistently worst). The order [7,5,4,1,3,6,8,9] was chosen from prior least-damage analysis and there is no evidence of ratio-dependent reordering. Testing one alternative order at 3x would be a useful robustness check but is not required for the current conclusion.
