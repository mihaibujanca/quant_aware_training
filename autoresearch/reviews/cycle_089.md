verdict: APPROVE with 1 issue (non-blocking)
round: 1
issues: 1
notes: 2

## R1: Premise validity — PASS

The additive vs multiplicative vs super-multiplicative framing is sound. These are the three natural regimes for error composition in a residual-stream architecture: (a) additive means per-block errors land in orthogonal subspaces and don't interact, (b) multiplicative means errors compound through the residual stream at a constant rate per block, (c) super-multiplicative means cross-block error amplification (e.g., one block's compression error pushes activations into a region where a downstream block's approximation is worse). The hypothesis structure correctly maps these to actionable outcomes: sub-multiplicative = stacking viable, super-multiplicative = hard ceiling. The "inconclusive" regime change at k>4 is a realistic middle ground.

Prior work is correctly differentiated: cycle 036 (unpolished k=1..6), cycle 059 (polished but only step-count sweep at fixed k=4), cycle 061 (extended polish steps, still k=4), cycle 087 (4-block mixed corpus). This is genuinely the first polished accumulation curve to k=8.

## R2: Null baseline — PASS

The null baseline is well-specified. k=0 = float baseline PPL (~57.6 from cycle 059/061). The multiplicative null prediction PPL(8) = baseline * 1.01^8 = +8.3% is grounded in empirical single-block polish data. Both additive and multiplicative fits will be reported with R^2, which allows distinguishing the regimes. The plan correctly identifies that anything worse than +8% at k=8 signals super-multiplicative interaction.

## R3: Statistical/sample adequacy — PASS

3 seeds [42, 123, 7] — consistent with the established seed set from cycles 059/061/062. 200 eval sequences x 128 tokens = 25,600 tokens per measurement. Cycle 059 seed std was ~0.3 PPL, and the effects of interest (+5-8% PPL at high k) are well above this noise floor. 48 measurements total (8 k-values x 2 orderings x 3 seeds) is adequate for curve fitting.

Block ordering is justified: forward = [7,5,4,1,3,6,8,9] from cycle 005 damage ranking (least-damage-first). Reverse tests the other extreme. The plan acknowledges the discrepancy with cycle 007's full order (block 2 easier than block 8) and uses the brief's order as-is. This is acceptable — the ordering tests whether damage sequence matters, not whether the optimal order is selected.

## R4: Method correctness — ISSUE 1 (non-blocking)

**Selection mode mismatch.** The agenda item 104 specifies "random+lstsq+500 Adam" but the plan uses "topk" selection_mode throughout (Step 2, P6). The referenced source scripts (`gradient_polish_stacked_gpt2.py`, `multiblock_accumulation_6x.py`) both default to topk. All prior polished stacking experiments (cycles 059, 061, 062) used topk. The plan is internally consistent and matches prior methodology — the agenda item's "random" appears to be a stale reference to an earlier recipe variant.

**Resolution:** topk is the correct choice for consistency with the polish baseline data that the null prediction is calibrated against. If the script uses random, the cycle 059/061 calibration numbers are invalid and the null prediction is wrong. The plan should proceed with topk. Suggest updating the agenda item text to say "topk+lstsq+500 Adam" to prevent future confusion.

Per-block polish (not joint) is the right choice for this experiment. Joint polish (cycle 061: ~1pp better than independent at 500 steps) would conflate two effects: composition and joint optimization. Per-block polish isolates the composition question. The plan correctly calls this out.

Adam polish applied per-block on a fresh model copy (step 3) then stacked (step 4) is correct. This avoids the cascaded polish confound where later blocks are polished in the context of already-compressed earlier blocks. The measured curve reflects pure composition of independently-optimized blocks.

## R5: Known pitfall scan — PASS

- MP1 (threshold above trivial): N/A — no detection threshold, continuous PPL metric.
- MP2 (activation regime mismatch): N/A — measuring PPL directly, not a proxy.
- MP3 (underdetermined fitting): Fitting 2-parameter models (additive: a+bk, multiplicative: a*(1+c)^k) from 8 data points. Ratio is 4:1, adequate.
- MP4 (domain mismatch): Same domain (GPT-2 FFN on OWT) as all prior cycles. No transfer.
- MP5 (insufficient capacity): Not comparing methods, measuring degradation curve.
- MP6 (verdict misclassification): The three-way verdict (confirmed/refuted/inconclusive) is well-mapped to quantitative criteria. Sub-multiplicative, super-multiplicative, and regime-change are all empirically distinguishable from the curve shape. Low risk.
- MP7 (post-LayerNorm): Not measuring geometry, measuring PPL. N/A.
- MP8 (representation vs functional rank): Not measuring rank. N/A.

No pitfalls apply.

## Note 1 (non-blocking): Monotonicity validation may be too strict

The validation section requires "PPL must be monotonically non-decreasing with k." This is true in expectation but may fail at individual seeds due to noise or beneficial error cancellation (compressed block A's error partially cancels compressed block B's error in the residual stream). The plan's fallback for non-monotonicity is correct (check per-seed, flag as anomaly), but if it occurs at 2+ seeds it could be a real phenomenon worth characterizing rather than dismissing.

## Note 2 (non-blocking): Forward <= reverse ordering assumption

The validation assumes "forward ordering PPL(k) <= reverse ordering PPL(k) for all k." This holds if single-block damage is a good proxy for compositional damage, but cross-block interactions could violate this — a "high-damage" block at position 9 might produce errors orthogonal to the residual stream's task-relevant subspace, while a "low-damage" block at position 7 could produce errors aligned with it. The plan will reveal this empirically, so the assumption is testable rather than load-bearing. Good.
