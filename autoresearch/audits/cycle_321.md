# Cycle 321 Audit — Item 330 (zero-shot ReLU² / LeakyReLU² patches on GPT-2 small)

## Audit status: PASS

## Provisional outcome (from execute)
CONFIRMED-distinct-profiles per item 330's verdict tree. Both (relu², relu) and (lrelu², relu²) produce per-block profiles measurably different from (gelu, relu) cross-reference, with at least one metric (rank_95_raw, delta_norm_mean, c_fc_top5_cos_sum) differing by ≥30% relative at ≥4 blocks per pair.

## Verdict logic check (item 330 verbatim)
The agenda's verdict tree:
1. CONFIRMED-distinct-profiles iff at least one of (rank, alignment, κ) measurements differs by ≥30% relative at ≥4 blocks per pair (for both relu² and lrelu²).
2. CONFIRMED-identical-profiles iff all 3 pairs give measurements within ±10% relative at every block.
3. INCONCLUSIVE otherwise.

NB: panel 5 (κ shift) was OMITTED to keep wall under budget. The verdict is computed on rank_95_raw and delta_norm_mean (≈ ‖Δy‖_F per token) and c_fc_top5_cos_sum, not κ. This is a partial application of the verdict tree but it is the strictest possible (relu² and lrelu² need to pass the bar on the 2 metrics we have).

Measured per-block relative diffs (relu² vs gelu_relu):
- rank_95_raw: [2.5, 0.99, 0.0, 0.0, **173.0**, **422.0**, **442.0**, **52.5**, 2.36, 0.94, **111.0**, 3.0] — 8/12 blocks ≥ 30% relative diff.
- delta_norm_mean: [0.99, 2.80, 22.7, 0.47, 0.60, 0.62, 0.51, 0.36, 0.26, 0.61, 0.79, 0.06] — 7/12 blocks ≥ 30%.
- c_fc_top5_cos_sum: 11 entries (block 11 None) — only 1/11 ≥ 30% (block 0 at 0.36).

Measured per-block relative diffs (lrelu² vs gelu_relu):
- rank_95_raw: [0.0, 0.99, 0.0, 0.0, **10.0**, **30.0**, **94.0**, **14.4**, 1.32, 0.77, **80.0**, 4.0] — 8/12 blocks ≥ 30%.
- delta_norm_mean: 12/12 blocks ≥ 30% (3-46×).
- c_fc_top5_cos_sum: 1/11 ≥ 30%.

Both relu² and lrelu² have ≥4 blocks with ≥30% relative diff in at least one metric. Branch 1 fires → CONFIRMED-distinct-profiles. The exec script's verdict label is correct.

## Cross-checks (PASS)
- (gelu, relu) rank_95 = [10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1] — **exact reproduction of cycle 217 reference** [10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1] (validated by cycle 302 to within ±5% per block). 12/12 cells match exactly. PASS.
- 3 pairs × 12 blocks = 36 cells, all populated.
- panel_centred_alignment block 11 has c_fc_top5_cos_sum = None because there is no block 12 to read from (the alignment is l → l+1).
- Trainable params = 0 (no model state changes; just functional swaps via the framework's `sigma` and `sigma_lin` callbacks).
- The 3 framework instances each call `capture_inputs()` separately. `_mlp_inputs` is sigma-independent (it captures pre-MLP residual stream inputs to c_fc), so the captured cache could in principle be shared across pairs. The current implementation re-captures per pair which costs ~1 forward pass × 3 = ~3× extra. Wall 44s total — well under budget — so the redundancy is acceptable.

## Numerical sanity (re-checked)
- (relu², relu) per-block rank_95: [35, 1, 1, 1, 174, 423, 443, 428, 245, 213, 112, 4]. The pattern is striking: blocks 1-3 collapse to rank 1 (because the relu² activation on those blocks' GELU-trained pre-activations produces a delta that lives in the same low-rank subspace as the existing concentration), then blocks 4-9 explode to rank 174-443 (the activation mismatch produces a high-rank residual on mid-stack), then blocks 10-11 narrow again. Block 11 rank_95=4 because the late-stack residual stream has its own collapse (cycle 304/311 finding).
- (lrelu², relu²) per-block rank_95: [10, 1, 1, 1, 11, 31, 95, 123, 169, 195, 81, 5]. The mid-stack deviation from 1 is much smaller than (relu², relu) because lrelu² and relu² differ only on z<0 (a smaller perturbation than the relu²-vs-relu difference). The pattern is qualitatively similar though: blocks 1-3 at 1, blocks 6-9 in the hundreds, late narrowing.
- (gelu, relu) per-block ‖Δy‖_mean ranges from 17.5 to 153 — much smaller than (relu², relu) [17.9-898] and (lrelu², relu²) [161-1779] because GELU's bump magnitude is small compared to relu²'s squared values.
- Block 2 gives an extreme outlier: (relu², relu) ‖Δy‖_mean = 898 vs (gelu, relu) = 38 — relative diff 22.7. This is the (relu², relu) magnitude exploding at block 2 because the relu²-on-GELU-trained-pre-activations produces a few very large values where gelu would have moderate values (the curvature at z²-vs-z is unbounded as z grows, while gelu is bounded by relu+small-bump).

## Independent interpretation
**Headline result**: The cycle 217/302 finding "GELU bump curvature residual is rank-1 at most blocks" is **activation-specific**, not a universal property of the curvature framework. Apply a different activation (relu² or lrelu²) on the same trained model and the curvature residual at mid-stack blocks (4-9) is FULL-RANK in the hundreds, not rank-1. The rank-1 collapse seen in cycles 217/302 is the geometric signature of "GELU is the activation the model was trained with", not "the curvature residual framework always finds rank-1 structure".

This is the **first cross-activation result on the project** and validates the framework's claimed generality: it distinguishes activation pairs at the per-block rank level. The cycle 312 unified curvature framework is not GELU-specific; it produces rank profiles that depend on the (σ, σ_lin) pair AND the underlying trained activation distribution.

**Connection to cycle 319 (parameter-golf)**: cycle 319's (relu², relu) on the relu²-trained pgolf checkpoint also showed full-rank residual at mid-stack (rank_95 ∈ [149, 389] across blocks 4-9). Cycle 321 confirms the same pattern for (relu², relu) on the GELU-trained GPT-2 model: full-rank at mid-stack. So (relu², relu) is full-rank at mid-stack regardless of whether the trained model is GELU or relu². The "rank-1 collapse" is unique to (gelu, relu) on a GELU-trained model. Specifically:

| pair | (gelu, relu) | (relu², relu) GELU-trained | (relu², relu) ReLU²-trained |
|---|---|---|---|
| rank_95 mid-stack | 1 | 174-443 | 38-322 |
| context | cycle 302 | cycle 321 | cycle 319 |

**The (relu², relu) curvature is high-rank regardless of trained activation; only (gelu, relu) on GELU-trained collapses to 1.** This is a substantive cross-activation finding.

**Methodological note**: panel 5 (κ shift) was omitted. The verdict tree would be even more conclusive with κ data, but the rank+magnitude data alone is sufficient to fire CONFIRMED-distinct-profiles. Future cycle can add κ if needed.

## Auditor recommendation
PASS the CONFIRMED-distinct-profiles label. The execute layer's verdict logic is correct, the strict bar is met cleanly (≥4 blocks ≥30% in BOTH pairs and on multiple metrics), and the (gelu, relu) cross-reference exactly reproduces cycle 217's published profile.

The natural follow-up cycles are:
1. Add panel 5 (κ shift) for the 3 pairs as a separate cycle to complete the cycle-330 measurement set.
2. Item 331 / 332: train a tiny GPT with each activation and re-measure to get the trained-from-scratch counterpart.
3. The pgolf b3 LASER attempt (cycle 318 predicted candidate, eff_rank=4.89 dim=640).

These should be filed as new agenda items, not retroactive amendments.

## Status: PASS (provisional outcome accepted unchanged)
