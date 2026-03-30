# Cycle 115 Audit — Cascade Correction for Stacked FFN Compression (Item 118)

## Verdict: CONFIRMED — ACCEPTED (with flags)

## G1: Code correctness

- **Correction form is correct.** `LowRankCorrection` implements `x + (x @ V^T) @ U^T` which is `(I + UV)x` — additive low-rank perturbation of the residual stream. U initialized to zeros so identity at start.
- **Placement is correct.** Forward hooks are registered on `model.transformer.h[bid]`, catching the full block output tuple `(hidden_states, present, ...)` and correcting `hidden_states`. This operates on the residual stream after attention+FFN+skip, which is the right injection point for inter-block cascade correction.
- **Baseline is the same model.** Uncorrected PPL is measured on `compressed_model` (line 451). Corrections are trained on `copy.deepcopy(compressed_model)` (line 483). Same corpus, different eval split. No contamination.
- **Only U, V trainable.** `base_model.requires_grad_(False)` at line 310, `corrections.requires_grad_(True)` at line 311. Verified.
- **Train/eval data are disjoint.** Corrections train on `corpus.probe_holdout` (200 seqs), eval on `corpus.eval_val` (200 seqs). No leakage. Polish also uses `probe_holdout` but trains different parameters — acceptable.
- **`_eval_corrected` hooks correctly.** Hooks installed before eval, removed after. `_evaluate_model` called with `corrected_model.base_model` which has the hooks attached to its blocks.

## G2: Seeds

3 seeds [42, 123, 7]. Per-seed results are consistent:
- Seed 42: baseline 57.65, uncorrected 95.95, corrected(r=8) 69.53 — reduction 74.5%
- Seed 123: baseline 46.88, uncorrected 72.03, corrected(r=8) 52.48 — reduction 77.8%
- Seed 7: baseline 50.45, uncorrected 74.08, corrected(r=8) 55.31 — reduction 79.5%

All 3 seeds show >72% reduction. Consistent.

Note: the per-seed "baseline PPL" variation (46.88 to 57.65) comes from different random shuffles of evaluation sequences. This is eval-set noise, not model variance.

## G3: Sniff test — PPL consistency

- **Uncorrected PPL 80.68 matches cycle 093** (k=4 stacked at 6x: PPL=80.68). Exact match confirms the compression pipeline is the same.
- **Cycle 114** shows post-polish PPL=76.69, ~4 PPL lower. The difference: cycle 114 does end-to-end polish on the stacked model, while cycle 115 does per-block polish then stacks. Per-block polish leaves ~4 PPL on the table. This is expected — not a bug.
- **Corrected PPL 59.1 vs baseline 51.66**: residual degradation of 7.45 PPL (+14.4%). This is a substantial recovery from +56% degradation to +14%.

## G4: Rank saturation — why rank-4 works despite 85%-dim error

r=4 gives 72.6%, r=8 gives 74.3%, r=16 gives 74.1%. Saturation between r=4 and r=8, with r=16 marginally worse (overfitting).

This is NOT contradictory with cycle 099's finding that error spans 85% of dims. The correction is not projecting out the error — it's a learned linear transform `(I + UV)` that steers the residual stream to compensate for downstream effects. The CE loss selects the few directions that matter most for next-token prediction, not for reconstructing the activation. The task-relevant error subspace can be much lower rank than the geometric error subspace.

## G5: What we learned

1. Per-block low-rank corrections at r=4-8 recover ~74% of stacking degradation with only 25K-49K params (0.04% of model).
2. The stacking catastrophe (c~0.41/block compounding) is largely addressable by cheap linear correction — it's not a fundamental nonlinear problem.
3. Correction capacity saturates at r=8 (74.3%). The remaining ~26% of degradation likely requires nonlinear correction or joint re-optimization.

## G6: Is this just LoRA distillation?

**Yes, structurally.** `x + U V x` is exactly LoRA applied to the identity (residual) connection after each compressed block. Training with CE loss on real data is implicit distillation (no explicit teacher logits, but the data provides the supervisory signal).

Key differences from standard LoRA:
- Only 4 corrections (compressed blocks), not all 12 blocks × 4 attention matrices
- Param budget: 49K at r=8, vs ~590K for standard LoRA r=8 on all attention (12x fewer)
- Applied to block output (residual stream), not inside attention

**This is a valid finding regardless.** The question the experiment answers is: "can cheap post-hoc correction recover stacking degradation?" The answer is yes, and the mechanism (LoRA-like adaptation) is the right one. The 0.04% param overhead is genuinely small.

However, a fair comparison would be: what happens if you spend the same 49K param budget on additional polish of W_down (which has 393K params per block)? If extended polish recovers similar PPL, then the correction adds no value over more training. This was not tested.

## Flags

1. **FLAG: LoRA-equivalent** — the correction is structurally identical to LoRA on the residual stream. Should be characterized as such in any writeup. This is not a geometric correction; it's a learned linear adaptation.
2. **FLAG: Missing ablation** — no comparison to extended polish (e.g., 2500 total polish steps instead of 500+2000 correction steps). This would disambiguate "corrections are useful" from "more training on compressed model is useful."
3. **FLAG: Per-block vs end-to-end polish gap** — the uncorrected baseline (80.68) uses per-block polish, while cycle 114 gets 76.69 with end-to-end polish. The correction might be partially compensating for this 4 PPL gap rather than addressing fundamental stacking error. A fairer experiment would apply corrections on top of the end-to-end polished model.
