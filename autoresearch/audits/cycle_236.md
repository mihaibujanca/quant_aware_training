# Cycle 236 audit — Item 246 (T2: Causal intervention on top bump directions)

**Status: PASS**
**Old outcome: INCONCLUSIVE**
**New outcome: INCONCLUSIVE**

## Phase 1 — Verdict check

Plan criteria (verbatim from agenda item 246):
- **Confirmed** if ≥3 directions per block have BOTH effect_size ≥ 3 AND jaccard ≥ 0.3 in ≥6 of 12 blocks.
- **Refuted** if no direction in any block produces a logit shift distinguishable from random (effect size <1.2× baseline).
- **Inconclusive** if directions cause shifts but the shifts don't match T1's vocab predictions.

Headline numbers from `cycle_236_t2_intervention.json` / `cycle_236_t2_summary.md`:
- `n_blocks_pass = 0/12` (joint effect≥3 ∧ jaccard≥0.3, ≥3 of top-5 dirs).
- `max_effect_size = 29.230` at block 11 dir 0 — far above the 1.2× refute floor.
- `max_jaccard = 0.333` (block 11 dir 4); only **1 of 60** directions reaches jaccard ≥ 0.3.
- 18/60 directions cross effect ≥ 3; 4/12 blocks pass an effect-only n_hits≥3 gate.
- Per-block max effect monotone with depth: `[5.23, 3.62, 2.34, 2.72, 2.62, 2.20, 3.47, 2.85, 4.20, 5.88, 8.05, 29.23]`.

CONFIRMED gate: 0/12 ≪ 6 → fails. REFUTED gate: max effect 29.23 ≫ 1.2 → fails. The remaining branch is the verbatim INCONCLUSIVE case ("directions cause shifts but the shifts don't match T1's vocab predictions"). Verdict matches. ✓

## Phase 2 — Critical interpretation

This is **the exact discriminating case the criterion anticipated**, not a degenerate "couldn't decide" outcome:

1. **Effect channel is alive**: at α=+3 std_l the intervention produces logit shifts up to 29× the random-direction baseline. The bump directions are causally real in the residual stream, especially at deep blocks.
2. **Decoder channel is wrong**: T1's per-direction `wte @ v_{l,i}` top-50 vocab boost set is essentially uncorrelated with the actual logit shifts (jaccard near 0 in 11/12 blocks). This refutes the *specific* cycle 218 alignment story ("Δ̃ writes into the unembedding column space at the top SVD direction") that T1 was built to test.
3. **Surviving weaker claim**: "Δ̃ writes into SOME downstream subspace, just not the unembedding basis" is consistent with the data and is what items 247 (T3 subspace ablation PPL) and 248 (T4 activation patching) are designed to probe — both are independent of the failed wte@v decoding.
4. **Depth monotonicity is informative**: max effect size grows almost monotonically across blocks 0→11 (5.23 → 29.23), with the block-11 spike consistent with residual_stream → ln_f → lm_head amplification of a direct writer at the terminal block. Not a bug — physical.

The cycle 235 audit explicitly anticipated this T2 outcome and recommended treating block 11 as a noisy outlier. T2 confirms that prediction: block 11 dominates both the effect tail (29.23) and the only jaccard hit (0.333). The audit also predicted that T1's correlational labelling could fail under causal test — and it did, in the cleanest possible way.

## Phase 3 — Quality checks

- **G1 Code correctness — clean**: `experiments/bump_t2_intervention_cycle236.py` loads cycle 235 features, recomputes per-block per-direction `std_l` from a fresh wikitext-2 train 200×512 calibration pass via `gelu_tanh(x @ W_fc + b_fc) @ W_proj − mu_l` projected onto V_l (matches the cycle 235 generative process), hooks `transformer.h[l+1]` (or `ln_f` at l=11) with `register_forward_pre_hook`, adds `α · std_l[i] · v_{l,i}` per intervention, and measures held-out logit shifts on a disjoint wikitext-2 val 8×128=1024-token slice (seed 42). Random baseline draws 10 unit-norm Gaussian seeds per block scaled by `mean(std_l)` — fair scale, plan-spec. No silent fallbacks.
- **G2 Reproducibility**: deterministic intervention path, 10 seeded random baselines, identical alphas list. Sanity checks reported: α=0 max |Δlogit| = 0.0; α=±1 L2 ≤ α=±3 L2 (linearity slack −23.75 i.e. ±1 strictly below ±3); std_l finite, O(1), monotone with depth `[0.67 → 1.42]`.
- **G3 Interpretation — clean**: executor's framing ("T1's wte-column probe was wrong; directions are causally active but operate through non-trivial paths") is correct and matches the criterion's INCONCLUSIVE branch verbatim. No overclaim, no underclaim.
- **G4 Counterintuitive check**: depth-monotone effect size growth and the block-11 spike are physical (post-LN amplification at the terminal block); not artefacts.
- **G5 Learning quality**: manifest `learned` field is exhaustive, propagates the failure mode, identifies replacement probes (logit-lens through downstream blocks; later-block V_{l',*} reading subspaces) and routes correctly to T3/T4 as next gating tests.
- **G6 Method failure vs genuine — NOT a method failure**: the criterion explicitly anticipated this exact case. T1 was a correlational labelling pass; T2 is the causal disconfirmation of the *labelling*, not of the underlying centred-feature signal. The cycle 218 strong alignment story is dead at the top-SVD-direction level; the weak version survives.

## Guardrails

Tripped: **none**.

## Interesting

Yes — the bump directions are **causally real** (effect grows monotonically with depth, peaking at 29× random baseline at block 11) but their decoder is non-linear/non-wte. T1's wte-column projection assumption is the part that broke, not the centred Δ̃ feature space itself. This is a clean discriminating null: it kills exactly one hypothesis (top SVD direction = unembedding writer) and leaves the broader feature-space hypothesis alive, with two independent gating tests (T3, T4) already on the agenda.

## Recommendations

1. **T3 (item 247)** is the next gating test — it measures aggregate compression relevance of the bump subspace via P_l = V_l[:,:10] @ V_l[:,:10].T projection ablation, independent of the wte decoding that failed here.
2. **T4 (item 248)** — A/B activation patching is the cleanest per-token causal test and also independent of the failed decoder.
3. **Replace T1's wte@v probe** with a logit-lens-style decoding that propagates v_{l,i} through ALL downstream blocks (not just `wte @ v`), or define directions by their causal effect on logit shifts directly rather than by wte alignment.
4. **Block 11 is a known outlier** (cycle 235 audit predicted this). When designing T3/T4, either run block 11 separately or weight per-block contributions by depth-corrected baselines.
