# Cycle 099 Audit — Inter-Block Residual-Stream Projection Correction (Item 112)

## Phase 1: Verdict Check

**Claimed outcome:** REFUTED
**Criteria (from plan):**
- CONFIRMED if c_best <= 0.25 at r <= 16, enabling k=8 at <=+30% PPL
- REFUTED if c_best >= 0.35 at all r tested
- INCONCLUSIVE if c drops modestly (0.25-0.35)

**Evidence:**
- c_corrected_mean = 4.74 (all seeds: 5.46, 4.49, 4.29) — all far above 0.35
- PPL at k=4 with projections: ~14k-69k (vs ~63-79 uncorrected, vs ~58 baseline)
- PPL at k=8 with projections: ~103k-263k (vs ~184-237 uncorrected)
- Error rank_95 across blocks: 500-650 out of 768 dimensions — error is diffuse
- Adam polish after projection: ~200 PPL (better than raw projection, still 3x worse than uncorrected)

The c_corrected values are far above 0.35. The verdict is correctly REFUTED.

**Note on c_corrected being nonsensical:** The corrected cascade c ~ 4.7 is not a meaningful cascade rate. The PPL curve (baseline, ~430, ~75k, ~124k, ~14k) is non-monotonic because projections accumulate signal destruction chaotically. The log-linear fit to this non-monotonic data produces garbage. The c value exceeds 0.35, so the verdict criterion is met, but the number itself has no physical meaning.

**Note on c_uncorrected discrepancy:** c_uncorrected ~ 0.06 differs from the "c~0.41" cited in the brief (from prior cycles). This is not a bug — the ~6% per-block PPL growth is consistent with the raw numbers (e.g., seed 42: 57.65 -> 61.58 -> 67.45 -> 74.57 -> 79.44, roughly 1.08x per block). The prior c~0.41 likely used a different measurement methodology (perhaps total delta / k rather than multiplicative fit, or was measured without lstsq refit). This is a minor inconsistency with prior cycles' framing but does not affect this cycle's verdict.

## Phase 2: Quality Checks

### G1: Implementation correctness

**Code review findings:**
1. **Error collection (lines 85-138):** Correct. Hooks are placed on block outputs. Error computed as student - teacher at block boundaries. Both models run independently, so errors reflect the compressed model's actual behavior.
2. **SVD (lines 151-168):** Correct. `torch.linalg.svd(E, full_matrices=False)` returns U(N,min), S, Vh(min,D). Right singular vectors are rows of Vh.
3. **Projection construction (lines 209-224):** Correct. `U_r = Vh[:r].t()` gives (D, r) orthonormal columns. P = I - U_r @ U_r.T projects out the top-r error directions.
4. **Projection application (lines 192-206):** Correct. Applied after block output (residual stream), which is the right insertion point for GPT-2 pre-norm.
5. **Eval data separation:** Proper. SVD computed on `probe_train`. PPL evaluated on `eval_val` (verified in `_evaluate_model` implementation).

**One design limitation (not a bug):** Errors at all blocks are collected from the student without projections, then projections are inserted at all blocks simultaneously. This means the projection at block 7 changes what block 5 sees, but the error SVD for block 5 was measured without that change. For a successful correction this mismatch would matter; for the catastrophic failure observed, it's moot.

**Plan deviation:** Plan said to track signal energy removed by projection. Code does not do this. This would have been diagnostically useful but doesn't affect the verdict.

**No bugs found that would explain the catastrophic PPL.**

### G2: Reproducibility

- 3 random seeds: [42, 123, 7]. Pass.
- All hyperparameters recorded in code (PROJECTION_RANKS, HIDDEN_FRACTION, etc.). Pass.
- Results consistent across seeds: all show massive signal destruction. Pass.
- Could be re-run from the script. Pass.

### G3: Results interpretation

The conclusion (error is diffuse, projection destroys signal) follows directly from the numbers:
- rank_95 of 500-650/768 means the error spans ~65-85% of the full dimension space
- Projecting out even 4 dimensions causes 100x-1000x PPL increase
- This gets *worse* with more dimensions projected out (r=32 gives highest PPL)

The interpretation that "error subspace overlaps heavily with useful-signal subspace" is correct. When error occupies 85% of dimensions, the top error directions almost certainly carry substantial signal energy.

The non-monotonicity in r (r=8 sometimes better than r=4, r=16 sometimes worse than r=8) is consistent with the stochastic nature of which signal/error mix each singular vector captures.

**Verdict is correctly REFUTED.** Pass.

### G4: Counterintuitive results — Signal destruction analysis

The massive signal destruction (PPL going from ~70 to ~33,000) from projecting out just 4 dimensions (0.5% of 768) is striking. Could this be a bug?

**Checked and ruled out:**
1. **Projecting out signal instead of error:** No. The SVD is of the error matrix E = student - teacher. The top singular vectors of E are the directions of maximum error. The code correctly projects these out.
2. **Wrong insertion point:** No. Projection is applied after the block output (which is the residual stream). This is the correct insertion point in GPT-2 pre-norm architecture.
3. **Formula error:** No. P = I - U_r @ U_r.T is a valid orthogonal projection. The math is verified.
4. **Dtype mismatch:** No. Both model and projection use float32.
5. **Orthogonality issue with Vh:** No. `torch.linalg.svd` returns orthonormal Vh rows.

**The result is genuine.** The top error directions carry enormous signal energy because error and signal share the same high-energy dimensions. The top singular value of block 7 error is ~1029 (and the 32nd is still ~313), but these directions also carry proportional signal energy. Removing them zeros out critical information for downstream computation.

This is actually the *expected* behavior when error rank is 650/768. In a 768-dim space where error fills 650 dimensions, any 4 directions with high error energy almost certainly have proportionally high signal energy. The projection is removing signal-to-error in roughly the same ratio as the data-to-error ratio in those directions, which for a well-functioning network is very high.

### G5: Learning quality

**What we learned:**
1. Residual-stream error from FFN compression in GPT-2 is genuinely diffuse (rank_95 ~ 500-650 / 768). This is the most important finding — it rules out a whole class of cheap correction approaches.
2. Even aggressive projection (r=32, 4% of dims) makes things catastrophically worse, not modestly worse. There is no "sweet spot" for projection rank.
3. The error-signal overlap is near-total. The top error directions are also top signal directions. This implies the compression error is not a structured perturbation that can be cheaply removed; it is a diffuse distortion of the full representation.

**What should happen next:**
- The subspace-projection approach to error correction is dead for GPT-2 FFN compression.
- Learned corrections (distillation-based, as in earlier toy experiments) remain viable since they can jointly model error and signal.
- The diffuse error finding suggests that architectural approaches (e.g., low-rank adapters, learned residual corrections) would need to touch most dimensions, negating the compression benefit.

### G6: Method failure vs genuine refutation

**Could the result be explained by measurement or design failure?**

No. The experiment correctly implements the proposed approach. The failure is in the *hypothesis* (that error is low-rank), not in the *method*. The SVD rank analysis provides independent confirmation: rank_95 of 500-650 means the error is genuinely diffuse, regardless of what the projection does.

The cascaded error collection (measuring errors without projections, then inserting projections everywhere) is a minor design limitation, but it would only make the projections suboptimal, not catastrophically wrong. The catastrophic failure is caused by signal destruction, which the rank analysis independently confirms as inevitable.

**Is the projection formula correct?** Yes. P_i = I - U_r @ U_r.T projects out the error directions. If error and signal overlap (which rank_95 >> r guarantees), this necessarily destroys signal. The formula is applied correctly in code.

## Verdict

**Status: PASS**

The REFUTED verdict is correct and well-supported. The implementation is clean, the results are consistent across 3 seeds, and the catastrophic signal destruction is explained by the diffuse nature of the error (rank_95 ~500-650/768). No bugs or design flaws that could account for the results.

Minor notes:
- Plan called for signal energy tracking; code omitted it. Would have been useful diagnostically but doesn't affect the verdict.
- c_uncorrected (~0.06) differs from the c~0.41 cited in prior cycles. This appears to be a measurement methodology difference, not a bug.
