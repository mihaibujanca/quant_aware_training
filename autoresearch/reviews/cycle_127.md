# Cycle 127 — Design Review: Item 144

**Reviewer**: design-review agent
**Verdict**: **REVISE**

---

## R1: Premise Validity — PASS

The hypothesis is well-formed: learned LoRA corrections at each block could align with (a) the raw compression error subspace, (b) the attention-amplified error subspace, or (c) bump-channel directions. These are geometrically distinct reference frames and the question — "what does the correction actually correct?" — is the right interpretability question for the correction line of work. The directional hypothesis (attention amplification > raw error) is grounded in the cycle 104 finding that cascade is attention-driven.

## R2: Measurement Soundness — REVISE

**Issue 1 (major): LayerNorm confound (MP7).** The plan hooks residual-stream activations to compute E_l = X_float - X_compressed. In GPT-2, the residual stream passes through LayerNorm before entering attention and FFN. If E_l is measured post-LayerNorm, the error directions are renormalized and may not reflect the true geometric structure. The plan must specify: hook the residual stream *before* LayerNorm (i.e., the raw skip-connection sum), and ensure the correction C = UV^T is compared against error in the same pre-LayerNorm space.

**Issue 2 (minor): Cosine threshold calibration.** The 0.5 threshold for "alignment" is stated but not justified. In d=768, the expected |cos| between two random unit vectors is ~sqrt(2/(pi*768)) ≈ 0.029. The null baseline (step 7) will calibrate this, which is good. But the *subspace* alignment (top-4 of an 8-d subspace vs top-4 of another 8-d subspace) has higher chance alignment than single-vector comparison. The null baseline must match the actual measurement protocol (max |cos(u_i, v_j)| over 4x4 pairs of random 8-d subspaces), not just random single vectors. The plan says "random rank-8 matrix" which is correct, but should explicitly state the null measures max-over-pairs, not mean.

**Issue 3 (minor): C = U @ V is not the effective correction.** The LoRA matrices U (768x8) and V (8x768) give C = U @ V, but the *effective* correction applied to the residual stream is C @ x (data-dependent). The SVD of C characterizes the linear map, but its top singular vectors may not align with the directions that actually carry correction energy on real data. Consider also computing a data-weighted version: SVD of C @ X^T (correction activations on holdout data) to see which directions are actually used. This is secondary but worth flagging.

## R3: Capacity and Statistical Adequacy — PASS

4 blocks x 8 singular vectors = 32 measurements per reference. The verdict criterion (cosine > 0.5 in >=3/4 blocks for top-4 vectors) is a clear counting rule. The plan acknowledges the fallback if alignment is low (<0.3 across the board → INCONCLUSIVE). Adequate.

## R4: Method-Architecture Alignment — PASS with note

The plan correctly identifies that cycle 104 and 110 data contain only scalar metrics and that direction vectors must be recomputed. This is honest and avoids a common trap of pretending prior artifacts contain more than they do. GPT-2 is GELU-based but the analysis is purely linear-algebraic (SVD of corrections and errors), so no GELU-specific issues arise.

## R5: Known Pitfall Scan

- **MP7 (post-LayerNorm)**: TRIGGERED — see R2 Issue 1. Plan does not specify hook placement relative to LayerNorm.
- **MP6 (verdict misclassification)**: Low risk. The INCONCLUSIVE fallback is well-defined.
- **MP1-5, MP8**: Not triggered.

---

## Required Revisions

1. **Specify hook placement relative to LayerNorm.** E_l must be computed in the same space as C. If corrections are applied to the residual stream after FFN output (pre-LayerNorm), then E_l must also be the residual-stream difference at that same point. Add explicit note: "hook at block output (post-FFN, pre-LayerNorm of next block)."

2. **Null baseline must use subspace-vs-subspace protocol.** The null should compute max |cos(u_i, v_j)| over 4x4 pairs from two random rank-8 subspaces in R^768, not single-vector alignment. State this explicitly in step 7.

## Suggested (non-blocking)

3. Consider adding a data-weighted SVD (SVD of C @ X_holdout^T) as a secondary measurement to check whether the correction directions that are geometrically prominent are also the ones that carry energy on real data.

---

**Verdict: REVISE** — fix LayerNorm hook placement specification and null baseline protocol, then re-submit.
