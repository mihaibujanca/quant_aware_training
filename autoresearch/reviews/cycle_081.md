verdict: APPROVE
round: 2
issues: none

feedback:

All four round 1 issues have been addressed in the revision:

1. **GN vs true Hessian (R1-Issue 1):** Resolved. Step 3c now implements Gauss-Newton explicitly via jvp/vjp on logits. True Hessian + Tikhonov is cleanly separated as fallback-only in step 6.

2. **Block order sensitivity (R1-Issue 2):** Resolved. Step 5 adds a second-pass staleness check (re-evaluate block 7 gradient after all blocks corrected). Step 3g logs per-block PPL after each correction. The plan acknowledges order is load-bearing without requiring a full permutation ablation.

3. **Inconclusive range mismatch (R1-Issue 3):** Resolved. Hypothesis thresholds are now internally consistent: confirmed <= +8% (within 2x of Adam's +4.9%), inconclusive 15-20%, refuted > +20%.

4. **OOM guard (R1-Issue 4):** Resolved. Step 3d wraps Hv in try/except with batch_size halving. Step 3c calls empty_cache() between blocks. CPU fallback documented at batch_size=1.

### Standard checks (round 2)

- **R1 (premise):** Sound. Gauss-Newton via CG is well-established; applying it block-by-block to polish compressed FFNs with full-model Jacobian is a well-motivated extension beyond per-block lstsq.
- **R2 (measurement):** PPL on held-out eval is correct. Three-way comparison (unpolished/Newton/Adam) is clean. Sanity checks on unpolished (~+30%) and Adam (~+4.9%) baselines will catch implementation errors.
- **R3 (capacity/stats):** 3 seeds, 25600 eval tokens, large effect sizes (5-25pp). Adequate.
- **R4 (method-architecture):** GELU is smooth (Jacobian well-defined). GN on logits is correct for cross-entropy. CG on PSD system is textbook.
- **R5 (pitfall scan):** MP1-MP8 reviewed. None applicable. Closest: MP3 (underdetermined fitting) does not apply since Newton step is optimization, not fitting. MP7 (post-LN) does not apply since we optimize weights through the full model.

### Minor note (non-blocking)

CG max_iter=50 on a ~98K parameter system is coarse, but the plan only seeks an approximate correction, checks convergence, and has fallbacks. Acceptable for a moonshot.
