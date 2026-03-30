# Cycle 127 — Design Review Round 2: Item 144

**Reviewer**: design-review agent
**Verdict**: **APPROVE**

---

## R1 Issue Check

### Issue 1: LayerNorm confound (MP7) — RESOLVED

Step 3 now explicitly specifies: "Hook placement: post-FFN, pre-LayerNorm of next block (raw residual-stream sum after `x = x + ffn(ln2(x))`)." The plan further notes this ensures E_l lives in the same pre-LayerNorm residual space as C = UV^T. This is the correct hook point and eliminates the MP7 confound.

### Issue 2: Null baseline protocol — RESOLVED

Step 7 now specifies: draw two random rank-8 matrices in R^768, SVD each to top-4, compute max |cos(u_i, v_j)| over the 4x4 pairs, repeat 1000 times, report mean and 99th percentile of max-over-pairs. This exactly mirrors the measurement protocol in step 6, which was the requirement.

### R1 Suggestion (data-weighted SVD) — ADOPTED

Added as step 8 (secondary, non-blocking). Good.

## Additional Checks

- **MP8 (representation vs functional rank)**: Not triggered. The plan SVDs the correction matrix C and error matrix E_l directly (function properties), not representation clouds.
- **MP3 (underdetermined fitting)**: Not triggered. No fitting occurs in this analysis cycle; P1 re-training uses the established cycle 125 protocol.
- **Validation gate**: PPL check (within 1 of ~69.5) guards against analyzing a misfit correction. Adequate.
- **Hypothesis specificity**: "cosine > 0.5 for top-4 SVs in >= 3/4 blocks" is a clear counting criterion with a calibrated null. The 0.5 threshold will be interpretable relative to the null 99th percentile (~0.1-0.15 expected for 4-d subspaces in 768-d).
- **Fallback**: INCONCLUSIVE path defined for all-low alignment. No risk of MP6 (false refutation).

No new issues identified.

---

**Verdict: APPROVE** — both required revisions addressed correctly. Proceed to implementation.
