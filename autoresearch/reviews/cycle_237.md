# Cycle 237 Design Review — item 247 (T3)

**Verdict: APPROVE**

## Summary

T3 ablates the bump subspace (top-10 right SVs of centred Δ̃ from cycle 235's `cycle_235_t1_features.pt`) on the residual stream after each of GPT-2's 12 blocks via `x_l ← x_l − x_l @ P_l`. Compares joint+per-block bump ablation to a 3-seed random-subspace baseline and a top-10 full-FFN-output SV baseline. Eval is full-model PPL on wikitext-2 val (N_SEQS=200, SEQ_LEN=512). Verdict thresholds copied verbatim from agenda item 247.

## Strengths

- Pre-flight P1–P4 are clean. Fresh `cycle_237_*` filenames; cycle 235 V's exist on disk.
- Hook removed-and-restored sanity check (re-eval baseline within 1e-4) is in Validation.
- Fallbacks anticipate the realistic walltime risk (drop per-block random, then drop full-FFN), MPS QR brittleness, and projector orientation flip.
- Random baseline uses QR of Gaussian → idempotency check `(P@P − P).abs().max() < 1e-4` is a good cheap correctness gate.
- Full-FFN-output top-10 SV baseline is the right comparator: tests whether the bump direction is structurally preferred over generic high-variance FFN output directions, not just generic random ones.
- Verdict logic matches the agenda item exactly (1.5× joint, 2.0× single-block on ≥3 blocks for CONFIRMED; 1.1× for REFUTED).

## Concerns

1. **HF GPT-2 block forward returns a tuple — flagged in the request.** The plan's hook factory branches on `isinstance(output, tuple)` and reconstructs `((new_hs,) + output[1:])`. This is correct: in HF transformers, `GPT2Block.forward` returns `(hidden_states, present)` when `use_cache=True` or `(hidden_states,)` / a tuple of length ≥1 always (since transformers ≥4.x). The branch handles both. The plan also asks to print `type(output)` on first call to verify — good. **Recommend additionally asserting `output[0].shape == hs.shape` after the rewrite to catch any silent broadcast.**

2. **`P_l = V_l @ V_l.T` is a projector ONLY if `V_l` has orthonormal columns.** The cycle 235 SVD should give orthonormal right singular vectors, so this should hold, but the plan should verify `(V_l.T @ V_l - I).abs().max() < 1e-5` once before building `P_bump_l`. Cheap. Add to Validation.

3. **Calibration mismatch risk for full-FFN baseline (step 11).** The full-FFN top-10 SV calibration uses wikitext-2 train 200×512 seed 42 to "match cycle 235 calibration". Confirm cycle 235 used the same split/size/seed for its Δ̃ SVD — otherwise the full-FFN comparator is biased toward whichever distribution had more capture coverage. Quick check of `cycle_235_t1_features.pt` metadata or the cycle 235 plan would be sufficient; not blocking.

4. **`x @ P` vs `P @ x` convention.** Plan writes `x_l − x_l @ P_l` and `P_l = V V^T` with `V: (768,10)`. For row-vector convention `(B, T, 768)`, `x @ P` projects each row onto col(P) = col(V), so subtraction removes the bump component. Correct. Sanity in fallbacks (`P @ V[:,0] ≈ V[:,0]`) covers the orientation slip. Fine.

5. **Walltime estimate.** 145 evals × ~30s/eval at SEQ_LEN=512 on MPS is plausible at ~70 min, but if the per-eval time slips to 40s the budget breaks. Fallbacks handle this. No change requested.

6. **Per-block monotonicity assumption (Validation).** "per_block_bump_ppl[l] < joint_bump_ppl for all l" is generally true but not guaranteed if any single block has a near-collapse mode that compounds nonlinearly. Treat as a warning, not a hard failure. Plan already calls it "expected", good.

## Minor

- Step 11's full-FFN hook captures `mlp` output, but in HF GPT-2 the residual addition pattern is `x = x + mlp(ln_2(x))`. The bump subspace is defined on Δ̃ which (per cycle 235 convention) is the FFN's output contribution to the residual — confirm cycle 235 measured `mlp(ln_2(x))` and not some other Δ. Match the two definitions or the baseline comparison loses meaning.
- Print verdict to stdout and also `joint_bump_ppl / random_joint_mean` ratio — useful for at-a-glance log inspection.

## Recommendation

Default APPROVE stands. Address concerns 1 (shape assert in hook), 2 (V orthonormality check), and the minor step-11 Δ̃ definition match before execute. Concerns 3 and 5 are bookkeeping/risk notes, not blockers.
