# Cycle 236 / item 246 (T2) — Design Review

**Verdict:** APPROVE

## Summary

T2 performs causal intervention on the top-5 bump directions per GPT-2 block (from
cycle 235's T1), injecting `alpha * std_l_i * v_l_i` into the residual stream entering
the next block for alpha in {-3, -1, 0, +1, +3}, and measures logit shifts against a
random-direction baseline plus Jaccard overlap vs T1's predicted boost list.

## Strengths

- Cleanly reuses cycle 235 artifacts (V_l, S_l, mean_delta_l, per-block coherent counts).
- std_l recomputed from identical calibration loader (same seed, N_SEQS, SEQ_LEN) —
  fair comparison with T1.
- Held-out batch is explicitly DISJOINT from calibration (wikitext-2 val, not train).
- Sanity checks are appropriate: alpha=0 no-op assertion, linearity (l2_alpha1 ≤
  l2_alpha3), std_l finiteness/magnitude, coherent-count cross-check.
- Verdict bands (confirmed / refuted / inconclusive) spelled out with thresholds from
  the item 246 hypothesis verbatim.
- Fallbacks concrete (reduce held-out tokens, drop top1000 indices, std_l=1.0 flag).

## Concerns

1. **Hook target for last block.** Step 5/6 says `forward_pre_hook on transformer.h[l+1]
   for l=0..10 and on transformer.ln_f for l=11`. This is correct: `h[12]` does not
   exist in GPT-2 small (12 blocks, indices 0..11). `ln_f` receives the residual stream
   after block 11 — the correct injection point for "after block 11". Already handled.

2. **Hook timing semantics.** `forward_pre_hook` on `h[l+1]` fires with
   `args[0] = residual stream BEFORE block l+1's LN_1`, i.e. the output of block l's
   residual add. This IS the agenda-specified "after block l" injection point.
   Equivalent for `ln_f` (residual stream after block 11, before final LN). Correct.

3. **Random baseline scale.** Step 6 random baseline says `scale v_rand by mean(std_l)
   to match scale`. Make sure this is applied: `delta_x = alpha * mean(std_l) *
   v_rand_unit` where `v_rand_unit` is unit-norm Gaussian. Without the `mean(std_l)`
   rescaling the baseline would be ~1/std smaller and effect_size would be inflated.
   Plan says it explicitly — keep a one-line assert in the code path.

4. **Effect size normalization.** Step 7 divides real `l2.mean()` by
   `random_l2_mean_alpha_3[l]`. With the rescaling above this is comparable. Without
   it, effect sizes would be 5-10x inflated and the 3× threshold trivially passed.
   Confirm at review time (code inspection) the rescale is applied.

5. **Jaccard computation.** T1's `top_vocab_boost` in decoded.json stores token_id
   lists (plan assumes "first 20 token_ids"). Verify field name/type when loading —
   if it's a list of dicts `{token_id, token_str, score}` extract `token_id` only.

6. **Memory bound on logit_shifts.pt.** Plan estimates 60 MB with only `mean_shift`
   (V,). With `top1000_indices/top1000_shifts` added, ~60 MB + 12×5×5×1000×8B ≈
   +2.4 MB — negligible. Keep the top1000 lists; useful for later analysis.

7. **alpha=0 skip.** Plan correctly skips hook registration at alpha=0 and uses
   `logits_clean` directly. Good — avoids a "no-op hook" that still moves fp32 bits.

## Minor

- Step 6 "Example records" uses last-50-char context — fine, but note this is
  character-level truncation of decoded text, not token-level. Non-issue for
  summary.md rendering.
- `std_l_i` stored per entry in intervention.json — good for reproducibility.
- Consider logging `n_hits[l]` and `max_effect_size[l]` in summary.md even for
  INCONCLUSIVE runs so the next-cycle agenda has signal.

## Approval

Default APPROVE. Concerns 3-5 are code-review items, not plan defects. Plan is
self-contained, deps satisfied, verdict bands concrete, fallbacks specific.
