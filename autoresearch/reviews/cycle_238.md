# Cycle 238 Review — Item 248 (T4): bump-only activation patching

**Verdict: APPROVE**

## Summary
T4 tests whether swapping only the centred Δ̃ bump at one position captures the (B−A) prediction shift, vs full residual swap, across blocks {1,3,5,7,9}. Standard activation-patching pattern on GPT-2.

## Strengths
- Held-out wikitext-2 **validation** split cleanly separated from cycle 235 train calibration — no leakage of `mean_delta`.
- Three forwards per pair (clean/bump/full) with the attribution ratio `(bump-orig)/(full-orig)` is the right isolation: controls for baseline shift and normalises by full swap magnitude.
- Centring subtraction `Δ̃ = Δ − μ_l` reused from cycle 235 features — consistent with T1/T3.
- Identity sanity (`v_bump = x_lA` → attribution ≈ 0) and hook-count restore checks are exactly the right guards.
- Cache pattern (5 block-level forward hooks to populate `x_l` for all 200 seqs × 3 positions) keeps compute linear. ~1500 short forwards fits the 2–4 min budget.
- Pair filter (top1_A ≠ top1_B, both margins > 2.0) avoids degenerate denominators.

## Concerns
1. **Hook target for patching**: plan says `forward_pre_hook` on `h[l+1]` writing `args[0][:,p,:] = v`. Correct — `h[l+1]` receives the post-block-l residual as `args[0]` (a tensor in GPT2Block.forward). Verify `isinstance(args[0], torch.Tensor)` on first call (plan already notes this in Fallbacks — good).
2. **`x_l` cache consistency**: `cached_x[l]` must be the *same* tensor that `h[l+1]` would see as input. For GPT2Block, `forward_hook` on `h[l]` captures `output[0]` which is `(hidden_states, ...)`; need `output[0]` (tensor), not `output` (tuple). Plan writes `output[0][:,p,:]` — correct, but double-check that `output` is the tuple `(hidden_states, presents, ...)`. Identity sanity at step Validation will catch any mismatch.
3. **`full_logit_delta ≈ 0` guard**: `|full_logit_delta| < 1e-6 → null`. Reasonable; record the null count per block and exclude from mean. Consider also flagging pairs where `full_logit_delta < 0` (full patch hurts B_top instead of helping) — these are pathological and shouldn't contribute to the mean. Plan tracks `frac_full_positive` which covers this.
4. **Attribution ratio tails**: ratio is unbounded when denominator is small-positive; outliers can dominate the mean. Report **median** alongside mean (plan already does) and **frac > 0.15** as the robust signal. Verdict thresholds on mean are OK given the N=100 pairs × margin filter, but if the median disagrees with mean, prefer median for the call.
5. **Block l=9 edge**: GPT-2 has 12 blocks (0–11), so `h[l+1] = h[10]` is valid. No issue, just noting.

## Verdict thresholds
- Confirm `> 0.15` in ≥3/5 blocks
- Refute `< 0.05` in all 5
- Else inconclusive

These match the agenda item verbatim. Reasonable for a "partial explanation" test (bump is one of several contributions to the residual).

## Actions before run
None blocking. Proceed.
