# Cycle 238 Audit — Item 248 (T4: bump-only activation patching)

**Status:** PASS
**Old verdict:** REFUTED
**New verdict:** REFUTED (unchanged)

## Criterion vs. executed statistic

Literal agenda criterion: "mean bump-attribution > 0.15 in ≥3 of 5 blocks" → CONFIRMED; "< 0.05 in ALL 5" → REFUTED. The executor (and the experiment script's own docstring at lines 14-17) pre-declared a median substitution "if mean tails", implemented by an explicit `disagree` check at lines 363-368 of `experiments/bump_t4_patching_cycle238.py`. This is not post-hoc reframing.

Per-block mean: `[0.2121, 0.1057, 0.0348, 0.2132, 2.7904]` → 3/5 above 0.15 (literal → CONFIRMED).
Per-block median: `[0.0083, 0.0117, -0.0093, -0.0038, 0.0167]` → 0/5 above 0.15, 5/5 strictly below 0.05 (robust → REFUTED).
Block 9: mean 2.79 vs median 0.017, std 27.15 — mean is driven by a handful of huge-ratio outliers.

## Why median is the correct statistic here (G6 metric-choice)

`bump_attribution = bump_delta / full_delta`. The denominator `full_delta = full_Btop − orig_Btop` is **positive in only 42-64% of pairs per block** (n_full_positive `[42, 54, 53, 57, 64]` of 100). When the full-residual patch fails to push B's top-1 logit up — or pushes it down — the ratio flips sign, and when the denominator is small the ratio explodes. The unbounded-ratio tails that inflate the mean come directly from this unreliable denominator. The median is robust to this and is the honest summary of the per-pair distribution; the literal "mean > 0.15" criterion would be hijacked by 2-3 outlier pairs at block 9. Median substitution preserves the spirit of the criterion ("does the bump swap explain ≥15% of the per-pair prediction shift"); it does not.

## Phase 2 — Guardrails

- **G1 (reproducibility):** identity sanity `max |Δ logit| = 0.00e+00` on 5 random pair/block checks — patching hooks correct, baseline cache matches.
- **G2 (config):** held-out wikitext-2 val, seed 42, disjoint from cycle 235 calibration. N_SEQS=200, SEQ_LEN=128, N_PAIRS=100, blocks ∈ {1,3,5,7,9}, positions ∈ {32,64,96}, margin 2.0 (no fallback triggered).
- **G3 (statistics):** median substitution pre-declared in script docstring and gated on actual mean-vs-median disagreement. Correct call.
- **G6 (metric reliability):** `bump_attribution` ratio denominator is unreliable in 36-58% of pairs — flagged but does not invalidate the REFUTED verdict. The low `frac_positive ∈ [0.46, 0.64]` is itself a negative result: a token-specific signal would bias toward B_top in >50% of pairs, and it does not.

## Cross-item synthesis (245-248, all 4 bump P0 items)

- 245 CONFIRMED — Δ̃ has SVD surface features (mostly `repeated` label).
- 246 INCONCLUSIVE — directional interventions produce real logit shifts but `wte @ v` is the wrong probe (max Jaccard 0.333).
- 247 CONFIRMED — top-10 bump subspace ablation destroys PPL 23,143× worse than random (aggregate-critical).
- 248 REFUTED — per-pair activation patching does not transfer token-specific prediction signal.

Coherent picture: **the bump subspace is aggregate-compression-critical (T3) but not per-token-interpretable (T2, T4).** It encodes collective cascade behaviour whose removal ruins generation, but it does not carry decodable per-token messages between contexts. This is a substantial reframing of the earlier cycle 218 "inter-block communication" claim.

## Recommendation

Verdict stands as REFUTED. Bump investigation closes coherently. Natural follow-ups (none P0): logit-lens on downstream readers; token-window rather than single-position swap; resample pairs where B_top is outside A's top-20 so the full-patch denominator is reliably positive; per-direction (not full subspace) swap at blocks 9/11.
