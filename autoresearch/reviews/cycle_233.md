# Review: cycle 233 / item 243

**Verdict: APPROVE**

## Summary

Plan sweeps three alternative 6-block subsets at 1 seed (Phase 1) and reruns
any Phase-1 winner (best_ppl < 59.0) with 3 seeds (Phase 2). Tests whether the
59.5 PPL ceiling observed in cycle 221 for BLOCKS=[1,2,4,5,7,8] is a
selection-artifact or a structural 6-block ceiling.

## Empirical-rule check (3-seed phase)

Step 4 of the plan explicitly states:
> "after Phase 1, if any subset's `best_ppl < 59.0`, rerun that subset with
> `SEEDS_P2=[42, 137, 2024]` and aggregate mean/std (reuse seed 42 result)."

This satisfies the agenda-item requirement that a winner triggers a 3-seed
rerun before an empirical claim ("CONFIRMED") can be issued. The verdict
logic in Step 5 keys "CONFIRMED if any subset's (Phase 2 if run, else Phase 1)
best <= 58.0" — the `(Phase 2 if run)` clause guarantees that a CONFIRMED
verdict against the cycle-221 ceiling is backed by >=3 seeds, consistent with
the empirical rule. REFUTED at Phase 1 alone is acceptable: "all three in
[59.0, 62.0]" is a negative structural claim about the ceiling, not a new
empirical ordering, so a single seed triples (one per subset) is adequate
for disconfirmation.

## Methodological notes

- Excluding blocks 0 and 11 is well-justified by cycles 198/201 structural
  uniqueness findings; subsets (a)/(b)/(c) span contiguous-early, contiguous-
  middle, and spread geometries which is a sensible coverage of 6-block
  selection strategies.
- Cross-reference of cycle 221 [1,2,4,5,7,8] best_mean=59.515 as an implicit
  fourth data point in the summary (Validation bullet 4) is appropriate and
  avoids a redundant rerun.
- Best-state snapshot/restore with assertion `abs(final_ppl_restored -
  best_ppl) < 1e-3` matches the cycle-221 protocol.

## Minor concerns

1. **Phase-2 gating threshold.** The gate is `best_ppl < 59.0` but the
   CONFIRMED criterion is `<= 58.0`. A subset with best_ppl in [58.0, 59.0)
   would enter Phase 2 but could not yield CONFIRMED even if the 3-seed mean
   held. This is fine (it still produces a valid 3-seed data point, reported
   as INCONCLUSIVE), but worth noting explicitly in the summary so the
   intermediate region is not misread as a failure.

2. **Fallback for middle subset [3,4,5,6,7,8].** The plan records init_ppl
   only if the cycle 228 gelu_noneg pathology triggers. This means that
   subset may not produce a comparable "best_ppl" and will be excluded from
   the REFUTED condition ("all three in [59.0, 62.0]"). Recommend that the
   verdict logic treat a pathological subset as "non-comparable" and require
   the remaining two (plus cycle-221 reference) to all lie in [59.0, 62.0]
   for REFUTED — otherwise it is INCONCLUSIVE. This should be spelled out in
   the results summary if it occurs.

3. **Phase 1 single-seed variance.** 1 seed per subset is acceptable for
   gating Phase 2, but the summary should flag that Phase-1-only numbers
   carry unknown seed variance (cycle 221 reported std ~0.3 PPL at 3 seeds).
   A 0.3 std means a subset with Phase-1 best_ppl = 59.2 could have true mean
   either side of 59.0, which is the gating boundary. The budget tradeoff is
   reasonable given the sweep breadth, but the summary should acknowledge it.

## Artifacts / bookkeeping

- New experiment file `experiments/adam_6block_subset_sweep_cycle233.py` is
  clean (no prior existence).
- Results paths `autoresearch/results/cycle_233_adam_6block_subset_sweep.json`
  and `cycle_233_summary.md` are clean.
- Manifest and agenda update steps are present.

## Decision

APPROVE with the minor concerns above noted as reporting recommendations (no
plan changes required). The 3-seed rerun handling is correct and the
empirical rule will be satisfied for any CONFIRMED claim.
