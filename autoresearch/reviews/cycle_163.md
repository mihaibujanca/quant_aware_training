# Cycle 163 Review — Per-block stacking profile + correction concentration (Item 173)

Item 173. Reviewer: design-review agent. Round 2.

## Round 1 recap
One blocking issue: plan only profiled compression stacking without testing
correction concentration, diverging from the agenda's confirmation criterion.

## Round 2 — blocking issue resolved

The revised plan adds Phase 2 with two budget-matched correction concentration
conditions (A: same rank fewer blocks; B: same budget higher rank) compared
against the uniform LoRA r=8 baseline on [1,4,7,11]. Verdict logic now requires
both Pareto structure AND correction concentration beating uniform by >= 1 PPL
for confirmation. This directly matches the agenda criterion.

## R1: Premise validity — PASS (unchanged)

## R2: Measurement soundness — PASS
Phase 1 marginals well-defined. Phase 2 comparisons are budget-matched
(condition A at ~37K vs uniform ~49K is actually budget-advantaged for uniform,
making a top-3 win more convincing; condition B at ~49K vs ~49K is exact match).

## R3: Capacity and statistical adequacy — PASS
3 seeds, mean +/- std, both phases. Per-seed top-3 ranking avoids averaging
away seed-specific variation.

## R4: Method-architecture alignment — PASS
Same recipe (topk+lstsq+LoRA) for both concentrated and uniform conditions.
Fair comparison.

## R5: Known pitfall scan — no triggers
MP1-MP8: none applicable to this PPL-profiling + correction-comparison design.

## Summary
Blocking issue from round 1 fully addressed. Plan is well-designed with
clean two-phase structure, budget-matched comparisons, and unambiguous
verdict logic.

**Verdict: APPROVE**
