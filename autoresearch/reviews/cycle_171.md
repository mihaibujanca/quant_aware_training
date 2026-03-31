# Cycle 171 Review — Item 181

## R1 — Does the plan answer the item?
YES. Tests 3 wider-shallower configs against 4@6x baseline. Verdict criteria match item.

## R2 — Confounds addressed?
YES. Polish config held constant. Block selection for 8-block config is noted as arbitrary.
Total params differ across configs (intentionally: testing the principle, not param-matching).

## R3 — Compute budget reasonable?
YES. 12-block probe + 4 configs × 3 seeds × polish = ~15 min on MPS. Within budget.

## R4 — Overlaps with prior work?
NO direct overlap. Cycle 094 (non-uniform, same blocks) is different axis. Cycle 093
(stacking rate c=0.41) provides the null hypothesis: 12 small errors compound worse.

## R5 — Pitfalls
- If baseline PPL differs from expected ~70.7 by >5 points, atlas or eval corpus may differ
  from prior runs. Use same OWT path and seed protocol.
- Polish may need more steps for 12-block models (more params). Watch for underfitting.
- The c=0.41 compounding rate suggests 12@3x will be WORSE: even light compression at 12
  blocks compounds to 0.59^12 = 0.002 of original signal. This experiment may refute quickly.
