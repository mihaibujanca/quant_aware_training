# Audit — Cycle 029

## Item
17 — Local-chart analytic compression inside one FFN block

## Phase 1: Verdict Check

Claimed: **refuted** — local-chart fails to improve over global in all 3 blocks.

Criteria: "Refuted if: local-chart compression fails to improve over the global analytic baseline or is less stable."

Evidence: all 6 comparisons (3 blocks × 2 K values) show local-chart worse than global.

**Verdict correct.**

## Phase 2: Quality Checks

### G1: Methodology
- 3 representative blocks (easy, medium, hard) — appropriate sample.
- Matched budget (384 total hidden channels) — fair comparison.
- k-means routing on FFN inputs — reasonable chart assignment.
- lstsq refit within each chart — correct.

The failure mode is clear: splitting 384 channels across K charts gives each chart too few channels. This is a capacity issue, not a signal issue.

### G3: Tautology Check
**Flagged**: The failure is architecturally expected when matching total budget. If each chart had the SAME budget as global (384 channels per chart, total = K×384), it would be a fair test of whether local specialization helps. The current setup tests budget-matched MoE, which penalizes the local method by construction.

However: the agenda item specifically says "at equal budget," so the execution is correct per spec.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: G3 (budget-matched comparison penalizes per-chart allocation by construction)
reason: Local-chart compression at matched budget is worse because per-chart channel allocation is too thin. The test is fair per the agenda criteria. If total budget were multiplied by K, local-chart might outperform — but that's a different experiment (K× budget MoE).
