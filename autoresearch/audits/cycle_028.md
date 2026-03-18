# Audit — Cycle 028

## Item
16 — Exact prune plus linear collapse baseline

## Phase 1: Verdict Check

Claimed: **refuted** — 0% dead channels, 0.008% near-linear, 1.000x compression. Need 1.1x.

Criteria: "Refuted if: exact or near-exact pruning yields negligible savings or immediately breaks block behavior."

**Verdict correct.** Zero savings from structural pruning. GELU channels are fully utilized.

## Phase 2: Quality Checks

### G1: Methodology
- Dead channel: max|activation| < 1e-5 — correct threshold for practical dead detection.
- Near-linear: |GELU(z) - z| / |z| < 0.05 for >95% tokens — reasonable soft threshold.
- lstsq refit after collapse — appropriate verification step.
- Result is consistent with cycle 006 (stable channels <1%).

### G3: Tautology Check
**Flagged**: The result is expected given cycle 006. This experiment adds implementation detail but the conclusion was already known. Still, the agenda required it as a mandatory baseline.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: G3 (result expected from cycle 006 — but agenda required this as baseline)
reason: GELU FFN channels in GPT-2 are fully utilized. Free structural compression is negligible. This closes the Serra/Berzins exact-pruning path for GELU transformers.
