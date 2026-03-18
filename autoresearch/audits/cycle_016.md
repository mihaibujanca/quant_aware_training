# Audit — Cycle 016

## Item
32 — GELU → 3-piece PWL approximation fidelity — toy GELU MLP

## Phase 1: Verdict Check

Claimed outcome: **inconclusive** — K=3/5 drop ~35pp (not confirmed), K=10 drops 2.6% (not refuted).

Criteria:
- Confirmed: 3-piece < 3%, 5-piece < 1%. NOT MET (both ~35pp drop).
- Refuted: 10-piece > 5%. NOT MET (10-piece = 2.59% drop).
- Inconclusive: recovery only with large piece count (>10). Matches — recovery at 10 pieces but not 3 or 5.

**Verdict is correct.** GELU is approximately piecewise-affine but at finer granularity than 3-5 pieces.

## Phase 2: Quality Checks

### G1: Methodology
- Float accuracy 86.75% with noise=0.5. The agenda doesn't specify noise, but prior experiments used noise=0.0 and achieved >95%. With noise=0.5, 86.75% may be near optimal. The PWL test is valid regardless of absolute accuracy level.
- 17% of activations outside [-3,3] — these are clamped to 0/x behavior in the PWL. This is a significant fraction and explains part of the K=3/5 failure (the PWL transition zone doesn't cover the full activation range).
- Both 2D and 100D models tested — good.

### G2: Reproducibility
Single seed. Acceptable — the effect is dramatic (35pp vs 2.6pp).

### G3: Tautology Check
Not tautological. The accuracy of PWL approximation depends on training dynamics and activation distribution.

### G4: Scope
Within scope. Directly tests whether ReLU theory can approximate GELU networks.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G1 (noise=0.5 differs from prior experiments' noise=0.0; doesn't invalidate result but worth noting)
reason: GELU needs ~10 pieces for faithful PWL approximation. At 3-5 pieces the approximation catastrophically fails. This means direct application of ReLU partition theory to GELU networks requires fine-grained PWL approximation, not coarse.
recommendations: Item 33 (GPT-2 PWL) remains eligible. Test with 10-piece PWL on GPT-2 — if it works, ReLU theory is applicable at moderate granularity. The 17% tail fraction suggests the [-3,3] window should be widened to [-5,5] for better coverage.
