# Audit — Cycle 020

## Item
38 — Manifold curvature trajectory — GPT-2

## Phase 1: Verdict Check

Claimed: **refuted** — curvature has no clear trend (flat at 0.045-0.047) and anti-correlates with damage (ρ=-0.42, p=0.17).

Agenda criteria: "Refuted if: curvature has no clear trend or anti-correlates with compression difficulty."

Evidence: curvature range 0.045-0.047 across 13 layers (<5% variation). Spearman ρ = -0.42 (anti-correlation). Both refuted conditions met.

**Verdict is correct.**

## Phase 2: Quality Checks

### G1: Methodology
- k=50 nearest neighbors, local PCA, unexplained variance as curvature proxy — standard approach.
- LayerNorm explanation is insightful: normalization at each block boundary collapses variance structure, making all layers look equally curved. This is a fundamental limitation of the metric, not a bug.

### G2: Reproducibility
Deterministic. Acceptable.

### G3: Tautology Check
The flat curvature IS somewhat expected given LayerNorm — each layer's representations are normalized to similar scale. But this wasn't obvious a priori and the finding is useful: it rules out local PCA-based curvature as a metric for layernorm-ed transformers.

### G4: Scope
Within scope.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: G3 (flat curvature partly expected from LayerNorm — metric structurally blind to inter-layer differences)
reason: Curvature proxy is confounded by LayerNorm. No compression signal.
recommendations: Any curvature metric for transformers must account for LayerNorm normalization. Pre-norm vs post-norm curvature might differ. Move to remaining items (39, 40).
