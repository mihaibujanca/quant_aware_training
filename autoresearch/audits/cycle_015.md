# Audit — Cycle 015

## Item
26 — Intrinsic dimension trajectory through depth (GPT-2)

## Phase 1: Verdict Check

Claimed outcome: **inconclusive** — ID varies but trend is non-monotonic (actually monotonically increasing) with no correlation to compression difficulty.

Agenda criteria:
- Confirmed: ID decreases ≥30% over final third AND Spearman ≥ 0.4. NOT MET (ID increases).
- Refuted: ID approximately constant (< 10% variation). NOT MET (ID varies from 1.64 to 7.46 = 355% range).
- Inconclusive: ID varies but trend is non-monotonic with no correlation. PARTIALLY MET — ID varies, correlation is near zero (ρ = -0.16, p = 0.62). Trend is actually monotonically increasing, not non-monotonic.

The outcome doesn't perfectly match any criterion. The closest is inconclusive. The ID trajectory is the opposite of predicted (increasing, not decreasing) and doesn't correlate with compression difficulty. **Verdict of inconclusive is correct.**

## Phase 2: Quality Checks

### G1: Methodology
- TwoNN: ID = N / Σ log(μ_i) is the correct MLE formula.
- 13 checkpoints (embedding + 12 blocks) — correct.
- 512+ tokens used for ID estimation — sufficient for TwoNN.

### G2: Reproducibility
Deterministic given fixed GPT-2 weights and corpus. Acceptable.

### G3: Tautology Check
Not tautological. The increasing ID trajectory is a genuine empirical finding about GPT-2's representation geometry.

### G4: Scope
Within scope. The finding is informative: residual stream accumulates information (new dimensions added by each block) rather than compressing.

### G5: Counterintuitive
The increasing ID is mildly surprising if you expect networks to progressively simplify representations. But for a transformer with residual connections, each block adds to the stream — new information = new dimensions. This is consistent with the "residual stream as growing representation" view.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: none
reason: ID increases monotonically (opposite of hypothesis), no correlation with compression difficulty. Residual stream accumulates rather than compresses. Inconclusive is the right call — the experiment rules out ID as a compression predictor but the increasing trend is itself informative.
recommendations: The rising ID suggests that late-layer blocks are adding genuinely new functional dimensions, which explains why all-block compression fails. Item 36 (Procrustes) tests a related but different question: whether any block's transformation is near-identity. Item 31 (CKA) would directly measure representational similarity between layers.
