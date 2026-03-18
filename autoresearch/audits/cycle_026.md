# Audit — Cycle 026

## Item
14 — Rank-only baseline versus geometry versus distillation

## Phase 1: Verdict Check

Claimed: **confirmed** — geometric method beats rank-only baseline in 11/12 (92%) block-budget pairs.

Criteria: "Confirmed if: the geometric method beats the rank-only baseline on full-model metrics in more than 50% of tested block-budget pairs at equal FFN parameter count."

Evidence: 11/12 = 92% >> 50%. Geometric wins by +0.1-3.3pp ppl across most comparisons.

**Verdict correct.**

## Phase 2: Quality Checks

### G1: Methodology
- Three methods compared at 4 blocks × 3 budgets = 12 comparisons.
- SVD-composed correctly identified as broken (GELU noncommutativity). Good diagnostic.
- SVD-norm (row-norm selection + lstsq) is the proper rank baseline — correct.
- Geometric (activity-score selection + lstsq) — the existing method.
- Distillation (200 steps SGD on probe data) — appropriate baseline.
- All methods evaluated on full-model perplexity — correct.

### G2: Reproducibility
Single seed. Effect sizes are moderate (1-3pp ppl difference). Some block-budget pairs may be sensitive to seed.

### G3: Tautology Check
Not tautological. Data-driven channel selection beating weight-magnitude selection is an empirical finding about the importance of activation statistics for compression.

### G4: Scope
Within scope. Core comparison experiment for the compression program.

### G5: Counterintuitive
SVD-composed catastrophic failure is notable — it reveals that rank reduction on the weight matrices alone (without accounting for nonlinearity) is fundamentally broken for GELU FFNs. This is consistent with cycle 025's GELU PWL result.

## Status: PASS

original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G2 (single seed, moderate effect sizes for some pairs)
reason: Clear confirmation. Geometric (data-driven channel selection) consistently outperforms rank-only baseline. The margin is moderate but consistent. The SVD-composed failure is an important negative result demonstrating why naive rank reduction fails for nonlinear FFNs.
recommendations: This validates the geometric method as better than rank-only. Next: item 16 (exact prune + linear collapse) to test whether exact pruning alone can achieve useful compression, and item 15 (centroid/Jacobian affinity) for better merge signals.
