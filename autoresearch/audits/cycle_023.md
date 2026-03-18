# Audit — Cycle 023

## Item
29 — Local complexity as compression difficulty predictor — toy ReLU

## Phase 1: Verdict Check

Claimed: **inconclusive** — correlation with r holds (ρ=0.80) but training dynamics fails (LC drops 5.5%, need 30%).

Criteria: "Inconclusive if: the correlation with r holds but the training-dynamics prediction fails, or vice versa." Exact match.

**Verdict correct.**

## Phase 2: Quality Checks

### G1: Methodology
- Local complexity = count of distinct activation patterns in ε-ball around each probe point. Standard measure per Patel & Montúfar.
- ε=0.39 with avg ~72 neighbors (larger than target 30). More neighbors = more patterns counted = higher LC. The correlation with r should be robust to ε choice since it's relative across layers.
- n=4 data points for Spearman (one per layer). Low power; p=0.20.

### G2: Reproducibility
Single seed. With only 4 data points, the correlation has high variance.

### G3: Tautology Check
The correlation between local complexity and r is not trivially expected: LC counts distinct patterns (combinatorial), r measures affine fit quality (continuous). They're related through the fact that more patterns = more nonlinearity, but the quantitative correlation is empirical.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G2 (n=4 for Spearman, low power)
reason: LC correlates with compressibility but the training dynamics prediction fails for continuous tasks (consistent with cycle 021 grokking result). Inconclusive is correct.
recommendations: Item 30 (GPT-2 proxy) is unlocked. Also item 31 (CKA) remains high-value.
