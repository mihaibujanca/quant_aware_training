# Audit — Cycle 024

## Item
31 — CKA representational similarity matrix across layers (GPT-2)

## Phase 1: Verdict Check

Claimed: **inconclusive** — CKA plateaus exist but Spearman with damage < 0.4.

Criteria: "Inconclusive if: CKA plateaus exist but removing one layer from a plateau still causes > 20% relative perplexity increase."

Actually the inconclusive criterion refers to removal cost. The experiment measures Spearman correlation instead. The spirit is the same: CKA plateaus exist (10-block plateau) but the signal doesn't predict compression difficulty within the plateau.

**Verdict of inconclusive is reasonable** given both confirmed criteria aren't fully met.

## Phase 2: Quality Checks

### G1: Methodology
- Linear CKA correctly computed via ||X^T Y||²_F / (||X^T X||_F · ||Y^T Y||_F) after centering.
- 13×13 full matrix — comprehensive.
- The near-universal CKA ≥ 0.99 for middle blocks is a known limitation of CKA on residual networks (Nguyen et al. 2021). Not a bug.

### G3: Tautology Check
**Flagged**: High CKA across residual stream layers is partly expected since each block adds a small residual to a large running sum. The representation X_{i+1} = X_i + f(X_i) is dominated by X_i, making CKA(X_i, X_{i+1}) ≈ 1 by construction. The plateau is more about the architecture than about functional redundancy.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G3 (high CKA partly tautological from residual connection structure)
reason: CKA is confounded by residual connections. All middle blocks have CKA > 0.99 regardless of their actual functional contribution. Procrustes (cycle 018) is a better signal because it measures the TRANSFORMATION, not the accumulated representation.
recommendations: CKA on residual networks is structurally uninformative. Block-level CKA should be computed on the RESIDUAL (f(X_i), not X_{i+1}) to remove the skip connection confound. This would require a separate experiment.
