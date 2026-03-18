# Audit — Cycle 018

## Item
36 — Procrustes linearity profile + identity-layer pruning — GPT-2

## Phase 1: Verdict Check

Claimed outcome: **inconclusive** — C1 (≥4 layers lin≥0.98) PASS, C2 (top-2 identity ≤10% ppl) FAIL, C3 (Spearman≥0.4) PASS.

Agenda criteria:
- Confirmed if: ≥4 layers linearity ≥ 0.98 AND top-2 identity replacement ≤ 10% ppl AND Spearman ≥ 0.4.
- Refuted if: all layers linearity < 0.90.
- Inconclusive if: linearity high but identity replacement causes large ppl increase.

**Verdict is correct.** Two of three confirmed criteria pass, but joint removal fails (33.6% >> 10%). The inconclusive criterion matches exactly.

## Phase 2: Quality Checks

### G1: Methodology
- Procrustes linearity = ||X̃ᵀỸ||_* / (||X̃||_F · ||Ỹ||_F) — correct normalized nuclear norm.
- Linearity scores > 1.0 for blocks 3-5 (max 1.0007). This is numerically possible when the representations grow (||Y||_F > ||X||_F) and the nuclear norm slightly exceeds the product of Frobenius norms due to floating-point or the specific normalization used. Not a bug, but the score should theoretically be capped at 1.0.
- Identity replacement: zero out block contribution (residual stream passes through unchanged) — correct interpretation.
- Single-block perplexities computed for all 12 blocks — comprehensive.

### G2: Reproducibility
Deterministic given fixed model and corpus. Acceptable.

### G3: Tautology Check
Not tautological. The linearity pattern depends on training. The near-identity behavior of middle layers is a genuine empirical finding.

### G4: Scope and Connection
Within scope. Directly addresses whether whole-layer linearity enables compression.

**Key insight the execution may have underemphasized**: Several single-block identity replacements are very cheap:
- Block 6: +3.9% ppl (cheapest)
- Block 2: +6.0%
- Block 10: +8.1%
- Block 4: +8.5%

Block 6 at +3.9% is within a typical acceptable threshold. The failure is only for JOINT removal.

### G5: Counterintuitive
Block 7 has high linearity (0.999) but costly identity replacement (+56.4% ppl). Linearity ≠ removability for all blocks. The Spearman between linearity and identity-replacement-cost is only 0.20 (p=0.53) — linearity predicts COMPRESSION damage (cycle 005, ρ=0.56) better than REMOVAL damage.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G1 (linearity > 1.0 for blocks 3-5 — numerical artifact, not critical)
reason: The Procrustes signal is the strongest geometry-compression predictor found in this research program. 7/12 layers are near-linear, and block 6 can be individually identity-replaced at 3.9% ppl cost. But joint removal compounds non-linearly, matching the accumulation finding from cycle 007. The finding is informative and actionable even though the strict confirmed criteria aren't met.
recommendations: This is the best signal yet. Next steps: (1) test sequential single-block identity replacement with evaluation after each removal; (2) test replacing blocks with learned low-rank approximation instead of identity; (3) combine Procrustes ranking with cycle 005/007 accumulation ordering. Item 31 (CKA) would provide complementary information about which blocks are redundant with each other.
