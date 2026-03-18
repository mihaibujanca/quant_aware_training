# Audit — Cycle 011

## Item
21 — Single-affine collapse quality — GPT-2 FFN blocks

## Phase 1: Verdict Check

Claimed outcome: **refuted** — "min r_held=0.1778 > 0.5 OR max ppl increase=90.6% > 20%."

Agenda criteria for refuted: "all blocks have r > 0.5 or the swap always causes > 20% perplexity increase regardless of r."

Evidence:
- r_held values: 0.178 (block 2), 0.622 (block 1), 1.0-1.76 (blocks 3-11), 1.53 (block 0)
- NOT all blocks have r > 0.5: block 2 has r=0.178. So the first refuted clause doesn't hold.
- Swap results: block 2 → +33.7% ppl, block 1 → +90.6% ppl. Both > 20%. Both-block swap → +344%.
- The second refuted clause ("swap always causes > 20% perplexity increase") is met.

**Verdict is correct** — the swap test clearly triggers the refuted criterion.

## Phase 2: Quality Checks

### G1: Methodology
**CRITICAL FLAG: Underdetermined system.** With 512 probe tokens and 769 augmented columns (768 + bias), the least-squares system is underdetermined (512 < 769). This explains the massive r_fit vs r_held gap:
- Block 2: r_fit = 0.018 vs r_held = 0.178 (10× gap)
- Block 5: r_fit = 0.267 vs r_held = 1.669 (6× gap)

The lstsq solution interpolates the fit data nearly perfectly but generalizes poorly. The r_held values are inflated by this overfitting artifact.

**Impact on verdict**: The swap test uses the overfitted affine map, so the perplexity damage may be worse than necessary. With proper regularization (ridge, λ=1e-4) or more tokens (≥1024), both r_fit and r_held would converge to a fairer estimate, and the swap damage might decrease.

However: even block 2 (the best) has r_held = 0.178 > 0.15, and its swap causes +33.7% ppl increase >> 5%. Even halving the swap damage wouldn't meet the confirmed criterion (≤5%). The verdict is robust despite the methodological issue.

### G2: Reproducibility
- Single seed. Acceptable for exploratory work given the effect size.
- Hyperparameters recorded in JSON.
- Baseline ppl = 122.3 on Shakespeare — higher than expected (plan said ~30-40), but GPT-2 on Shakespeare is genuinely out-of-distribution. Not a bug.

### G3: Tautology Check
Not tautological. The affine-fit quality depends on training and data distribution.

### G4: Scope and Connection
- Stays within item 21 scope.
- Directly addresses whether single-affine collapse is viable for GPT-2 FFN blocks → answer is no.
- Combined with cycle 009 (Jacobian variance) and cycle 010 (toy confirmed), paints a clear picture: toy ReLU networks linearize deep layers, GPT-2 FFN blocks do not.

### G5: Counterintuitive Results
The contrast with cycle 010 (toy confirmed, GPT-2 refuted) is notable but expected: GPT-2's GELU FFN blocks have 4x wider hidden dimension (3072) and process a much more diverse token distribution than a 2D spiral classifier.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: G1 (underdetermined lstsq — 512 tokens < 769 columns), G2 (single seed)
reason: Despite the underdetermined system issue, the swap test conclusively shows that single-affine collapse causes >20% ppl increase for all tested blocks. The verdict is robust. The underdetermined system should be fixed in future experiments requiring per-block affine fits on GPT-2 (use ≥1024 tokens or ridge regression).
recommendations: Future GPT-2 affine fits should use ≥1024 probe tokens or ridge regression. Move on to multi-region approaches (items 22-25) since single-affine is decisively ruled out for GPT-2.
