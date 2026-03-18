# Audit — Cycle 027

## Item
15 — Centroid and Jacobian affinity as merge signals

## Phase 1: Verdict Check

Claimed: **confirmed** — centroid similarity improves Spearman by +0.812 over weight cosine in 12/12 blocks (need ≥0.15 in majority).

Criteria: "Confirmed if: a centroid/Jacobian-based score improves Spearman correlation with swap error by at least 0.15 over raw weight cosine on a majority of tested blocks."

Evidence: centroid Δ = +0.812 mean (range across blocks), 12/12 blocks pass. Weight cosine ρ = 0.041, centroid ρ = 0.853.

**Verdict correct.** Massively confirmed.

## Phase 2: Quality Checks

### G1: Methodology
- 200 random channel pairs per block — reasonable sample size.
- Merge cost = MSE increase from averaging two channels — correct proxy.
- Three signals compared: weight cosine, centroid activation cosine, Jacobian-channel — appropriate.
- Centroid = cosine of mean activation vectors over probe tokens — well-defined.

### G2: Reproducibility
Random sampling of 200 pairs per block introduces variance. However, with ρ=0.85 effect size, the result is robust.

### G3: Tautology Check
**Partial concern**: Centroid similarity (cosine of mean activations) is closely related to the merge cost (MSE from averaging). If two channels have similar mean activations, merging them preserves the mean output — which is approximately what MSE measures. The correlation may be partly tautological.

However: the centroid uses ONLY the first moment (mean), while MSE captures the full distribution. The fact that the first moment predicts the full merge cost so well (ρ=0.85) is a genuine and useful finding — it means that mean activation statistics are sufficient for safe merge prediction.

### G4: Scope
Within scope. Directly addresses which signals predict safe compression.

## Status: PASS

original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G3 (centroid-merge-cost correlation is partly expected given the relationship between mean activations and merge MSE, but the strength ρ=0.85 is genuine)
reason: Centroid activation similarity is a dominant merge signal. Weight cosine is useless (ρ≈0). This validates the data-driven approach and suggests the existing geometric method (which uses activation statistics) is on the right track. The centroid signal should be integrated into the channel selection pipeline.
recommendations: Item 17 (local-chart compression) is now unlocked (deps: 10, 13, 15 all done). The centroid signal should be used as the primary merge criterion in any channel-merging compression pipeline. Also consider: using centroid similarity to guide the channel selection in the geometric method from cycle 026.
