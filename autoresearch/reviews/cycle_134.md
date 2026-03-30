# Cycle 134 Review — Affine Map Extraction (Item 135)

Item 135. Reviewer: design-review agent.

## R1: Premise validity — OK

The MASO/spline viewpoint (Balestriero & Baraniuk 2018, Walker et al. 2025) predicts piecewise-affine structure in ReLU nets. GELU is smooth, so "piecewise-affine" is approximate — the Jacobian varies continuously, not discretely. The plan correctly uses cosine similarity thresholds rather than assuming hard partition boundaries. This is the right approach for GELU FFNs.

## R2: Measurement soundness — OK

Spectral gap + greedy threshold clustering is a reasonable two-pronged approach. The functional interchangeability check (step 3) is the critical validation: high cosine similarity of flattened Jacobians does NOT guarantee functional equivalence (two matrices can have similar direction but different scale, and the bias term matters). The plan correctly checks ||J_i @ x_j + b_i - FFN(x_j)|| which captures both issues.

## R3: Scope reduction — OK with note

128 tokens x 2 blocks is sufficient for a SCREENING experiment. However, 128 tokens from a SINGLE sequence is not representative of the data distribution. The plan should sample tokens from multiple sequences (e.g., 8 sequences x 16 tokens) to avoid correlation artifacts from within-sequence positional structure. If all 128 tokens come from one passage about, say, sports, the Jacobians may appear more similar than they would across diverse inputs.

**Fix**: sample 128 tokens from >=8 different OWT sequences.

## R4: Method soundness — MINOR FIX NEEDED

**Greedy clustering is order-dependent.** The plan uses greedy assignment (step 2g) which gives different group counts depending on token ordering. This is acceptable for a screening experiment but the plan should: (a) sort tokens by norm before clustering (deterministic), or (b) run greedy 3 times with random permutations and report mean group count.

**Jacobian computation through LayerNorm.** Step 2a says ffn_fn includes ln_2. LayerNorm's Jacobian depends on the full-batch statistics (or at least the token's own norm). Since `torch.func.jacrev` computes per-token Jacobians, the LayerNorm Jacobian is correct for that token but the combined Jacobian is data-dependent even for linear layers. This is fine — it's the point of the experiment — but document that the "local affine map" includes the LayerNorm linearization, not just W_fc and W_proj.

## R5: Known pitfall scan — OK

- **MP6 (dimensionality curse):** cosine similarity of random 589,824-dim vectors concentrates at 0. Any positive signal is meaningful. The null baseline is well-defined.
- **MP5 (capacity):** not applicable (analysis, not training).
- **Timing risk:** 256 Jacobians x ~1s each = ~4-5 min. The 20-min abort threshold is generous but appropriate.

## Summary

Two minor fixes, neither blocking:
1. Sample 128 tokens from >=8 sequences, not a single sequence.
2. Address greedy clustering order-dependence (sort by norm or run 3 permutations).

One advisory (no code change):
3. Since centroid-affine compression is deferred, the strongest possible outcome is INCONCLUSIVE-positive. Make sure the verdict language reflects this.

**Verdict: ACCEPT with minor fixes.**
