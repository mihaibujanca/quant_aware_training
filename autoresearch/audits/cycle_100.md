# Audit: Cycle 100 — Low-rank Jacobian factorization for FFN compression

**Item:** 103
**Claimed outcome:** REFUTED
**Audited outcome:** REFUTED (confirmed, with caveats)
**Status:** PASS

## Phase 1: Verdict Check

The refutation gate is: median relative Frobenius error > 0.5 at r=128. Reported median = 0.809. Per-block r=128 errors: {0.718, 0.809, 0.848, 0.883}. `torch.median` on a 4-element tensor returns the lower of the two middle values (0.809), while the statistical median is (0.809+0.848)/2 = 0.829. Either way, well above 0.5. **Refutation gate correctly triggered.**

PPL further confirms: oracle S(x) (exact diagonal on SVD basis) gives +13.7% to +74.0% degradation. Predicted S(x) nearly matches oracle. Random U,V bases with oracle S are dramatically better than SVD bases (random: +3-12% vs SVD: +13-74%).

## Phase 2: Quality Checks

### G1: Implementation correctness

**Jacobian computation: CORRECT.** `J = W_out @ diag(GELU'(z)) @ W_in`, shapes (768,3072) * diag(3072) * (3072,768) = (768,768). GELU derivative uses the GPT-2 tanh approximation, formula verified correct. Weight transpose convention matches GPT-2 Conv1D (weight stored as (in, out), so `.T` gives the standard matrix form).

**SVD procedure: CORRECT.** Full SVD of mean Jacobian J_bar, truncated to rank r. Sanity check (full-rank reconstruction ~2e-6) passes.

**Oracle S(x): CORRECT.** `S(x) = diag(U^T @ J(x) @ V)`, vectorized via precomputed `M = (U^T @ W_out) * (W_in @ V)^T`. Shapes and broadcasting verified.

**CRITICAL CONCEPTUAL ISSUE:** The factored FFN replacement computes `f(x) = U @ diag(S(x)) @ V^T @ x + b_out`. This is the rank-r truncated Jacobian applied to x: `J_approx(x) @ x`. But the actual FFN output is `W_out @ GELU(W_in @ x + b_in) + b_out`. These are NOT equal even at full rank (r=768), because `GELU'(z) * (W_in @ x) != GELU(z)` when `b_in != 0` or when the activation isn't homogeneous of degree 1. The experiment effectively tests whether a bilinear factored form `U @ diag(S(x)) @ V^T @ x` can represent a nonlinear FFN — the answer is no, but this is partially baked in by the approach.

**Ridge predictor: CORRECT.** RidgeCV with 5 alpha values, train/eval split, rel_MSE < 0.5%. Prediction is accurate — the factorization structure itself is the bottleneck.

### G2: Reproducibility

**PASS.** 3 seeds (42, 123, 7) for all PPL evaluations. Seed variance ~10-15 PPL points, which is typical for Shakespeare-based evaluation. All trends are completely consistent across seeds: oracle always catastrophic, random always better than SVD, topk always near baseline. Hyperparameters fully recorded in script constants.

### G3: Results interpretation

**Refutation gate correctly applied.** Median r=128 error = 0.809 > 0.5. Even the best block (11) has 0.718 > 0.5 at r=128.

**PPL results support conclusion.** Oracle PPL degradation is massive (blocks 1,4: +72-74%; block 7: +14%; block 11: +33%). Predicted S(x) closely tracks oracle (within 0.2%), confirming the predictor works but the factorized form fails.

**TopK+lstsq comparison (k=96):** Near-baseline PPL (+0.09% to +6.3%), providing context that conventional compression succeeds where Jacobian factorization fails.

### G4: Counterintuitive results — Random beats SVD

**This is genuine, not a bug.** Random orthogonal U,V with oracle S gives much lower PPL than SVD-derived U,V with oracle S:
- Block 1: random +5.4% vs SVD +74.0%
- Block 4: random +3.0% vs SVD +72.4%
- Block 7: random +8.1% vs SVD +13.7%
- Block 11: random +12.5% vs SVD +33.2%

**Explanation:** The mean-Jacobian SVD captures directions that maximize variance of J_bar, but per-token Jacobians vary enormously (median rank_95 = 270-470). The mean's top singular vectors are a poor shared basis for individual tokens. Random orthogonal projections distribute error more uniformly across dimensions rather than concentrating on the mean's dominant (but token-unrepresentative) directions. This is related to the Johnson-Lindenstrauss phenomenon: random projections can be better than data-dependent ones when the data has high effective dimensionality and the mean is unrepresentative.

Additionally, the factored form `J_approx(x) @ x` is already a poor representation of the FFN (see G1). With SVD bases that concentrate on the mean Jacobian's top directions, the error in those directions gets amplified through `V^T @ x`. With random bases, the error is spread across all directions and may partially cancel.

### G5: Learning quality

**Key insights:**
1. Per-token Jacobians of GPT-2 FFNs are nearly full-rank (median rank_95: 270-470 / 768), confirming the Jacobian does not have useful low-rank structure at the per-token level.
2. The mean Jacobian is a poor representative of per-token Jacobians — its SVD basis is worse than random for the factored-form approximation.
3. The factored linear form `U @ diag(S(x)) @ V^T @ x` fundamentally cannot represent GELU FFNs because J(x)@x != FFN(x) (bias term + non-homogeneity of GELU).
4. Ridge predictor achieves <0.5% relative MSE for S(x), so the prediction side works — the factorization structure is the bottleneck.

### G6: Method failure vs genuine refutation

**Partially both.** Two distinct failures are conflated:

1. **Genuine finding (high Jacobian rank):** Per-token Jacobians have rank_95 of 270-470, meaning there's no exploitable low-rank structure in the Jacobian at the per-token level. This is a genuine property of GPT-2 FFNs that any Jacobian-based factorization would face.

2. **Method limitation (J(x)@x != FFN(x)):** The factored form `J(x) @ x` is not the FFN output due to GELU non-homogeneity and bias terms. A full-rank oracle (r=768) was not tested, so we cannot separate "rank truncation error" from "linearization error." The massive PPL degradation even with oracle S likely reflects both errors.

**Would column-stacked SVD help?** Possibly for the basis selection (addressing the "random beats SVD" issue), but it cannot fix the fundamental `J(x)@x != FFN(x)` problem. A better approach might factor the actual FFN output (e.g., SVD of stacked per-token FFN outputs) rather than the Jacobian.

**Verdict: REFUTED stands.** The Jacobian factorization approach fails for GELU FFNs. However, the experiment could be strengthened by: (a) testing full-rank oracle to isolate linearization error from rank truncation error, (b) testing on a ReLU network where J(x)@x = FFN(x) - bias_correction.

## Guardrail Flags

- **G6 partial method failure:** The `J(x)@x != FFN(x)` issue means the experiment conflates two distinct failure modes. The refutation is valid but the mechanism is partially confounded.

## Recommendations

1. If pursuing Jacobian-based approaches: test on a ReLU-swapped GPT-2 (links to item 107) where J(x)@x = ReLU(W_in@x) for b_in=0. This isolates rank structure from activation non-homogeneity.
2. More promising: factor the actual FFN output `y(x) = W_out @ GELU(z)` directly, not the Jacobian. SVD of stacked output vectors would test low-rank structure in the function values rather than derivatives.
3. Consider this result alongside items 107-109 (GELU->ReLU swap). If ReLU swap is viable, the Jacobian approach might work on the ReLU version.
