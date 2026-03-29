# Audit — Cycle 088

- **Item**: 102 — Input-space token clustering + local Jacobian — GPT-2 FFN
- **Cycle**: 088
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Implementation correct: first-order Taylor expansion y_approx = J_c @ (x - x_c) + f(x_c) with J_c = W_out @ diag(GELU'(W_in @ x_c + b_in)) @ W_in is the right formula. GELU' validated against autograd (max diff < 1e-4). Hooks correctly capture ln_2 output (FFN input) and mlp output. Conv1D weight transpose handled. K=64 errors 153-213% (far above 5% threshold), K=1 baseline 193-268% — clustering provides only 21-31% improvement over a single global linearization. Refutation criteria clearly met (K* > 64 for all 4/4 blocks, errors >> 10%).
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Cycle 085 (output-space GELU pattern clustering) + cycle 088 (input-space Jacobian clustering) together close the piecewise-linear centroid compression direction for GPT-2 FFNs. The effective rank numbers (rank95: 319-478 out of 768) also cast doubt on item 103 (low-rank Jacobian factorization) — if individual Jacobians need 319-478 dimensions to capture 95% energy, rank-64 factorization has slim odds. Consider: (a) marking item 103 as low-priority given these rank numbers, (b) pivoting FFN compression efforts toward structured pruning or knowledge distillation rather than geometric/Jacobian approaches, (c) attention compression (item 104) may be more tractable since attention heads have explicit low-rank structure (d_head << d_model).
interesting=true
interesting_reason=Combined with cycle 085, this closes the piecewise-linear compression direction from both sides (input and output clustering). The rank95=319-478 numbers quantify why: GPT-2 FFNs operate at near-full rank locally, meaning there is no low-dimensional Jacobian subspace to exploit. Each token's linearization is genuinely high-dimensional. This is a definitive negative result that should redirect compression strategy away from Jacobian-based methods toward other approaches.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Taylor reconstruction formula `y_approx = J_c @ (x - x_c) + f(x_c)` is correct first-order expansion. Jacobian `J_c = (W_out * d_c.unsqueeze(0)) @ W_in` correctly computes `W_out @ diag(GELU'(z_c)) @ W_in`. Analytic `gelu_grad` (tanh approximation derivative, line 56-61) validated against PyTorch autograd with max diff < 1e-4 (line 198). Hooks: `register_forward_hook` on `model.transformer.h[b].ln_2` captures post-LayerNorm FFN input, `register_forward_hook` on `model.transformer.h[b].mlp` captures FFN output. GPT-2 Conv1D `.weight.T` transpose is correct (line 139). Linearity sanity check at line 188 verifies `J_c @ 0 = 0`. Affine offset `f(x_c) = W_out @ GELU(W_in @ x_c + b_in) + b_out` is exact at centroid.

**G2 (Reproducibility)**: PASS. `random_state=42` for all MiniBatchKMeans, `n_init=3`. All hyperparameters recorded in JSON: N_TOKENS=4096, N_TRAIN=3072, N_HOLDOUT=1024, K_VALUES=[4,8,16,32,64], blocks=[7,5,4,1]. Mixed corpus from 4 sources (Shakespeare, WikiText-2, OWT, C4). Deterministic given same data files.

**G3 (Results)**: PASS. Numbers match JSON precisely. K=64 mean errors: Block 7=1.531, Block 5=1.899, Block 4=2.134, Block 1=1.816. K=1 baselines: 1.931, 2.375, 2.683, 2.619. Improvement ratio K64/K1: 0.79, 0.80, 0.80, 0.69 — clustering captures only 20-31% of the nonlinearity. Error monotonically decreases with K for all blocks (verified in code, line 253-257). Refutation criterion "K* > 128 still > 10% error for majority" is met: K* > 64 for all 4/4 blocks at 153-213% >> 10%. REFUTED is the correct verdict.

**G4 (Surprising?)**: NOT SURPRISING given cycle 085. Cycle 085 showed 2045/2048 unique binary GELU patterns with low Hamming distance — output-space diversity rules out centroid approximation. This cycle confirms the same from input space: tokens that are nearby in 768D input space still have very different FFN responses. The effective rank numbers (rank95: 319-478) provide the mechanistic explanation: local Jacobians span most of the 768D space, so the function varies in many directions simultaneously. Block 1 has the highest rank (rank95=478, rank99=608), consistent with early blocks being more "general purpose." The combined picture — diverse output patterns (085) + high-rank input Jacobians (088) — is internally consistent and definitively closes piecewise-linear clustering.

**G5 (Learning quality)**: HIGH. This is a clean negative result with clear implications. Two complementary experiments (output-space clustering at K up to 256, input-space Jacobian clustering at K up to 64) both fail by wide margins. The effective rank measurements provide a quantitative bound: any method that assumes local low-rank Jacobian structure in GPT-2 FFNs will need to explain away rank95 of 319-478. This redirects the compression research program: Jacobian-based methods (items 97, 98, 102, and likely 103) are not viable for GPT-2 FFNs. Attention compression (item 104) or entirely different FFN compression strategies (structured pruning, distillation) should be prioritized.

**G6 (Method)**: PASS. The Taylor approximation is mathematically exact to first order — verified by the autograd cross-check. The affine offset `f(x_c)` is exact at the centroid; all error comes from the linear approximation `J_c @ (x - x_c)` being insufficient over the cluster radius. The fact that K=64 barely improves over K=1 (only 21-31% reduction) confirms the error is fundamentally due to high curvature in many dimensions simultaneously, not a bug in the offset computation. The method is sound; the hypothesis is genuinely refuted.
