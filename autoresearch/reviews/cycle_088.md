verdict: APPROVE
round: 1
issues: 0
notes: 2

## R1: Premise validity — PASS

Clustering in 768D input space is a well-motivated pivot from cycle 085's 3072D GELU pattern clustering (REFUTED: K* > 256 for all blocks, near-unique binary patterns). The core argument is sound: GELU' patterns are 3072D and near-maximally diverse, but the *input* tokens live in 768D on a low-dimensional manifold (cycle 086: steep eigenspectrum, isotropy ratios 0.004-0.016). LayerNorm normalization concentrates inputs onto a sphere-like manifold, which *helps* k-means by removing norm variance and leaving only angular diversity to partition. The first-order Taylor reconstruction around centroids is consistent linearization (no Jensen gap issue from cycle 084, since GELU' is evaluated at the centroid point, not averaged over cluster members).

## R2: Null baseline — PASS

K=1 (global linearization at dataset mean) is explicitly included as the null baseline (step 6), with the correct validity gate: K=1 error must exceed 5%, otherwise the FFN is globally linear and the experiment is uninformative. Monotonicity check (K=64 strictly better than K=4) is also included.

## R3: Statistical/sample adequacy — PASS

3072 train tokens for k-means, 1024 holdout for evaluation. At K=64: 48 tokens per cluster. K-means centroid estimation is not subject to the curse of dimensionality the way nearest-neighbor methods are — it needs O(K) samples to converge, not O(K * d). 48 per cluster is adequate. The plan flags degenerate clusters (<5 members) and uses MiniBatchKMeans. Holdout split prevents overfitting cluster assignments. No seeds/repeats, but this is an analysis experiment with deterministic Jacobian computation; the only randomness is k-means initialization, which MiniBatchKMeans handles with multiple inits.

## R4: Method correctness — PASS

The reconstruction formula `y_approx = FFN(x_c) + J_c @ (x - x_c)` with `J_c = W_out @ diag(GELU'(W_in @ x_c + b_in)) @ W_in` is the correct first-order Taylor expansion of the FFN around x_c. The analytic Jacobian formula is exact for this architecture (two linear layers with elementwise GELU). GPT-2 uses tanh-approx GELU — the plan correctly notes this. Validation includes a spot-check against torch.func.jacrev (step in Validation section) and a linearity sanity check (J_c @ 0 = 0). Weight extraction (step 2) correctly identifies c_fc.weight.T as W_in (3072, 768) and c_proj.weight.T as W_out (768, 3072).

## R5: Known pitfall scan — PASS

- MP7 (post-LayerNorm geometry): Hooking h[b].ln_2 output IS post-LayerNorm, but correctly so — this is the FFN's actual input domain. We are not measuring geometric signatures that LayerNorm erases; we are clustering the domain the FFN operates on. No issue.
- MP8 (representation vs functional rank): Effective rank is measured on J_c (the Jacobian), which IS functional rank. Correct application.
- MP3 (underdetermined fitting): No fitting — k-means + analytic Jacobian. N/A.
- MP1, MP2, MP4, MP5, MP6: Not applicable (analysis experiment, no thresholds, no cross-domain transfer, no capacity comparison, no verdict-at-risk).

No pitfalls apply.

## Note 1 (non-blocking): GELU curvature heterogeneity across blocks

The Taylor remainder is O(||x - x_c||^2 * max_eigenvalue(Hessian)). GELU curvature peaks near z=0 and vanishes for |z| >> 1. Different blocks have different pre-activation statistics (early blocks may have more near-zero pre-activations), so the same K may give very different reconstruction quality across blocks. The plan tests 4 blocks and reports per-block K*, which will capture this. But if block 1 needs K=128 while block 7 needs K=8, the confirmation criterion (>= 3/4 blocks at K <= 64) may mask a useful partial result. Consider noting block-level trends in the results discussion.

## Note 2 (non-blocking): Relative error denominator

The error metric is ||y_approx - y_true|| / ||y_true||. If any holdout token has ||y_true|| near zero (FFN output close to origin), the relative error blows up. This is unlikely for GPT-2 FFN (residual stream offset), but a median or trimmed-mean might be more robust than plain mean. The plan reports both mean and p95, which partially addresses this.
