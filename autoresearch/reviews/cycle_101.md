verdict: APPROVE
round: 1

## R1: Premise validity
Cycle 085 tested local linearity of GELU via Taylor linearization — not whether derivative-similar tokens produce similar outputs. The correction (cluster on GELU'(z), measure actual output similarity via cluster-mean reconstruction) is the right methodology. Sound premise.

## R2: Measurement soundness
k=1 null baseline included. Train/holdout split (3072/1024) prevents cluster-mean overfitting. One gap: if k=1 error is only ~8%, the 5% threshold is nearly trivial. Plan should report improvement relative to k=1, not just absolute 5%. Minor — does not block.

## R3: Statistical adequacy
At k=256 with 1024 holdout tokens, only ~4 tokens/cluster in holdout. A single outlier dominates error estimates. **[I2]** Must report median error alongside mean, and exclude or flag clusters with <5 holdout tokens when computing K*. Also **[I1]**: no mention of k-means seeds. Sklearn default n_init=10 may suffice, but plan should state it explicitly and verify it's used.

## R4: Method-architecture alignment
Clustering on GELU'(z) is reasonable — groups tokens by gate profile. Subtlety: GELU' saturates at ~1 for large z, so tokens with z=5 and z=50 both have GELU'≈1 but produce very different outputs. **[A1]** Report within-cluster z-norm variance as a diagnostic to detect this failure mode.

## R5: Known pitfall scan
- MP1 (threshold above trivial): k=1 null included. Verify 5% is well below null. Pass.
- MP3 (underdetermined): No fitting, just cluster means. N/A.
- MP6 (verdict misclassification): If holdout cluster sizes are tiny, K* is unreliable. I2 addresses this.
- MP7 (post-LayerNorm): Measurements are on pre-activation z and FFN outputs, not post-LN. Pass.
- MP8 (representation vs functional rank): Not applicable — measuring output similarity, not rank.

## Issues

**I1 (moderate):** No explicit seed/n_init control. K-means initialization variance could affect K* determination. Fix: state that sklearn n_init=10 (default) is used, or run 3 explicit seeds and report variance.

**I2 (moderate):** Holdout has ~4 tokens/cluster at k=256. Mean error over 4 samples is noisy. Fix: report median reconstruction error alongside mean, exclude clusters with <5 holdout tokens from K* computation, and report fraction excluded.

## Advisory

**A1:** GELU'(z) saturation — tokens with large |z| all have GELU'≈1 regardless of z magnitude. Report within-cluster z-norm std as a diagnostic. If high, derivative clustering is grouping dissimilar-output tokens together.

**A2:** If k=1 null error is <10%, the 5% K* threshold is nearly trivial. Also report the k at which error drops to 50% of k=1 error as a secondary metric.

## Notes
- Plan correctly identifies that cycle 085 k=256 error should be strictly higher than this cycle's k=256 error (cluster-mean of actuals is a better approximation than linearized). Good cross-check.
- Validation step (assertion that hooked FFN output matches manual gelu(z)@W_out+b_out) is well-designed and catches potential dropout/residual issues.
- Fallback for OOM is reasonable. MiniBatchKMeans fallback for speed is practical.
