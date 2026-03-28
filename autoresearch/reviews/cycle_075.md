verdict: APPROVE
round: 1
issues: 0 blocking, 2 minor

## R1: Premise
Sound. Item 34 was INCONCLUSIVE on toy ReLU (Gini=0.408, top-20%=48.2%). GPT-2 has 12 blocks providing Spearman statistical power for the Gini-vs-difficulty correlation. The Jacobian-clustering proxy for "linear regions" is well-motivated: GELU FFNs don't have discrete regions, but per-token Jacobians capture the local linear map, and clustering these is the natural analog. The plan explicitly uses SVD singular value profiles (768D) to avoid the 589K-dimension problem.

## R2: Measurement
SVD profiles as region fingerprints are appropriate (consistent with MP8: measure functional rank via Jacobian SVD). Null baseline (uniform random assignment) addresses MP1. The Gini coefficient and top-20% coverage metrics are well-defined and match the agenda criteria verbatim.

## R3: Seeds
3 seeds [42, 123, 7], cross-seed std < 50% guard. Adequate.

## R4: Method-architecture match
K-means on normalized SV profiles is reasonable. K=50 default with fallback sweep [10, 20, 50, 100] mitigates K sensitivity. Normalization removes scale but clusters on shape — correct for identifying distinct "behavior types." The variance computation in step 4 uses unnormalized Jacobians, which is consistent.

## R5: Pitfalls
- MP1: Null baseline included. Mitigated.
- MP7: Hooking mlp.c_fc (pre-GELU, inside FFN). Jacobian computed analytically. No LayerNorm contamination. Not triggered.
- MP8: Using Jacobian SVD, not representation geometry. Correctly addressed.
- MP4: Toy ReLU has discrete regions; GPT-2 GELU has continuous local maps. The Jacobian-clustering proxy is the bridge. Plan should note this is an analog, not an exact correspondence. Minor.

## Minor observations

1. **Reconstruction error should use profile space.** Step 4 describes "output variance contribution = ||J_c||_F^2 * volume_c" but the agenda says "cumulative reconstruction error as clusters added." More precisely: use K-means within-cluster SS in the 768D SVD-profile space. Cumulative coverage = 1 - (remaining within-cluster SS / total SS from global mean). This avoids storing full 768x768 Jacobians per token for the variance step and is internally consistent with the clustering. The plan has been revised to reflect this.

2. **Domain mismatch acknowledgment.** GELU FFNs don't have discrete linear regions — the Jacobian varies continuously. The K-means clustering imposes discretization. This is a reasonable proxy but the experiment cannot distinguish "Pareto because of discrete region structure" from "Pareto because K-means always produces skewed clusters on smooth data." The null baseline partially addresses this, but the plan should note that Pareto structure exceeding the null is necessary but not sufficient to confirm discrete-like region structure.
