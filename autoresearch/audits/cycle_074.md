status=PASS
original_outcome=INCONCLUSIVE
audited_outcome=INCONCLUSIVE
guardrail_flags=G1 (minor — null baseline flawed), G3 (interpretation gap), G4 (null > real is expected artifact), G6 (method limitation)
interesting=true

## Phase 1: Verdict Check

INCONCLUSIVE is correct. PCA variance at 30D is 72.4% for blocks 0-1 (seeds 7, 123), below the 80% threshold. The code correctly gates on min PCA variance across all blocks and seeds (line 286-289, 305). Even if we relaxed this and looked only at blocks 2-11 (all >=80%), TP monotonically INCREASES (0/11 decreasing pairs), which is the opposite of the confirmed criterion (needs >=60% decreasing). So the outcome would be INCONCLUSIVE or trend-toward-refuted regardless. Verdict: INCONCLUSIVE is defensible.

## G1: Implementation Correctness

**Hook point: CORRECT.** `block.register_forward_hook` captures `output[0]` from GPT2Block.forward, which is the residual stream after attention + MLP residual additions but BEFORE the next block's ln_1. This is the un-normalized residual stream between blocks — the correct signal. Norm std values (5.5-133) confirm un-normalized activations (post-LN would be ~1.0).

**PCA: CORRECT.** sklearn PCA on (512, 768) per block, projecting to 30D. Variance explained tracked per block per seed.

**ripser: CORRECT.** Euclidean pairwise distance matrix passed with `distance_matrix=True`, maxdim=1, thresh at 95th percentile. Total persistence sums lifetimes above 1% of max_r threshold. Standard usage.

**Null baseline: FLAWED.** `compute_null_baseline` (line 120-131) generates isotropic Gaussian points normalized to match block-6's mean PCA-projected norm, creating points approximately on a sphere in 30D. This maximizes pairwise distance diversity and yields artificially high TP. Cycle 047 audit explicitly recommended matching the per-layer covariance structure, not just norm. This recommendation was ignored. The null being larger than real data (11/12 blocks) is a direct consequence of this design flaw, not a genuine finding.

**Total persistence computation: CORRECT.** H0 and H1 computed separately, combined as sum. Lifetime threshold filters noise features.

## G2: Reproducibility

PASS. Three seeds (42, 123, 7), each with different 512-token windows from Shakespeare. Cross-seed CV is low (max 0.091), demonstrating good reproducibility. The monotonic TP increase is consistent across all 3 seeds.

## G3: Results Interpretation

**PCA variance by block:** Blocks 0-1 fail (72-80%), blocks 2-11 all pass (81-96%). The low PCA variance at early blocks makes sense: embedding + first block have less redundancy in 768D than deeper blocks where residual stream accumulates structure. The 80% threshold is conservative but reasonable as a data quality gate.

**TP monotonically increases:** 0/11 decreasing pairs across ALL seeds. This is the OPPOSITE of what item 28 hypothesized (Betti complexity predicts compression difficulty, implying TP should correlate with or decrease as representations stabilize). Instead, TP grows 6.3x from block 0 to block 11 (mean: 6518 to 40831).

**The rho=0.448 (p=0.072) correlation is driven by block 0 leverage.** Block 0 has both the lowest TP (6518) and highest damage (0.177 — 8x outlier). Removing it: rho jumps to 0.882 (p=0.0002). But this is N=11 with a monotonic predictor vs a nearly-flat response (damage_8x for blocks 1-11 ranges only 0.002-0.067). The high rho without block 0 is spurious — any monotonically increasing variable would correlate with the slight upward drift in damage for later blocks.

**Null TP > block TP (11/12 blocks):** This is NOT a surprising finding about the data. It is a consequence of the flawed null (spherical points in 30D have maximal distance spread). With a covariance-matched null, the comparison would be meaningful. As is, it tells us only that GPT-2 hidden states are more structured (lower-dimensional) than isotropic noise at the same norm — which we already know from the PCA variance being <100%.

## G4: Counterintuitive Results

The monotonic TP increase is not counterintuitive once you consider what drives it: activation norms grow through depth (norm_std: 5.5 at block 0 to 131 at block 10), and TP scales with the diameter of the point cloud. Without norm normalization, TP growth is dominated by the scale effect. This is EXACTLY the same artifact that cycle 047 (toy ReLU) identified and cycle 065 failed to fix. Three cycles of persistent homology experiments have now all suffered from the same confound.

## G5: Is the Topology Direction Closed?

Three persistent homology experiments have been run:
- Cycle 047 (item 27, toy ReLU): REFUTED. Audit identified scale artifact.
- Cycle 065 (item 28 predecessor): INCONCLUSIVE. Audit found same scale artifact unfixed.
- Cycle 074 (item 28, GPT-2): INCONCLUSIVE. Same scale artifact persists.

The fundamental problem is not specific to any experiment — persistent homology on raw activations is confounded by activation scale growth through depth. To make this direction viable, you would need: (a) per-layer norm normalization or covariance-matched null, AND (b) evidence that the normalized TP signal carries information beyond what simpler metrics (spectral norm, condition number, effective rank) already provide.

Given that item 80 (Jacobian effective rank) already provides a direct, interpretable measure of block complexity with clear operational meaning for compression, persistent homology adds complexity without demonstrated marginal value. The topology direction should be considered low-priority unless someone proposes a specific, testable mechanism by which homological features predict compression damage that spectral features miss.

## G6: Method Failure vs Genuine Finding

Method failure. The unresolved scale confound (3 cycles running) means we cannot distinguish "topology genuinely evolves" from "norms grow." The experiment is correctly gated as INCONCLUSIVE by the PCA threshold, but even relaxing that gate, the evidence points toward the scale artifact explanation rather than a genuine topological signal.

## Recommendations

1. Do NOT rerun this direction without first implementing per-layer covariance-matched null or activation normalization (divide by Frobenius norm per layer before PCA+ripser). This has been recommended since cycle 047.
2. The INCONCLUSIVE outcome is correct. The data does not support or refute the hypothesis because the method is confounded.
3. Mark item 28 as dead end (method limitation) rather than keeping it as a potential rerun candidate. Three attempts is enough.
4. The block 0 leverage effect (high damage + low TP) is worth noting but is a single-point observation, not a basis for conclusions.
