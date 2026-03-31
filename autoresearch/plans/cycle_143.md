# Cycle 143 Plan: Eigenstructure-Aware Channel Selection (Item 155)

## Task
Test whether incorporating the learned covariance structure of FFN activations
(whitening) improves channel selection beyond topk and raw importance pruning.

## Hypothesis
Cycle 129 showed 7-8x isotropy increase in blocks 4-7. If the covariance structure
is informative, whitened importance (which upweights independently informative
channels and downweights redundant ones) should beat activation-magnitude topk.
If it doesn't, the isotropy structure is irrelevant to channel selection.

## Pre-flight Results
- P1 (duplicate): no prior cycle tested whitened importance.
- P2 (deps): no hard deps. Cycle 129 (isotropy) and 141 (importance) completed.
- P3 (scope): 3 methods x 4 blocks x 3 seeds = 36 comparisons. Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): float PPL = 57.65. topk k=512 PPL from cycle 141 available.
- P6 (architecture): covariance is 3072x3072, eigendecomp is O(n^3) ~ 29B ops.
  On CPU this takes ~5-10s per block. No GPU needed.
- P7 (seeds): [42, 137, 271] matching cycle 141.

## Covariance Computation
For each block b in [1, 4, 7, 11]:
1. Collect GELU(z) activations over probe_holdout tokens (same 2048+ tokens as
   cycle 141). Shape: (n_tokens, 3072).
2. Center: h_c = h - mean(h).
3. Compute C = (1/n) h_c^T h_c. Shape: (3072, 3072).
4. Eigendecompose: C = V diag(lambda) V^T.
5. Regularize: lambda_reg = max(lambda, eps * max(lambda)) with eps=1e-6.
6. C^{-1/2} = V diag(1/sqrt(lambda_reg)) V^T.
7. Whitened importance_j = ||C^{-1/2}[:,j]||_2 * rms(GELU(z_j)).

The ||C^{-1/2}[:,j]||_2 term measures the "effective independent variance" of
channel j. Channels with high covariance with others get downweighted because
their information is redundant.

## Steps
1. Copy `experiments/importance_pruning_k4_cycle141.py` to
   `experiments/eigenstructure_channel_selection_cycle143.py`.
2. Add `compute_covariance_whitening(model, tokens, block_id, device)` function:
   - Reuse the activation hook from `compute_importance`.
   - Compute C, eigendecompose, regularize, return C^{-1/2}.
3. Add `compute_whitened_importance(C_inv_sqrt, rms_gelu)` function:
   - whitened_importance_j = ||C_inv_sqrt[:, j]||_2 * rms_gelu_j.
4. Fix k=512 (matching working recipe 6x). Remove the PRUNE_FRACS sweep.
5. For each seed, block:
   a. Compute raw importance ranking (cycle 141 replication).
   b. Compute whitened importance ranking.
   c. Build compressed FFN for each of: topk, importance, whitened.
   d. Evaluate pre-polish PPL (after lstsq refit).
   e. Polish 500 steps, evaluate post-polish PPL.
   f. Log overlap between the three channel sets (Jaccard index).
6. Save results JSON with per-seed, per-block, per-method PPL (pre/post).
7. Print summary table and apply verdict.

## Validation
- Sanity: all three methods at k=512 must have PPL < uncorrected (~96).
- Topk and importance numbers should approximately match cycle 141 at k=2458
  (20% prune = 2458 channels). Our k=512 is a much more aggressive compression
  so absolute PPLs will differ, but relative ordering should be informative.
- Covariance eigenvalue spectrum: print top-5 and bottom-5 eigenvalues per block.
  If condition number is small (~1-2x), whitening has minimal effect and the
  experiment is expected to be inconclusive by design.
- Channel overlap: print Jaccard(topk, whitened) and Jaccard(importance, whitened).
  If overlap is >90%, the methods are essentially identical despite different
  scoring, and the experiment tests a distinction without a difference.

## Criteria (verbatim from agenda)
- **Confirmed** if: whitened importance beats topk by >=2% relative PPL post-polish
  on >=2/4 blocks.
- **Refuted** if: whitened importance matches topk or is worse.
- **Inconclusive** if: advantage is <2% or block-dependent.

## Artifacts
- `experiments/eigenstructure_channel_selection_cycle143.py`
- `autoresearch/results/cycle_143_eigenstructure_selection.json`

## Fallbacks
- If C is near-singular (smallest eigenvalue < 1e-10 * largest): increase
  regularization eps to 1e-4. If still unstable, use pseudoinverse with rank
  cutoff at 99% cumulative variance.
- If covariance computation OOMs on 3072x3072 (unlikely on CPU with float32 =
  ~36MB): subsample to 1024 tokens.
- If whitened == importance (Jaccard > 0.95 at all blocks): report as
  INCONCLUSIVE with note that covariance structure doesn't change selection
  at this compression ratio.
