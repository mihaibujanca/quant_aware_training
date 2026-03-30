# Cycle 133 Plan: Direct Regression Hybrid (Item 127)

## Task
Test hybrid compression: direct rank-r regression (r=256) for middle blocks (4, 7) + topk+lstsq (k=512) for edge blocks (1, 11). Compare stacked k=4 PPL against all-topk and all-direct baselines.

## Hypothesis
Middle blocks are near-linear (cycle 107): direct regression is equally efficient. Edge blocks need GELU nonlinearity: topk wins. A hybrid combining the best method per block should outperform either uniform strategy.

## Pre-flights
- P1 (duplicate): cycle 114 tested ratio allocation (REFUTED), not method selection. Different axis. Not a duplicate.
- P2 (deps): none. Cycle 107 already completed.
- P3 (scope): 3 conditions x 3 seeds = 9 stacking runs + polish. Single cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): float PPL ~57.65. All-topk k=4 stacked+polished ~80 PPL (from STATUS).
- P6 (architecture): direct regression replaces FFN with W_approx = V_r @ A + b (rank-r SVD of input, lstsq for output). Must use same hook interface as topk for stacking.
- P7 (seeds): [42, 123, 7], 200 eval seqs OWT per agenda.

## Steps
1. Create `experiments/hybrid_regression_topk_cycle133.py` (new file).
2. Implement two FFN replacement strategies sharing a common hook interface:
   - `topk_lstsq(block_idx, k=512)`: existing topk+lstsq from prior cycles.
   - `direct_regression(block_idx, r=256)`: rank-r SVD of input activations, lstsq map to output.
3. Define three conditions:
   - **all-topk**: blocks [1, 4, 7, 11] with topk+lstsq, k=512.
   - **all-direct**: blocks [1, 4, 7, 11] with direct regression, r=256.
   - **hybrid**: blocks [4, 7] direct r=256, blocks [1, 11] topk k=512.
4. For each condition and seed:
   a. Collect 4096 probe tokens per block (OWT).
   b. Fit per-block approximations.
   c. Stack all 4 blocks simultaneously (hooks replace all 4 FFNs).
   d. Evaluate PPL on 200 OWT sequences (pre-polish).
   e. Run 500 Adam steps (lr=1e-4) jointly polishing all approximation parameters.
   f. Evaluate PPL on 200 OWT sequences (post-polish).
5. Save results JSON: per-(condition, seed): pre-polish PPL, post-polish PPL, per-block reconstruction error.
6. Save bar chart: 3 conditions x pre/post polish, error bars from seeds.
7. Apply verdict criteria, update agenda.md, commit.

## Param Matching
- topk k=512: each block stores a 768x512 + 512x768 = 786,432 param approximation.
- direct r=256: each block stores a 768x256 + 256x768 + 768 = 393,984 params.
- These are NOT matched. The agenda says r=256 and k=512 — follow as written. Note the param asymmetry in the results (direct uses ~50% fewer params per block). If hybrid wins, this actually UNDERSTATES the advantage (direct is cheaper). If hybrid loses, the param gap is a confound.

## Validation
- Sanity: all-topk post-polish PPL should be ~80 (matching STATUS baseline). If >90, pipeline is broken.
- Reconstruction: per-block MSE for direct vs topk should match cycle 107 trends (direct lower MSE, but PPL may diverge for edge blocks).
- Seeds: check std across seeds. If std > 5 PPL, results are noisy — note in verdict.

## Criteria (verbatim from agenda)
- **Confirmed** if: hybrid outperforms both all-topk and all-direct by >=2% relative PPL at k=4 stacking.
- **Refuted** if: all-topk matches or beats hybrid (topk is robust enough that block-adaptive method selection doesn't help).
- **Inconclusive** if: hybrid beats one baseline but not the other, or <2% difference.

## Risk Assessment
Cycle 114 showed ratio allocation REFUTED (<1% improvement). Polish erased pre-polish differences. Strong prior that this will also show polish convergence, yielding REFUTED or INCONCLUSIVE. The experiment is still worth running because method selection (nonlinear vs linear approximation) is a fundamentally different axis than ratio allocation.

## Artifacts
- `experiments/hybrid_regression_topk_cycle133.py`
- `autoresearch/results/cycle_133_hybrid_regression.json`
- `autoresearch/results/cycle_133_hybrid_regression.png`
