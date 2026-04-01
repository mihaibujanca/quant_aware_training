# Cycle 176 Plan: Sign-pattern-conditioned lstsq (Item 183)

## Pre-flights

### P1 (duplicate check)
No prior cycle tested piecewise lstsq conditioned on sign-pattern clusters. Cycle 167 ran the census; this is the follow-up empirical test. NOT a duplicate.

### P2 (dependency validity)
Item 177 (cycle 167) was REFUTED: sign patterns don't cluster tractably (all 4096 unique, K* > 256). The dependency is marked "not blocking" and the item text explicitly says "can use k-means directly." Proceeding is valid, but we should expect this to be refuted given the census showed high within-cluster variance even at k=128. The experiment is still informative: it directly tests whether piecewise lstsq helps despite poor clustering.

### P3 (scope)
3 seeds x 4 configurations (baseline + K=4,8,16) x (probe + lstsq + polish + eval) = 12 pipeline runs. Each takes ~2-3 min on GPU. Total ~30 min. Fits in one cycle.

### P4 (program.md compliance)
Empirical type, GPT-2 proxy benchmark, FFN-only, evaluation ladder: single-block (implicit in lstsq) then stacked. Standard blocks [1,4,7,11]. No program.md violations.

## Hypothesis
Piecewise lstsq, where each cluster of tokens (grouped by k-means on pre-GELU sign patterns) gets a separate output projection, will improve stacked PPL over the single-lstsq baseline. The argument: tokens in different GELU activation regimes use different effective linear maps, so fitting separate projections should reduce approximation error.

Counter-hypothesis (favored by cycle 167): sign patterns are too diverse for k-means to find meaningful clusters. The piecewise projections will overfit to small cluster populations and the routing overhead will not pay off.

## Design

### Approach
For each block, after selecting topk channels (same k=512 for all):
1. Compute pre-GELU sign patterns from `preactivation_sample` (z > 0).
2. Run k-means on sign patterns (float32 cast) with K clusters.
3. For each cluster c, select the tokens assigned to c and fit a SEPARATE lstsq: W_c = lstsq(H_c, Y_c) where H_c is the topk hidden activations for cluster-c tokens and Y_c is their float FFN outputs.
4. At inference, compute the compressed hidden activations, determine sign pattern, route to nearest centroid, and apply the corresponding W_c.

### Inference routing
The compressed block stores K centroids (from sign patterns) and K output projections. At inference:
1. Compute z = W_in @ x + b_in (same for all clusters — shared c_fc)
2. Sign pattern s = (z > 0).float()
3. Assign to nearest centroid: c = argmin_k ||s - centroid_k||
4. Apply W_c[c] @ GELU(z[topk_indices]) + b_c[c]

This is a piecewise-linear compressed block.

### Configurations
- **baseline**: single lstsq (K=1), same as standard pipeline
- **K=4**: 4 clusters per block
- **K=8**: 8 clusters per block (primary test per item criteria)
- **K=16**: 16 clusters per block

### Param accounting
- Baseline: 4 blocks x (512 x 768 W_out + 768 bias) = 4 x 394,240 = 1,576,960 c_proj params
- K=8: 4 blocks x 8 x 394,240 = 12,615,680 c_proj params (8x more)
- Plus K centroids per block (3072-dim): 4 x K x 3072 (negligible vs c_proj)
- c_fc params are SHARED across clusters (same topk channels)

This is a heavy param budget. The item criteria require >= 3 PPL improvement to justify.

### Minimum cluster size guard
With 4096 tokens and K=16, mean cluster size = 256. Some clusters may be very small. If any cluster has < 32 tokens, lstsq will be underdetermined (512 features, 768 outputs). Guard: if cluster_size < 64, merge with nearest cluster for lstsq fitting. Flag this in results.

## Steps
1. Copy `experiments/lstsq_target_alt_cycle172.py` to `experiments/sign_pattern_lstsq_cycle176.py`
2. Modify to:
   a. After probing, extract `preactivation_sample` for sign patterns
   b. For each K in [1, 4, 8, 16], run k-means on sign patterns, fit per-cluster lstsq
   c. Build a `PiecewiseCompressedMLP` that stores K centroids + K output projections
   d. Swap into model, evaluate stacked PPL
3. Joint polish: polish ALL K projections simultaneously (each sees only its cluster's loss contribution, but backprop flows through the routing)
4. Evaluate on 200 OWT sequences, 3 seeds
5. Report: baseline PPL, K=4/8/16 PPL, delta vs baseline, cluster size stats
6. Apply verdict criteria

## Validation
- Baseline (K=1) must match prior cycles (~59-60 PPL stacked) within 2 PPL
- All K>1 configs must have PPL < uncorrected (~88 stacked)
- Report min/max cluster sizes to check for degenerate clusters
- Report per-block routing distribution

## Artifacts
- `experiments/sign_pattern_lstsq_cycle176.py`
- `autoresearch/results/cycle_176_sign_lstsq.json`
- `autoresearch/results/cycle_176_sign_lstsq.png`

## Verdict Criteria (verbatim from agenda)
- CONFIRMED: K=8 piecewise lstsq improves stacked PPL >= 3 points over single lstsq
- REFUTED: < 1 PPL improvement
- INCONCLUSIVE: 1-3 PPL improvement
