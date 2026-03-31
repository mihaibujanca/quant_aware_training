# Item 155: Eigenstructure-Aware Channel Selection — GPT-2 FFN

## Item
Cycle 129 confirmed isotropy is learned (7-8x in blocks 4-7). Current topk selects
channels by activation magnitude, ignoring the learned covariance structure. Test
eigenstructure-aware selection: compute data covariance C = E[h h^T] of FFN
intermediate activations (3072x3072), then select channels by whitened importance:
importance_j = ||C^{-1/2}[:,j]||_2 * rms(act_j). Compare to topk and raw importance
pruning at k=512 (6x) on blocks [1, 4, 7, 11]. lstsq refit + 500 Adam polish.
3 seeds, 200 OWT eval sequences.

- **Confirmed** if whitened importance beats topk by >=2% relative PPL post-polish
  on >=2/4 blocks.
- **Refuted** if whitened importance matches topk or is worse (learned isotropy
  doesn't help channel selection).
- **Inconclusive** if advantage is <2% or block-dependent.

## Context
- **Cycle 129** (item 100): isotropy learned. Covariance condition number 7-8x
  higher than random baseline in blocks 4-7. The FFN activations have meaningful
  non-isotropic structure.
- **Cycle 141** (item 153): importance pruning (weight norm * activation RMS) vs
  topk. Result: INCONCLUSIVE. Importance beats topk pre-polish by 0.2-2% (block/
  prune-level dependent) but advantage is inconsistent across blocks and frac levels.
  Polish narrows or erases the gap.
- **Working recipe**: topk k=4 (512/3072 channels) + lstsq + 500 Adam -> PPL ~70.

## Dependencies
- None hard. Cycle 129 (isotropy) provides motivation. Cycle 141 (importance) is
  the direct predecessor and provides the comparison structure.
- `aleph/lm_geometric_compression.py`: compression infrastructure.
- `experiments/importance_pruning_k4_cycle141.py`: adapt for whitened scoring.

## Key Files
- `aleph/lm_geometric_compression.py` — compression pipeline
- `experiments/importance_pruning_k4_cycle141.py` — cycle 141 code (importance
  metric, topk comparison, polish loop)
- `autoresearch/results/cycle_141_importance_pruning.json` — baseline numbers

## Constraints
- CPU-only, ~20-30 min wall time. Covariance computation is the only new cost
  (3072x3072 matrix + eigendecomposition per block).
- k=512 fixed (6x compression), matching working recipe.
- Must report PPL at two checkpoints: pre-polish (after lstsq) and post-polish.
- Three selection methods: topk, raw importance (cycle 141 replication), whitened
  importance (new).

## Handoff
Adapt cycle 141's experiment script. Add covariance computation from probe_holdout
activations, compute C^{-1/2} via eigendecomposition (add small regularizer for
numerical stability), compute whitened importance scores. Run three-way comparison
(topk vs importance vs whitened) at k=512 on blocks [1,4,7,11], 3 seeds, pre/post
polish. Save results JSON + comparison table.
