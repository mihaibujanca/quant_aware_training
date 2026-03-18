# Cycle 032 Brief — Item 47: GPT-2 Dead Channel Census v2 (GELU-Correct)

## Selected Item
**Item 47** — Dead channel census — GPT-2 FFN with GELU-correct definitions (v2, supersedes item 16)

## Context
- Cycle 028 (item 16) used `DEAD_THRESHOLD=1e-5` on **post-GELU** activations.
- Problem: GELU(-3) ≈ -0.004, far above 1e-5. This threshold caught nothing (mean dead=0.00%).
- Item 47 redesigns with three GELU-correct definitions:
  1. **Pre-GELU saturated**: max(h_j) < T for T ∈ {-2,-3,-4,-5} (always in GELU's near-zero tail)
  2. **Output-contribution dead**: ‖W_down[:,j]‖ × rms(act_j) as fraction of total (catches low-impact channels regardless of magnitude)
  3. **Near-linear**: min(h_j) > T for T ∈ {2,3,4} (always in GELU≈identity regime)
- Plus smooth pruning curve: remove bottom K% by contribution for K ∈ {1,2,5,10,15,20,25,30}%.

## Key Prior Finding
- Cycle 026: geometric channel selection (by centroid activation similarity) beats SVD.
  This census tests whether there's any "free" structural compression before heuristic merging.
- If contribution-based pruning removes 5%+ channels at <1% PPL cost → significant finding.

## Script
`experiments/exact_prune_linear_collapse_gpt2_v2.py` (untracked, needs cycle-number fixes → applied)

## Confirmed if
≥5% channels dead/removable (pre-GELU T=-3 + near-linear T=+3) AND smooth pruning at 5% ≤1% PPL increase.

## Refuted if
<1% removable at any threshold AND pruning >2% always causes >1% PPL increase.
