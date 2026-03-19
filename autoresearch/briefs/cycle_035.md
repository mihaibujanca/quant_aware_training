# Item
52. Multi-seed confirmation of single-block compression tolerance at 6x — GPT-2 FFN

# Context
Cycle 034 found all 6 easiest blocks (7,5,4,1,3,6) tolerate 6x compression within 5% PPL on single seed (42). Non-monotonic PPL observed (sub-baseline at high compression). Audit downgraded result to INCONCLUSIVE — 50 eval sequences too small to confirm if sub-baseline is real or noise. Result needs replication on 3 seeds with 200 eval sequences each.

# Dependencies
- Depends on: item 49 (CONFIRMED, cycle 034)
- Blocks identified from cycle 005/033: [7,5,4,1,3,6]
- Method: topk channel selection + centroid similarity merge + lstsq output refit (cycles 026-027)

# Key Files
- `experiments/block_ratio_sweep_gpt2.py` — existing ratio sweep harness (use as template)
- `autoresearch/results/cycle_034_ratio_sweep.json` — single-seed baseline (seed 42, blocks [7,5,4,1,3,6], ratios {2,3,4,5,6})
- `aleph/lm_geometric_compression.py` — geometric compression API

# Constraints
- Replicate cycle 034 conditions exactly: blocks [7,5,4,1,3,6], ratio=6x only
- Add seeds 123, 7 in addition to 42
- Eval: 200 sequences each (vs 50 in cycle 034)
- Report per-block mean and std of relative PPL across 3 seeds
- Check for sub-baseline mean PPL (non-monotonicity was noise if all 3 seeds show baseline or above)

# Handoff
**Goal**: Confirm cycle 034 6x tolerance is not a single-seed artifact. Mean relative PPL at 6x must be ≤5% for ≥5 of 6 blocks, and NO block should show sub-baseline mean PPL (criterion: non-monotonicity = noise). If std exceeds 2% for majority of blocks, verdict = INCONCLUSIVE (measurement noise despite more sequences). Success unblocks item 53 (heterogeneous multi-block accumulation) and item 54 (8x/10x ceiling scan).
