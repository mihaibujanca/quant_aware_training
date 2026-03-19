# Cycle 038 Plan — Block Ordering Optimization for 5-Block Stacking

## Goal
Test whether the k=5 wall (~17% PPL) is ordering-dependent by: (A) all C(6,5)=6 combinations of 5 blocks from the 6 easiest, each at 6x; (B) reverse insertion order for the best combo.

## Rationale
Cycle 037 only tested adding block 3 or 6 to the fixed base {7,5,4,1}. The wall may be driven by which 4 blocks form the base, not which is 5th. Testing all 6 combos covers this. Reverse insertion tests whether compress-order matters (compress hardest first while model is still intact).

## Script
New file: `experiments/ffn_block_ordering_k5.py` (adapt from `ffn_param_budget_hetero_k5.py`).

## Procedure

### Part A — All 6 five-block combos at 6x (seed 42)
The 6 easiest blocks: [7,5,4,1,3,6]. All C(6,5) = 6 combos (each drops one block):
1. {5,4,1,3,6} (drop 7)
2. {7,4,1,3,6} (drop 5)
3. {7,5,1,3,6} (drop 4)
4. {7,5,4,3,6} (drop 1)
5. {7,5,4,1,6} (drop 3)
6. {7,5,4,1,3} (drop 6)

All 5 blocks at 6x (hidden=512). Single seed (42) — 12 evals total is sufficient for 5pp gaps.

### Part B — Reverse insertion for best combo (seed 42)
Take the best combo from Part A. Compress blocks in reverse damage order (hardest first). This requires sequential per-block compression:
1. Compress only the hardest block (by cycle 005 ranking) while other 4 stay float.
2. Then compress the 2nd hardest, etc., through all 5.
3. Evaluate PPL after all 5 are compressed.

Compare to Part A's forward-order result for the same combo. If reverse order PPL differs by >2pp, ordering matters.

### Part C — Heterogeneous ratio on best combo (seed 42)
If the best Part A combo differs from cycle 037's {7,5,4,1,3/6}, test that combo with heterogeneous 5th-block ratios (the dropped block's replacement at 4x) to see if the new base set also benefits from lower 5th-block ratio.

## Verdict
- **CONFIRMED**: any 5-block config achieves <12% relative PPL.
- **REFUTED**: all 5-block configs >15% regardless of combo or ordering.
- **INCONCLUSIVE**: best is 12-15%.

## Key API
```python
budget = CompressionBudgetSpec(hidden_fraction=1/6, selection_mode="topk")
# For reverse order: compress blocks one-at-a-time using per_layer_hidden
# to set only the current block, then swap and probe next.
```

## Outputs
- `autoresearch/results/cycle_038_block_ordering.json`
- `autoresearch/results/cycle_038_block_ordering.png`

## Estimated cost
~10 min: 6 combo evals (Part A) + 1 reverse eval (Part B) + 2-3 hetero evals (Part C) + baseline.
