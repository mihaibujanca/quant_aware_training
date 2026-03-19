# Cycle 037 Plan — Total FFN Parameter Reduction Budget at <10% PPL

## Goal
Compute total FFN param reduction for k=4 at 6× (reuse cycle 036 data), then test heterogeneous k=5 configs where the 5th block uses a lower ratio (2×/3×/4×) to probe whether the k=4→k=5 cliff can be softened.

## Block selection rationale
Only blocks [7,5,4,1,3,6] are candidates — the 6 easiest blocks from cycle 005 single-block damage ranking. Blocks 0,2,8,9,10,11 are excluded because their single-block compression damage exceeds the safety threshold.

## Script
New file: `experiments/ffn_param_budget_hetero_k5.py`

## Procedure

### Part A — k=4 budget baseline (2 seeds: 42, 123)
Re-run k=4 blocks [7,5,4,1] at 6× with seeds 42 and 123. Cycle 036 used seed 42 only; cycle 035 confirmed single-block 6× across 3 seeds, but composition effects need multi-seed validation.
- Record PPL per seed + mean/std.
- Compute param budget: % saved = 1 - (4×compressed + 8×source) / (12×source).

### Part B — heterogeneous k=5 sweep (2 seeds: 42, 123)
Base 4 blocks [7,5,4,1] at 6× (hidden=512). 5th block candidates: block 3, block 6.
- 5th block ratios: 2× (hidden=1536), 3× (hidden=1024), 4× (hidden=768)
- Total configs: 2 blocks × 3 ratios × 2 seeds = 12 PPL evaluations
- Use `per_layer_hidden` overrides for the 5th block.

### Part C — verdict
- **Confirmed**: k=4@6× mean PPL <10% (both seeds) AND ≥25% param reduction AND any k=5 hetero config stays <12% (both seeds).
- **Refuted**: k=4@6× exceeds 10% on either seed OR all k=5 configs exceed 15% on both seeds.
- **Inconclusive**: k=4 safe but all k=5 configs fall in 12-15% range, or result depends on which block is 5th.

Note: the 3pp gap between confirmation (12%) and refutation (15%) for k=5 is tight. Two seeds mitigate noise; if seeds disagree by >2pp on any config, flag as inconclusive.

## Key API usage
```python
budget = CompressionBudgetSpec(
    hidden_fraction=1/6,
    selection_mode="topk",
    per_layer_hidden={3: 1024},  # 5th block override; rest use hidden_fraction
)
```

## Outputs
- `autoresearch/results/cycle_037_param_budget.json` — all configs, seeds, param counts, PPL
- `autoresearch/results/cycle_037_param_budget.png` — bar chart: config vs (rel_ppl%, param_reduction%)

## Estimated cost
~18 min: 2 k=4 evals (Part A) + 12 k=5 evals (Part B) + 1 baseline.
