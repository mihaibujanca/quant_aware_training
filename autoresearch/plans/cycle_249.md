# Cycle 249 — Item 259

## Hypothesis
Heterogeneous K complement — K=512 on {7,8}, K=128 on {1,2,4,5}. Tests cycle 247's residual-MSE-driven hypothesis: blocks 7,8 had HIGHER per-block MSE under K=128 than blocks 4,5. Upgrade THOSE blocks to K=512 instead.

## Rationale
Cycle 247 placed K=512 on {4,5} and got 64.453 (38.6% gap recovery). Per-block MSE seed-42 showed blocks 7=0.581, 8=0.818 dominate vs blocks 4=0.162, 5=0.224. If MSE drives the binding constraint, swapping K=512 onto {7,8} should recover more of the 67.562 → 59.515 gap.

## Method
Copy `experiments/hetero_k_6block_cycle247.py` to `experiments/hetero_k_complement_cycle249.py`. Change only `K_PER_BLOCK = {1: 128, 2: 128, 4: 128, 5: 128, 7: 512, 8: 512}`. Update docstring/output paths to cycle 249. Trainable param budget identical to cycle 247 (~1.18M).

## Protocol
- BLOCKS=[1,2,4,5,7,8], W_out+b_out trainable, W_fc frozen
- Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore
- Disjoint calib (200) / eval (100), seeds [42, 137, 2024]
- Output: `autoresearch/results/cycle_249_hetero_k_complement.json` + summary

## Decision Rule
- Confirmed: best_mean ≤ 61.0 (or significantly better than 64.453)
- Refuted: best_mean ≥ 65.0
- Inconclusive: 61 < best_mean < 65

## Files
- New: `experiments/hetero_k_complement_cycle249.py`
- Template: `experiments/hetero_k_6block_cycle247.py`
- Output: `autoresearch/results/cycle_249_hetero_k_complement.json`

## Risks
Per-block MSE may not predict ppl impact if errors compound non-linearly across cascade. Result may be similar to cycle 247 (mid-60s), making the MSE-driven hypothesis itself inconclusive rather than the swap.

## Verification
Confirm K_PER_BLOCK dict matches spec, BLOCKS list unchanged, trainable param count matches cycle 247. Run summary line emits best_mean across 3 seeds.
