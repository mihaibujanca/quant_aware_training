# Cycle 043 Plan — Multi-point Residual Correction

## Goal
Fit N sequential affine corrections (one per compressed block boundary) and measure PPL recovery vs cycle 039's single-point baseline.

## Script
New file: `experiments/ffn_multipoint_correction.py` (adapt from `ffn_residual_correction_k5.py`).

## Procedure

### Step 1 — Setup
Load GPT-2, build probe corpus (200 train / 50 holdout / 50 eval, Shakespeare), compute baseline PPL. Compress blocks using standard pipeline. Configs:
- k=4: blocks [7,5,4,1] at 6x
- k=5: blocks [7,5,1,3,6] at 6x (same as cycle 039)

### Step 2 — Multi-point correction fitting (SEQUENTIAL, layer order)
For k=4 blocks [7,5,4,1], the correction points in layer order are:
- After block 1 (hook on block 2 input)
- After block 4 (hook on block 5 input)
- After block 5 (hook on block 6 input)
- After block 7 (hook on block 8 input)

**Critical: fit corrections in layer order, cascading each before fitting the next.**

```
for block_id in sorted(compressed_block_ids):
    correction_layer = block_id + 1
    # 1. Collect h_float and h_compressed at correction_layer input
    #    (compressed model already has all prior corrections applied)
    # 2. Fit: h_float = h_compressed @ W_i.T + b_i via lstsq
    # 3. Install correction hook on compressed model at correction_layer
    # 4. Proceed to next block
```

Each correction is fit on ~25K probe tokens (200 seqs * 128 tokens). The lstsq system is [N, 769] -> [N, 768], well-determined at ~32:1 ratio.

### Step 3 — Evaluate
Measure PPL on eval_val with all corrections installed. Report per-correction-point MSE reduction.

### Step 4 — Also test k=5 multi-point
Same procedure for blocks [7,5,1,3,6], sorted = [1,3,5,6,7], correction points at blocks [2,4,6,7,8].

### Step 5 — Comparison table
| Config | Uncorrected | Single-point (039) | Multi-point (043) |
|--------|------------|-------------------|-------------------|
| k=4@6x | +9.59% | +8.35% | ? |
| k=5@6x | +16.12% | +12.53% | ? |

### Step 6 — Diagnostic: per-correction MSE breakdown
For each correction point, report:
- MSE before correction at that point
- MSE after correction at that point
- ||W_i - I||_F and ||b_i|| (correction magnitude)
This reveals which blocks contribute most error and whether later corrections shrink (diminishing returns) or stay large (compounding error).

## Verdict Criteria
- **CONFIRMED**: k=5 multi-point <= +12% PPL OR k=4 multi-point <= +6.85% (>= 1.5pp gain over +8.35%).
- **REFUTED**: neither k=4 nor k=5 improves over single-point.
- **INCONCLUSIVE**: k=4 improves but k=5 does not, or gains < 0.5pp.

## Outputs
- `autoresearch/results/cycle_043_multipoint_correction.json`
- `autoresearch/results/cycle_043_multipoint_correction.png`

## Estimated cost
~12 min: load + probe + compress + 2x (4 or 5 hidden-state collections per correction point) + lstsq fits + 4 evals.
