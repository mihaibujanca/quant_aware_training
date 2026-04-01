# Item 187: SoLA-style split compression — GPT-2 FFN k=4

## Item
Instead of topk+lstsq (keep top-k channels, drop rest), keep top-k channels full-rank AND SVD-compress the remaining channels at low rank. Sweep r in {2, 4, 8, 16}. Compare stacked PPL vs topk+lstsq.
- **Confirmed** if: stacked PPL <= 68.7 at matched params (>= 2 PPL improvement over topk+lstsq 70.7)
- **Refuted** if: stacked PPL >= 72.7 or worse than topk+lstsq at all rank settings
- **Inconclusive** if: within +/-2 PPL of topk+lstsq

## Context
- **Baseline**: topk+lstsq k=4 (512 channels), blocks [1,4,7,11], stacked PPL ~70.7 (no polish), ~59.4 (polished).
- **Key design issue**: keeping full c_fc for tail channels costs 2560x768=2M params per block, making strict param matching impossible. We solve this by: (a) fitting lstsq from h_topk + SVD-compressed tail activations, (b) at inference using original c_fc (not pruned), (c) comparing against topk+lstsq at SAME total params (higher k) for fairness.
- **Cycle 176 (item 183)**: piecewise lstsq by sign pattern was catastrophic. This is different — we're not partitioning tokens, we're partitioning channels.
- **Cycle 171 (item 181)**: wider-shallower (more blocks, less per block) was catastrophic. This is orthogonal — same block count, richer per-block representation.

## Key Files
- `experiments/sign_pattern_lstsq_cycle176.py` — template for experiment structure
- `aleph/lm_geometric_compression.py` — FfnLayerGeometry has hidden_sample, preactivation_sample, output_sample

## Constraints
- 3 seeds [42, 137, 7], 200 OWT eval sequences, blocks [1,4,7,11]
- 500 Adam polish steps per configuration
- SVD on calibration hidden activations for tail channel compression
