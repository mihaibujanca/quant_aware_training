# Item 194: Compression Ratio Sweep — GPT-2 FFN k={2,4,6,8} on blocks [1,4,7,11]

## Item
Map the PPL-vs-compression tradeoff curve by running topk+lstsq+polish at k=2 (12x), k=4 (6x), k=6 (4x), k=8 (3x) on the standard 4 blocks [1,4,7,11]. Reports stacked PPL, total FFN params, and compression ratio for each k.
- **Confirmed** if: PPL decreases monotonically with k AND k=6 or k=8 achieves PPL <45 (practical operating point exists).
- **Refuted** if: diminishing returns — k=8 still >55 PPL (compression approach has a floor unrelated to ratio).
- **Inconclusive** if: non-monotonic or only marginal improvement at higher k.

## Context
- **k=4 baseline** (cycle 183, 3 seeds): topk+lstsq+polish stacked PPL = 59.4 +/- 8.2. STUDENT_HIDDEN=512, 6x compression.
- **Teacher GPT-2** float PPL: ~30 (200 eval seqs) or ~50-58 (16 seq probe). OWT sample corpus.
- **Working recipe** (unchanged since ~cycle 100): topk channel selection, lstsq output projection, 500-step Adam polish of c_proj weights. Code: `aleph/lm_geometric_compression.py`.
- **k mapping**: k=2 -> hidden=256 (12x), k=4 -> 512 (6x), k=6 -> 768 (4x), k=8 -> 1024 (3x). GPT-2 FFN hidden=3072, groups of 128 channels each (3072/128=24 groups, top-k groups).
- **Composition barrier** (known): local per-block quality does not predict stacked quality. Stacked PPL is the metric that matters.

## Dependencies
- None. Reuses existing code from cycle 183 (`_derive_topk_pack`, `_joint_polish` patterns).

## Key Files
- Library: `aleph/lm_geometric_compression.py` (probe_ffn_geometry, derive_ffn_compression, swap_compressed_ffns, _evaluate_model)
- Reference experiment: `experiments/distillation_baseline_cycle183.py` (topk+lstsq+polish section)
- Cycle 183 results: `autoresearch/results/cycle_183_distillation_baseline.json`

## Handoff
Sweep k in {2,4,6,8}, 3 seeds each, stacked PPL after polish. This is a parameter sweep — no new algorithmic ideas. The question is purely whether the recipe is operating in a steep or flat region of the compression curve.
