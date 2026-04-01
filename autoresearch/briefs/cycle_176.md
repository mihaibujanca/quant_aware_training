# Item 183: Sign-pattern-conditioned lstsq — GPT-2 FFN k=4

## Item
Fit SEPARATE lstsq per k-means cluster of pre-GELU sign patterns, instead of one global lstsq per block. Test K={4,8,16} clusters. Each cluster gets its own topk+lstsq with k=512 channels. At inference, route tokens by nearest cluster centroid. Compare stacked PPL vs single-lstsq baseline.
- **Confirmed** if: K=8 piecewise lstsq improves stacked PPL >= 3 points over single lstsq.
- **Refuted** if: < 1 PPL improvement.
- **Inconclusive** if: 1-3 PPL improvement.

## Context
- **Cycle 167 (item 177) REFUTED** the sign-pattern census: all 4096 tokens produce unique binary sign patterns per block. K* > 256 in 11/12 blocks. k-means on sign patterns retains 68-96% within-cluster FFN variance at k=128. Sign patterns are too diverse for tractable piecewise partition.
- **Dependency note**: item says "177 informative but not blocking." We can still run k-means and test whether piecewise lstsq helps despite the high diversity. Cycle 167 does NOT prove piecewise is useless — it proves that sign patterns lack natural clusters. k-means may still find useful groupings that reduce lstsq fitting error.
- **Baseline**: topk+lstsq+500 Adam k=4@6x gives PPL ~59-60 stacked (cycles 157, 172).
- **Param overhead**: K clusters means K× the c_proj parameters. K=8 means 8× c_proj, which is significant. The item requires >= 3 PPL improvement to justify.

## Key Files
- `experiments/lstsq_target_alt_cycle172.py` — template for topk+lstsq pipeline
- `autoresearch/results/cycle_167_sign_census.json` — sign pattern census results
- `aleph/lm_geometric_compression.py` — FfnLayerGeometry has `preactivation_sample`

## Constraints
- 3 seeds [42, 137, 7], 200 OWT eval sequences, blocks [1,4,7,11], k=4 (hidden=512)
- 500 Adam polish steps per configuration
