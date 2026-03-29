# Cycle 094: Non-Uniform Compression Ratio Allocation — GPT-2 FFN

**Item:** 110
**Type:** Empirical
**Task:** Allocate per-block compression ratios non-uniformly to minimize PPL, given a fixed parameter budget.

## Context

- **Cycle 089** measured single-block damage: blocks [7,5,4,1,3,6,8,9] (increasing damage order)
- **Cycle 093** confirmed: stacking error coefficient c≈0.41 is **ratio-independent** (3×, 4×, 6× all yield c≈0.41)
- **Current recipe:** topk channel selection + lstsq + 500 Adam polish
- **Observation:** single-block damage varies widely (block 7 = −3%, block 9 = +7%), but uniform 6× ignores this structure

## Experiment Design

1. **Baseline:** k=4 blocks at uniform 6× (blocks [7,5,4,1])
2. **Budget:** Match total parameters = 4 × (original_block_params / 6)
3. **Greedy allocation:**
   - Start all at 6×
   - Iteratively reduce highest-damage block (weaker compression needed)
   - Increase lowest-damage block (can tolerate stronger compression)
   - Maintain constant parameter budget throughout
4. **Test:** k=4 blocks, 3 seeds, OWT, 200 eval seqs, topk+lstsq + 500 Adam steps

## Success Criteria

- **Confirmed:** ≥10% relative PPL improvement over uniform 6×
- **Refuted:** <3% improvement (single-block damage differences don't matter after polish)
- **Inconclusive:** 3–10% improvement

## Related Code

- `experiments/compositional_accumulation_curve_gpt2.py` (base infrastructure)
- `aleph/lm_geometric_compression.py` (CompressionBudgetSpec for non-uniform ratios)
