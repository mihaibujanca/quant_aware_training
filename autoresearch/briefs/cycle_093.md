# Cycle 093: Lower Compression Ratio Stacking — GPT-2 FFN

**Item:** 108
**Type:** Empirical
**Task:** Does lowering per-block compression ratio from 6× to 3×–4× reduce multiplicative error compounding enough to viably compress 6–8 blocks?

## Context

- **Cycle 089** found: error compounds at **~41%/block at 6×**, capping k≤4 as practical ceiling
- **Recipe:** topk channel selection + lstsq + 500 Adam steps (lr=1e-4, W_down only)
- **Parameter budget trade-off:** 4× on 8 blocks = **2× total params** vs 6× on 4 blocks = **1.5×**
  - Question: does 8-block stacking at 4× outperform 4-block at 6×?

## Experiment Design

1. **Three compression ratios:** {3×, 4×, 6×}
2. **Ordered accumulation curves:** blocks added in damage order [7,5,4,1,3,6,8,9]
3. **Measurements per ratio:**
   - (a) Single-block PPL delta for blocks [1-11] individually
   - (b) Stacked PPL at k={4,6,8} blocks combined
   - (c) Fit multiplicative coefficient c (PPL(k) = baseline · (1+c)^k)
4. **Data:** 3 seeds, OWT, 200 eval seqs, topk+lstsq + 500-step Adam polish

## Success Criteria

- **Confirmed:** c drops to ≤0.20 at 3× AND k=8 stacked PPL ≤ +20% (viability proven)
- **Refuted:** c ≥ 0.30 at 3× (compression barely helps)
- **Inconclusive:** c improves but k=8 PPL > +30%

## Related Code

- `experiments/compositional_accumulation_curve_gpt2.py` (cycle 089 stacking infrastructure; extend to 3× and 4×)
- `aleph/lm_geometric_compression.py` (CompressionBudgetSpec for non-uniform ratios)
