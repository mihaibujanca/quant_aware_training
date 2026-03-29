# Cycle 089: Compositional Error Accumulation Curve

**Item:** 104
**Type:** Empirical
**Task:** Measure how compression error compounds when stacking k=1..8 FFN blocks simultaneously.

## Background

- **Cycle 087** (mixed-corpus stacking): single-block gains attenuate in 4-block stacking
- **Cycles 033–059**: gradient polish reduces k=4 stacking from +30% to +5% PPL
- **Damage order established:** [7, 5, 4, 1, 3, 6] (least-damage-first from cycle 005)
- **Current recipe:** random projection + lstsq + 500 Adam steps (lr=1e-4, W_down only)

## Experiment Design

1. **Setup:** Compress blocks [7,5,4,1,3,6,8,9] (increasing damage order) one at a time
2. **Apply compression at 6×** using the recipe (random+lstsq+500 Adam polish)
3. **Measure PPL** at k=1,2,...,8 blocks compressed **simultaneously** (all-or-nothing mode)
4. **Two orderings:**
   - Forward (least-damage-first): [7, 5, 4, 1, 3, 6, 8, 9]
   - Reverse (most-damage-first): [9, 8, 6, 3, 1, 4, 5, 7]
5. **Data:** OWT corpus, 200 eval seqs, 3 seeds, batch_size=4

## Success Criteria

- **Confirmed:** Error is sub-multiplicative (PPL(k) < baseline·(1+δ)^k) → >4 viable
- **Refuted:** Super-multiplicative error (cross-block amplification) → hard ceiling
- **Inconclusive:** Additive for k≤4 but regime change at k>4

## Related Code

- `experiments/gradient_polish_stacked_gpt2.py` (template for polish loop)
- `experiments/multiblock_accumulation_6x.py` (legacy k=4 config, upgrade to k=8)
- `aleph/lm_geometric_compression.py` (core library: swap_compressed_ffns, derive_ffn_compression)

## Key Decision

If error is sub-multiplicative, next steps are k=8 at 6×. Otherwise, k=4 is hard ceiling; pivot to attention blocks (item 105).
