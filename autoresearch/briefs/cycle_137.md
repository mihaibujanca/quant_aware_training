# Item 151: Bottleneck AE at k=512 (6x) — Param-Matched Comparison — GPT-2 FFN

## Item
Cycle 131 showed AE beats topk by 2-3.4% at k=384 (8x) but with 2.5x more params
(1.48M vs 0.59M). This confounds the conclusion. Test at k=512 (6x, the standard
recipe compression ratio) with three methods:
- **topk+lstsq**: baseline (512*768 = 393K params)
- **full AE**: 3072->512->3072, same arch as cycle 131 (~3.15M params)
- **param-matched AE**: d_bn~64 so total AE params ~ 393K (matches topk)

Blocks [1, 4, 7, 11]. 3 seeds. Pre- and post-500-step Adam polish.
- **Confirmed** if param-matched AE beats topk+lstsq by >=3% relative PPL on >=2/4
  blocks, POST-polish.
- **Refuted** if param-matched AE matches or loses to topk+lstsq post-polish.
- **Inconclusive** if full AE beats topk but param-matched doesn't, or advantage
  disappears post-polish only.

## Context
- Cycle 131 (item 88): AE 2-3.4% better, INCONCLUSIVE (param confound, no polish).
- Cycle 133 (item 127): polish erases pre-polish differences in hybrid methods.
- Working recipe: topk+lstsq+500 Adam at k=4@6x -> PPL ~70.

## Dependencies
None. Standalone follow-up to cycle 131.
