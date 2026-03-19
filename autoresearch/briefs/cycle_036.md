# Item 53: Heterogeneous-ratio multi-block accumulation — GPT-2 FFN

## Item
Cycle 033 stacked 4 blocks at uniform 4x (+8.84%). Item 52 confirms per-block tolerance varies.
Assign each block its max safe ratio from cycle 035 multi-seed, then run easiest-first accumulation.
Compare PPL curve vs uniform 4x. Tests whether per-block ratio optimization beats uniform.

## Context
- **Confirmed dependency 48**: Easiest-first order [7,5,4,1,3,6] is safe at 4x (033)
- **Inconclusive dependency 52**: 6/6 blocks within 5% at 6x; 5/6 sub-baseline (regularization effect)
- **Cycle 033**: Uniform 4x: k=4 at +8.84%, k=5 at +17.46% (super-additive breakdown)
- **Cycle 035**: Per-block 6x PPL (multi-seed means): block 7: +0.51%, block 5: -0.20%, block 4: -0.13%, block 1: -1.00%, block 3: -1.55%, block 6: -0.57%
- All blocks beat random baseline; topk beats noise; sub-baseline PPL is real (channel selection regularizes)

## Dependencies
- Item 48 (CONFIRMED): Confirms easiest-first ordering safe at 4x
- Item 52 (INCONCLUSIVE): Per-block max-safe ratios available; all 6 blocks tolerate 6x within 5%

## Key Files
- `experiments/multiblock_accumulation_6block.py` — baseline implementation (cycle 033)
- `autoresearch/results/cycle_035_multiseed_6x.json` — per-block 6x tolerances with 3 seeds
- `autoresearch/results/cycle_033_accumulation.json` — uniform 4x curve for reference

## Constraints
- Use per-block max ratios from cycle 035: [6, 6, 6, 6, 6, 6] (all blocks pass 6x at 5% threshold)
- Accumulate in easiest-first order: [7, 5, 4, 1, 3, 6]
- Measure full-model PPL at k=1..6; compare vs cycle 033 uniform 4x curve
- Null: heterogeneous ratio optimization yields worse or equal results vs uniform 4x

## Handoff
Run heterogeneous-ratio sweep using per-block 6x tolerances from 035. Replace uniform 4x per-block ratios in `multiblock_accumulation_6block.py` with block-specific values [6,6,6,6,6,6]. Execute accumulation k=1..6 with same probe/eval setup (200 train, 50 eval seq). Chart PPL curve vs 033 baseline. **Verdict**: CONFIRMED if k=4 achieves ≤8% relative PPL (matching 033's 8.84%) while avg compression ratio >4x across blocks. REFUTED if k=4 PPL >12% or per-block ratios are not compositionally safe.
