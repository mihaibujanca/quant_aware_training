# Cycle 033 Brief — Item 48: Easiest-First Multi-Block Accumulation (Up to 6 Blocks)

## Selected Item
**Item 48** — Easiest-first multi-block accumulation up to 6 blocks — GPT-2 FFN

## Context
- Cycle 007 ran a 12-block easiest-first accumulation with the OLD method.
  - k=1: -2.08% (block 7), k=2: +0.64%, k=3: +5.79%, k=4: +9.89%, k=5: +21.76%
- Cycles 026-027 established the IMPROVED geometric method:
  - topk channel selection beats SVD in 11/12 blocks (cycle 026)
  - centroid similarity (ρ=0.85) is the right merge signal (cycle 027)
- Item 48 re-runs the 6-block accumulation with the current best method.

## Key Question
Can the improved method (cycles 026-027) keep 4-block PPL ≤10%? Can it extend to 5-6 blocks?

## Easiest Blocks (from cycle 007 data)
Order from cycle 007 easiest-first: [7, 5, 4, 1, 3, 6]

## Method
- Probe geometry for the 6 target blocks on Shakespeare corpus
- Compress each at 4× compression (hidden_fraction=0.25) using topk channel selection
- Accumulate: at step k, swap blocks [7], [7,5], [7,5,4], ..., [7,5,4,1,3,6] into model
- Measure full-model PPL at each step

## Confirmed if
4-block swap ≤ baseline × 1.10 (≤10% PPL increase from teacher).

## Refuted if
PPL exceeds baseline × 1.20 at the 3-block mark.

## Inconclusive if
3-block within 10% but 4-block exceeds it.
