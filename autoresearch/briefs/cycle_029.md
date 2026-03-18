# Cycle 029 Brief — Local-Chart Analytic FFN Compression

## Item
17. Local-chart analytic compression inside one FFN block

## Goal
Replace global top-k channel selection with a local-chart mixture:
cluster FFN inputs into K charts, fit a small compressed block per chart,
route tokens at inference. Test whether per-chart specialisation improves
over the global analytic baseline by ≥20% on ≥3 representative blocks.

## Key Prior Context
- Cycle 026: global geometric (top-k + lstsq refit) beats SVD in 11/12 pairs.
  8× compression ΔPPL: block 2 +1.2%, block 5 -0.2%, block 8 +0.9%, block 11 +4.8%.
- Cycle 027: centroid similarity is best merge signal (ρ=0.85).
- Cycle 005: block 0 hardest, blocks 3-8 easy, block 11 hardest surviving block.
- Budget: global keeps 384/3072 channels; local K=4 keeps 96/chart (same total).

## Hypothesis
Different input regions activate different channel subsets; global selection
wastes capacity on channels rarely active for a given input region.
Per-chart selection should exploit this locality to reduce block MSE.

## Metrics
- Block MSE (geometric baseline vs local K=4 vs local K=8)
- Full-model PPL (baseline 122.35)
- % improvement in PPL over geometric baseline

## Confirmed if
Local-chart ≥20% PPL improvement over geometric on ≥3 of 3 tested blocks.

## Script
`experiments/local_chart_ffn_compression.py`
