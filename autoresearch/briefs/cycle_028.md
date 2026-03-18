# Cycle 028 Brief — Exact Prune Plus Linear Collapse Baseline

## Item
16. Exact prune plus linear collapse baseline

## Goal
Quantify how much "free" (no-SGD) FFN compression is achievable by:
(a) pruning dead channels (max|act| < threshold),
(b) collapsing always-active / near-linear GELU channels to linear,
(c) optionally refitting remaining weights via lstsq.

## Key Prior Context
Cycle 006 found <1% stable channels in 10/12 GPT-2 blocks (max 1.79%).
That makes meeting the 1.1x compression threshold unlikely.
This experiment confirms the exact baseline number and tests whether
near-linear collapse (5% GELU linearity criterion) adds anything.

## Metrics
- Fraction dead / always-active / near-linear per block
- Perplexity: baseline, prune-only, prune+collapse, prune+collapse+refit
- Parameter reduction ratio (must reach 1.1x to confirm)

## Confirmed if
Pruning or collapse yields ≥1.1x FFN param reduction with ≤1% ppl increase.

## Expected outcome
REFUTED — cycle 006 ceiling (~1.8%) cannot reach 1.1x (10% reduction needed).
Near-linear threshold may find more channels but at a large ppl cost.

## Script
`experiments/exact_prune_linear_collapse_gpt2.py`
