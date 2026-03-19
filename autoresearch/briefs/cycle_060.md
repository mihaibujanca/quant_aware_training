# Cycle 060 Brief — Iterative Cascaded Compression

**Item**: 76 (P0)
**Goal**: Reduce stacking degradation from +30.2% to ≤+10% by re-probing through compressed blocks sequentially.

## Prior Context

- **Cycle 057**: Single-block 6× on OpenWebText = −3 to −5% PPL (sub-baseline). No single component dominates the +48% gap.
- **Cycle 058**: Stacking k=4@6× = +30.2% PPL. Error pattern non-monotonic; ratio metric ill-defined (first single-block is negative). **Stacking is the sole failure mode.**
- **Cycle 059**: Gradient polish (50 Adam steps) reduces stacking degradation 39% relative: +30.2% → +18.4%. Not enough to close the gap.

## Hypothesis

Block N's lstsq refit assumes float inputs, but at inference receives distorted outputs from compressed blocks 1..N-1. **Fix**: cascade the refit. After compressing block i, forward all probe data through compressed blocks 1..i to get actual (distorted) activations at block i+1's input. Use those distorted activations for topk + lstsq on block i+1. This decouples the "compression per se" from the "compounding interference."

## Experiment

- **Pipeline**: blocks [7, 5, 4, 1] at 6× on OpenWebText (probe + eval).
- **Variants**: (1) Independent (baseline): each lstsq fitted on float inputs. (2) Cascaded: lstsq fitted on distorted inputs from prior compressed blocks.
- **Metric**: Stacked k=4 PPL degradation (target: ≤+10% vs +30.2% independent).
- **Seeds**: 3 ([42, 123, 7]).

## Success Criteria

- **Confirmed**: cascaded ≤ +10% (vs +30.2% independent).
- **Refuted**: cascaded ≥ +25%.
- **Inconclusive**: cascaded between +10% and +25%.

## Artifacts

- Experiment: `experiments/cascaded_compression_gpt2.py` (new)
- Results: `autoresearch/results/cycle_060_cascaded.json`, `.png`
