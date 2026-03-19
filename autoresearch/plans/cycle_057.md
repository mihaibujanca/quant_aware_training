# Cycle 057 Plan: Failure Decomposition — Channel Selection vs lstsq Refit vs Stacking

## Pre-flight
- P1 (duplicate): No prior cycle isolates selection/refit/stacking components. Cycle 054 tested full pipeline; cycle 056 swept ratios but always used matched probes.
- P2 (dependencies): Cycle 054 (+48% PPL at k=4@6x on OWT), cycle 056 (single-block OWT 6x: block4 +6.6%, block5 +8.2%, block7 +9.4%). OWT sample cached. Pipeline unchanged.
- P3 (concrete): Threshold: one component >= 60% of gap. Three explicit comparisons per block.
- P4 (feasible): 3 blocks x 3 variants x 3 seeds = 27 configs, plus 3 Jaccard computations. ~30 min total.
- P5 (null baseline): Single-block OWT 6x from cycle 056 already shows +6.6-9.4% PPL. Stacked 6x from cycle 054 = +48%. Gap is ~39-41pp, so stacking contributes at least 80% if single-block results hold.
- P6 (alignment): Same model, blocks, ratio as prior cycles.
- P7 (power): 3 seeds per variant; Jaccard is deterministic given probes.

## Hypothesis
Stacking (error compounding across 3 blocks) accounts for >= 60% of the +48% PPL gap, since single-block OWT 6x is already < 10%.

## Method
Create `experiments/failure_decomposition_gpt2.py`. For blocks [7, 5, 4] at 6x, 3 seeds:

1. **Probe both corpora**: Build OWT probe corpus AND Shakespeare probe corpus (same seed). Use `max_tokens_per_layer=4096` (fix cycle 056 audit: overdetermination >= 2.5x at 6x means 512 channels + 1 = 513 cols, 4096/513 = 8x overdetermined).
2. **Jaccard overlap**: For each block, run `_select_hidden_units` on Shakespeare atlas and OWT atlas with `hidden_fraction=1/6`. Compute `|S_shk & S_owt| / |S_shk | S_owt|`. This tests channel selection stability across corpora.
3. **Variant A — OWT-native single-block**: Probe on OWT, select on OWT, lstsq on OWT, eval on OWT. Single-block swap. This is the cycle 056 baseline with higher probe budget.
4. **Variant B — Cross-selection refit**: Use Shakespeare-selected channels, but refit lstsq on OWT probes, eval on OWT. If B matches A, selection doesn't matter; if B is much worse, selection is the bottleneck.
5. **Variant C — Full stacked 3-block**: Same as variant A but swap all 3 blocks simultaneously. This measures stacking overhead.

## Key Parameters
- `target_blocks = [7, 5, 4]`, `hidden_fraction = 1/6` (6x)
- `max_tokens_per_layer = 4096`, `probe_train_sequences = 200`, `eval_val_sequences = 200`
- Seeds = [42, 123, 7]. Shakespeare text from `data/shakespeare.txt`, OWT from `data/openwebtext_sample.txt`.

## Output Files
- `experiments/failure_decomposition_gpt2.py`
- `autoresearch/results/cycle_057_decomposition.json` — per-block Jaccard, per-variant per-seed PPL
- `autoresearch/results/cycle_057_decomposition.png` — grouped bar chart: variant x block

## Success Criteria
- **Confirmed**: one component >= 60% of the +48pp gap (>= 28.8pp attributable)
- **Refuted**: all three <= 40% each, no single bottleneck
- **Inconclusive**: single-block OWT already < 10% (stacking is sole problem, not per-block method)

## Risks
- Pre-flight P5 already suggests stacking is dominant (~80%+). If so, the "confirmed" verdict is likely trivial and the Jaccard/refit comparisons become secondary diagnostics rather than discriminating tests.
- Shakespeare text may be too short for 4096-token probes at `probe_train_sequences=200`. Fall back to available token count and document actual overdetermination ratio.
