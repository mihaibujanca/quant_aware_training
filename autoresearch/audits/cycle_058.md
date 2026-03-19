# Cycle 058 Audit — Multi-block Stacking Error Budget: GPT-2 FFN

**Item**: 75 — Multi-block stacking error budget: additive vs compounding degradation — GPT-2 FFN
**Claimed verdict**: INCONCLUSIVE
**Audit verdict**: INCONCLUSIVE — agree; primary criterion (ratio stacked/sum-of-singles) is inapplicable because all single-block deltas are negative

## Guardrail Flags

| Check | Status | Note |
|-------|--------|------|
| G1: Implementation correctness | PASS | Progressive stacking probes with full teacher, swaps all k blocks simultaneously via `swap_compressed_ffns`. Single-block baselines are independent. Marginals and second differences computed correctly. |
| G2: Reproducibility (3 seeds) | **WEAK** | 3 seeds present. Marginal k=1 has sign flip across seeds (seed 42: +8.7pp, seeds 123/7: negative) giving SNR=0.30. Marginals k=2..4 are tight (std 1.4-1.7pp) — stacking signal is robust, but the k=1 baseline is noise-dominated. |
| G3: Results interpretation | **FLAG** | The INCONCLUSIVE verdict is technically correct per criteria, but understates the qualitative finding. Additive prediction at k=4 is -20.9% (sum of singles); actual is +30.2% — a 51pp gap. This is unambiguously super-additive compounding regardless of the ratio metric's sign problem. The narrative should state this more directly. |
| G4: Single-block PPL consistency | PASS | Block 7/5/4 deltas match cycle 057 Variant A within 0.1pp. Block 1 (-7.6%) is new but follows the same pattern. Baseline PPL (57.653) is identical. Not a bug — the negative deltas are the known lstsq regularization effect. |

## Key Issues

1. **Criteria design flaw, not experiment flaw.** The original criteria assumed single-block swaps would degrade PPL (positive deltas), making the ratio metric well-defined. The negative-delta reality (lstsq implicit regularization) breaks the criterion. The experiment itself is well-executed; the conclusion template just cannot express the answer.

2. **Qualitative answer is clear despite INCONCLUSIVE.** The k=1 to k=2 transition (marginal jumps from -3.3 to +10.4pp, second diff +13.7pp) demonstrates a sharp nonlinear interaction onset at 2 blocks. Errors compound through the residual stream — MSE at block 7 grows from 0.66 (k=1) to 1.83 (k=4), confirming upstream error propagation. This is super-additive by any reasonable interpretation.

## Recommendations

- Accept INCONCLUSIVE as formally correct. Add a note to the agenda that the qualitative answer is "super-additive compounding" even though the ratio metric is ill-defined.
- Future stacking experiments should use marginal convexity as the primary criterion, not the ratio metric, since single-block deltas can be negative.
