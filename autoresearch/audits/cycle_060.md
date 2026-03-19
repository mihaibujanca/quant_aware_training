# Audit — Cycle 060: Iterative Cascaded Compression (GPT-2 FFN)

**Item**: 76
**Verdict**: REFUTED
**Status**: PASS (with minor bookkeeping issue)

## Guardrail Assessment

| Guardrail | Status | Notes |
|-----------|--------|-------|
| G1: Implementation | PASS | Cascaded is truly sequential: `run_cascaded` iterates `TARGET_BLOCKS_LAYERORDER` in order, deepcopies the teacher model, replaces FFN modules in-place via `_replace_ffn_modules` after each block, and re-probes through the partially-compressed `working_model`. Block 1 MSE is identical in both conditions (0.0875), confirming no prior distortion exists at the first block — a correct sanity check. |
| G2: Reproducibility | PASS | 3 seeds (42, 123, 7). All seeds show the same pattern: cascaded is ~2pp better than independent. Std is high (~15%) but the **direction** is consistent across all seeds. |
| G3: Results interpretation | FLAG | The 1.83pp improvement is correctly called "negligible," but the manuscript should note that 1.83pp sits well within the ±15pp std — it is statistically indistinguishable from zero improvement. The real signal is the per-block MSE reduction (10-20%) that fails to translate to PPL, which is the key finding. |
| G4: Refutation criterion | PASS | Criterion: cascaded >= +25% = REFUTED. Cascaded degradation = +28.37%, exceeding the threshold. The criterion tests whether cascaded *overall degradation* is acceptable, not whether it beats independent. This is correct — even with cascaded re-probing, 4-block stacking at 6x still exceeds +25%. |

## Issues Found

1. **Bookkeeping error in agenda.md (line 161-162)**: Item 76 under "### Cycle 060" header lists "Cycle: 059" and references cycle_059 artifacts (`gradient_polish_stacked_gpt2.py`, `cycle_059_gradient_polish.json`). These belong to item 75, not item 76. The actual cycle 060 artifacts are correctly listed in `cycle_060_manifest.txt`. This is a copy-paste error — cosmetic, not a data integrity issue.

2. **Independent baseline uses easiest-first order constant** (`TARGET_BLOCKS_STACKING = [7,5,4,1]`) defined but never used — `run_independent_baseline` probes all 4 blocks simultaneously via `TARGET_BLOCKS_LAYERORDER`. Not a bug (simultaneous probing is order-independent), but dead code.

## Recommendations

- Fix the agenda.md bookkeeping: update lines 161-162 to reference cycle 060 and the correct artifacts.
- Remove unused `TARGET_BLOCKS_STACKING` constant from the experiment script.
- The finding that per-block MSE improves but PPL does not is the most valuable insight — it cleanly separates the FFN input-mismatch hypothesis from the residual-stream-propagation hypothesis.
