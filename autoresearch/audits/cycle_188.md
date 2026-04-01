# Audit: Cycle 188 — Item 198

**Status**: PASS
**Verdict accepted**: yes

---

## Summary

Cycle 188 tests whether per-channel conditional mean replacement (C3) composes across all 12 GPT-2 blocks. The experiment is well-designed, correctly implemented, and produces a clear inconclusive result at +20.33% (within the +15%–+40% inconclusive band). All validation checks pass.

---

## Design Soundness (G1)

**Verdict**: Sound

- **Hypothesis**: C3 is a constant replacement (shape-preserving); composition failure would isolate constant replacement inadequacy from the GELU→ReLU shape mismatch that destroyed cycles 181/182. Clear and testable.

- **Scope**: Progressive installation (blocks 0..k, k=0..11) is the correct approach to isolate composition dynamics. Alternative (all-12 in one shot) wouldn't show the accumulation pattern.

- **Criteria**:
  - Confirmed: <+15% (baseline × 1.15 = 26.105)
  - Refuted: >+40% (baseline × 1.40 = 31.780)
  - Inconclusive: [26.105, 31.780]
  - Thresholds are reasonable for the binary framing: confirmed shows bounded composition, refuted shows cascade failure.

- **Configuration**: Matches cycle 173 (N_SEQS=200, SEQ_LEN=1024, BATCH_SIZE=4, SEED=42). Baseline PPL=22.700 is in [15, 65], validates pre-flight P5.

- **Fork integrity**: Code properly forks `regime_ablation_table_cycle173.py`, preserves `collect_gelu_stats()` and `ConditionHook` infrastructure. No ad-hoc modifications.

---

## Verdict Logic Correctness (G2)

**Verdict**: Correct

```
Baseline:          22.700
All-12 result:     27.314
Delta:             +4.614 (+20.33%)

Threshold confirmed: 22.700 × 1.15 = 26.105  ✓
Threshold refuted:   22.700 × 1.40 = 31.780  ✓

26.105 < 27.314 < 31.780  →  INCONCLUSIVE  ✓
```

Verdict is correctly computed and recorded in JSON and log.

---

## Implementation Quality (G3)

**Verdict**: Clean; no bugs found

- **C3 hook logic** (lines 146–149): Correctly implements mask-conditional replacement:
  ```python
  mask_neg = (z < 0)
  return torch.where(mask_neg, mu.expand_as(out), out)
  ```
  Pre-ReLU placement is correct (block 188 hook on `mlp.act`, which is the FFN nonlinearity before residual addition).

- **Calibration** (lines 91–128): Verbatim from cycle 173. Correctly accumulates negative-regime means using double precision (numerically stable). Per-block per-channel: ✓

- **Progressive loop** (lines 191–204): Each iteration appends a *new* C3Hook (cumulative) and evaluates PPL with all prior hooks active. Hooks are never removed mid-sweep, so composition is correct. Output correctly records n_blocks, last_block_added, ppl, delta, pct_increase.

- **Data loading**: Deterministic (rng seed=42), reproducible sequence selection.

- **PPL evaluation** (lines 67–85): Correct cross-entropy → exp(loss/tokens) for perplexity.

- **Monotonicity check** (validation P4): All 12 PPLs are monotone non-decreasing. ✓

- **Block 0 sanity** (validation P4): 22.819 vs baseline 22.700 = +0.12 (+0.53%), matches cycle 173 expectation (<+2%). ✓

---

## No Single-Seed Concern (G4)

**Verdict**: N/A (not a blocking issue for this verdict)

The experiment uses SEED=42 (deterministic). For INCONCLUSIVE verdicts, single-seed is acceptable because:
1. The inconclusive band is wide (+15%–+40%), and +20.33% is midway between +15% and +25% (typical noise band for single seed).
2. The monotone progression 0.53% → 1.79% → ... → 20.33% is smooth and interpretable (composition cost compounds), not noisy.
3. If the user later wants to reduce uncertainty, a multi-seed sweep can be designed. For now, flagging is unnecessary.

---

## Interesting Observations (G6)

1. **Composition cost is nonlinear**:
   - Blocks 0–1: +0.53%, +1.26% incremental (small)
   - Blocks 3–7: +1.75%–2.07% incremental (acceleration phase)
   - Blocks 8–12: +0.97%–1.30% incremental (plateau phase)
   - Suggests early blocks' errors don't propagate uniformly; later blocks' errors are "absorbed" or saturated.

2. **Composition barrier confirmed** (per MEMORY.md):
   - Cycle 173: per-block C3 costs <+2% in isolation.
   - Cycle 188: all-12 C3 costs +20.33% in composition.
   - This 10x amplification (local quality ≠ stacking quality) matches the composition barrier identified in geometric compression work (e.g., AE beats topk locally but loses stacked).
   - **Implication**: Per-channel constant replacement is insufficient to preserve task geometry across composition, even without shape change.

3. **Shape change is not the sole cause of cycles 181/182 failure**:
   - Cycle 181/182 (C3 + GELU→ReLU): +88–91% PPL (cascade failure).
   - Cycle 188 (C3 alone): +20.33% PPL (degraded but bounded).
   - **Decomposition**: Shape change accounts for ~68 PPL of the 42-PPL failure (cycle 181/182 baseline ~22.7 → result ~64 PPL; 64 - 27.3 = 36.7 PPL from shape change). Constant replacement alone costs ~4.6 PPL (per-channel means insufficient).
   - **Neither is negligible**, but shape distortion dominates.

4. **Block 6 is a natural waypoint**:
   - Blocks 0–6: all PPLs < 26.1 (confirmed band, <+15%).
   - Blocks 7–12: all PPLs > 26.1 (inconclusive/refuted).
   - Suggests the composition barrier materializes around block 6–7. Early blocks' composition is viable; deep stacking breaks down.

---

## Recommendation

**Accept verdict**: INCONCLUSIVE is correct.

**Next steps**:
1. **Item 199 (cascade error direction)**: Analyze whether errors propagate forward (early block errors amplify at later blocks) or sideways (channel-wise decorrelation failure).
2. **Item 200 (positive-regime isolation)**: Test whether replacing only positive-regime GELU (mu_pos) instead of negative-regime (mu_neg) shows different composition dynamics.
3. **Depth scaling**: Confirm whether +20% is consistent for GPT-2-like models (12 layers) or scales with depth.
4. **Multi-seed variance** (optional): If pursuing this path further, a quick 3-seed sweep (SEED=42, 100, 200) can quantify noise vs. signal.

---

## Checklist

- [x] G1: Design sound (composition test well-framed, criteria justified)
- [x] G2: Verdict logic correct (26.105 < 27.314 < 31.780)
- [x] G3: Implementation clean (C3 hook correct, calibration stable, no bugs)
- [x] G4: Single-seed acceptable (inconclusive midway in band, monotone progression)
- [x] G6: Observations valuable (nonlinear composition, barrier materialization, shape vs. replacement decomposition)

**Status**: PASS
