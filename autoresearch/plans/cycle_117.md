# Cycle 117 Plan: Cascade Corrections at k=5 and k=8 — GPT-2 FFN

**Item:** 133 | **Script:** `experiments/cascade_correction_k5_k8_gpt2.py` (new, derived from cycle 115)

## Criteria (verbatim from agenda)
- **Confirmed if:** corrections recover ≥30% at k=5 AND ≥20% at k=8 with r≤16.
- **Refuted if:** <10% recovery at k=5.
- **Inconclusive if:** works at k=5 but fails at k=8, or needs r>16.

## Pre-flights
**P1 -- Prior:** Cycle 115 tested corrections at k=4 only. k=5 and k=8 are untested.
**P2 -- Cost:** 2 k-values × {r=8} × 3 seeds + fallback r=16 at k=8. ~90 min MPS.
**P3 -- Abort:** Float PPL outside [25, 65]. Uncorrected k=5 PPL outside [55, 120].
**P4 -- Param count:** Assert total = k * 2 * 768 * r for each condition.

## Config
- k=5 blocks: `[7, 5, 4, 1, 6]` (best config from cycle 106 audit)
- k=8 blocks: `[7, 5, 4, 3, 2, 1, 0, 6]` (cycle 089/099 order)
- Compression: 6x (hidden_fraction=1/6), topk+lstsq+500 Adam per block
- Corrections: C_l(x) = x + U_l @ (V_l @ x), U_l=0 init (identity at start)
- Correction training: Adam lr=1e-4, 2000 steps, batch=4, seq_len=128, CE loss
- Primary rank: r=8. Fallback: r=16 at k=8 if r=8 < 20% recovery.
- Seeds: [42, 123, 7], eval: 200 seqs OWT

## Steps

### Phase 1: Baselines (1 pass)
1. Load GPT-2, eval float PPL on 200 OWT seqs. Abort if outside [25, 65].
2. Compress k=5 blocks, topk+lstsq+500 Adam. Eval uncorrected_ppl_k5.
3. Compress k=8 blocks, topk+lstsq+500 Adam. Eval uncorrected_ppl_k8.
   Abort if k=5 PPL outside [55, 120] or k=8 PPL outside [100, 1000].

### Phase 2: Corrections (all seeds)
4. For each k in {5, 8}, each seed in {42, 123, 7}:
   a. Deep-copy compressed model. Freeze all params.
   b. Insert LowRankCorrection(768, r=8) on each compressed block.
   c. Train corrections 2000 steps (Adam lr=1e-4, batch=4, CE loss on OWT).
   d. Eval corrected PPL on 200 eval seqs. Log loss curve every 100 steps.
   e. Record: corrected_ppl, param_count.

### Phase 3: Fallback (conditional)
5. If mean recovery at k=8 with r=8 is <20%, re-run Phase 2 for k=8 with r=16.
   Log that fallback was triggered.

### Phase 4: Verdict
6. For each k:
   - degradation = uncorrected_ppl - float_ppl
   - residual = corrected_ppl - float_ppl
   - recovery_pct = (degradation - residual) / degradation * 100
7. Verdict:
   - recovery_k5 >= 30 AND recovery_k8 >= 20 (at r<=16): CONFIRMED
   - recovery_k5 < 10: REFUTED
   - else: INCONCLUSIVE

### Phase 5: Output
8. JSON: `autoresearch/results/cycle_117_cascade_k5_k8.json`
9. Plot: grouped bar (float, uncorr_k5, corr_k5, uncorr_k8, corr_k8) with seed error bars.
10. Manifest: `autoresearch/results/cycle_117_manifest.txt`

## Note on overfitting (from cycle 116 audit)
B/D conditions in cycle 116 reached best PPL ~58.5 at step 1000 before overfitting to 62+
by step 2500. The LoRA parameterization (UV with U=0 init) may provide implicit regularization
that prevents this. Monitor correction training loss curves for signs of U-shape. If present,
the 2000-step budget may still be appropriate since cycle 115 showed 74% recovery at 2000 steps
without degradation.
