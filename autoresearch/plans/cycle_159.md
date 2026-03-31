# Task
Cycle 159 / Item 169: Correction-first compression. Reverse the pipeline: train LoRA corrections on the compressed model first, extract the effective correction matrix C, compute residual R = W_float - C, then compress R via topk+lstsq. Compare to current best (compress then correct).

# Hypothesis
- **Confirmed if:** correction-first PPL < 57 (beats current best ~59 by >=3%).
- **Refuted if:** PPL > 62 or corrections can't transfer to the modified compressed weights.
- **Inconclusive if:** 57-62 (marginal), OR lstsq residual on corrected target > 2x baseline residual (see Validation).

# Pre-flight Results
- P1-P7: all PASS (see brief for details).
- Baselines: topk+lstsq+polish = 70.7, topk+lstsq+corrections = 59.0, float = 51.7.

# Steps
1. **Copy scaffold** from `experiments/cascade_correction_stacking_gpt2.py`. New file: `experiments/correction_first_cycle159.py`.
2. **Phase A — Baseline compress + correct.** topk+lstsq k=4 blocks [7,5,4,1] at 6x, 500 Adam polish, then per-block LowRankCorrection r=8, 2000 Adam steps lr=1e-4. Record PPL per seed. After training, for each block `b`, extract correction matrix: `C_b = LoRA_B.weight.T @ LoRA_A.weight.T` (shape 768x768). Verify orientation: on a probe batch, assert `(W_compressed + C_b) @ x ≈ corrected_output` (cosine > 0.99). If assertion fails, transpose C_b and re-check.
3. **Phase A diagnostic — OOD check.** For each block, compute cosine similarity between compressed-model residuals and float-model residuals on the probe batch. Report per-block. If cosine < 0.5 for any block, corrections are OOD for float residuals: skip Phase B first-order, go directly to iterative fallback (step 5).
4. **Phase B — Correction-first compression.** For each block `b`: collect float activations X and float FFN output Y_float. Compute `Y_target = Y_float - C_b @ (residual_in_float + Y_float)` (first-order approx). Fit topk+lstsq k=4 to Y_target. Record lstsq residual. Compare to Phase A lstsq residual on original Y_float — if ratio > 2x for any block, verdict = INCONCLUSIVE with the specific ratio.
5. **Phase B eval.** Swap correction-first compressed FFNs into model, apply same corrections, eval PPL on 200 OWT seqs, 3 seeds.
6. **Iterative fallback** (if Phase A diagnostic triggers OOD, or Phase B PPL > 80). Re-train corrections on the new compressed model from Phase B. Re-extract C', re-compute R' = W_float - C', re-compress. Cap at 2 iterations.
7. **Report**: Save `autoresearch/results/cycle_159_correction_first.json` with both pipelines' PPL, per-seed values, correction norms, per-block cosine similarities, lstsq residual ratios. Save bar chart to `autoresearch/results/cycle_159_correction_first.png`.
8. **Agenda update**: Move item 169 from NEXT to PAST in `autoresearch/agenda.md`.
9. **Commit**: `[autoresearch] cycle 159: correction-first compression`

# Validation
- Phase A PPL must be within 2% of known baseline (~59.0) to confirm protocol fidelity.
- Phase A correction matrices C_b must have non-trivial norm (>0.01).
- Correction orientation verified on probe batch (step 2).
- Per-block cosine similarity between compressed/float residuals reported (step 3). Cosine < 0.5 triggers iterative fallback.
- lstsq residual gate: if Phase B residual > 2x Phase A residual on any block, verdict = INCONCLUSIVE (not REFUTED), with the specific ratio reported.

# Artifacts
- `experiments/correction_first_cycle159.py`
- `autoresearch/results/cycle_159_correction_first.json`
- `autoresearch/results/cycle_159_correction_first.png`

# Fallbacks
- OOD corrections (cosine < 0.5) or PPL > 80: iterative re-training (step 6), cap 2 iterations.
- Corrections don't transfer (PPL > 100 with correction-first weights): REFUTED — correction subspace is coupled to specific compression.
