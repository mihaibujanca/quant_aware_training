# Cycle 099 Plan: Inter-Block Residual-Stream Projection Correction (GPT-2 FFN)

**Item:** 112
**Script:** `experiments/residual_projection_correction_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** Residual-stream error after compressed FFN blocks is diffuse (not low-rank), so projection correction at r<=16 removes <10% of error energy and does not reduce cascade rate.

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: c drops to ≤0.25 at r≤16, enabling k=8 at ≤+30% PPL (vs current +914%).
- Refuted if: c ≥0.35 at all r (error is not concentrated in a low-rank subspace of the downstream input).
- Inconclusive if: c drops modestly (0.25-0.35) — helps but doesn't break the wall.

**P3 -- Novelty vs replication:** Cycles 089/093/094 established c~0.41 and showed Adam polish does not reduce it. This is the first no-training inter-block correction attempt.

**P4 -- Confounds:** (1) LayerNorm after each block may redistribute error across dimensions, inflating apparent rank. Measure error rank BOTH pre-LN and post-LN. (2) Probe data distribution matters: use OWT calibration (same as polish cycles). (3) Projection removes error but may also remove useful signal — track both error energy removed and signal energy removed.

**P5 -- Cost estimate:** 1 forward pass for float refs, 4 ranks × (k=4 + k=8) = 8 stacking evals, plus cascade refit. ~15 min on MPS.

**P6 -- Data dependencies:** `data/openwebtext_sample.txt` (same as cycles 089/093/094).

**P7 -- Abort conditions:** Baseline PPL outside [25, 65]. Projection makes PPL worse than uncorrected (signal destruction exceeds error removal).

## Implementation

1. **Baseline.** Load GPT-2, eval PPL on OWT, 200 seqs, 3 seeds [42, 123, 7]. Verify ~51.66.

2. **Collect error vectors.** For k=4 blocks [7,5,4,1] at 6x (topk+lstsq+500 Adam polish):
   - Hook residual stream AFTER each compressed block's FFN output (pre-LN).
   - Run same 200 sequences through both float and compressed models.
   - Compute E_i = residual_compressed - residual_float at each block boundary.
   - E_i shape: (n_tokens, 768). Store per block.

3. **Error rank analysis.** For each block i:
   - SVD of E_i: U, S, V = torch.linalg.svd(E_i).
   - Report: rank at 50%, 80%, 90%, 95% energy. Plot singular value spectrum.
   - If rank_95 > 32 at most blocks: error is diffuse, expect refutation.

4. **Build projection corrections.** For r in {4, 8, 16, 32}:
   - P_i = I - U_i[:, :r] @ U_i[:, :r].T (project out top-r error directions).
   - Insert P_i as a linear layer (no bias, frozen) after block i's FFN output, before LN.

5. **Evaluate corrected PPL.** For each r:
   - k=4 blocks [7,5,4,1]: eval PPL with projections inserted.
   - k=8 blocks [7,5,4,3,2,1,0,6] (cycle 089 order): compress all, insert projections, eval.

6. **Refit cascade rate.** For the best r:
   - Run accumulation curve (1, 2, ..., k blocks) with projections.
   - Fit c from log-PPL vs k. Compare to uncorrected c~0.41.

7. **Verdict logic.**
   ```
   c_best = min(c across r in {4,8,16,32})
   r_best = argmin r such that c <= 0.25
   ppl_k8_corrected = PPL at k=8 with projections at r_best
   delta_k8 = (ppl_k8_corrected - baseline) / baseline * 100

   if c_best <= 0.25 and r_best <= 16 and delta_k8 <= 30.0:  CONFIRMED
   elif c_best >= 0.35 for all r:                              REFUTED
   else:                                                        INCONCLUSIVE
   ```

8. **Output:**
   - JSON: `autoresearch/results/cycle_099_residual_projection.json` — per-block error SVD spectra, per-r PPL values (k=4, k=8), fitted c values, verdict.
   - Plot 1: singular value spectra per block (4 lines, one per compressed block).
   - Plot 2: PPL vs r for k=4 and k=8 (two lines). Save as `..._residual_projection.png`.

9. **Agenda update + commit** per standard autoresearch flow.
