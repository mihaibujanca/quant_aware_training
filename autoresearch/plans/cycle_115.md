# Cycle 115 Plan: Per-Block Cascade Correction — GPT-2 FFN

**Item:** 118 | **Script:** `experiments/cascade_correction_stacking_gpt2.py` (new)

## Criteria (verbatim from agenda)
- **Confirmed if:** cascade corrections reduce stacked PPL degradation by >=30% relative at k=4 with r<=16.
- **Refuted if:** corrections reduce degradation by <10% even at r=16.
- **Inconclusive if:** 10-30% reduction, or requires r>16.

## Pre-flights
**P1 -- Prior:** Cycle 063 tested full affine (d x d) lstsq corrections — different mechanism
(lstsq vs CE-trained, full-rank vs low-rank). This is genuinely new: CE training finds
task-relevant directions, not MSE-minimizing ones.
**P2 -- Cost:** 1 baseline compression + 3 ranks x 3 seeds x ~2000 CE steps = moderate. ~60 min MPS.
**P3 -- Abort:** Baseline PPL outside [25, 65]. Any corrected PPL > 2x baseline float.
**P4 -- Param count:** Assert param_count == k * 2 * 768 * r for each rank.

## Implementation

1. **Baseline.** Load GPT-2, eval 200 seqs OWT. Record float_ppl. Abort if outside [25, 65].

2. **Compress k=4.** Standard recipe: blocks [7,5,4,1] at 6x, topk+lstsq+500 Adam per block.
   Eval stacked compressed model. Record uncorrected_ppl. This is the baseline for % reduction.

3. **Insert corrections.** For each compressed block l, add C_l applied to the residual stream
   after the block's output is added back. C_l(x) = x + U_l @ (V_l @ x), where U_l is
   (768, r) and V_l is (r, 768). Initialize U_l = 0 so C_l = identity at start.

4. **Train corrections.** Freeze all model params. Train only {U_l, V_l} for l in compressed
   blocks. Optimizer: Adam lr=1e-4, 2000 steps, batch_size=4, seq_len=128. Loss: next-token CE
   on OWT train data. Log loss curve every 100 steps.

5. **Eval.** For each rank r in {4, 8, 16}, repeat steps 3-4 with 3 seeds. Eval corrected PPL
   on 200 eval seqs.

6. **Param verification.** Per rank: total_correction_params = 4 * 2 * 768 * r.
   r=4: 24,576. r=8: 49,152. r=16: 98,304. Log and assert.

7. **Verdict logic.**
   - degradation_base = uncorrected_ppl - float_ppl (absolute PPL increase from compression).
   - For each rank: degradation_corrected = corrected_ppl - float_ppl.
   - reduction_pct = (degradation_base - degradation_corrected) / degradation_base * 100.
   - Best rank = rank with highest reduction_pct among r <= 16.
   - If best reduction_pct >= 30%: CONFIRMED.
   - If best reduction_pct < 10%: REFUTED.
   - Else: INCONCLUSIVE.

8. **Output.**
   - JSON: `autoresearch/results/cycle_115_cascade_correction.json`
   - Plot: bar chart of PPL (float, uncorrected, r=4, r=8, r=16) with error bars across seeds.
   - Training loss curves per rank.
   - Manifest: `autoresearch/results/cycle_115_manifest.txt`

## Hook placement detail
The correction C_l is applied to the *residual stream* after the compressed FFN block's
contribution is added. Concretely, hook into the block's output (after residual add) and
apply x_new = x + U_l @ (V_l @ x). This is equivalent to a rank-r perturbation of the
identity on the residual stream. It does NOT go inside the FFN or before any nonlinearity.
