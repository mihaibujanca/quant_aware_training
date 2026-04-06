# Cycle 224 Audit — Item 229 (KD distillation baseline, W_out+W_fc)

**Status: CORRECTED (REFUTED → INCONCLUSIVE — method failure per G6)**

## Verdict check
- Criteria: best_mean ≤ 44.0 → CONFIRMED; ≥ 47.0 → REFUTED; else INCONCLUSIVE.
- Reported: best_mean = 60.938 ± 0.103 (seeds 61.075 / 60.915 / 60.825), init = 61.746.
- Literally 60.938 ≥ 47.0 → REFUTED.
- BUT the metric did not vary in the right range: best is only 0.6–0.9 PPL below init
  across all 3 seeds. The optimizer barely moved off init. G6 method failure: a verdict
  that depends on the trainable surface staying frozen does not refute the hypothesis,
  it refutes the loss formulation.

## Root cause — KD loss normalization
- `experiments/distillation_baseline_cycle224.py:221`:
  `kd_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (TAU ** 2)`
- Student/teacher logits are shape `(B, T, V) = (1, 128, 50257)`. With `reduction='batchmean'`
  PyTorch divides only by the leading dim `B=1`, so the returned KL is the SUM over
  T·V = 6,432,896 entries. CE meanwhile uses `F.cross_entropy(... reshape(-1, V), ...)`
  which averages over B·T = 128 tokens.
- Empirical scale (per executor): kd_loss ≈ 30–50, ce_loss ≈ 4. The 0.5/0.5 weighting is
  effectively ~7–12× more KD than CE. Adam steps are dominated by a KL gradient against
  softened teacher logits at τ=2 (a weak target), so the W_out/W_fc surface barely moves.
- Best PPL minus init: −0.67 / −0.83 / −0.92 across seeds — consistent with "no real
  optimization happened on this loss".

## Evidence the run is method-bound, not hypothesis-bound
- Same trainable surface under pure CE (cycle 213): 46.424 PPL — proves the surface CAN
  reach the threshold. The loss, not the parameter set, is the bottleneck.
- Best-step distribution: 2500 / 5000 / 3000 (mean 3500 ± 1080), with no monotone descent
  in any seed history — pure noise around init, not learning.

## Decision (G6)
- Downgrade REFUTED → INCONCLUSIVE. The result does NOT prove "KD cannot beat CE on
  this surface"; it proves "this particular un-normalized KD+CE mix does not train".
- Document the gotcha so the next attempt fixes the loss before re-running.

## Recommendations
- Reshape logits to `(B*T, V)` before `F.kl_div` (or use `reduction='mean'`, or normalize
  by `T*V`). Then the KD term is per-token KL, comparable in scale to CE.
- Alternative: weight the existing call by `1 / (T * V)` and recover token-mean KL.
- After normalization, sweep KD/CE ratios (0.9/0.1, 0.5/0.5, 0.1/0.9) and τ ∈ {1, 2, 4}
  on a short budget before committing to a 10k-step run.
- Verify Δ vs init is at least comparable to cycle 213's CE descent before declaring a verdict.

## Bookkeeping changes
- agenda.md: Cycle 224 PAST entry → INCONCLUSIVE (corrected from REFUTED, method failure).
- cycle_224_manifest.txt: outcome → inconclusive; learned/next updated.
- experiments.tsv row 224: outcome refuted → inconclusive.
- flagged.jsonl: appended audit entry.
