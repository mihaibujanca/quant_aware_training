verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim. Verdict logic (plan step 7): reduction_pct = (degrad_base -
degrad_corrected) / degrad_base * 100. >=30% at r<=16 -> CONFIRMED, <10% at r=16 -> REFUTED,
else INCONCLUSIVE. Matches agenda. Pass.

## R2: Measurement soundness
PPL comparison with corrections vs without, same compressed baseline. Corrections are additive
to the residual stream (identity at init), trained under CE loss. The degradation metric is
relative to float PPL, which is the right reference. Param counts are explicit and verified.
Pass.

## R3: Known pitfalls
- **MP3 (underdetermined fitting):** r=16 correction per block: 2*768*16 = 24,576 params.
  Trained on 2000 steps * 4 seqs * 128 tokens = 1,024,000 tokens. Total correction params
  across 4 blocks: 98,304. Overdetermination ratio ~10:1. Not triggered.
- **MP5 (insufficient capacity):** The base compression (k=4@6x) is the established working
  recipe (~+5% PPL). The corrections are tested on top of a regime that works. Not triggered.
- **MP6 (verdict misclassification):** PPL is informative and varies across conditions. The
  uncorrected baseline has measurable degradation (~+5% = ~2.5 PPL points). A 30% reduction
  would be ~0.75 PPL points — small but above noise with 3 seeds. If degradation_base is
  too small to measure reductions reliably, the experiment is INCONCLUSIVE, not REFUTED. Pass.
- **MP1, MP2, MP4, MP7, MP8:** Not applicable (no detection thresholds, no activation regime
  mismatch, no domain transfer, no post-LayerNorm geometry, no representation vs functional
  rank confusion).

## R4: Cost and feasibility
1 compression run + 3 ranks x 3 seeds x 2000 CE steps. ~60 min MPS. Reuses standard
compression infrastructure. Feasible.

## R5: Prior probability assessment
Cycle 063 tested full affine lstsq corrections (d x d) and found some benefit but with
overfitting risk. This uses CE-trained low-rank corrections — different optimization target
(task loss vs MSE) and constrained rank. The 85%-of-dims error span (cycle 099) works
against low-rank corrections, but CE training may find the ~15% of directions that matter
for task loss. Estimated prior: ~35% confirmed, ~35% refuted, ~30% inconclusive.

## Advisory

**A1 (Small degradation baseline).** If k=4@6x degradation is only ~2.5 PPL points, a 30%
reduction is ~0.75 points. With 3 seeds, confirm that seed-to-seed variance is smaller than
this. If variance > 1 PPL point, the experiment may be underpowered. Consider adding a
diagnostic: if std(uncorrected_ppl) > 0.5 * degradation_base, flag as underpowered.

**A2 (Hook placement).** Plan specifies corrections on the residual stream after the block
output add. This is correct — it avoids the pre-ReLU pitfall (MEMORY.md). But verify that
the hook fires after LayerNorm + Attention + FFN + residual, not just after FFN inside the
block. GPT-2 block output = residual + attn + ffn, so hooking the block output captures the
full residual. Confirm in implementation.
