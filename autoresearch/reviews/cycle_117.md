verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim in plan. Verdict logic (step 7) maps directly:
recovery_k5 >= 30 AND recovery_k8 >= 20 at r<=16 -> CONFIRMED, recovery_k5 < 10 -> REFUTED,
else INCONCLUSIVE. The fallback (r=16 at k=8) is correctly gated on r=8 failing the 20%
threshold, consistent with the agenda's r<=16 allowance. Pass.

## R2: Measurement soundness
Baselines and corrections share the same compressed model per seed (deep-copy for corrections).
Eval data (200 seqs) is disjoint from training data. Recovery fraction is computed relative to
the same float baseline. Abort bounds prevent nonsensical results from propagating. Pass.

## R3: Known pitfalls
- **MP3 (underdetermined fitting):** k=5 r=8: 5*2*768*8 = 61,440 params trained on
  2000*4*128 = 1,024,000 tokens. Ratio ~17:1. k=8 r=8: 8*2*768*8 = 98,304 params on same
  tokens. Ratio ~10:1. Both above 2x. Not triggered.
  k=8 r=16 fallback: 196,608 params on 1M tokens. Ratio ~5:1. Marginal but above 2x. Flag.
- **MP5 (insufficient capacity):** At k=8, 8/12 FFN blocks are compressed at 6x. The model
  may have insufficient remaining capacity for the task itself (not just the correction).
  However, the corrections add capacity back. The abort bound (k=8 uncorrected PPL < 1000)
  catches catastrophic capacity loss. Not triggered.
- **MP6 (verdict misclassification):** Recovery fraction is well-defined (degradation > 0
  guaranteed by compression). The 30%/20%/10% thresholds produce distinct ranges. At k=8,
  even 20% of a ~470 PPL gap is ~94 PPL improvement — easily detectable above seed noise.
  Not triggered.
- **MP1, MP2, MP4, MP7, MP8:** Not applicable (no thresholds, no activation-regime mismatch,
  same domain as prior cycles, no post-LayerNorm measurements, no dimension/rank confusion).

## R4: Cost and feasibility
2 k-values x 3 seeds x 2000 correction steps = 6 training runs. Plus optional r=16 fallback
(3 more runs). Reuses cycle 115 infrastructure. ~90 min MPS. Feasible.

## R5: Prior probability assessment
k=5 has moderate uncorrected degradation (+22-24%). Cycle 115 showed 74% recovery at k=4.
Recovery likely decreases with more blocks but 30% is a low bar — estimated 70% confirmation
at k=5. At k=8, uncorrected PPL ~524 (+914%). The correction must absorb error from 8 blocks
with only rank-8 per block. Estimated 40% that r=8 meets 20%, rising to 55% with r=16 fallback.
Overall: ~50% confirmed, ~15% refuted, ~35% inconclusive.

## Advisory

**A1 (Block list verification).** k=5 uses [7,5,4,1,6] and k=8 uses [7,5,4,3,2,1,0,6].
Verify these are passed correctly to compression and correction insertion. The k=5 list
must NOT include block 3 (that was the other config from cycle 106).

**A2 (k=8 compression order).** All 8 blocks compressed independently (per-block polish),
then stacked. The stacking order matters for the uncorrected baseline. Use the cycle 089
order for consistency with prior results.

**A3 (Correction training stability at k=8).** With 8 corrections all training simultaneously
on CE loss, gradient interactions could cause instability. Monitor for loss spikes. If training
diverges at k=8 but not k=5, reduce lr to 5e-5 before declaring failure.
