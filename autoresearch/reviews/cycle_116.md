verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim. Verdict logic (plan step 8): relative gap computed as
|ppl_C - ppl_X| / ppl_C * 100 for X in {B, D}. >=5% both → CONFIRMED, either within
2% → REFUTED, 2-5% → INCONCLUSIVE. Matches agenda. Pass.

## R2: Measurement soundness
All four conditions start from the same compressed checkpoint (topk+lstsq, no polish).
This ensures the only variable is the optimization strategy. Condition D resets the
optimizer at step 500 to match C's two-phase structure. PPL is evaluated on the same
200 eval seqs across all conditions and seeds. Pass.

## R3: Known pitfalls
- **MP3 (underdetermined fitting):** Condition C trains 49,152 correction params on
  2000 steps * 4 seqs * 128 tokens = 1,024,000 tokens. Ratio ~20:1. Not triggered.
  Conditions B and D train existing W_down params — no new params. Not triggered.
- **MP5 (insufficient capacity):** All conditions build on the working k=4@6x recipe.
  Condition A is the established baseline (~80 PPL). Not triggered.
- **MP6 (verdict misclassification):** The 5%/2% relative PPL thresholds are applied to
  values around 59-80 PPL. A 5% gap at PPL 59 is ~3 PPL points, well above seed-to-seed
  variance (~7 PPL std from cycle 115, but that was absolute — the relative ordering
  across conditions within a seed should have much lower variance). Not triggered.
- **MP1, MP2, MP4, MP7, MP8:** Not applicable.

## R4: Cost and feasibility
4 conditions x 3 seeds. Most expensive is B (2500 Adam steps per block per seed).
Total ~90 min MPS. Reuses cycle 115 infrastructure. Feasible.

## R5: Prior probability assessment
Cycle 115 showed corrections saturate at r=4 (72.6%) vs r=8 (74.3%) — only 1.7pp
improvement from doubling rank. This suggests the correction is capturing a low-dimensional
signal, which extended polish on W_down could plausibly also capture. On the other hand,
corrections act on the residual stream (inter-block), while W_down polish acts intra-block
— different error channels. Estimated prior: ~40% confirmed, ~30% refuted, ~30% inconclusive.

## Advisory

**A1 (Block config mismatch).** Agenda item 132 specifies blocks [1,4,7,11]. Cycle 115
used [7,5,4,1]. Plan correctly uses [1,4,7,11]. The implementation must match — do not
copy cycle 115's block list. Verify that the uncorrected baseline (condition A) PPL is
in the expected range (~80 PPL) with these blocks.

**A2 (Optimizer reset for D).** Plan specifies optimizer reset at step 500 for condition D.
This is important: if the optimizer accumulates momentum from the first 500 steps, D's
second phase has an unfair advantage (warm start) vs C's correction training (cold start
on new params). The reset makes D and C's second phases comparable. Verify in implementation.

**A3 (Seed-to-seed variance).** Cycle 115 reported std ~7-10 PPL across seeds, but this is
dominated by eval noise on 200 seqs. The relative ordering (C vs D within same seed) should
have much lower variance. Still, report per-seed results alongside means for transparency.
