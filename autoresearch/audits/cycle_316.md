# Cycle 316 Audit — Item 326 (b10, b11) deep dive

## Audit status: PASS

## Provisional outcome (from execute)
INCONCLUSIVE per item 326's strict three-way verdict tree, with substantive findings sharper than INCONCLUSIVE: M3 (joint Adam saddle) outright REFUTED; M1 (residual stream redundancy) PARTIAL — top-2 directions share the leading residual-stream subspace but the rank-r tail diverges; M2 (LN coupling) indirectly REFUTED by M3 ablation A.

## Verdict logic check (item 326 verbatim)
The brief's verdict tree:
1. `sequential_ppl ≤ 55` → CONFIRMED-M3-pathology
2. `sequential_ppl ≥ 60 AND mean_angle < 30°` → CONFIRMED-M1-redundancy
3. `sequential_ppl ≥ 60 AND mean_angle > 60°` → REFUTED-mechanism-unknown
4. else → INCONCLUSIVE

Measured:
- `sequential_best_ppl` = 61.2059 (≥ 60, NOT ≤ 55)
- `min_mean_angle_deg` = 39.63 (NOT < 30, NOT > 60 — middle band)

Branch 1 fails (61.21 > 55). Branch 2 fails (39.63 ≥ 30). Branch 3 fails (39.63 ≤ 60). Therefore the strict tree returns INCONCLUSIVE. The execute layer's verdict label "INCONCLUSIVE" is correct.

The exec script's logic matches the brief: I traced the threshold constants `M3_PATHOLOGY_LE = 55.0`, `DESTRUCTIVE_GE = 60.0`, `M1_ANGLE_LT = 30.0`, `M1_REFUTE_GT = 60.0` and the four-branch elif chain in `experiments/b10_b11_deep_dive_cycle316.py`. They reproduce item 326 verbatim.

## Cross-checks (all PASS)
- Float PPL 53.8949 vs ref 53.895 (|Δ| = 5e-5; assert <0.5 holds).
- Install init 76.6535 matches cycle 310 init 76.6535 to 4 decimals (assert |Δ| < 0.5 holds).
- Ablation A init 59.6834: this is the b11 r=22 no-retrain init with b10 at float. Cycle 270 reports b11 r=22 noretrain = 59.683 — exact match within rounding.
- Ablation B init 59.9179: this is the b10 r=5 noretrain init with b11 at float. Cycle 282 reports b10 r=5 noretrain ≈ 60 (the cycle 282 single-block diagonal 51.47 is the RETRAINED anchor; the noretrain at r=5 sits much higher). Within band.
- Ablation A best 54.7724 vs cycle 270 retrained reference 54.4687 (Δ+0.30): consistent with the cycle 270 published 4000-step / 3-seed / SEED-set anchor compared to a 2000-step / 1-seed measurement here. The +0.30 PPL gap is within the budget-shortfall band cycle 310 already documented (cycle 310 (1,11) at 2000 steps showed +0.70 PPL vs cycle 284 4000-step matched anchor).
- Ablation B best 51.4783 vs cycle 282 b10 r=5 single-block diagonal 51.4659 (Δ+0.01): essentially exact reproduction.
- Per-phase `|restored - best| < 1e-3`: asserted by `adam_fit` for all 4 Adam runs (phase A, phase B, ablation A, ablation B). PASS.
- Per-block Vh orthonormality: asserted with off-diag tolerance 1e-4 in M1 test. PASS.
- Trainable counts: phase A = 4,608 = 5×768+768 ✓; phase B = 17,664 = 22×768+768 ✓; M3A = 17,664 ✓; M3B = 4,608 ✓.

## Numerical checks I re-ran
1. Reloaded `cycle_316_b10_b11_deep_dive.json`. Verdict logic block reproduces cleanly:
   - sequential_best_ppl 61.2059 ≥ 60: TRUE
   - min_mean_angle 39.63 < 30: FALSE
   - min_mean_angle 39.63 > 60: FALSE
   - sequential_best_ppl 61.2059 ≤ 55: FALSE
   ⇒ INCONCLUSIVE — matches.
2. Re-checked the M1 cosines: raw 5×22 cos = [0.967, 0.865, 0.718, 0.619, 0.539]. arccos(0.967) = 14.77°, arccos(0.865) = 30.10°, arccos(0.718) = 44.13°, arccos(0.619) = 51.79°, arccos(0.539) = 57.38°. Sum = 198.17, mean = 39.63°. Matches.
3. Phase B init = phase A best = 64.6726: this confirms the executor restored phase-A's best state into the model before phase B started (otherwise phase B init would be either the install init 76.6535 or some intermediate value). The phase boundary is correctly handled.
4. Ablation A best step 2000 (last eval) and ablation B best step 1500 (mid-trajectory) — both consistent with Adam still descending at the budget limit.

## Independent interpretation
- **M3 refuted** is unambiguous. Sequential 61.21 exceeds joint 60.72, and the bar for CONFIRMED-M3 (≤55) is missed by 6 PPL — well outside any noise envelope. The brief's hypothesis that "joint Adam can't escape a saddle the sequential fits would" is wrong: sequential is strictly worse, plausibly because phase A's b10 fit shapes the residual stream around an upstream-conditioned target that phase B's b11 fit then has to undo from a worse starting position than joint Adam's intermediate state.
- **M1 partial** is the most informative finding. The top-1 angle of 14.77° (cos 0.967) is a strong signal that b10 and b11 share a near-identical leading direction in the residual stream. The top-2 angle of 30.10° (cos 0.865) is right at the M1 boundary. Beyond rank 2 the alignment drops sharply (rank 3 at 44.14°, rank 4 at 51.79°, rank 5 at 57.38°). This is consistent with "the two latest concentrators write the same 1-2 leading directions" but not with "the rank-5/22 subspaces are essentially the same". The rank-r mean's 39.63° sits between because the top-2 strong overlap is averaged with the mid/tail divergence. The brief's threshold (<30° on the rank-r mean) is too tight for this geometry; a "top-1 or top-2 below 30°" rule would have classified this as M1.
- **M2 (LN coupling) indirectly refuted by ablation A**: if LN coupling were the dominant mechanism, b11's r=22 retrain on top of float-b10 should still be poisoned (because the LN normalises a perfectly correct upstream and maps it to the same target distribution). Ablation A shows b11 reaches 54.77 vs cycle 270 anchor 54.47 — essentially perfect single-block performance with float upstream. So LN coupling with float b10 is not the issue. The damage only appears when b10 is ALSO compressed. This is consistent with the M1 reading "shared subspace competition" rather than the M2 reading "LN normalisation propagates b10 noise to b11".
- **Cross-config table** in the manifest is sharp:
  - float                       53.8949
  - b10 r=5 alone               51.4783 (-2.42 vs float — bottleneck removal!)
  - b11 r=22 alone              54.7724 (+0.88)
  - joint cycle 310             60.7161 (+6.82)
  - sequential                  61.2059 (+7.31)
  Each block alone is fine (b10 alone improves on float; cycle 282 already documented this regularisation effect). The joint result is +6.82 above float, +9.24 above the better single config. The destruction is purely interactive.

## Audit verdict on the substantive interpretation
The execute layer's substantive findings (1)-(6) in the manifest are accurate and well-supported by the data. In particular:
- "M3 outright REFUTED" — supported by sequential 61.21 vs threshold 55 (6 PPL margin).
- "M1 partial" — supported by the top-2 cosines (0.967, 0.865) crossing the 30° boundary even though the rank-r mean does not.
- "M2 indirectly refuted by ablation A" — supported by ablation A reaching cycle 270's anchor within 0.30 PPL.
- "destruction is purely interactive" — supported by both ablations reaching their single-block anchors.

## Auditor recommendation
PASS the INCONCLUSIVE verdict. The execute and manifest layers are honest about the strict-tree outcome and explicit about the substantive sharper findings. No corrections needed to the verdict label, the manifest, the agenda PAST entry, the JSON, or the experiment script.

The natural follow-up (for a future cycle) is to widen the M1 test to either:
(a) a "top-k angle" rule (top-1 or top-2 below 30°), OR
(b) a higher-rank b10 or b11 single-fit measurement to test whether the destruction shifts when one block has more residual-stream capacity, OR
(c) a 2D rank grid (b10 ∈ {5, 10, 22}, b11 ∈ {22, 44, 88}) to localise the rank threshold at which the destruction disappears.

These should be filed as new agenda items, not retroactive amendments to cycle 316.

## Status: PASS (provisional outcome accepted unchanged)
