verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim in plan header. Verdict logic (step 8) maps directly:
gain_B >= 3% in >= 2 eligible blocks -> CONFIRMED; all |gain_B| <= 1% -> REFUTED;
else INCONCLUSIVE. Matches agenda. Pass.

## R2: Measurement soundness
PPL is the gold-standard end-to-end metric. gain_B = relative PPL improvement, well-defined.
The probe/fit split (step 2-3 vs 4-6) prevents b_leak from seeing lstsq data. Condition C
isolates filtering from bias correction. The lstsq target fix (y - b_leak for B, not y)
is the whole point of the corrected item and is correctly specified in step 5. Pass.

## R3: Known pitfalls
- **MP3 (underdetermined fitting):** lstsq fits 512 x 768 = 393K params from 100 seqs x 128 tokens =
  12800 samples x 768 outputs. 12800 > 2 x 512 (input dim). Not triggered.
- **MP7 (post-LayerNorm):** Channel census measured on pre-GELU activations (post-W_in).
  b_leak computed from same. Not post-LN geometry. Not triggered.
- **MP1 (trivial threshold):** neg_frac threshold is 0.5 (natural split). Not triggered.
- **MP5 (insufficient capacity):** All conditions use same k (or matched k_eff). Fair. Not triggered.
- **MP6 (verdict misclassification):** PPL varies meaningfully across conditions (prior: cycle 105
  plan identified this experiment as informative). Not triggered.
- **MP2, MP4, MP8:** Not applicable.

## R4: Cost and feasibility
36 runs, ~25 min on MPS. Reuses topk+lstsq+Adam infra from prior cycles. Feasible.

## R5: Verdict logic soundness
Three-way verdict with clean gap (1% refuted ceiling, 3% confirmed floor). Eligible-block
exclusion for N_pos < 256 is conservative. gain_C reported alongside for attribution. Sound.

## Advisory

**A1 (k_eff matching).** If N_pos < 512, Condition A also uses k_eff (step 2 abort guard).
This means A is artificially weakened relative to its unconstrained capacity. The comparison
is fair at matched k but the absolute PPL numbers for that block won't match the known
k=512 baseline. Not blocking -- the relative comparison (B vs A) is still valid.

**A2 (neg_frac variability across blocks).** Cycle 111 showed block 1 has neg_frac = 73%
vs ~85% for others. This means block 1 has more positive channels (~810 vs ~460 for deep
blocks). Some deep blocks might hit the N_pos < 512 guard. Worth logging in results.
