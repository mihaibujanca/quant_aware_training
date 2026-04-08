# Cycle 309 Audit — Item 319

**Audit status: PASS (with substantive correction)**

The literal verbatim verdict CONFIRMED is correct. The interpretive claim
attached to CONFIRMED ("Δy basis is the right basis at b8, W_proj basis is
the wrong one") is rejected by the cycle's own paired baseline. Auditor
endorses both findings: CONFIRMED-by-threshold and comparative-claim-REFUTED.

## Verdict verification

Item 319 verbatim conditions:
- **CONFIRMED** iff r=1 Δy-aligned PPL ≤ 60 ("Δy basis is the right basis at
  b8, W_proj basis is the wrong one")
- **REFUTED** iff r=1 Δy-aligned PPL ≥ 110 ("the b8 53% unprojected energy is
  not recoverable from any rank-1 cytoplasm; b8 needs higher rank regardless
  of basis")
- **INCONCLUSIVE** otherwise

Measured (from `cycle_309_block8_dy_aligned_laser.json::verdict_logic`):
- dy_r1 retrain best_mean = **54.3185** PPL
- 54.3185 ≤ 60.0 → CONFIRMED. Margin: 5.68 PPL below threshold.
- All 3 dy_r1 seeds converged to {54.3142, 54.3169, 54.3245}; std=0.0044.
  No seed is anywhere near 60. The CONFIRMED literal-threshold conclusion
  is robust.

The script's threshold logic (lines 461–466 of the experiment script):
```
if dy_r1_best_mean <= CONFIRMED_RETRAIN_LE:    # 60.0
    verdict = "CONFIRMED"
elif dy_r1_best_mean >= REFUTED_RETRAIN_GE:    # 110.0
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
```
matches the verbatim wording exactly. Auditor independently re-derives
CONFIRMED from the JSON. ✓

## Comparative-claim correction

The verbatim CONFIRMED branch makes the *interpretive* claim "Δy basis is
the right basis at b8, W_proj basis is the wrong one". This is a comparative
statement; the verdict threshold tests it only one-sidedly (Δy-aligned
absolute PPL ≤ 60). The plan and the cycle 306 brief required a paired
W_proj-basis baseline in the same cycle to actually adjudicate the comparison.

Paired baseline measurements:

| basis | r | no-retrain | retrain best_mean ± std | Δfloat |
|---|---:|---:|---:|---:|
| dy   | 1 | 61.6614 | 54.3185 ± 0.0044 | +0.424 |
| dy   | 2 | 61.1611 | 53.6040 ± 0.0042 | −0.291 |
| wproj | 1 | 60.8573 | 53.3034 ± 0.0071 | −0.591 |
| wproj | 2 | 60.4958 | 52.8538 ± 0.0126 | −1.041 |

- wproj_r1 (53.3034) beats dy_r1 (54.3185) by **+1.02 PPL**.
- wproj_r2 (52.8538) beats dy_r2 (53.6040) by **+0.75 PPL**.
- All seed std ≤ 0.013. The wproj > dy ordering is robust across seeds.

The CONFIRMED branch's interpretive claim is rejected. The substantive
finding is that **the basis change does not close the gap; the W_proj
basis is the better choice at both ranks**. This is a stronger statement
than the verbatim CONFIRMED label, and is the correct framing for follow-up
work.

Per program.md §5 (audit), the audit status is PASS (the verbatim verdict
is correct as stated and the cycle ran cleanly), with the substantive
finding upgraded in the manifest, exec_result, and PAST entry to reflect
the comparative refutation. The verdict label in the agenda PAST entry is
"CONFIRMED verbatim, comparative claim REFUTED" — the literal-threshold
verdict is preserved and the comparative correction is added.

## Reproduction fidelity

### Float PPL vs cycle 270/280/305 reference
- Float PPL on eval slice = **53.8949**.
- Reference: 53.895 (cycles 270, 280, 281, 282, 285, 288, 289, 292, 305).
- |delta| < 0.001 << 0.5 (asserted in script). ✓

### Δy at b8 vs cycle 306 reference
- top1_E measured: 0.9141 (4 dp).
- top1_E cycle 306 ref: 0.9141 (4 dp).
- |delta| < 1e-3 (asserted in script). ✓

### W_proj_8 SVD top-5 vs cycle 306 reference
- Measured: [31.0758, 24.4946, 18.5351, 16.7729, 16.4667]
- Cycle 306 (`cycle_306_block8_deep_dive.json::per_block.8.wproj_singvals_top10[:5]`):
  [31.0758, 24.4946, 18.5351, 16.7729, 16.4667]
- Exact match. ✓

### Δy SVD top-5 vs cycle 306 reference
- Measured: [5316.4561, 327.3077, 239.6714, 226.7930, 216.3731]
- Cycle 306: [5316.4561, 327.3077, 239.6714, 226.7930, 216.3731]
- Exact match. ✓

### LowRankCProj wproj branch vs cycle 280 init
- wproj_r1 init PPL = 60.8573 (from `noretrain.wproj_r1.ppl`).
- This is the no-retrain rank-1 SVD truncation of c_proj_8.weight,
  bit-for-bit identical construction to cycle 280's no-retrain init for
  block 8 r=1.
- Per-seed retrain init PPLs all equal 60.8573 within 1e-2 (asserted
  per seed in script). ✓

## Geometric construction check

For each (basis, r):
- dy: `A_y = W_proj_cpu @ Vhy[:r, :].T` (shape (3072, r), frozen);
       `B_init = Vhy[:r, :].clone()` (shape (r, 768), trainable);
       At init: `A_y @ B_init = W_proj_cpu @ Vhy[:r].T @ Vhy[:r]` =
       orthogonal projection of W_proj's columns onto span(Vhy[:r]).
- wproj: `A_w = (Uw[:, :r] * Sw[:r])` (shape (3072, r), frozen);
         `B_init = Vhw[:r, :].clone()` (shape (r, 768), trainable);
         At init: `A_w @ B_init = Uw[:, :r] @ diag(Sw[:r]) @ Vhw[:r]` =
         standard cycle 280-style rank-r SVD truncation of W_proj.

Both branches use the same `LowRankCProj` template (cycle 270/280/292):
forward = `addmm(bias, x.view(-1, 768), A @ B)`; A is a buffer (frozen);
B and bias are trainable Parameters. Trainable param count per fit:
- r=1: 1*768 + 768 = 1,536 (asserted)
- r=2: 2*768 + 768 = 2,304 (asserted)

Auditor confirmed via JSON `retrain_cells.{cell}.n_trainable` field. ✓

## Cross-basis projection sanity (Vh_y vs W_proj energy)

The script reports the Frobenius energy fraction of W_proj that lives in
span(Vh_y[:r]):
- r=1: ||W_proj @ Vhy[:1].T @ Vhy[:1]||_F² / ||W_proj||_F² = **0.0127** (1.27%)
- r=2: ||W_proj @ Vhy[:2].T @ Vhy[:2]||_F² / ||W_proj||_F² = **0.0220** (2.20%)

This is the geometric explanation for the comparative refutation. Δy is
rank-1 dominated on its own SVD (top1_E=0.914), but the rank-1 axis Vh_y[:1]
points into W_proj's *low-energy tail* on the input side (W_proj's top-5
SVs are [31.08, 24.49, 18.53, 16.77, 16.47], Δy's top-5 SVs are [5316,
327, 240, 227, 216] — the ratios are 5316/31 ≈ 170 for Δy/W_proj at the
top, 216/16 ≈ 13 at rank 5; the Δy spectrum is concentrated and W_proj's
spectrum is flat).

The geometric reading: replacing c_proj_8.weight with `W_proj @ P_y` (where
P_y projects R^768 onto span(Vh_y[:r]) on the *output side*) preserves the
top-r Δy directions on the output side but kills 98%+ of the W_proj operator
on the input side. The trainable B (r, 768) cannot recover what was thrown
away on the input side because A is frozen at a low-energy slice of W_proj's
input space. wproj's A_w preserves W_proj's *highest-energy input directions*
exactly, which is why wproj wins.

The cycle 306 "53% unprojected energy" framing measures Δy's energy fraction
captured by Vh_w[:2] **on the output side**. It is a measurement about the
Δy → W_proj output map, not about how to construct an optimal rank-r
replacement of c_proj. The substantive finding of cycle 309 is that these
two concepts (output-side Δy alignment and input-side LASER capacity) are
geometrically distinct, and the cycle 306 → cycle 309 inference ("optimize
output-side alignment to fix the LASER") is wrong.

## Best-state restore + monotonicity

Per-seed:
- All 12 (cell, seed) pairs report best_step = **4000** (the budget end).
  Training was still descending at every cell. The script logged the
  per-500-step trajectory; for dy_r1 seed 42, the trajectory is monotone
  decreasing (58.77 → 57.02 → 56.05 → 55.47 → 55.06 → 54.76 → 54.51 → 54.31).
  Same shape for all other cells.
- Restored best_state matches best_ppl within 1e-3 per seed (asserted).
- final_ppl == best_ppl per seed (because best_step = N_STEPS).

This means **all 4 cells' absolute PPLs are upper bounds** on the trainable
floor at this rank assignment. Auditor checked whether the wproj > dy
ordering could flip with more steps:
- The descent slope at the last 500 steps (step 3500 → 4000) for dy_r1
  was 54.5097 → 54.3142 (Δ = −0.196, average −0.000391/step). For wproj_r1
  (similar inspection): 53.4862 → 53.3133 (Δ = −0.173). Both slopes are
  comparable; extrapolating either by even 4000 more steps would not
  bring dy down by the 1.02 PPL needed to match wproj. The ordering is
  robust to budget extension within the order-of-magnitude window.

## Plan adherence

Plan (`autoresearch/plans/cycle_309.md`) vs execution:
- 8 H2 headings (Item 319 verbatim Hypothesis, Geometric construction, Paired
  baseline, Files, Config, Verification, Verdict logic, Outputs). ✓
- ≤ 45 lines (36 lines). ✓
- Item 319 verbatim Hypothesis section reproduced unchanged from agenda. ✓
- New script `experiments/block8_dy_aligned_laser_cycle309.py`. ✓
- Reuses `UnifiedCurvatureFramework` (cycle 302/306) for `compute_Dy(8)`. ✓
- Reuses `LowRankCProj` template (cycle 270/280/292). ✓
- Paired W_proj-basis baseline run in the SAME cycle on the SAME calib /
  eval slices and SAME 3 seeds (the brief and the agenda item explicitly
  required this). ✓

Design review (`autoresearch/reviews/cycle_309.md`) marked APPROVE; the
review explicitly flagged that the verbatim threshold gates only the
absolute Δy PPL and that the W_proj baseline is what discriminates the
interpretive claim. The exec_result and manifest both lift the comparative
refutation finding above the literal verdict label.

## Convergence check (CLAUDE.md verify-task-performance rule)

Float PPL on the 100-seq held-out eval slice = 53.8949, matches cycle
270/280/305 baseline 53.895 to <0.001. Off-the-shelf gpt2, no task
training. The compression analysis is meaningful because the task baseline
is correct.

## Trainable param count

| cell | trainable | breakdown |
|---|---:|---|
| dy_r1   | 1,536 | 1*768 (B) + 768 (bias) |
| dy_r2   | 2,304 | 2*768 (B) + 768 (bias) |
| wproj_r1 | 1,536 | 1*768 (B) + 768 (bias) |
| wproj_r2 | 2,304 | 2*768 (B) + 768 (bias) |

A is a frozen buffer in all 4 cells. Asserted in script via
`assert n_trainable == r * 768 + 768`. ✓

## Wall time

~50 min total on mps:
- 4 no-retrain cells: ~5 s total
- 4 retrain cells × 3 seeds × ~2.2 min/seed (~26 min)
- ~20 min model loading / SVD / framework setup / per-seed init evaluation

Within the 60-min cycle cap. The plan estimated 41 min — overshoot of
~22% is from model reloads (one per seed, 12 reloads total) plus framework
setup. Acceptable for a single cycle.

## Cross-cycle sanity

- Float PPL 53.8949 matches cycle 270/280/305 ref 53.895 to <0.001. ✓
- Δy top1_E at b8 = 0.9141 matches cycle 306 ref 0.9141 to 4 dp. ✓
- W_proj_8 top-5 SVs match cycle 306 to all 4 reported decimals. ✓
- Δy top-5 SVs match cycle 306 to all 4 reported decimals. ✓
- wproj_r1 init PPL = 60.8573 is the no-retrain rank-1 SVD truncation of
  c_proj_8 (paired-baseline analog of cycle 280's no-retrain rank-r init
  at block 1).

## Corrections

- **Verdict label**: PASS at the literal-threshold level. The label
  "CONFIRMED verbatim, comparative claim REFUTED" is the corrected version
  recorded in the agenda PAST entry, the manifest header, and the
  exec_result. No correction to the dy_r1 measurement itself.

## Artifact existence

- `experiments/block8_dy_aligned_laser_cycle309.py` — exists, ~520 lines,
  single driver script reusing cycle 270/280/292 LowRankCProj + cycle
  302/306 UnifiedCurvatureFramework.
- `autoresearch/plans/cycle_309.md` — exists, 36 lines, marked `# APPROVE`,
  8 H2 headings.
- `autoresearch/reviews/cycle_309.md` — exists, design review APPROVE.
- `autoresearch/results/cycle_309_block8_dy_aligned_laser.json` — exists,
  valid JSON, full per-cell histories for all 12 (cell, seed) pairs.
- `autoresearch/results/cycle_309_summary.md` — exists, no-retrain table,
  retrain table, per-seed table, verdict logic, geometric construction.
- `autoresearch/results/cycle_309_manifest.txt` — exists, with verdict label
  "CONFIRMED (verbatim) but COMPARATIVE HYPOTHESIS REFUTED" in the header.
- `autoresearch/results/cycle_309_exec_result.txt` — exists, lifts the
  comparative refutation above the literal CONFIRMED label.
- `autoresearch/audits/cycle_309.md` — this file.
- `autoresearch/results/cycle_309_audit.txt` — to be written.
- `autoresearch/logs/cycle_309_run.log` — exists, contains all 12 per-seed
  trajectories and the final cell summary, plus the geometric sanity prints.
- `autoresearch/agenda.md` — item 319 deleted from NEXT P0; PAST Cycle 309
  entry inserted before Cycle 308 PAST entry.
- `autoresearch/logs/experiments.tsv` — row 309 appended with
  `outcome=confirmed-comparative-refuted`.

## Final status

**PASS** with substantive correction lifted into the agenda PAST entry.
The literal verbatim verdict is CONFIRMED (dy_r1 retrain best_mean
54.3185 ≤ 60.0). The comparative interpretive claim attached to CONFIRMED
is rejected by the paired W_proj-basis baseline (wproj > dy at both ranks
by 0.75-1.02 PPL, all seed std ≤ 0.013). The geometric reading: Δy is
rank-1 dominated on its own SVD but the rank-1 axis lies in W_proj's
input-side tail (1.27% of W_proj's Frobenius energy at r=1), so the
Δy-aligned LASER throws away the input-side energy that wproj keeps. The
b8 LASER gap is governed by W_proj's input-side spectrum, not by Δy's
output-side alignment. The cycle 306 → 309 inference path is closed.

Auditor recommends:
1. Update `STATUS.md` to reflect cycle 309 (latest) and lift the
   substantive finding "wproj > dy at b8" into the geometric thread. The
   cycle 305 destructive composition at b8 needs a different geometric
   reading; the basis-misalignment hypothesis is closed.
2. The agenda P0 set is now {320, 321}. Item 320 (pairwise LASER
   composition matrix at b1/b2/b3/b10/b11) is now the next eligible P0
   pick — note that block 8 is NOT in the cycle 282 family, so the
   cycle 305 destructive composition story for b8 specifically is the
   open thread, not addressable by item 320.
3. No follow-up to item 319 is recommended unless a new geometric framing
   appears: the b8 LASER gap is now characterised as a W_proj-spectrum
   property, not a basis-choice problem.
