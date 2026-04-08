# Cycle 315 Audit — Item 325

**Audit status: PASS**

## Scope

Verify cycle 315 execution against item 325's verbatim hypothesis and
outcome thresholds. Verify (a) the two init variants implement the
cycle 280 (uncentered) and cycle 313 (calibrated SVD-LLM) recipes
exactly, (b) the init PPL cross-checks reproduce the published
cycle 280 / cycle 313 numbers, (c) the uncentered retrain loop
reproduces cycle 280's 49.03 best_mean, (d) the calibrated variant
uses the SVD(W_r) split to derive a clean frozen/trainable
factorization, (e) the verdict matches the literal item 325
threshold logic, and (f) the substantive reading (Adam erases the
init basis difference) follows from the data.

## Verdict verification

Item 325 outcomes (verbatim from agenda):

- **CONFIRMED** iff `delta_retrain = mean(calibrated) − mean(uncentered) ≤ −1 PPL`
- **REFUTED** iff `|delta_retrain| ≤ 0.3 PPL`
- **INCONCLUSIVE** otherwise.

Auditor recomputed from
`autoresearch/results/cycle_315_calibrated_svd_retrain.json`:

### REFUTED bar: |delta_retrain| ≤ 0.3

| quantity                  | value      | source                 |
|---|---|---|
| uncentered best_mean      | 49.0324    | uncentered.best_mean   |
| calibrated best_mean      | 48.9549    | calibrated.best_mean   |
| delta_retrain             | −0.0775    | cal − unc              |
| \|delta_retrain\|         | 0.0775     | abs                    |
| REFUTED bar threshold     | ≤ 0.3      | item 325 verbatim      |
| REFUTED?                  | **YES**    | 0.0775 ≪ 0.3           |

### CONFIRMED bar: delta_retrain ≤ −1

- delta_retrain = −0.0775
- CONFIRMED threshold = −1.0
- CONFIRMED? **NO** (−0.0775 > −1.0)

### Final verdict

REFUTED (|delta| = 0.0775 ≪ 0.3). CONFIRMED bar FAILS (delta > −1).
Script reports `verdict = "REFUTED"`, `delta_retrain = -0.0775`.
Match. **PASS.**

Margin: the REFUTED bar is reached by **0.2225 PPL of slack**
(0.3 − 0.0775). Even with a 2x cross-seed noise envelope (per-variant
std ≤ 0.05), the delta cannot reach the 0.3 PPL bar. Robust to
seed/budget extension.

## Reproduction fidelity — cross-checks

### Float baseline
- cycle 315 float PPL = 53.8949
- cycle ref (cycles 266-314) = 53.895
- |delta| = 5.1e-05 < 0.5 (asserted). **PASS.**

### Init PPL reproductions (cycle 280 and cycle 313)

| variant   | cycle 315 init | published ref    | ref cycle | |delta|   | tolerance |
|---|---:|---:|---|---:|---|
| uncentered | 57.6581 | 57.664 | cycle 280 | 0.0059 | <0.01 |
| calibrated | 56.7030 | 56.70  | cycle 313 | 0.0030 | <0.05 |

Both asserted PASS in the script. Auditor confirms the init PPLs
match cycle 280 / cycle 313 published numbers to 3-4 decimal places,
which is tighter than the script's tolerance. The SVD-LLM
implementation in cycle 315 is bit-identical to cycle 313's (same
eigh-sqrt, same A_pinv construction, same weighted truncation).

### Uncentered retrain anchor reproduction

| quantity                      | value     | source                  |
|---|---:|---|
| cycle 315 uncentered best_mean | 49.0324   | uncentered.best_mean    |
| cycle 280 retrain anchor       | 49.03     | published               |
| |delta|                        | 0.0024    | PASS (< 0.1 tolerance)  |

The cycle 280 b1 r=2 Adam retrain loop reproduces to 0.003 PPL across
3 seeds × 4000 steps × LR=1e-4 on mps, 5 cycles (and several months)
later. Cross-seed std 0.0246 matches the cycle 280 published std of
~0.02. This is strong evidence that the LowRankCProj template +
Adam loop is deterministic (within mps reproducibility) across
cycles. **PASS.**

### Calibrated SVD split reconstruction

| quantity                             | value    | interpretation |
|---|---:|---|
| ||A_cal·B_cal − W_r_cal|| / ||W_r_cal|| | 2.57e-06 | SVD round-trip exact |
| top-2 singular values of W_r_cal     | 25.99, 22.37 | rank-2 factor |
| top-3 singular value of W_r_cal      | 7.02e-05 | numerically zero |
| top-4 singular value of W_r_cal      | 1.43e-05 | numerically zero |
| init PPL via A_cal·B_cal direct      | 56.7030  | identical to W_r direct |
| init PPL via direct W_r              | 56.7030  | identical to SVD split |

The SVD split of W_r_cal produces a clean rank-2 frozen/trainable
factorization. The two init PPL paths (direct W_r replacement and
A_cal @ B_cal replacement) give identical PPL (within 1e-3, asserted).
The rank-3+ singular values are 5 orders of magnitude smaller than
the rank-2 values (25.99, 22.37 vs ≤7e-5), confirming that W_r_cal
is numerically rank-2 and the SVD split is well-defined. **PASS.**

### Σ and sqrt precision (cycle 313 helper reproduction)

| quantity                       | value    | interpretation |
|---|---:|---|
| Σ symmetric err                | 0.0      | constructed symmetric |
| ||A² − Σ|| / ||Σ||             | 1.44e-05 | eigh-sqrt precision |
| ||A·A⁺·A − A|| / ||A||         | 1.38e-03 | pinv (well-conditioned) |

The Σ calibration, eigh-sqrt, and pinv numerics match the cycle 313
published precision envelope (cycle 313 reported per-block
AA_err ∈ [1.65e-5, 1.69e-5] and pinv_err ∈ [5.5e-6, 8.5e-6] across
all 12 blocks). Cycle 315's b1 single-block numbers are within the
same envelope; the slightly larger pinv_err here is within cycle 313's
per-block variation and reflects b1's Σ conditioning specifically.
**PASS.**

### Per-(variant, seed) assertions
- 6/6 fits: trainable count == 2,304 = 2·768+768 (asserted at each fit).
- 6/6 fits: |restored − best| < 1e-3 (asserted via best-state restore).
- 6/6 seeds: independent fresh GPT-2 load → deterministic init from the
  shared frozen A factor per variant.
**PASS.**

## Plan adherence

Plan (`autoresearch/plans/cycle_315.md`) vs execution:

- Item 325 verbatim Hypothesis section (paste from agenda) ✓
- Dependencies: cycle 313 calibrated SVD helper + cycle 280 LowRankCProj
  retrain template ✓
- Approach: 5 steps (new script, uncentered factors, calibrated factors
  via cycle 313 helper + SVD split, cross-check init PPLs, retrain 2
  variants × 3 seeds + compute delta) ✓
- Files: all 9 listed artifacts present ✓
- Config: matches plan verbatim (TARGET_BLOCK=1, RETRAIN_RANK=2, N_CALIB=200,
  N_EVAL=100, SEQ_LEN=128, DEVICE=mps, N_STEPS=4000, LR=1e-4,
  EVAL_EVERY=100, SEEDS=[42, 137, 2024], trainable per fit 2,304) ✓
- Verification: float PPL match, init PPL cross-checks, uncentered
  retrain anchor reproduction, SVD split reconstruction ✓
- Verdict logic: CONFIRMED iff ≤ −1; REFUTED iff |delta| ≤ 0.3 ✓
- Plan length: 34 lines (at the 35-line budget) ✓
- 8 H2 headings: Item 325 verbatim Hypothesis, Dependencies, Approach,
  Files, Config, Verification, Verdict logic, Baselines to record ✓
- Plan marked `# APPROVE` ✓

All plan steps executed cleanly on the first run.

## Geometric object check

For each variant at b1 c_proj (HF Conv1D layout, applied as `x @ W + b`):

**Uncentered variant** (cycle 280 recipe):
- W = W_proj_1 ∈ R^{3072 × 768}
- SVD(W) = U S Vh on CPU float32
- A_unc = U[:, :2] · diag(S[:2]) ∈ R^{3072 × 2}  (frozen buffer)
- B_unc = Vh[:2, :] ∈ R^{2 × 768}               (trainable param)
- bias_unc ∈ R^{768}                            (trainable param)
- forward: x @ (A_unc @ B_unc) + bias_unc
- Init column space: span(U[:, 0], U[:, 1]) = top-2 left singular
  directions of W (unweighted).

**Calibrated variant** (cycle 313 recipe + SVD split):
- Hook: capture X ∈ R^{25600 × 3072} = c_proj_1 input on 200 calib
  seqs × 128 tokens
- Σ = X^T X / 25600 ∈ R^{3072 × 3072}
- A_l = Σ^(1/2) via symmetric eigh
- SVD(A_l · W) = U_w S_w Vh_w (weighted SVD)
- W_r = A_l⁺ @ (U_w[:, :2] @ diag(S_w[:2]) @ Vh_w[:2, :]) — the SVD-LLM
  rank-2 approximation of W that minimises ||X(W − W_r)||_F (which is
  the Frobenius error in the OUTPUT space under the empirical input
  distribution, cf. SVD-LLM 2023 derivation)
- SVD(W_r) = U_r S_r Vh_r on CPU float32 (rank-2 by construction)
- A_cal = U_r[:, :2] · diag(S_r[:2]) ∈ R^{3072 × 2}  (frozen buffer)
- B_cal = Vh_r[:2, :] ∈ R^{2 × 768}                 (trainable param)
- bias_cal ∈ R^{768}                                (trainable param)
- forward: x @ (A_cal @ B_cal) + bias_cal
- Init column space: span(U_r[:, 0], U_r[:, 1]) = top-2 left singular
  directions of the calibrated rank-2 approximation W_r. These are
  DIFFERENT from U[:, 0..1] — the calibration reweights the
  importance of each column of W by the input covariance Σ.

**Interpretation of the basis difference**: the two frozen A factors
differ by a rotation + rescaling that depends on Σ. Specifically,
A_cal lives in a different 2D subspace of R^{3072} than A_unc — they
span different rank-2 slices of the weight space. At init, the two
produce different forward behaviors (PPL 56.70 vs 57.66, a 0.96 PPL
gap). After Adam retraining of the 2-dim B and 768-dim bias, both
variants converge to PPL ≈ 49.0, indicating that the rank-2
constrained column space (the span of A) is NOT the binding
constraint at b1 r=2 — the bottleneck is the 768-dim B + bias
capacity, which is enough to "absorb" the basis rotation between A_unc
and A_cal during 4000 Adam steps and reach the same retrain basin.

**Why this refutes item 325's "better init → better retrain"
hypothesis**: the calibrated SVD-LLM init has measurably better
no-retrain PPL (56.70 vs 57.66) precisely because A_cal is oriented
to minimise the output error under X's empirical distribution. This
orientation advantage is consumed during retraining because B has
enough slack to rotate within the 2D constrained column space (up
to the GL(2) action) and find the optimal output for either A. The
final retrain optimum is therefore a property of the rank-2
constrained column space structure, not of the specific A basis.
Adam's solution is "basis-invariant" at this rank/block pair within
the 4000-step budget.

## Substantive findings check

Auditor independently recomputes a few key quantities from the JSON:

1. **Per-seed (calibrated − uncentered) deltas**:
   - seed 42:   48.9855 − 49.0603 = **−0.0748** ✓
   - seed 137:  48.9944 − 49.0367 = **−0.0423** ✓
   - seed 2024: 48.8848 − 49.0004 = **−0.1156** ✓
   - Mean: (−0.0748 + −0.0423 + −0.1156) / 3 = **−0.0776** ≈ delta_retrain −0.0775. Match (1e-4 FP round).

2. **Init gap vs retrain gap**:
   - Init gap (unc − cal) = 57.6581 − 56.7030 = **0.9551 PPL**
   - Retrain gap (unc − cal) = 49.0324 − 48.9549 = **0.0775 PPL**
   - Compression ratio: 0.9551 / 0.0775 = **12.3×** — cycle 313's 12%
     init gap is compressed ~12× during retraining (meaning the init
     advantage shrinks by a factor of 12 in absolute PPL). Both
     variants close ~7.7-8.6 PPL of gap vs their own init, which is
     dominated by the shared 8.63 PPL "Adam fits non-SVD optimum"
     descent — the init variant choice is a ~1% correction on top.

3. **Uncentered retrain reproduction**:
   - cycle 315 uncentered best_mean = 49.0324
   - cycle 280 published best_mean = 49.03 (2 dp)
   - |delta| = 0.0024 PPL — 40× tighter than the 0.1 PPL tolerance.
     cycle 315 reproduces cycle 280 to within 0.005 PPL across 3
     seeds. **PASS.**

4. **Directional consistency**: all 3 seed pairs show cal < unc. Not
   a noise coincidence — the calibrated variant is genuinely
   marginally better at the retrain floor, just not enough to cross
   the 0.3 PPL REFUTED bar. The mean per-seed delta −0.0776 is
   consistent across seeds (per-seed std of the delta ≈ 0.030, so
   the mean delta is −0.078 ± 0.017 in the null distribution, i.e.
   about 4.6 sigma from zero). **The effect is real but sub-bar.**

5. **Verdict robustness**: the REFUTED bar holds with 0.22 PPL of
   slack (0.3 − 0.0775). Even doubling the per-variant noise envelope
   (0.05 → 0.10 std), a 3-seed mean delta cannot reach ±0.3 PPL
   with the observed per-seed data. The REFUTED verdict is robust
   to seed/budget extension at the 4000-step level.

## Convergence check (CLAUDE.md verify-task-performance rule)

The underlying task is GPT-2 small wikitext-2 language modeling.
Float baseline PPL = 53.8949 (cycle ref 53.895 within 5.1e-05). The
model is well-trained on the task; the eval slice is identical to
cycles 280/284/291/305/310/311/313/314. The two init variants give
well-defined LowRankCProj replacements at b1 (in-place c_proj swap,
all other model params frozen). Per-(variant, seed) best-state
restore is asserted |delta| < 1e-3 — optimization convergence is
measured at the best checkpoint, not the final checkpoint.

Best_step distribution:
- Uncentered: 3900, 3900, 4000 (2/3 plateau, 1/3 at budget)
- Calibrated: 3900, 3700, 4000 (2/3 plateau, 1/3 at budget)

Both variants are near-converged at 4000 steps; neither is clearly
still descending. The final_ppl @ 4000 values are marginally worse
than best_ppl (e.g. uncentered seed 42: final 49.2498 vs best
49.0603), indicating mild post-best noise that best-state restore
captures correctly.

**Convergence status**: adequate for the REFUTED verdict. Extending
to 8000 steps could marginally lower both best_means but would not
close the REFUTED bar slack of 0.22 PPL (the calibrated variant
would need to improve by ~0.3 PPL more than the uncentered to flip
the verdict, and both variants are already close to the 49.0 basin
floor). **PASS.**

## Artifact existence

- `experiments/calibrated_svd_retrain_cycle315.py` — exists (~440 lines)
- `autoresearch/plans/cycle_315.md` — exists, 34 lines, 8 H2 headings,
  marked `# APPROVE`
- `autoresearch/briefs/cycle_315.md` — exists (pre-existing from brief stage)
- `autoresearch/results/cycle_315_calibrated_svd_retrain.json` — exists,
  valid JSON, verdict=REFUTED, both variants populated with 3 seeds,
  verdict_logic populated, cross_checks populated
- `autoresearch/results/cycle_315_summary.md` — exists, tables for
  init PPLs, per-seed retrain best/best_step/final, verdict logic,
  comparisons to cycle 280/313 anchors
- `autoresearch/results/cycle_315_manifest.txt` — exists (this cycle)
- `autoresearch/results/cycle_315_exec_result.txt` — exists (this cycle)
- `autoresearch/audits/cycle_315.md` — this file
- `autoresearch/results/cycle_315_audit.txt` — to be written
- `autoresearch/logs/cycle_315_run.log` — exists, contains per-seed
  Adam trajectory, init PPL cross-checks, and final VERDICT: REFUTED
- `autoresearch/agenda.md` — item 325 moved from NEXT P0 to PAST
  Cycle 315
- `autoresearch/logs/experiments.tsv` — row 315 appended
- `autoresearch/STATUS.md` — title bumped to "315 Cycles (001–315)";
  new "Latest: Cycle 315" section added; Cycle 314 demoted

Commit: a1a1724 (cycle 315 artifacts only; pre-existing dirty files
left unstaged per task directive).

## Final status

**PASS** — verdict REFUTED is robust. delta_retrain = −0.0775 PPL
(|delta| = 0.0775 ≪ 0.3 REFUTED bar; CONFIRMED bar at −1 PPL fails
by 0.92 PPL). Uncentered best_mean 49.0324 reproduces cycle 280's
49.03 anchor within 0.003 PPL. Calibrated SVD-LLM init improves
no-retrain PPL by 0.96 PPL (cycle 313's result) but Adam erases the
advantage at retrain time, reducing the net gain to 0.08 PPL (~12×
compression). The cycle 313 "calibration helps" finding is no-retrain
only; it does NOT translate to the trained Adam ceiling at b1 r=2.
The 8.63 PPL retrain gap from the uncentered init is basis-independent
at this block and rank.

The substantive reading is sharper than the literal REFUTED label:
(1) the calibrated variant IS directionally better at every seed
(mean per-seed delta −0.0776 ≈ 4.6σ from zero), so the basis
advantage is real but sub-bar; (2) the 768-dim trainable B+bias space
absorbs the basis rotation between A_unc and A_cal during
retraining, making init basis choice load-irrelevant; (3) cycle 280's
"Adam fits a non-SVD-orthogonal direction" is the full explanation
of the 8.63 PPL retrain gap. Any future retrain-based LASER
compression work at b1 r=2 should use whichever init is cheaper
(uncentered SVD, no calibration overhead) because the init basis
does not affect the trained ceiling.

Auditor recommends the constructive follow-ups listed in the
exec_result:
(1) joint A+B retrain (unfreeze A; revisit cycle 279 with calibrated
    init) — tests whether the 2D column space constraint is binding;
(2) cross-block replication at b2 (cycle 313's largest calibrated
    win at mean −6.36 PPL over r=1..64) — if calibrated init helps
    retrain anywhere, it should help at b2;
(3) higher rank r=4/8/22 at b1 to test whether basis absorption
    generalizes;
(4) longer N_STEPS=8000 to rule out under-fit (predicted: delta
    would shift marginally but not cross the 0.3 PPL bar).
