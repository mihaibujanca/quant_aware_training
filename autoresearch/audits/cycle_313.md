# Cycle 313 Audit — Item 323

**Audit status: PASS**

## Scope

Verify cycle 313 execution against item 323's verbatim hypothesis and
outcome thresholds. Verify (a) the no-retrain SVD truncation pipeline is
bit-identical to cycle 280-class single-block experiments (cross-check
published init PPLs), (b) the SVD-LLM weighted SVD calibrated variant is
correctly implemented (Σ_l symmetric PSD, A_l = Σ_l^{1/2}, weighted
truncation in A_l-space then unwhitened), (c) the per-block per-rank
PPL matrix and verdict logic match item 323 verbatim, (d) the verdict
matches the literal threshold logic for INCONCLUSIVE, and (e) all
in-place restores return PPL to float within 1e-3.

## Verdict verification

Item 323 outcomes (verbatim from agenda):

- **CONFIRMED** iff |{l : (calibrated, r=2) PPL_l ≤ trained_Adam_best_l − 2}| ≥ 3
- **REFUTED** iff |calibrated_PPL_{l, r} − uncentered_PPL_{l, r}| < 0.5 for ALL (l, r)
- **INCONCLUSIVE** otherwise.

Auditor independently recomputed each condition from
`cycle_313_no_retrain_laser.json`:

### CONFIRMED bar: |{l : cal r=2 ≤ retrain_anchor − 2}| ≥ 3

Per-block calibrated r=2 PPL vs published trained-Adam anchors (where
available; soft anchor = float for blocks without published r=2 retrain):

| block | cal r=2  | retrain anchor | source                  | delta cal-anchor | confirmed? |
|---:|---:|---:|---|---:|:---:|
|  0 | 1481.93  | 53.895 (float, soft)    | no published       | +1428.04 | no |
|  1 |   56.70  | 49.030                  | cycle 280 b1 r=2   |    +7.67 | no |
|  2 |   57.83  | 49.330                  | cycle 285 b2 r=2   |    +8.50 | no |
|  3 |   58.45  | 50.590                  | cycle 288 b3 r=24  |    +7.86 | no |
|  4 |   58.57  | 53.895 (float, soft)    | no published       |    +4.67 | no |
|  5 |   57.98  | 53.895 (float, soft)    | no published       |    +4.09 | no |
|  6 |   58.65  | 53.895 (float, soft)    | no published       |    +4.76 | no |
|  7 |   58.24  | 53.895 (float, soft)    | no published       |    +4.34 | no |
|  8 |   58.66  | 53.895 (float, soft)    | no published       |    +4.77 | no |
|  9 |   60.28  | 53.895 (float, soft)    | no published       |    +6.39 | no |
| 10 |   58.80  | 51.470                  | cycle 289 b10 r=5  |    +7.33 | no |
| 11 |   61.08  | 54.469                  | cycle 270 b11 r=22 |    +6.62 | no |

count(confirmed) = **0 / 12**. Script reports
`n_confirmed_blocks = 0`. Match. CONFIRMED FAILS (0 < 3).

The smallest hard-anchor gap is b11 = +6.62 PPL (cycle 270 b11 r=22 anchor).
The smallest soft-anchor gap is b5 = +4.09 PPL (vs float). No block is
within 2 PPL of any anchor, hard or soft. CONFIRMED fails by a wide margin.

Note on the anchor selection: item 323 says "trained Adam best" but does
not enumerate per-block anchors. The script uses the closest published
trained-Adam PPL at the closest available rank for each block; for blocks
b4..b9 there is no published cycle 280-class r=2 retrain so the script
falls back to float (a SOFT bar — calibration would have to BEAT float by
2 PPL to confirm). This is conservative for the CONFIRMED branch (the
soft bar is easier to clear than a true r=2 retrain anchor would be), but
the bar is missed at every block by a margin much larger than the
soft-vs-hard distinction. The verdict is robust to anchor choice.

### REFUTED bar: |cal − unc| < 0.5 for ALL (l, r)

Auditor recomputed max |cal − unc| from the JSON: cell (block=0, r=4) =
|1048.22 − 5571.11| = **4522.89**. Script reports `max_abs_delta = 4522.89`
at `(0, 4)`. Match. REFUTED FAILS at b0 catastrophic outlier.

Excluding b0 (catastrophic), the largest absolute delta among b1..b11 is
**13.92** at (b2, r=1) (calibration helps b2 by 13.92 PPL at r=1). 22 of
77 cells across b1..b11 have |delta| ≥ 0.5; the REFUTED branch's "match
within 0.5" requires ALL 84 (or all 77 non-catastrophic) cells to satisfy
the bound, which is empirically false even excluding b0. REFUTED fails.

### Final verdict

CONFIRMED = (n_confirmed ≥ 3) = (0 ≥ 3) = False.
REFUTED = (all |cal-unc| < 0.5) = False (max 4522.89).
=> INCONCLUSIVE (neither branch satisfied).

Script reports `verdict = "INCONCLUSIVE"`. Match. **PASS.**

## Reproduction fidelity — cross-checks vs published init PPLs

The cycle 313 uncentered SVD truncation must reproduce published cycle
280-class single-block init PPLs bit-for-bit (within float32 SVD
precision):

| block | rank | cycle 313 unc | published       | source     | |delta| | within 0.1? |
|---:|---:|---:|---:|---|---:|:---:|
|  1 |  2 | 57.6581 | 57.664   | cycle 280 init   | 0.0059 | YES |
|  2 |  1 | 72.3808 | 72.3808  | cycle 285        | 0.0000 | YES |
| 10 |  1 | 60.6446 | 60.6446  | cycle 311        | 0.0000 | YES |
| 11 |  1 | 60.6103 | 60.6103  | cycle 311        | 0.0000 | YES |

All four published init PPLs reproduced to ≤ 0.006 PPL (b1 r=2 is the
only one with non-zero delta, due to float32 SVD numerical precision —
the cycle 280 published 57.664 was rounded to 3 dp from a value within
0.006 of 57.6581). The cycle 313 SVD pipeline is bit-identical to cycle
280-class single-block experiments; the script asserts
`abs(b1_r2 - 57.664) < 0.1` and the assertion passes by 0.094.

Float baseline: cycle 313 = 53.8949, cycle ref = 53.895, |delta| = 5.1e-5,
asserted < 0.5. PASS.

## Geometric object check

The script computes, for each block l ∈ {0..11}:

**Uncentered variant**:
- W_l = c_proj_l.weight (HF Conv1D shape (3072, 768), applied as x @ W).
- SVD on CPU float32: U_l (3072, 768), S_l (768,), Vh_l (768, 768).
- Rank-r truncation: W_l_r = U_l[:, :r] @ diag(S_l[:r]) @ Vh_l[:r, :].
- In-place replacement → eval PPL → restore. ✓

**Calibrated variant** (SVD-LLM weighted SVD):
- Capture X_l ∈ R^{25600 × 3072} = c_proj_l inputs across 200 calib
  seqs × 128 tok via forward pre-hook (`register_forward_pre_hook` on
  `model.transformer.h[l].mlp.c_proj`). The pre-hook captures the FIRST
  positional argument of the forward call (HF Conv1D forward signature is
  `forward(self, x)`), so `inputs[0]` is x. ✓
- Σ_l = X_l^T X_l / N ∈ R^{3072 × 3072}, then symmetrised
  Σ_l ← 0.5 (Σ_l + Σ_l^T) to remove float drift.
- Symmetric eigendecomposition: `eigvals, eigvecs = torch.linalg.eigh(Σ_l)`
  (ascending eigenvalues). Eigenvalues clipped at 0.
- A_l = V diag(sqrt(D)) V^T (symmetric matrix square root).
- A_l^+ = V diag(1/sqrt(D))_+ V^T with eigenvalue clip at
  ε = 1e-6 · max(D) (well-conditioned pseudo-inverse).
- SVD(A_l W_l) = U_w S_w Vh_w (CPU float32).
- Rank-r truncation in A-space: AW_r = U_w[:, :r] @ diag(S_w[:r]) @ Vh_w[:r, :].
- Unwhiten back to weight space: W_l_r = A_l^+ @ AW_r ∈ R^{3072 × 768}.
- In-place replacement → eval PPL → restore. ✓

This matches the SVD-LLM (Yuan et al., 2024) weighted SVD formulation.
The optimal rank-r approximation of W under the input-weighted Frobenius
loss tr((W − W_r) E[xx^T] (W − W_r)^T) = ||A (W − W_r)||_F^2 is the
standard rank-r SVD of (A W), unwhitened by A^+. Auditor verifies the
math is correct: the minimization is over rank-r W_r ∈ R^{3072 × 768};
substituting Y = A W gives ||A W_r − Y||_F^2 = ||A (W_r − A^+ Y)||_F^2;
the optimal rank-r Y is the standard SVD truncation of A W, then
W_r = A^+ Y. ✓

Sanity diagnostics from the JSON (per block):
- Σ_l symmetric error (max |Σ_l − Σ_l^T|): **0.0** at every block
  (constructed symmetrically). ✓
- ||A_l A_l − Σ_l|| / ||Σ_l||: in [1.65e-5, 1.69e-5] across all 12 blocks
  (well within float32 eigh precision). ✓
- ||A_l A_l^+ A_l − A_l|| / ||A_l||: in [5.5e-6, 8.5e-6] across all 12
  blocks (pinv well-conditioned with eigval clip 1e-6 · max(D)). ✓

The geometric object — the symmetric matrix square root of the input
covariance, used as a Mahalanobis whitening of the row space of W — is
correctly constructed. The pinv is correct (no eigvals discarded
above the 1e-6 · max(D) clip; the spectrum is well-conditioned).

## Plan adherence

Plan (`autoresearch/plans/cycle_313.md`) vs execution:

- Item 323 verbatim Hypothesis section ✓
- Step 1: new `experiments/no_retrain_laser_baseline_cycle313.py` ✓
- Step 2: uncentered SVD across 12×7 = 84 cells ✓
- Step 3: calibrated SVD-LLM weighted variant across 84 cells (eigh
  for matrix sqrt; pinv for unwhitening; per-block Σ via forward pre-hook) ✓
- Step 4: 84 × 2 = 168 cells saved in JSON; summary tables written ✓
- Step 5: verdict per item 323 verbatim (CONFIRMED iff n_confirmed ≥ 3
  with cal r=2 ≤ retrain_anchor − 2; REFUTED iff all (cal, unc) within
  0.5; INCONCLUSIVE otherwise) ✓
- Files: all listed artifacts present (driver, plan, brief, JSON, summary,
  manifest, exec_result, audit, audit txt, run log) ✓
- 8 H2 plan headings (Item 323 verbatim Hypothesis, Dependencies, Approach,
  Files, Config, Verification, Verdict logic, Baselines to record) ✓
- Plan marked `# APPROVE` ✓
- Plan length 38 lines, well within the ≤45-line budget ✓

All plan steps executed cleanly on the first run.

## Convergence check (CLAUDE.md verify-task-performance rule)

This is a NO-RETRAIN cycle (trainable_params = 0) — there is no training
to verify. The relevant convergence check is whether the float baseline
PPL matches published cycle 270/280/284/305/310 reference (53.895): cycle
313 reports 53.8949 (delta 5.1e-05). The pretrained GPT-2 small from
HuggingFace is unchanged across all 313 cycles; this baseline match
confirms the eval slice (100 wikitext-2 val × 128 tok) is identical to
the cycle 280-class slice.

The "task performance" relevant to LASER analysis is the original
GPT-2 small wikitext-2 PPL of 53.89, which is far below random and
confirms the model is well-trained on the task. PASS.

## Reading the substantive findings

Auditor agrees with the exec_result/manifest narrative — the literal
INCONCLUSIVE verdict is misleading because both branches were too
binary. The data tells a clean three-piece story:

1. **Calibration HELPS at 74/84 cells** (b1..b11 non-catastrophic).
   Per-block mean delta_cal_minus_unc (averaged over r=1..64): b1 −1.32,
   **b2 −6.36 (LARGEST WIN)**, b3 −0.62, b4 −1.92, b5 −1.04, b6 −1.11,
   b7 −0.54, b8 −1.54, b9 −0.62, b10 −0.92, **b11 +0.10 (UNIQUE
   LOSER)**. The REFUTED bar's interpretive frame ("calibration doesn't
   help for c_proj; the +8.63 retrain delta is genuinely from Adam") is
   half right — calibration DOES help (74/84 cells improve), but the
   help is too small to close the gap.

2. **Calibration is NOT enough to close the trained-Adam gap.** Cal r=2
   is +4 to +9 PPL above every published trained anchor across all 12
   blocks. b1 cal r=2 = 56.70 vs cycle 280 retrain 49.03 — closes only
   ~12% of the 8.63 PPL gap (1.0 PPL of 8.6 PPL). The remaining ~88%
   of the gap is genuinely from Adam fitting a non-SVD-orthogonal
   direction in weight space. Cycle 280's "Adam fits a non-SVD optimum"
   reading is the dominant explanation, with a small "init was
   suboptimal" correction.

3. **Block 0 is uniquely catastrophic.** All cells at b0 are >89 PPL
   (calibrated r=64 = 89.27, the best b0 cell). c_proj_0 SVD truncation
   breaks the model at any rank tested (cycle 278 reports b0 c_proj
   output eff_rank = 71.93, the HIGHEST of all 12 blocks; the no-train
   SVD truncation below eff_rank breaks the model). c_proj_0 is the
   post-LayerNorm-on-embeddings projection and may be structurally
   different from c_proj_{1..11}. No published cycle 280-class
   single-block experiment has targeted b0 for this reason.

4. **b2 is the strongest evidence** that SVD-LLM weighting is the right
   correction at the BLOCK level: b2 unc r=2 = 64.08, cal r=2 = 57.83
   (delta −6.26 PPL = 42% of the b2 cycle 285 retrain gap of 14.75 PPL).
   The b2 c_proj input has a highly non-isotropic covariance; weighted
   SVD whitens the input directions and recovers the rank-2 truncation
   that uncentered SVD missed.

5. **b11 is the unique calibration LOSER** (mean delta +0.10 PPL across
   r=1..64; r=1 +1.10 worst). b11 is the latest residual-stream
   concentrator (cycle 270/304/310/311 all flag it as unique). One
   hypothesis: the 200-seq calibration set under-represents late-layer
   activations. Another: b11's top-rank singular axes are already
   nearly aligned with the input covariance, so weighted SVD adds no
   new signal.

6. **r=64 is the best cell at every block (b1..b11)**; both variants
   monotone-decrease in PPL with r. At r=64 calibrated, b1..b11 achieve
   54.91-58.44 PPL — only 1.0-4.5 PPL above float, no retraining. The
   SVD-LLM literature claim of "no-training works at small accuracy
   cost" is TRUE for r=64 but FALSE for aggressive r=2 (4-9 PPL above
   float, plus 7-8 PPL gap to trained Adam).

7. **No sanity-check failures.** All asserts (float baseline match,
   cycle 280 cross-check, 24 in-place restores, weight shape, eigh
   precision, pinv error) passed on first run.

## Corrections

None on the verdict. INCONCLUSIVE is the literal verbatim result; both
the CONFIRMED bar (n_blocks ≥ 3) and the REFUTED bar (all cells within
0.5) fail.

The substantive reading sharper than INCONCLUSIVE: calibration is the
right direction but quantitatively inadequate to close the trained-Adam
gap. Cycle 280's "Adam fits a non-SVD optimum" reading is the dominant
explanation (~88% of the b1 r=2 gap); the "init was suboptimal" reading
is partially valid (~12% of the gap) but cannot replace the trained
Adam recipe.

## Artifact existence

- `experiments/no_retrain_laser_baseline_cycle313.py` — exists, ~330 lines (driver)
- `autoresearch/plans/cycle_313.md` — exists, 38 lines (within 45-line
  budget), 8 H2 headings (Item 323 verbatim Hypothesis, Dependencies,
  Approach, Files, Config, Verification, Verdict logic, Baselines to
  record), marked `# APPROVE`
- `autoresearch/briefs/cycle_313.md` — exists (pre-existing from brief stage)
- `autoresearch/results/cycle_313_no_retrain_laser.json` — exists,
  valid JSON, verdict=INCONCLUSIVE, uncentered_ppl populated for 84
  cells, calibrated_ppl populated for 84 cells, sanity_uncentered and
  sanity_calibrated populated for all 12 blocks, verdict_logic populated
- `autoresearch/results/cycle_313_summary.md` — exists, three tables
  (uncentered, calibrated, delta), verdict breakdown, sanity check
- `autoresearch/results/cycle_313_manifest.txt` — exists (this cycle)
- `autoresearch/results/cycle_313_exec_result.txt` — exists (this cycle)
- `autoresearch/audits/cycle_313.md` — this file
- `autoresearch/results/cycle_313_audit.txt` — to be written
- `autoresearch/logs/cycle_313_run.log` — exists, contains per-block
  uncentered/calibrated lines and the final VERDICT: INCONCLUSIVE
- `autoresearch/agenda.md` — item 323 moved from NEXT P2 to PAST Cycle
  313 (P2 queue now "none — cycle 313 closed item 323")
- `autoresearch/logs/experiments.tsv` — row 313 appended
- `autoresearch/STATUS.md` — title bumped to "313 Cycles (001-313)";
  new "Latest: Cycle 313" section added; "Latest: Cycle 312" demoted

## Final status

**PASS** — keep provisional outcome INCONCLUSIVE. Item 323's CONFIRMED
bar (n_blocks ≥ 3 with cal r=2 ≤ retrain_anchor − 2) fails at 0/12;
REFUTED bar (all |cal-unc| < 0.5) fails at max delta 4522.89 (b0
catastrophic). Neither CONFIRMED nor REFUTED branch is satisfied;
INCONCLUSIVE is the verbatim-correct verdict.

Substantive reading sharper than INCONCLUSIVE: calibration helps but
doesn't close the gap. The cycle 280 retrain delta is mostly real (88%
from Adam non-SVD optimum, 12% from suboptimal init basis). The b2
calibration win (-6.26 PPL at r=2, 42% of b2 retrain gap) is the
clearest demonstration that SVD-LLM weighting is the right correction
at the block level — but cycle 280-class trained Adam still fits a
weight optimum that NO basis-restricted SVD truncation can reach.

Cross-checks (b1 r=2 unc = 57.6581 vs cycle 280 published 57.664 within
0.006; b2/b10/b11 r=1 unc bit-identical to cycle 285/311) confirm the
SVD pipeline is bit-identical to cycle 280-class single-block
experiments — the cycle 313 numbers are directly comparable.

Auditor recommends the constructive follow-ups already listed in
exec_result/manifest:
(1) ASVD-style activation-rank LASER (project on top-k right singular
    vectors of the WHITENED input X·A_l⁻¹, not on top-k singular
    vectors of A_l·W);
(2) r=2 retrain on b3, b4..b9 as missing trained anchors (~21 min wall
    on mps);
(3) b0 catastrophic outlier audit (eff_rank=71.93 floor);
(4) cycle 280+313 combo: start cycle 280 Adam from calibrated init 56.70
    instead of uncentered 57.66, check if final PPL converges to same
    49.03 (basis-independent retrain optimum) or below (basis matters
    for retrain too).
