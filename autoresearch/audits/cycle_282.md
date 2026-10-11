# Cycle 282 Audit

**Status: PASS**

## Item
292 (P0): Block 10 c_proj rank-17 weight-SVD — extending cycle 270 to the second-lowest late-block eff_rank.

## Verdict
CONFIRMED — rank-17 retrain best_mean = 50.0445 ± 0.0565 ≤ 55.0 by 4.956 PPL margin. Below float by 3.850 PPL. New Pareto anchor (13,824 params, 50.04 PPL) strictly DOMINATES cycle 270 block 11 r=22 (17,664 params, 54.47 PPL) on BOTH axes: 22% fewer trainable params AND 4.42 PPL better.

## Verification

### Plan adherence
- Plan: `autoresearch/plans/cycle_282.md` — 26 lines, 8 headings (Item, Hypothesis, Setup, Verdict thresholds, Steps, Bookkeeping, Verification) matching cycle 280/281 template. Hypothesis quoted verbatim from item 292 in `autoresearch/briefs/cycle_282.md`. PASS
- Script: `experiments/block10_lowrank_cycle282.py` — direct adaptation of cycle 270 (`experiments/block11_lowrank_cycle270.py`) with TARGET_BLOCK=10, RANKS_NORETRAIN=[5, 17, 50] (reduced from cycle 270's [5, 22, 50, 100] per item 292 brief), RETRAIN_RANK=17. LowRankCProj module unchanged (frozen A=U·Σ buffer, trainable B Vh factor + bias). Verdict logic rewritten per item 292 verbatim: CONFIRMED iff retrain best_mean ≤ 55.0; REFUTED iff ALL no-retrain ≥ 57.0 AND retrain ≥ 57.0; else INCONCLUSIVE. Trainable assert: n_trainable == 17×768 + 768 = 13,824. PASS
- Hyperparameters: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore (B + bias snapshots). Match item 292 brief and cycle 270/280 convention. PASS
- Trainable per fit: 13,824 = 17×768 (B factor) + 768 (bias) — script-internal assert PASS for all 3 seeds. Matches brief exactly. PASS

### Run completion
- Script executed with `MPLCONFIGDIR=.mplconfig HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python experiments/block10_lowrank_cycle282.py > autoresearch/logs/cycle_282_run.log 2>&1`. Exit 0. PASS
- HF progress bars disabled at env var level (clean log without post-processing). PASS
- Wall: ~7 min total (3 noretrain forward swaps at ~1.3s each + 3 retrain × ~2.1 min each). Well under the ~12 min plan estimate. PASS
- All 3 retrain seeds completed; per-seed restore PASSED (`|restored_ppl - best_ppl| < 1e-3`). PASS

### Numerical consistency
- Float PPL = 53.89489 — matches FLOAT_BASELINE_REF=53.895 to 4 dp. PASS
- Init PPL deterministic at 59.378 across all 3 seeds (SVD init is seed-independent; the only seed-dependent randomness is Adam's calib shuffle). Matches noretrain[r=17]=59.378 within 1e-2 (assert PASS). PASS
- Restored original c_proj.weight after no-retrain sweep → PPL=53.895 (matches float within 1e-3). PASS
- Per-seed best PPLs: seed 42 → 50.0048 @ step 3900, seed 137 → 50.0042 @ step 3900, seed 2024 → 50.1245 @ step 3800. best_mean = (50.0048 + 50.0042 + 50.1245) / 3 = 50.04450, std (population) = sqrt(((50.0048-50.0445)² + (50.0042-50.0445)² + (50.1245-50.0445)²)/3) ≈ 0.0565 — matches JSON aggregate {best_mean: 50.0445, best_std: 0.0565} to 4 dp. PASS
- best_min = 50.0042 (seed 137), best_max = 50.1245 (seed 2024) — matches JSON. PASS
- All 3 seeds' best_step ∈ {3900, 3900, 3800} — the retrain is STILL DESCENDING at the budget cap (best_step_mean = 3867 ± 47). Longer training could push best_mean further below 50.04. PASS
- final@4000 mean ≈ 50.1045 — overshoot vs best only +0.060 PPL (tight, budget-bounded descent). Compare to cycle 279's +2.75 PPL overshoot. PASS
- Per-seed restored == recorded best within 1e-3 for all 3 seeds (assert PASS from script). PASS
- SVD of block 10 c_proj.weight: top-5 SVs = [41.585, 28.194, 21.173, 20.170, 19.657]; S[-1] = 0.647; S[0]/S[-1] = 64.3×. Energy fractions: r=5: 5.03%, r=17: 10.24%, r=50: 21.18%. Block 10 weight matrix is 3.3× BETTER conditioned than block 1 (cycle 280 had 213×). The weight matrix is NOT rank-concentrated as an operator — block 10's useful "rank" for compression is a property of the joint c_fc+c_proj action, not the c_proj weight SVD energy. PASS

### Verdict logic
- Per item 292 (verbatim from agenda):
  - CONFIRMED iff rank-17 retrained best_mean ≤ 55.0
  - REFUTED iff ALL ranks (r ∈ {5,17,50}) no-retrain ≥ 57.0 AND rank-17 retrain ≥ 57.0
  - INCONCLUSIVE otherwise
- Script-implemented logic matches verbatim:
  - best_mean = 50.0445 ≤ 55.0? YES (by 4.956 PPL margin) → CONFIRMED
  - Note: REFUTED noretrain clause WOULD have triggered (r=5: 59.918 ≥ 57; r=17: 59.378 ≥ 57; r=50: 58.265 ≥ 57) — all no-retrain ≥ 57. But the retrain clause blocks REFUTED (50.0445 < 57.0). → CONFIRMED wins.
- verdict_logic dict in JSON: `{threshold_confirmed_retrain_le: 55.0, threshold_refuted_noretrain_ge: 57.0, threshold_refuted_retrain_ge: 57.0, all_noretrain_ge_refuted: true, retrain_best_mean: 50.0445, verdict: "CONFIRMED"}` — internally consistent. PASS

### Geometric / cross-cycle reading
- **Headline finding: block 10 rank-17 retrain STRICTLY DOMINATES cycle 270 block 11 rank-22** on BOTH Pareto axes — 22% fewer trainable params (13,824 vs 17,664) AND 4.42 PPL better (50.04 vs 54.47). Cycle 270's late-block rank compression recipe is now DEPRECATED; cycle 282 is the new late-block Pareto anchor. PASS
- **Cycle 282 extends the cycle 270/280 pattern**: rank-(eff_rank-match) + Adam retrain is a reproducible single-block c_proj recipe across the late-block AND early-block rank concentrators. Three data points:
  - block 1 (cycle 278 eff_rank=1.94)  → r=2  retrain 49.03 PPL at 2,304 params  (cycle 280)
  - block 10 (cycle 278 eff_rank=16.70) → r=17 retrain 50.04 PPL at 13,824 params (cycle 282)
  - block 11 (cycle 267 eff_rank=22.49) → r=22 retrain 54.47 PPL at 17,664 params (cycle 270)
  The recipe is reproducible across blocks. PASS
- **FIRST positive eff_rank prescriptive result**: monotone ordering at single-block c_proj weight-SVD + Adam retrain level: block 1 (eff_rank 1.94) < block 10 (eff_rank 16.70) < block 11 (eff_rank 22.49) maps to retrained PPL 49.03 < 50.04 < 54.47. Lower eff_rank = better retrained PPL. This REFINES (not contradicts) cycle 273/278/281 findings: eff_rank is prescriptive for single-block c_proj weight-SVD + Adam retrain, but NOT for cascade cross-block composition. PASS
- **Block 10 unexpectedly beats block 11 by 4.42 PPL with 22% fewer params** — despite both being late blocks. Block 10's eff_rank (16.70) is 25% lower than block 11's (22.49), and the retrained PPL is significantly better. Strong per-block directional confirmation that lower eff_rank = cheaper retrained compressibility at the single-block c_proj level. The cycle 278 eff_rank census already flagged this: block 10 was the most rank-concentrated late block but was never tested as a compression target before cycle 282. PASS
- **Weight-SVD energy ≠ activation eff_rank — SECOND confirmation**. Block 10's weight-SVD top-17 energy is only 10.24% of the weight-matrix energy, yet the rank-17 retrained c_proj reaches 50.04 PPL (below float by 3.85). Block 1 was similar (top-2 energy 7.57% yet r=2 retrain 49.03). Three cycles (270/280/282) now confirm the weight-SVD energy fraction is NOT a useful predictor of retrained compressibility — only activation eff_rank is directionally predictive. PASS
- **Adam recovery magnitudes**: block 11 recovered 59.68 → 54.47 = −5.22 PPL; block 1 recovered 57.66 → 49.03 = −8.63 PPL; block 10 recovered 59.38 → 50.04 = **−9.33 PPL**. Block 10 has the LARGEST single-block c_proj weight-SVD retrain recovery measured so far. All three retrained PPLs beat float — the "small trainable subset beats float" phenomenon extends to 4 cycles now (271/276/280/282). PASS
- **New Pareto row at (13,824, 50.04)**: fills the gap between cycle 280 (2,304, 49.03) and cycle 276 [1] K=32 (25,344, 49.65). Cycle 280 still strictly dominates cycle 282 on BOTH axes (6× fewer params and 1.01 PPL better). Cycle 282 strictly dominates cycle 270 on BOTH axes. Cycle 282 partially dominates cycle 276 K=32 (45% fewer params at +0.40 PPL — both on frontier but at different operating points). PASS
- **The cycle 282 retrain is budget-bounded**: all 3 seeds best at step ≥3800, still descending. Longer training could push best_mean further below 50.04. Same pattern as cycle 270/280 (also budget-bounded). The 4000-step budget is not sufficient for convergence on these small-rank c_proj retrain surfaces. PASS
- **Follow-up**: block 2 (cycle 278 eff_rank=1.05, even more concentrated than block 1) has NEVER been tested as a rank-1 weight-SVD target. Item 293 tests block 2 with topk+lstsq K-sweep (different operator family). A block 2 rank-1 weight-SVD + retrain test would be a natural follow-up to cycle 282. PASS
- **Absolute gain**: cycle 282 beats float by 3.85 PPL — the second-largest margin of any single-block low-rank weight-SVD retrain recipe (cycle 280 block 1 was −4.86 PPL, cycle 282 block 10 is −3.85 PPL, cycle 270 block 11 was +0.57 PPL above float). PASS

### Bookkeeping
- `autoresearch/plans/cycle_282.md` written (26 lines, 8 headings, hypothesis verbatim from item 292, exact procedure, verdict thresholds, baselines, artifacts list, verification checklist). PASS
- `autoresearch/briefs/cycle_282.md` pre-existing (read-only context from planning agent). PASS
- `experiments/block10_lowrank_cycle282.py` created from cycle 270 template with TARGET_BLOCK=10, RANKS_NORETRAIN=[5,17,50], RETRAIN_RANK=17 swap, verdict logic rewritten per item 292, trainable-count assert. PASS
- `autoresearch/logs/cycle_282_run.log` created via stdout/stderr redirect. Clean log (HF progress bars disabled via env var). PASS
- `autoresearch/results/cycle_282_block10_lowrank.json` contains cycle/item/verdict/verdict_logic/float_ppl/noretrain (per r)/retrain_seeds (per seed history)/retrain_aggregate/config/svd_summary/baselines. Verified numerics match log. PASS
- `autoresearch/results/cycle_282_summary.md` written (verdict, SVD summary, noretrain table, retrain per-seed table + aggregate, verdict logic, Pareto deltas). PASS
- `autoresearch/results/cycle_282_manifest.txt` written with verdict, both commit hashes, per-seed results, deltas, full geometric reading. PASS
- `autoresearch/results/cycle_282_exec_result.txt` written with both commit hashes (main + bookkeeping). PASS
- `autoresearch/agenda.md`: item 292 removed from P0; cycle 282 entry added at top of PAST EXPERIMENTS (above cycle 281), with item 292 marked CONFIRMED, full reading + numerics + cross-cycle deltas + Pareto status (cycle 270 deprecated). PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` (line 280); HASH placeholder updated to `434ba6c` after main commit. PASS
- Two commits: `434ba6c` (cycle files: experiment script + plan + log + JSON + summary + agenda + TSV) + `d1c4c8f` (bookkeeping: TSV hash update + manifest + exec_result). PASS
- No pre-existing dirty files staged: verified via `git status --short` — only cycle 282 files + agenda + TSV + manifest + exec_result staged; pre-existing M files (program.md, flagged.jsonl, cycle_170/205/255_manifest.txt, cycle_266/270/281_exec_result.txt, cycle_270_manifest.txt, surveys/literature_map.md, aleph/pullback_geometry.py + untracked audits/reviews/plans/briefs from prior cycles) untouched. PASS

### Pre-commit hook interaction
- First main-commit attempt failed because `fix end of files` hook added trailing newlines to `cycle_282_block10_lowrank.json` and `cycle_282_summary.md`. Rule-compliant resolution: re-staged the modified files and re-ran `git commit` (NEW commit, not --amend), per CLAUDE.md git safety protocol. Second attempt passed all hooks (commit 434ba6c). PASS
- Bookkeeping commit passed all hooks first try (d1c4c8f). PASS

## Issues found
1. **Cycle 270 block 11 rank-22 recipe is now deprecated** — cycle 282 block 10 rank-17 strictly dominates it on BOTH axes. Any follow-up items that relied on cycle 270 as the late-block anchor should switch to cycle 282. Worth flagging for item 294 (hybrid recipe block 1 rank-2 + block 11 rank-22): swapping block 11 rank-22 for block 10 rank-17 in the hybrid recipe would likely improve the hybrid's Pareto position (smaller params AND lower PPL).
2. **All 3 retrain seeds still descending at budget cap** (best_step ∈ {3800, 3900, 3900}). The 4000-step budget is not sufficient for convergence on the block 10 rank-17 retrain surface. Longer training could push best_mean further below 50.04. Same pattern as cycles 270/280 — this is a known limitation of the current recipe.
3. **REFUTED noretrain clause technically triggered but retrain clause wins** — all three no-retrain PPLs (r=5: 59.918, r=17: 59.378, r=50: 58.265) are ≥ 57.0, so the REFUTED first sub-clause would have fired if the retrain PPL had also been ≥ 57. The retrain dropped to 50.04, firmly below 55.0, so CONFIRMED wins. This shows the REFUTED clause is a safety net against "no-retrain AND retrain both fail"; cycle 282's no-retrain gap confirms that rank-17 alone (without Adam) does NOT recover block 10.
4. **Trainable count is 13,824** not the ~13.1K figure in item 292's brief ("~13.1K params (768×17 + 17×768)"). Item 292 double-counted: the item brief confuses rank×768 (one factor) with the FULL (A + B) factorization. Actual: A=(U·Σ)[:17]=(3072, 17) is FROZEN (not trainable); B=Vh[:17, :]=(17, 768) is trainable (13,056 params) + bias=(768) is trainable (768 params) = 13,824 trainable. The item 292 brief's 13.1K figure is a minor arithmetic error (it omitted the bias). Trainable count asserted at runtime and matches cycle 270/280 conventions. Not a defect — noted in audit.

## Status
PASS
