# Cycle 280 Audit

**Status: PASS**

## Item
290 (P0): Block 1 c_proj rank-r weight-SVD sweep — r ∈ {1, 2, 5, 10}. Block 1 c_proj output eff_rank=1.94 (cycle 278) → does it translate to a rank-2 no/low-cost patch?

## Verdict
CONFIRMED (retrain clause only) — rank-2 retrain best_mean = 49.0324 ± 0.0246 ≤ 54.0 by 4.968 PPL margin. No-retrain clause (any r ≤ 55.0) FAILS at min 56.730 (r=10). REFUTED clause (all r noretrain ≥ 60.0 AND retrain ≥ 55.0) does NOT trigger — max noretrain is 59.751 at r=1, just below the 60.0 refute threshold, and retrain best_mean 49.03 is far below the 55.0 refute threshold.

## Verification

### Plan adherence
- Plan: `autoresearch/plans/cycle_280.md` — 30 lines, 8 headings (Item, Hypothesis, Activation-vs-weight SVD caveat, Setup, Verdict thresholds, Steps, Bookkeeping, Verification). Hypothesis quoted verbatim from item 290 in the brief. Plan flags the weight-SVD ≠ activation-SVD prior from cycle 270. PASS
- Script: `experiments/block1_lowrank_cycle280.py` — adapted from cycle 270 template: TARGET_BLOCK=1, RANKS_NORETRAIN=[1,2,5,10], RETRAIN_RANK=2, CONFIRMED_RETRAIN_LE=54.0 (not 54.5 as in cycle 270), REFUTED_RETRAIN_GE=55.0 (not 58.0), trainable-count assertion updated to `RETRAIN_RANK*768 + 768 = 2304`. Verdict logic rewritten to `any r noretrain ≤ 55.0` and `all r noretrain ≥ 60.0` clauses per item 290 verbatim. LowRankCProj wrapper identical to cycle 270 (HF Conv1D semantics: frozen A=U[:,:r]·Σ[:r] buffer, trainable B=Vh[:r,:], trainable bias). PASS
- Hyperparameters: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. Match item 290 brief and constraints. PASS
- Trainable params per fit: 2,304 = 2*768 (B factor) + 768 (bias) — script-internal assert `n_trainable == expected_trainable == 2304` PASS for all 3 seeds. Matches brief exactly. PASS

### Run completion
- Script executed with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/block1_lowrank_cycle280.py > autoresearch/logs/cycle_280_run.log 2>&1`. Exit 0. PASS
- Log post-cleaned by stripping HF `Loading weights:` and `Map:` progress bars (1270 lines → 78 lines). PASS
- Wall: ~8 min total (4 forward-only PPL evals at 1.3s each + 3 retrain fits at 2.6 min each + SVD + reload overhead). Well under the ~18-min cap in the brief. PASS
- All 4 no-retrain ranks evaluated; restore PASSED back to float exactly. PASS
- All 3 retrain seeds completed; per-seed restore matches recorded best within 1e-3. PASS

### Numerical consistency
- Float PPL = 53.89489 — matches FLOAT_BASELINE_REF=53.895 to 4 dp. PASS
- No-retrain rank sweep monotone-decreasing in r (59.751 → 57.658 → 56.833 → 56.730). PASS
- Energy fractions monotone-increasing in r (0.0459 → 0.0757 → 0.1167 → 0.1523). PASS
- Restored original c_proj.weight after sweep → PPL exactly = 53.895 (assert PASS). PASS
- Init PPL deterministic across all 3 retrain seeds at 57.6581, matches noretrain[r=2]=57.6581 — the lstsq-style SVD factor init is seed-independent, only Adam shuffle differs. PASS
- Per-seed best PPLs: 49.0603 / 49.0367 / 49.0004 — best_mean = (49.0603 + 49.0367 + 49.0004) / 3 = 49.03246, std (population) = sqrt(((49.0603-49.03246)^2 + (49.0367-49.03246)^2 + (49.0004-49.03246)^2)/3) = 0.02465 — matches JSON aggregate {best_mean: 49.0324496019933, best_std: 0.02464882698467303} to 4 dp. PASS
- All 3 best_step values are at step 3900/3900/4000 (still descending at the budget cap) — indicates no overfit. PASS
- Per-seed restored matches recorded best to 4 dp. PASS

### Verdict logic
- Per item 290 (verbatim from agenda):
  - CONFIRMED iff any r ∈ {1,2,5,10} no-retrain ≤ 55.0 OR r=2 retrain best_mean ≤ 54.0
  - REFUTED iff all r ∈ {1,2,5,10} no-retrain ≥ 60.0 AND r=2 retrain best_mean ≥ 55.0
  - INCONCLUSIVE otherwise
- Script-implemented logic matches verbatim:
  - any noretrain ≤ 55.0? NO (min 56.730 at r=10; 56.730 > 55.0)
  - all noretrain ≥ 60.0? NO (r=10 is 56.730 < 60.0; none of the 4 ranks is ≥ 60)
  - r=2 retrain best_mean = 49.0324 → ≤54.0? YES (passes by 4.968 PPL); ≥55.0? NO
  - First clause: any noretrain ≤ 55 (FAIL) OR retrain ≤ 54 (PASS) → CONFIRMED. PASS
  - Refuted clause: all noretrain ≥ 60 (FAIL) AND retrain ≥ 55 (FAIL) → does NOT trigger (correct skip). PASS
- → CONFIRMED. PASS

### Geometric / cross-cycle reading
- **Headline finding**: the (2,304, 49.03) point is a new dominant Pareto anchor. It strictly dominates cycle 270's (17,664, 54.47) on BOTH axes (7.7× fewer trainable params AND −5.44 PPL). It is also the smallest-trainable recipe found so far that BEATS float (−4.86 PPL). PASS
- **Weight-SVD ≠ activation-SVD, generalised**: cycle 278 measured c_proj_1 OUTPUT activation eff_rank = 1.94 / top-1 energy 92.94%. Cycle 280 measures the literal weight SVD: top-2 weight energy only 7.57%, rank-2 no-retrain still costs +3.76 PPL. Second independent confirmation (after cycle 270 block 11) that these are different objects. The brief's prediction "rank-2 no-retrain ≤ 55" is WRONG by 2.66 PPL. PASS
- **Per-param Adam recovery 25.7× higher at block 1**: cycle 270 recovered 5.22 PPL with 17,664 params = 2.96e-4 PPL/param; cycle 280 recovers 8.63 PPL with 2,304 params = 3.75e-3 PPL/param. Eff_rank IS directionally predictive of retrained compressibility at the single-block c_proj weight-SVD level, refining cycle 273's "eff_rank not prescriptive at cascade level" finding. PASS
- **Rank-cost curve at block 1**: r=1 → +5.86, r=2 → +3.76, r=5 → +2.94, r=10 → +2.84. The curve is concave-down and flattens between r=5 and r=10 (only −0.10 PPL gain from doubling rank), suggesting the useful weight-SVD rank is small (~5-10) but the residual is nontrivial (+2.84 PPL even at r=10). Adam retrain at r=2 more than compensates — driving 57.66 → 49.03. PASS
- **All 3 retrain seeds best at step ≥3900 (still descending)**: longer training could push best_mean further below 49. Budget-bounded verdict. PASS

### Bookkeeping
- `autoresearch/plans/cycle_280.md` written (30 lines, 8 headings, hypothesis verbatim from item 290, activation-vs-weight SVD caveat, exact procedure, verdict thresholds, baselines, artifacts list, verification checklist). PASS
- `autoresearch/briefs/cycle_280.md` pre-existing (read-only context). PASS
- `experiments/block1_lowrank_cycle280.py` created from cycle 270 template with BLOCK/rank swaps and verdict-logic rewrite. PASS
- `autoresearch/logs/cycle_280_run.log` created, cleaned of tqdm progress bars (78 lines final). PASS
- `autoresearch/results/cycle_280_block1_lowrank.json` contains cycle/item/verdict/float_ppl/noretrain/retrain_seeds (per-seed history)/retrain_aggregate/verdict_logic/config/svd_summary/baselines. Verified. PASS
- `autoresearch/results/cycle_280_summary.md` written. PASS
- `autoresearch/results/cycle_280_manifest.txt` written with verdict, both commit hashes, all per-r noretrain values, all per-seed retrain values, deltas, geometric reading, artifact list. PASS
- `autoresearch/results/cycle_280_exec_result.txt` written with both commit hashes. PASS
- `autoresearch/agenda.md`: item 290 removed from P0; cycle 280 entry added at top of PAST EXPERIMENTS above cycle 279, with item 290 marked CONFIRMED, full reading + numerics + cross-cycle deltas + Pareto + open follow-ups. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` (line 278); HASH placeholder updated to `1109243` after main commit. PASS
- Two commits: `1109243` (cycle files: experiment script + plan + brief + log + JSON + summary + agenda) + `c47da2a` (bookkeeping: TSV + manifest). PASS
- No pre-existing dirty files staged: verified via `git status --short` — only cycle 280 files + agenda + TSV + manifest staged; pre-existing M files (program.md, results/cycle_170/205/255_manifest.txt, results/cycle_266/270_exec_result.txt, surveys/literature_map.md, logs/flagged.jsonl, aleph/pullback_geometry.py + untracked audits/reviews/plans/briefs) untouched. PASS

### Pre-commit hook interaction
- First main-commit attempt failed because `fix end of files` hook added trailing newlines to `cycle_280_summary.md` and `cycle_280_block1_lowrank.json`. Rule-compliant resolution: re-staged the modified files and re-ran `git commit` (NEW commit, not --amend), per CLAUDE.md git safety protocol. Second attempt passed all hooks. PASS

## Issues found
1. **No-retrain hypothesis was wrong by 2.66 PPL.** Item 290 predicted "rank-2 weight-SVD no-retrain ≤ 2 PPL vs float 53.895" — actual +3.76 PPL. Same structural reason as cycle 270: cycle 278 measured activation eff_rank=1.94, not weight-SVD rank. The "weight-SVD ≠ activation-SVD" finding from cycle 270 is now confirmed a second time at a different block.
2. **Retrain verdict passes CONFIRMED clause by a huge margin (4.968 PPL)** — not a tight threshold call like cycle 270's 0.031 margin. The result is very robust.
3. **All 3 retrain seeds best at step ≥3900 (step 3900/3900/4000)** — the 4000-step budget is still budget-bounded; longer training could likely push best_mean further below 49. Recording in the audit; not a defect, just a note for possible follow-up.
4. **HF `Loading weights:` and `Map:` tqdm progress bars polluted the log** (~1192 lines of progress bar output vs 78 lines of real content). Cleaned post-run by stripping those lines with a python3 one-liner before staging. Worth considering redirecting HF progress bars to /dev/null at the script level in future cycles (e.g. setting HF_HUB_DISABLE_PROGRESS_BARS=1).

## Status
PASS
