# Cycle 290 Audit — Item 300

**Audit status: PASS**

## Scope
Verify cycle 290 execution adheres to item 300's verbatim Hypothesis, verdict thresholds, and the cycle 287 template it derives from; verify arithmetic and cross-check against the raw JSON; cross-check cycle 290 step-4000 evals against cycle 286 (matched-config reproducibility check, the unique sanity test for a budget extension).

## Verdict verification

Item 300 thresholds (verbatim):
- CONFIRMED iff best_mean ≤ 42.5 (block 2 beats cycle 287 [1] K=512 8k=42.785 by ≥0.29 PPL — new dominant PPL Pareto winner at 394K params)
- REFUTED iff best_mean ≥ 43.0 (block 2 buys < 0.56 PPL from extension; [1] still wins)
- INCONCLUSIVE otherwise (dead zone (42.5, 43.0))

Measured:
- best_mean = 42.75211 (from JSON `aggregates['2'].best_mean`)

Arithmetic:
- 42.75211 ≤ 42.5  → FALSE (over by 0.25211) → CONFIRMED clause NOT triggered
- 42.75211 ≥ 43.0  → FALSE (under by 0.24789) → REFUTED clause NOT triggered
- → **INCONCLUSIVE** ✓ (matches manifest, exec_result, JSON)

The verdict sits almost exactly in the middle of the dead zone (0.252 from CONFIRMED, 0.248 from REFUTED).

## Per-seed and aggregate sanity

Seeds (JSON `subset_results['2']`):
- 42  : init 55.46118 → best 42.7767 @ step 8000 → final@8000 42.7767 → restored 42.7767
- 137 : init 55.46119 → best 42.7723 @ step 7800 → final@8000 42.7793 → restored 42.7723
- 2024: init 55.46118 → best 42.7074 @ step 8000 → final@8000 42.7074 → restored 42.7074

Aggregate (population std, recomputed from per-seed bests):
- mean(42.7767, 42.7723, 42.7074) = (42.7767 + 42.7723 + 42.7074)/3 = 128.2564/3 = 42.75213 ✓
  (matches reported 42.75211 within rounding)
- deviations: (+0.02457, +0.02019, −0.04471)
- squared:    (0.0006037, 0.0004076, 0.0019990); sum = 0.0030103; /3 = 0.0010034
- std = sqrt(0.0010034) = 0.03168 ✓ (matches reported 0.03169 within 5dp)
- min = 42.7074 (seed 2024) ✓; max = 42.7767 (seed 42) ✓

All three seeds: init PPL ≈ 55.46118 (max |Δ| across seeds = 9e-7) — deterministic per cycle 286, MPS noise level ✓
All three seeds: |restored − best| < 1e-3 (exact for seeds 42 and 2024 since best=final; seed 137 best at 7800 < final 7800.00696 at 8000, restore exact) ✓
best_step distribution: {8000, 7800, 8000} → best_step_mean = 7933.33 — 2/3 still BUDGET-BOUNDED at 8000 cap, 1 (seed 137) overfit slightly past 7800. Aggregate is budget-bounded.

## Cross-checks vs cycle 286 (matched-config reproducibility — UNIQUE to budget extensions)

Step-4000 evals must match cycle 286 within 1e-4 per seed (the only diff is N_STEPS=8000 vs 4000):

| seed | cycle 290 step-4000 | cycle 286 step-4000 | abs diff   | tolerance | status |
|------|---------------------|---------------------|------------|-----------|--------|
| 42   | 43.530286           | 43.530270           | 1.54e-05   | 1e-4      | PASS   |
| 137  | 43.593368           | 43.593362           | 5.51e-06   | 1e-4      | PASS   |
| 2024 | 43.564445           | 43.564442           | 2.49e-06   | 1e-4      | PASS   |

Init PPLs must match cycle 286 within 1e-4 per seed:

| seed | cycle 290 init      | cycle 286 init      | abs diff   | tolerance | status |
|------|---------------------|---------------------|------------|-----------|--------|
| 42   | 55.4611762          | 55.4611872          | 1.10e-05   | 1e-4      | PASS   |
| 137  | 55.4611853          | 55.4611822          | 3.17e-06   | 1e-4      | PASS   |
| 2024 | 55.4611763          | 55.4611872          | 1.08e-05   | 1e-4      | PASS   |

Reproducibility cross-checks PASS at 1.5e-5 worst-case (well within the 1e-4 tolerance the plan asked for). The 1e-5 noise is MPS non-determinism on identical operations across runs. The −0.811 PPL gain (43.563 → 42.752) is therefore unambiguously attributed to the extra 4000 Adam steps and NOT to any silent config drift.

## Config / plan adherence

Plan (cycle_290.md) vs execution:
- SUBSETS=[[2]] ✓
- K=512 ✓
- N_STEPS=8000, EVAL_EVERY=200 ✓
- LR=1e-4 ✓
- SEEDS=[42, 137, 2024] ✓
- best-state restore ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, disjoint ✓
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit = 512×768 + 768 = 393,984 asserted in-script (line 150 of script) ✓
- Float PPL 53.895 within 0.5 of reference 53.895 ✓ (exact)
- Template: `experiments/one_block_1_k512_8k_cycle287.py` adapted verbatim. Diff vs cycle 287: SUBSETS=[[1]]→[[2]], OUTPUT_PATH/SUMMARY_PATH cycle_287→cycle_290, baseline constants (286/287/271 vs 271/286), thresholds (≤42.5 / ≥43.0 vs ≤43.0 / ≥43.7), JSON metadata ("cycle"/"item"/"verdict logic" labels). Same CompressedMLP, collect_mlp_io, fit_block_lstsq, eval_ppl, run_seed loop, snapshot/restore. ✓

## Baseline cross-checks (vs JSON / manifest)

| baseline                          | recorded   | from JSON  | match |
|-----------------------------------|------------|------------|-------|
| float                             | 53.89500   | 53.89489   | ✓ (within 0.0001) |
| cycle 286 [2] K=512 4k            | 43.56269   | 43.56269   | ✓ |
| cycle 287 [1] K=512 8k            | 42.78533   | 42.78533   | ✓ |
| cycle 271 [1] K=512 4k            | 43.82045   | 43.82045   | ✓ |

Delta cross-checks (from manifest, recomputed):
- 42.75211 − 53.89489 = −11.14278 → manifest reports −11.143  ✓
- 42.75211 − 43.56269 =  −0.81058 → manifest reports −0.811  ✓ (pure budget gain on block 2)
- 42.75211 − 42.78533 =  −0.03322 → manifest reports −0.033  ✓ (matched 8k vs [1])
- 42.75211 − 43.82045 =  −1.06834 → manifest reports −1.068  ✓

## Hypothesis falsification audit

Item 300 stated Hypothesis: "at 8k steps, [2] K=512 drops below 42.0 (matches or beats cycle 287 [1] K=512 by ≥0.78 PPL)".

Two clauses to test:
1. best_mean < 42.0?  → measured 42.752 > 42.0 by 0.752 PPL. FALSE.
2. margin over [1] ≥ 0.78 PPL?  → measured margin 42.785 − 42.752 = 0.033 PPL.
   Required ≥ 0.78. Measured / required = 0.033/0.78 = 0.042 (4.2% of predicted).
   FALSE by a factor of 23.

Both Hypothesis clauses FALSIFIED in magnitude. The directional claim ([2] beats [1]) is preserved in sign (+0.033 in [2]'s favor) but the magnitude is so far below the predicted 0.78 PPL that it sits inside 1-sigma of either run's std (cycle 287 best_std=0.0125, cycle 290 best_std=0.0317; neither exceeds 0.033). Statistically a TIE.

The "cycle 286 4k advantage of [2] over [1] (-0.258) is BUDGET ARTIFACT" interpretation is supported:
- At 4k, [2] beat [1] by 0.258 PPL (cycle 286 vs cycle 271).
- At 8k, [2] beats [1] by 0.033 PPL (cycle 290 vs cycle 287).
- Gap shrinks by factor 7.8× when both are doubled to 8k.
- At 8k, both subsets sit at the same statistical floor (within noise) at the 393,984-param tile.

## Pareto reading — audit

Cycle 290 (393,984 params, 42.752 PPL) vs existing anchors:
- **TIE/co-winner** with cycle 287 [1] K=512 8k (393,984, 42.785): nominal +0.033 PPL in [2]'s favor at SAME param count, but margin INSIDE 1-sigma of either run. No clear PPL Pareto winner at this tile.
- **STRICTLY DOMINATES** cycle 286 [2] K=512 4k (393,984, 43.563): SAME param count, −0.811 PPL. Cycle 286 superseded.
- **STRICTLY DOMINATES** cycle 271 [1] K=512 4k (393,984, 43.821): SAME param count, −1.068 PPL.
- All four data points sit at the SAME 393,984-param tile. Among them, cycle 290 and cycle 287 are co-winners; cycles 286 and 271 are now dominated.

Net: cycle 290 does NOT add a new Pareto point at a new param tile. It adds a co-winner at an existing tile and supersedes the prior 4k entries at that tile.

## Geometric reading — audit

(i) **Block 2 channel-K topk + Adam at K=512 8k matches block 1 within noise.** Cycle 278 measured block 2 c_proj OUTPUT eff_rank = 1.05 (extreme INPUT concentration; the 1D-active subspace dominates output covariance). Cycle 285 b2 r=2 weight-SVD = 49.33 PPL (DOMINATED by cycle 280 b1 r=2 = 49.03). The "structure lives in INPUT not WEIGHT for block 2" reading from cycle 286 holds — channel-K topk on block 2 reaches 42.75 while weight-SVD on block 2 plateaus at ~49.

(ii) **The per-K [2]-beats-[1] trend at small K does NOT extrapolate to large K + matched budget.** Cycle 283 measured K∈{16, 32, 64} with [2] beating [1] by 1.75/0.53/(extrapolated) PPL. Cycle 286 at K=512 4k preserved this trend with -0.258 PPL. Cycle 290 at K=512 8k collapses it to -0.033. The trend is K-and-budget conditional: at small K and matched budget the input concentration of block 2 matters because the topk subset is sparse; at K=512 the 512-channel subset is wide enough that block 1 and block 2 both saturate their respective floors and the difference vanishes.

(iii) **Both subsets are at the same floor at K=512 8k.** Combining cycle 287 (best_std=0.0125) and cycle 290 (best_std=0.0317), the joint std is roughly 0.034 PPL, comparable to the gap. This is the signature of two configurations sitting in the same loss basin within seed-to-seed noise — there is no information in the +0.033 sign.

(iv) **Budget-extension family observation.** Both K=512 single-block extensions buy ~0.8-1.0 PPL on doubling (cycle 287 −1.035, cycle 290 −0.811). Cycle 287 buys slightly more because cycle 271 was further from the floor than cycle 286. The two extensions converge to within 0.033 PPL by step 8000.

(v) **Auditor recommendation for next agenda items**: cycle 290 closes the [1]-vs-[2] K=512 question with a TIE verdict. Future single-block K=512 work should not pursue [1] vs [2] further. Open directions: (a) push K higher (K=1024) to test if the floor moves further; (b) add EVEN MORE budget (16k steps) to test whether [1] and [2] separate at deeper Adam; (c) test 2-block [1, 2] K=512 8k joint, since the two are now known to be co-winners — cycle 281 INCONCLUSIVE was at 4k.

## Artifact existence

- experiments/block2_k512_8k_cycle290.py — exists, 376 lines
- autoresearch/plans/cycle_290.md — exists, ≤25 lines, 8 top-level headings
- autoresearch/briefs/cycle_290.md — exists (from brief stage)
- autoresearch/results/cycle_290_block2_k512_8k.json — exists, valid JSON, verdict="INCONCLUSIVE"
- autoresearch/results/cycle_290_summary.md — exists
- autoresearch/results/cycle_290_manifest.txt — exists, commit=df14bf0
- autoresearch/results/cycle_290_exec_result.txt — exists, commit=df14bf0
- autoresearch/logs/cycle_290_run.log — exists
- Primary commit df14bf0 exists in git log
- experiments.tsv row 290 appended with 8 columns ✓
- agenda.md: item 300 deleted from NEXT P0, added to PAST as Cycle 290 ✓

## Corrections

None. Provisional outcome holds.

## Final status

**PASS** — keep provisional outcome INCONCLUSIVE. No agenda fix required beyond the move from NEXT to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = INCONCLUSIVE with best_mean = 42.7521 ± 0.0317 (dead zone 42.5–43.0).

Auditor note: cycle 290 closes the [1]-vs-[2] K=512 question by demonstrating a STATISTICAL TIE at matched 8k budget. The cycle 286 4k advantage of [2] was a BUDGET ARTIFACT — block 1's 4k cap was binding more tightly. The hypothesis ("[2] should still win at K=512 IF given the same Adam budget") was preserved in sign (+0.033 PPL) but falsified in magnitude (>20× smaller than predicted). Cycle 290 supersedes cycle 286 at the 393,984-param tile. The next natural follow-up is the joint [1, 2] K=512 8k experiment (cycle 281 INCONCLUSIVE was at 4k), since both [1] and [2] are now known to be co-winners at this tile.
