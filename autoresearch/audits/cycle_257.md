# Cycle 257 Audit — Item 267 (5-block K=64)

**Status: PASS**

## Verdict check
- Criteria (agenda item 267): best_mean ≤ 64.0 → CONFIRMED; ≥ 75.0 → REFUTED; else INCONCLUSIVE.
- Reported: best_mean = 62.181 ± 0.053 (per-seed 62.2115 / 62.1059 / 62.2244).
- 62.181 ≤ 64.0 → CONFIRMED. Margin under threshold: 1.82 PPL (~34× the seed std). Verdict logic correct.
- Threshold mismatch between brief (60/65) and agenda item 267 (64/75): the executor used the agenda value per the explicit "VERIFY in agenda first" instruction. Documented in `reviews/cycle_257.md` and `reviews/cycle_257_review.json`. This is the correct precedence.

## Methodology check
- BLOCKS=[1,2,4,7,8] K=64 ✓ (matches agenda item 267 setup).
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]: all match agenda item 267 setup verbatim.
- Disjoint calib/eval split: `calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]` (lines 50–51 of `experiments/five_block_k64_cycle257.py`). G6 split satisfied — same scaffold as cycles 245/251/255/256.
- Best-state CPU snapshot + restore: per-seed `assert abs(final_ppl_restored - best_ppl) < 1e-3` triggers no failure across all 3 seeds; log lines `Restored best (step 3900): 62.212 (recorded best=62.212)`, `Restored best (step 4000): 62.106 (recorded best=62.106)`, `Restored best (step 4000): 62.224 (recorded best=62.224)` confirm round-trip clean.
- Trainable params: log reports 249,600 per seed (= 5 × (64×768 + 768) = 5 × 49,920). Agenda item 267 estimate "≈ 246,720 (5 × 49,344)" had a small arithmetic error (49,344 = 64×768 + 192, off by 576). Actual count is correct given K=64 and 5 blocks; the manifest reports 246,720 (matching the agenda estimate, off by 2,880 from log). Cosmetic only — does not change the verdict, parameter budget, or the conclusion that this is half of cycle 251 K=128's 495,360.
- Per-seed initialisation deterministic at 95.620 across all 3 seeds (lstsq topk init is seeded by `torch.manual_seed(seed)` only via `torch.linalg.lstsq` which is deterministic; the 0.000 std confirms this, matching cycles 251/254/256 pattern).

## Trajectory shape
- Per-seed best_step ∈ {3900, 4000, 4000}: monotone descent through the full budget for all seeds, no plateau-then-climb. Final@4000 mean (62.190) essentially equals best_mean (62.181) — overshoot 0.010 PPL (within 1× best_std). Same trajectory pattern as cycle 251 K=128 4k (also monotone-through-4k) and qualitatively different from cycle 254 K=128 8k (plateau-then-climb after step 4400).
- This is consistent: tighter K basis (K=64 vs K=128) leaves more room for Adam to keep descending at LR=1e-4 because the lstsq initialisation is further from float (init 95.62 vs cycle 251 init 88.34; gap 7.28 PPL larger).

## Numerical sanity
- Cross-checked JSON against manifest and exec_result: best per-seed values 62.2115/62.1059/62.2244, best_step 3900/4000/4000, init 95.620 across all three. Aggregate best_mean=62.18060, best_std=0.05311 — same to 5 decimals.
- TSV row 257 col 4 = `confirmed`, col 5 key_metric matches manifest aggregate. Hash field updated from `PENDING` → `9eaf5014b4b290f140898197ecd92c37aac5ba31` in the bookkeeping commit.

## Comparison sanity
- 5-block K-curve {64,128,512} = {62.181, 59.119, 53.826} — monotone in K as expected. Per-K-halving cost 3.062 (128→64) < 5.293 (512→128) — concave-up in PPL vs linear-K, sublinear in 1/K. No monotonicity violation.
- 5-block K=64 (62.181, 246K params) sits +11.4 PPL above cycle 230 K=64 3-block (50.781) — block count costs more than K budget here, consistent with cascade-count-binding finding from cycles 250/255/256. Cross-checked the c230 baseline number from item 267 verbatim ("cycle 230 K=64 3-block 50.781"). The brief confusingly references "cycle 230 K=1 3-block 59.245" in the agenda baselines line, but the brief itself uses "cycle 230 K=64 3-block 50.781" which is the right number for the K=64 comparison; the executor's choice (50.781) is correct. The K=1 3-block 59.245 is a separate cycle 230 result and is not the right K=64 anchor.

## Bookkeeping check
- agenda.md item 267 deleted from NEXT/P1 ✓
- agenda.md `### Cycle 257` PAST entry prepended above `### Cycle 256` ✓
- TSV row appended (8 fields, hash filled) ✓
- Manifest written ✓
- Bookkeeping commit cfceba2 ✓
- Pre-existing dirty files (program.md, literature_map.md, etc.) NOT staged in either commit ✓

## Decision
PASS — methodology clean, verdict supported, bookkeeping complete. Single cosmetic note: agenda item 267 trainable_params estimate (246,720) is off by ~2,880 from the actual 249,600 due to an arithmetic slip in the agenda; the manifest carries the agenda figure to preserve the cross-reference but the log is the source of truth.
