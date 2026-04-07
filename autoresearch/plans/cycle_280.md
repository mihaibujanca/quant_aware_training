# Cycle 280 Plan — Item 290: Block 1 c_proj rank-r weight-SVD sweep

## Item (verbatim)
290 (P0): Block 1 c_proj rank-r weight-SVD sweep — r ∈ {1, 2, 5, 10}. Block 1 c_proj output eff_rank=1.94 (cycle 278) → rank-2 should capture most of its signal.

## Hypothesis (item 290 verbatim)
Block 1 is structurally cheaper to compress at the c_proj weight level than block 11 → any r ≤ 10 no-retrain ≤ 55 PPL OR r=2 retrain best_mean ≤ 54 PPL. Rationale: cycle 278 measured c_proj_1 OUTPUT eff_rank=1.94 (top-1 energy 92.94%) — the most concentrated block of its kind in the early stack — so the literal rank-r weight approximation should require far less than the r=22 cycle 270 needed at block 11.

## Activation-vs-weight SVD caveat
Cycle 270 showed at block 11 that weight-SVD ≠ activation-SVD: c_proj_11 OUTPUT eff_rank=22.49 did NOT translate into a cheap rank-22 WEIGHT approximation (rank-22 no-retrain PPL=59.68, +5.79 above float; only Adam retrain recovered to 54.47 by a 0.031 margin). This cycle tests whether block 1 behaves the same way or whether its much more extreme eff_rank=1.94 DOES translate into a cheap weight approximation. The "weight SVD is a different object from activation SVD" lesson is the prior — confirmation would be notable.

## Setup
New script `experiments/block1_lowrank_cycle280.py`. Adapt cycle 270 template verbatim with TARGET_BLOCK=1, RANKS_NORETRAIN=[1, 2, 5, 10], RETRAIN_RANK=2. `LowRankCProj` wrapper unchanged (HF Conv1D c_proj.weight shape (3072, 768), A=U[:, :r]·Σ[:r] frozen buffer, B=Vh[:r, :] trainable, bias trainable). For r=2: trainable = 2×768 (B) + 768 (bias) = 2304 params. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (B + bias snapshots), 3 seeds {42, 137, 2024}, disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128, MPS GPT-2 small wikitext-2 val. ~1.5 min/fit × 3 + 4 no-retrain evals ≈ 6 min wall.

## Verdict thresholds (item 290 verbatim)
- CONFIRMED iff any r ∈ {1,2,5,10} no-retrain PPL ≤ 55.0 OR r=2 retrain best_mean ≤ 54.0.
- REFUTED iff all r ∈ {1,2,5,10} no-retrain PPL ≥ 60.0 AND r=2 retrain best_mean ≥ 55.0.
- INCONCLUSIVE otherwise.

## Steps
1. Create `experiments/block1_lowrank_cycle280.py` from the cycle 270 template: swap TARGET_BLOCK to 1, RANKS_NORETRAIN to [1, 2, 5, 10], RETRAIN_RANK to 2, update thresholds and output paths, adjust trainable-count assertion to `RETRAIN_RANK*768 + 768 = 2304`, add cycle 278 block-1 eff_rank=1.94 + top-1 energy 0.9294 baseline, and keep FLOAT_BASELINE_REF=53.895.
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/block1_lowrank_cycle280.py 2>&1 | tee autoresearch/logs/cycle_280_run.log`.
3. Save `autoresearch/results/cycle_280_block1_lowrank.json` with per-r noretrain PPL+energy, per-seed retrain history, aggregate best_mean±std, verdict + verdict_logic, config, SVD summary (top-5 SVs, S[0], S[1], S[4], S[9], S[-1], energy fractions), cycle 278 eff_rank=1.94, cycle 270 block-11 rank-22 baseline (54.4687), cycle 266 [1,4] K=512 (43.852), float (53.895).
4. Save `autoresearch/results/cycle_280_summary.md` with verdict, no-retrain table, retrain seed table, verdict logic, cross-cycle deltas.

## Bookkeeping
Delete item 290 from NEXT/P0; insert `### Cycle 280` at top of PAST EXPERIMENTS (above `### Cycle 279`) with verdict + cross-cycle deltas. Append TSV row to `autoresearch/logs/experiments.tsv` (initial HASH placeholder, update after commit). Stage ONLY cycle 280 files (no pre-existing dirty files). Primary commit `[autoresearch] cycle 280: block 1 rank-r c_proj — <verdict>`, capture hash, update TSV hash, write `autoresearch/results/cycle_280_manifest.txt`, bookkeeping commit. Write `autoresearch/results/cycle_280_exec_result.txt`. Audit: `autoresearch/audits/cycle_280.md` + `autoresearch/results/cycle_280_audit.txt`.

## Verification
JSON+log present; each r ∈ {1,2,5,10} no-retrain PPL recorded; per-seed r=2 retrain history recorded; restore matches recorded best (script assert); verdict matches verbatim threshold logic; trainable param count asserted == 2304; init PPL deterministic and equals noretrain[r=2] (assert).
