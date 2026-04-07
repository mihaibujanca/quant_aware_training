# Cycle 293 Plan — Item 303

## Dependencies
None (cycle 278 eff_rank census; cycle 282/288 templates already on disk).

## Item (verbatim Hypothesis)
rank-72 retrain reaches ≤55 PPL at 56,064 trainable params (72×768 + 768) — extends the eff_rank-recipe to the largest concentrator and tests whether early-block depth helps or hurts.

## Approach
Adapt `experiments/block3_lowrank_cycle288.py` verbatim with `TARGET_BLOCK=0`. Stage A: no-retrain forward swap of block 0 `c_proj.weight` with rank-r SVD approximations for r ∈ {10, 30, 72}. Stage B: retrain at `RETRAIN_RANK=72`, joint Adam over `(B, bias) = (72,768)+(768) = 56,064` trainable params, 3 seeds {42,137,2024}, best-state restore. Cycle 198 establishes block 0 transforms embedding diversity (input rank 269) and cycle 187 shows dropping it breaks the model (PPL 1923) — block 0 is irreplaceable, but the question here is whether its `c_proj` matrix is rank-compressible like later concentrators.

## Files
- `experiments/block0_lowrank_cycle293.py` (new)
- `autoresearch/results/cycle_293_block0_lowrank.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`, `_audit.txt`
- `autoresearch/plans/cycle_293.md`, `autoresearch/audits/cycle_293.md`
- `autoresearch/logs/cycle_293_run.log`

## Config
TARGET_BLOCK=0; RANKS_NORETRAIN=[10,30,72]; RETRAIN_RANK=72; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 72*768 + 768 = 56,064. Wall ~8-10 min.

## Verification
- Float PPL within 0.5 of 53.895.
- SVD shapes: U(3072,768), S(768,), Vh(768,768); print top-5 SVs and energies at r∈{10,30,72}.
- Restore original `c_proj.weight` between sweep stages; assert restore PPL ≈ float (<1e-3).
- Per-seed init PPL identical (deterministic SVD init) and equal to noretrain r=72 PPL.
- Trainable param count == 56,064 asserted; everything else `requires_grad_(False)`.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.

## Verdict logic (item 303 verbatim)
- CONFIRMED iff rank-72 retrained best_mean ≤ 55.0.
- REFUTED iff rank-72 retrained best_mean ≥ 58.0.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 = 49.0324 (2,304 params); cycle 282 b10 r=17 = 50.0445 (13,824 params); cycle 285 b2 r=2 = 49.3290 (2,304 params); cycle 288 b3 r=24 = 50.5935 (19,200 params); cycle 270 b11 r=22 = 54.4687 (17,664 params); cycle 292 b1 r=1 = 50.0794 (1,536 params); cycle 278 b0 c_proj eff_rank = 71.93; cycle 187 dropping b0 → PPL 1923; cycle 198 b0 input rank 269.
