# Cycle 284 Plan — Item 294

## Item (verbatim Hypothesis)
since BOTH blocks are weight-SVD patches, joint Adam should be able to compose them additively, unlike cycle 275's heterogeneous mix. Target: match cycle 270's 54.47 PPL with slightly more params but a "bracketing" structure that captures early+late concentration simultaneously.

## Approach
Compose cycle 280 (block 1 r=2, A1=U[:,:2]·Σ[:2] frozen, B1=Vh[:2,:] + bias trainable) with cycle 270 (block 11 r=22, A11 frozen, B11 + bias trainable). Replace BOTH c_proj modules with `LowRankCProj` instances on a fresh model. Joint Adam over [B1, b1, B11, b11] only — every other param frozen. Same calib/eval/seed splits as cycles 270/280.

## Files
- `experiments/rank_hybrid_1_11_cycle284.py` (new)
- `autoresearch/results/cycle_284_rank_hybrid_1_11.json`, `_summary.md`, `_manifest.txt`
- `autoresearch/logs/cycle_284_run.log`

## Config
TARGET_BLOCKS=(1,11); RANKS=(2,22); LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = (2·768 + 768) + (22·768 + 768) = 2,304 + 17,664 = 19,968. ~5 min × 3 ≈ 15 min wall.

## Verification
- Float PPL within 0.5 of 53.895.
- SVD on each block independently; assert top singular values reproduce cycle 270 / cycle 280 references.
- init_ppl deterministic per seed (lstsq init seed-independent).
- Trainable param count == 19,968 asserted; everything else `.requires_grad_(False)`.
- Per-seed best-state restore (B1, b1, B11, b11): assert `|restored - best| < 1e-3`.
- Restore original c_proj weights between sweep stages.

## Verdict logic (item 294 verbatim)
- CONFIRMED iff best_mean ≤ 55.0 (matches cycle 270 band → composition is non-destructive).
- REFUTED iff best_mean ≥ 58.0 (composition is destructive, same as cycle 275).
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 270 b11 r22 = 54.4687; cycle 280 b1 r2 = 49.0324; cycle 282 b10 r17 = 50.0445; cycle 275 hetero hybrid REFUTED 48.2004 (different scoring axis).
