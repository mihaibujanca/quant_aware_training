# Cycle 311 Plan — Item 321

## Item 321 (verbatim Hypothesis)
cycle 292 showed b1 at rank-1 trains to PPL 50.08 (vs float 53.89 — note cycle 292 is BELOW float, an over-fit-on-calib effect; vs eval-slice float 53.8949 the gap is small). Cycle 304 showed only b11 has k₉₅=2 in pure curvature energy. The question: at the **trained-Adam** rank-1 floor (cycle 280-style A_frozen=U[:,:1]·S[:1] + B_trainable + bias_trainable, NOT the pure SVD truncation of cycle 304), how does each concentrator block compare? **Setup**: For each block in {b2, b3, b10, b11}: install single-block LowRankCProj with r=1 (cycle 292 template); freeze rest of model; Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, SEEDS=[42, 137, 2024]; calibrate on 200 wikitext-2 val × 128 tok; eval on the same 100 wikitext-2 val × 128 tok slice as cycles 280/284/291/305. Report `r1_best_mean_ppl[4]` and trainable param count per block (768·1 + 768 = 1536). Compare each against the cycle-282-family multi-rank single-block anchor (b2: 2 → ?, b3: 24 → ?, b10: 5 → ?, b11: 22 → ?). **Verbatim conditions**: ≥3/4 blocks reach PPL ≤ 60 at r=1 → CONFIRMED ("rank-1 is a universal concentrator floor at the trained-Adam level; cycle 304's k₉₅ census understates the trainable rank-1 capacity"); ≤1/4 blocks reach r=1 PPL ≤ 60 → REFUTED ("b1 is the unique trained-rank-1 concentrator alongside b11; the cycle 282 family is heterogeneous in trainable rank-1 capacity"); else INCONCLUSIVE.

## Dependencies
- `experiments/block1_rank1_cycle292.py` — direct rank-1 retrain template (LowRankCProj, SVD, joint Adam, best-state restore).
- Multi-rank anchors: b2 r=2 → 49.3290 (c285); b3 r=24 → 50.5935 (c288); b10 r=5 → 51.4659 (c289); b11 r=22 → 54.4687 (c270). Cycle 292 b1 r=1 → 50.0794. Cycle 285 b2 r=1 noretrain ref = 72.3808. Float ref 53.8949.

## Approach
1. New script `experiments/rank1_all_concentrators_cycle311.py`. Copy cycle 292 LowRankCProj / eval_ppl / SVD / Adam loop verbatim. Outer loop over `BLOCKS = [2, 3, 10, 11]`.
2. Load GPT-2 once for the float baseline; for each block reload fresh per seed (cycle 292 pattern).
3. For each block: SVD c_proj.weight on CPU float32; r=1 no-retrain check; for each seed in [42, 137, 2024] install LowRankCProj(r=1), freeze rest, Adam LR=1e-4 4000 steps, EVAL_EVERY=100, best-state restore (assert |restored-best|<1e-3), record init/best/best_step/final/restored/history.
4. Aggregate per block: best_mean, best_std, best_min, best_max over 3 seeds. Verdict per item 321 (count blocks with best_mean ≤ 60).
5. Cross-check b2 r=1 no-retrain vs cycle 285 ref (72.3808) within 0.01.

## Files
- `experiments/rank1_all_concentrators_cycle311.py` (new)
- `autoresearch/results/cycle_311_rank1_all_concentrators.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_311_run.log`, `autoresearch/audits/cycle_311.md`, `autoresearch/results/cycle_311_audit.txt`

## Config
BLOCKS=[2,3,10,11]; RETRAIN_RANK=1; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; DEVICE=mps; gpt2; wikitext-2 val. Trainable per fit = 1·768 + 768 = 1,536. Wall ~4 blocks × 3 seeds × ~1.3 min ≈ 16 min on mps.

## Verification
- Float PPL within 0.5 of 53.8949.
- Per (block, seed): trainable count == 1,536; init PPL == r=1 no-retrain PPL within 1e-2; |restored − best| < 1e-3.
- b2 r=1 no-retrain matches cycle 285 ref 72.3808 within 0.01.

## Verdict logic (item 321 verbatim)
n_pass := |{b ∈ {2,3,10,11} : r1_best_mean_ppl[b] ≤ 60}|. CONFIRMED iff n_pass ≥ 3. REFUTED iff n_pass ≤ 1. INCONCLUSIVE otherwise.

## Baselines to record
float 53.8949; b1 r=1 (c292) 50.0794; b2 r=2 49.3290; b3 r=24 50.5935; b10 r=5 51.4659; b11 r=22 54.4687; b2 r=1 noretrain (c285) 72.3808.

# APPROVE
