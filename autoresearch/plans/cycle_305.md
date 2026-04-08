# Cycle 305 Plan — Item 315

## Item 315 (verbatim Hypothesis)
joint Adam over b1 r=2 + b2 r=2 + b3 r=24 + b10 r=5 + b11 r=22 = 42.5K trainable reaches ≤51 PPL — additive composition (each block absorbs its own damage via per-block rank slack), establishing a new Pareto tile at the intersection of 5 recipe anchors. A destructive outcome (≥55 PPL) would say the single-block floors are INCOMPATIBLE: the model cannot simultaneously give up the low-rank subspaces at all concentrators. A middling outcome (51-55 PPL) quantifies the composition penalty per concentrator.

## Dependencies
- `experiments/rank_hybrid_1_11_cycle284.py` — 2-block homogeneous-hybrid template (LowRankCProj, SVD cache, joint Adam loop, best-state restore).
- Cycle 280 (b1 r=2 → 49.0324, 2,304 params), cycle 285 (b2 r=2 → 49.3290, 2,304), cycle 288 (b3 r=24 → 50.5935, 19,200), cycle 289 (b10 r=5 → 51.4659, 4,608), cycle 270 (b11 r=22 → 54.4687, 17,664). Cycle 284 hybrid b1+b11 → 52.8803 (CONFIRMED). Cycle 291 hybrid b1+b10 → 51.8008 (REFUTED).

## Approach
1. New script `experiments/multi_block_laser_cycle305.py`. Adapt cycle 284 verbatim with TARGET_BLOCKS=(1, 2, 3, 10, 11), RANKS=(2, 2, 24, 5, 22). HF Conv1D `c_proj.weight` is (3072, 768). For each block, SVD `W = U diag(S) Vh` on CPU float32; `A_l = (U[:, :r_l] * S[:r_l])` (3072, r_l) frozen buffer; `B_l = Vh[:r_l, :]` (r_l, 768) `nn.Parameter`; `bias_l` (768,) `nn.Parameter` initialised from original `c_proj.bias`. Forward `x @ (A_l @ B_l) + bias_l`.
2. Per seed in {42, 137, 2024}: reload GPT-2 small, install all 5 LowRankCProj modules, freeze every other parameter, joint Adam over the 5 (B_l, bias_l) pairs.
3. Joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore (assert `|restored - best| < 1e-3` per seed). Disjoint N_CALIB=200, N_EVAL=100, SEQ_LEN=128, MPS.
4. Verdict per item 315 verbatim thresholds.

## Files
- `experiments/multi_block_laser_cycle305.py` (new)
- `autoresearch/results/cycle_305_multi_block_laser.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_305_run.log`, `autoresearch/audits/cycle_305.md`, `autoresearch/results/cycle_305_audit.txt`

## Config
TARGET_BLOCKS=(1, 2, 3, 10, 11); RANKS=(2, 2, 24, 5, 22); LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42, 137, 2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = sum_l (r_l·768 + 768) = (2+2+24+5+22)·768 + 5·768 = 55·768 + 3,840 = 42,240 + 3,840 = 46,080. (Brief said ~42.5K — actual exact 46,080.) Wall ~10 min.

## Verification
- Float PPL within 0.5 of 53.895.
- Trainable param count == 46,080 asserted; everything else `requires_grad_(False)`.
- Per-seed init PPL deterministic (combined 5-block no-retrain rank product).
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.

## Verdict logic (item 315 verbatim)
- CONFIRMED iff best_mean ≤ 51.0.
- REFUTED iff best_mean ≥ 55.0.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; b1 r=2 49.0324; b2 r=2 49.3290; b3 r=24 50.5935; b10 r=5 51.4659; b11 r=22 54.4687; cycle 284 hybrid b1+b11 52.8803; cycle 291 hybrid b1+b10 51.8008.

# APPROVE
