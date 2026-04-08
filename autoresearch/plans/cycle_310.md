# Cycle 310 Plan — Item 320

## Item 320 (verbatim Hypothesis)
cycle 284 showed b1+b11 composes (52.88, +0.88 PPL above worst-single, NOT destructive). Cycle 305 showed b1+b2+b3+b10+b11 destructs (+18 PPL). Run all C(5,2)=10 pairwise LASER stacks at the cycle-282-family ranks (b1 r=2, b2 r=2, b3 r=24, b10 r=5, b11 r=22) and identify which pair(s) first introduce destructive composition. **Setup**: For each of 10 pairs (b1,b2), (b1,b3), (b1,b10), (b1,b11), (b2,b3), (b2,b10), (b2,b11), (b3,b10), (b3,b11), (b10,b11): install both LowRankCProj replacements simultaneously, freeze every other model parameter, joint Adam LR=1e-4, N_STEPS=2000 (half of cycle 305 budget — pairs are smaller), EVAL_EVERY=100, best-state restore, SEEDS=[42, 137, 2024]. Float PPL ref 53.8949; cycle 284 (b1+b11) ref 52.88. Report `pair_best_mean_ppl[10]` and `pair_delta_vs_worst_single[10]` (where worst_single per pair is max of the two single-block anchors). **Verbatim conditions**: ≥6/10 pairs have delta_vs_worst_single ≥ +5 PPL → CONFIRMED; ≤2/10 pairs have delta ≥ +5 PPL → REFUTED; else INCONCLUSIVE. Identify the worst pair as a follow-up target. **Wall budget**: 10 pairs × ~3 min = ~30 min total.

## Dependencies
- `experiments/rank_hybrid_1_11_cycle284.py` — 2-block homogeneous LowRankCProj hybrid template (SVD cache, joint Adam, best-state restore).
- `experiments/multi_block_laser_cycle305.py` — 5-block variant used for `BLOCKS_RANKS` dict semantics.
- Single-block anchors: b1 r=2→49.0324 (c280); b2 r=2→49.3290 (c285); b3 r=24→50.5935 (c288); b10 r=5→51.4659 (c289); b11 r=22→54.4687 (c270). Cycle 284 pair b1+b11→52.8803; cycle 291 pair b1+b10→51.8008.

## Approach
1. New script `experiments/pairwise_laser_cycle310.py`. Copy cycle 284 LowRankCProj / eval_ppl / SVD-cache / joint-Adam loop verbatim. Config: `BLOCKS_RANKS = {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}`, `PAIRS` = sorted itertools.combinations of the 5 block keys (10 pairs).
2. Load GPT-2 once, compute SVDs of 5 c_proj weights once, cache A/B_init/b_init per block.
3. For each pair: for each seed in [42]: reload GPT-2, install 2 LowRankCProj modules at the pair's blocks at their BLOCKS_RANKS ranks, freeze rest, joint Adam LR=1e-4, N_STEPS=2000 (item 320 verbatim — half cycle 305 budget), EVAL_EVERY=100, best-state restore, |restored-best|<1e-3 assert. Record init/best/final/restored/history.
4. SEED=42 ONLY for all 10 pairs (budget cap). Do NOT extend to 3 seeds unconditionally; item 320 brief sanction says extend only if budget permits / any pair < worst-single. Implement `EXTEND_SEEDS = False` flag off by default; leave extension as a follow-up cycle if time over.
5. Build the 5×5 matrix: off-diagonal cells filled by pair best_ppl (symmetric); diagonal cells filled by the 5 single-block anchor constants above. Persist as `pair_ppl_matrix[5][5]` (row/col order [1,2,3,10,11]).
6. Verdict per item 320 conditions (counting pairs where delta_vs_worst_single ≥ +5 PPL). Identify `worst_pair` (argmax delta_vs_worst_single).

## Files
- `experiments/pairwise_laser_cycle310.py` (new)
- `autoresearch/results/cycle_310_pairwise_laser.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_310_run.log`, `autoresearch/audits/cycle_310.md`, `autoresearch/results/cycle_310_audit.txt`

## Config
BLOCKS_RANKS={1:2, 2:2, 3:24, 10:5, 11:22}; PAIRS=C(5,2)=10; LR=1e-4; N_STEPS=2000; EVAL_EVERY=100; SEEDS=[42]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; DEVICE=mps; gpt2; wikitext-2 val. Trainable per pair = (r_i+r_j+2)·768. Wall ~10 pairs × 1.5 min ≈ 15 min at SEED=42 (half of cycle 284 per-seed wall since N_STEPS halved).

## Verification
- Float PPL within 0.5 of 53.895.
- Per pair: expected trainable asserted; per-seed |restored-best|<1e-3 asserted.
- `pair_ppl_matrix` symmetric (off-diag), diagonal == single-block anchors.
- Cross-check: cycle 310 (b1,b11) best vs cycle 284 52.8803 (must be in the same band; cycle 310 uses N_STEPS=2000 vs cycle 284 N_STEPS=4000 — expect slightly worse, record delta as sanity not hard-assert).

## Verdict logic (item 320 verbatim)
n_destructive := |{pair : delta_vs_worst_single ≥ +5 PPL}|. CONFIRMED iff n_destructive ≥ 6. REFUTED iff n_destructive ≤ 2. INCONCLUSIVE otherwise. Record `worst_pair` as follow-up target.

## Baselines to record
float 53.8949; b1 r=2 49.0324; b2 r=2 49.3290; b3 r=24 50.5935; b10 r=5 51.4659; b11 r=22 54.4687; cycle 284 (b1,b11) 52.8803; cycle 291 (b1,b10) 51.8008; cycle 305 5-stack 73.0060.

# APPROVE
