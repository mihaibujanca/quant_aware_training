# Cycle 317 Plan — Item 327

## Item 327 (verbatim Hypothesis)
cycle 310 showed 9/10 pairs non-destructive but one destructive (b10, b11). Cycle 305 showed the 5-stack destructive at +18 PPL. The smallest destructive tuple must be some k ∈ {2, 3, 4, 5}. Cycle 310 established k=2 has only one destructive cell; measuring k=3 localises where the higher-order interaction kicks in. Specifically: does every triple that contains (b10, b11) destruct? Do triples that EXCLUDE both b10 and b11 (e.g. (b1, b2, b3)) compose benignly?

## Dependencies
- `experiments/pairwise_laser_cycle310.py` — joint LowRankCProj template + 5×5 pair matrix (REFUTED, (b10,b11)=60.7161 unique destructive cell; 9/10 non-destructive with delta_v_worst ∈ [-0.89, +4.87]).
- `experiments/multi_block_laser_cycle305.py` — 5-stack best_mean=73.0060 (+18.54 vs worst-single).
- Single-block anchors: b1 r=2 → 49.0324 (cycle 280), b2 r=2 → 49.3290 (cycle 285), b3 r=24 → 50.5935 (cycle 288), b10 r=5 → 51.4659 (cycle 289), b11 r=22 → 54.4687 (cycle 270).

## Approach
1. New script `experiments/triples_laser_cycle317.py`, adapted directly from `experiments/pairwise_laser_cycle310.py`. Replace PAIRS with TRIPLES = sorted(itertools.combinations([1,2,3,10,11], 3)) — exactly 10 triples.
2. For each triple install 3 `LowRankCProj` modules simultaneously at cycle-282-family ranks {1:2, 2:2, 3:24, 10:5, 11:22}. A_l frozen = U_l[:,:r_l]·diag(S_l[:r_l]) (3072,r_l), B_l trainable = Vh_l[:r_l,:], bias_l trainable = W_proj_l.bias (cycle 310 template bit-identical).
3. Freeze every other model parameter. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, SEEDS=[42].
4. Per triple: report `init_ppl`, `best_mean`, `best_step`, `delta_vs_worst_single = best_mean − max_{l∈triple} anchor_l`, `destructive_at_5ppl` bool.
5. Cross-check 2 anchors: cycle 310 (b10,b11) pair 60.7161 and cycle 305 5-stack 73.0060 as "triple ⊂ pair/5-stack" frame references. Also record all 10 cycle 310 pair deltas for each triple's 3 constituent pairs (report mean pair-delta per triple).

## Files
- `experiments/triples_laser_cycle317.py` (new)
- `autoresearch/plans/cycle_317.md`, `autoresearch/reviews/cycle_317.md`, `autoresearch/reviews/cycle_317_review.json`
- `autoresearch/results/cycle_317_triples.json`, `autoresearch/results/cycle_317_summary.md`, `autoresearch/results/cycle_317_manifest.txt`, `autoresearch/results/cycle_317_exec_result.txt`
- `autoresearch/logs/cycle_317_run.log`, `autoresearch/audits/cycle_317.md`, `autoresearch/results/cycle_317_audit.txt`

## Config
TRIPLES=C({1,2,3,10,11},3) (10); BLOCKS_RANKS={1:2,2:2,3:24,10:5,11:22}; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; calib=all_ids[:200], eval=all_ids[200:300]; DEVICE=mps; gpt2; wikitext-2 val (same slice as 280/284/291/305/310/311/313/314/316). Expected wall ~40-50 min (10 triples × ~4-5 min each at 4000 steps; heavier than cycle 310's 12.9 min at 2000 steps).

## Verification
- Float PPL within 0.5 of 53.8949.
- Per triple: `n_trainable == (r_i+1)·768 + (r_j+1)·768 + (r_k+1)·768` asserted exact.
- Per fit: `|restored − best| < 1e-3`.
- `(b10,b11,X)` triples init PPL ≳ cycle 310 (b10,b11) init 76.65 (these triples include the destructive pair).
- Sanity: (b10,b11,b1) / (b10,b11,b2) / (b10,b11,b3) all contain the (b10,b11) destructive pair; best_mean of each expected ≥ 60 (inherit the pair destruction).

## Verdict logic (item 327 verbatim)
- (All 4 triples containing (b10,b11) destruct AND all 6 triples NOT containing (b10,b11) non-destructive with delta ≤ +2 PPL) → CONFIRMED-b10b11-carries-destruction.
- `n_destructive ≥ 8 / 10` → CONFIRMED-triple-level-pervasive.
- `n_destructive ≤ 2 / 10` → REFUTED.
- Else → INCONCLUSIVE.

## Baselines to record
float 53.8949; single-block anchors {b1:49.03, b2:49.33, b3:50.59, b10:51.47, b11:54.47}; cycle 310 pair_ppl_matrix (all 10 pairs); cycle 305 5-stack 73.0060.

# APPROVE
