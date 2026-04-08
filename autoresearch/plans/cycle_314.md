# Cycle 314 Plan — Item 324

## Item 324 (verbatim Hypothesis)
cycle 310 used the heterogeneous cycle-282-family ranks (b1 r=2, b2 r=2, b3 r=24, b10 r=5, b11 r=22), total 55 rank units, and found only (b10, b11) destructive at the +5 PPL bar (9/10 non-destructive). Cycle 311 then established that every single concentrator block admits a 1,536-param rank-1 LASER floor below PPL 60. Open question: at the **universal r=1 level** (2 × 1,536 = 3,072 trainable per pair, ~6× smaller than cycle 310's heterogeneous pairs), does the (b10, b11) destruction persist? Three possible outcomes: (a) (b10, b11) r=1 still destructs → the destruction mechanism is basis-geometric, not capacity-limited; (b) (b10, b11) r=1 no longer destructs → cycle 310's destruction was capacity competition at r=22, not intrinsic to the block pair; (c) several new pairs now destruct at r=1 → rank-1 is too aggressive for composition and the cycle-311 universal floor does not stack. **Setup**: BLOCKS=[1, 2, 3, 10, 11]; PAIRS=sorted(itertools.combinations(BLOCKS, 2)) (10 pairs); for each pair install both LowRankCProj(r=1) modules (cycle 292/311 template: `A_l = U_l[:, :1] * S_l[:1]` (3072, 1) frozen from full SVD of W_proj_l; `B_l` (1, 768) trainable; `bias_l` (768,) trainable), freeze everything else, joint Adam LR=1e-4, N_STEPS=2000 (matches cycle 310), EVAL_EVERY=100, best-state restore, SEEDS=[42] (single seed to match cycle 310 budget; 3-seed audit if the pair matrix shows new destructive cells). N_CALIB=200, N_EVAL=100, SEQ_LEN=128, MPS, GPT-2 small, wikitext-2 val (cycle 280/284/291/305/310/311 slice). Trainable per pair = 2 × (1·768 + 768) = 3,072. Report `pair_best_mean_ppl[10]`, `pair_delta_vs_worst_single_r1[10]` where worst_single_r1 is max(b_i r=1 anchor from cycle 311/292, b_j r=1 anchor). Single-block r=1 anchors: b1=50.08 (cycle 292), b2=49.75, b3=55.05, b10=52.91, b11=59.71 (cycle 311). **Verbatim conditions**: (b10, b11) r=1 delta_vs_worst_single ≥ +5 PPL → CONFIRMED-persistent-destruction ("the (b10, b11) destruction is basis-geometric and survives rank-1 reduction; both late residual-stream concentrators cannot share a low-rank projection"); (b10, b11) r=1 delta_vs_worst_single ≤ +1 PPL AND ≤1/10 other pairs destruct → REFUTED-at-r1 ("cycle 310's (b10, b11) destruction was capacity competition at r=22; rank-1 reduces the shared capacity enough to disambiguate"); else INCONCLUSIVE.

## Dependencies
- `experiments/pairwise_laser_cycle310.py` — pairwise template (SVD cache, joint Adam, 5×5 assembly, verdict block).
- `experiments/rank1_all_concentrators_cycle311.py` — rank-1 infra + diagonal anchors b1=50.0794 (c292), b2=49.7539, b3=55.0485, b10=52.9107, b11=59.7074 (c311).

## Approach
1. New script `experiments/rank1_pairwise_laser_cycle314.py`. Adapt cycle 310 pairwise template with `BLOCKS_RANKS = {1:1, 2:1, 3:1, 10:1, 11:1}`, BLOCK_ORDER=[1,2,3,10,11], PAIRS = C(5,2) = 10.
2. Load GPT-2 once, cache SVDs (rank-1 slice A=(U[:, :1]·S[:1]), B_init=Vh[:1, :], bias=orig) per block.
3. For each pair: reload GPT-2, install two LowRankCProj(r=1) modules, freeze rest, joint Adam LR=1e-4, N_STEPS=2000, EVAL_EVERY=100, best-state restore, SEED=42 only. Assert trainable == 3,072 per pair.
4. Build symmetric 5×5 matrix (off-diag = best_mean; diagonal = cycle 292/311 r=1 anchors). Write JSON + summary + verdict per item 324 verbatim logic.

## Files
- `experiments/rank1_pairwise_laser_cycle314.py` (new)
- `autoresearch/results/cycle_314_rank1_pairwise_laser.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_314_run.log`, `autoresearch/audits/cycle_314.md`, `autoresearch/results/cycle_314_audit.txt`

## Config
BLOCKS_RANKS={1:1,2:1,3:1,10:1,11:1}; PAIRS=C(5,2)=10; LR=1e-4; N_STEPS=2000; EVAL_EVERY=100; SEEDS=[42]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; DEVICE=mps; gpt2; wikitext-2 val. Trainable per pair = 2·(1·768+768) = 3,072. Wall ~10 pairs × ~1 min ≈ 10-15 min.

## Verification
- Float PPL within 0.5 of 53.895. Per pair: trainable count == 3,072 asserted; |restored-best|<1e-3 asserted.
- 5×5 matrix symmetric (off-diag |A_ij - A_ji| < 1e-9); diagonal == cycle 292/311 r=1 anchors.
- Sanity vs cycle 310 (b10,b11) heterogeneous r_10=5 r_11=22 best_mean = 60.7161 at +6.2474; the cycle 314 r=1+r=1 delta is the primary substantive signal (basis-geometric vs capacity-limited).

## Verdict logic (item 324 verbatim)
CONFIRMED iff (b10,b11) r=1 delta_vs_worst_single ≥ +5 PPL. REFUTED iff (b10,b11) r=1 delta_vs_worst_single ≤ +1 PPL AND ≤1/10 other pairs destruct at +5 PPL bar. INCONCLUSIVE otherwise. Secondary: n_destructive_at_5 over all 10 pairs.

## Baselines to record
float 53.8949; r=1 anchors: b1=50.0794 (c292), b2=49.7539, b3=55.0485, b10=52.9107, b11=59.7074 (c311); cycle 310 (b10,b11) r=5+r=22 best=60.7161 (+6.2474 destructive); cycle 310 9/10 non-destructive; cycle 305 5-stack +18 PPL.

# APPROVE
