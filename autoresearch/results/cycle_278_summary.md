# Cycle 278 — Item 288: All-blocks c_proj output eff_rank census

**Verdict: INCONCLUSIVE** (CONFIRMED clause 1 triggers but clause 2 fails; REFUTED clause fails)

## Verdict logic (item 288 verbatim)
- CONFIRMED iff (block 4 ≤ 50 OR block 7 ≤ 50) AND ALL of {2, 3, 6, 8, 9, 10} ≥ 100
- REFUTED iff blocks 4 AND 7 BOTH ≥ 100
- INCONCLUSIVE otherwise

  - block 4 eff_rank = **49.04** ≤ 50 → CONFIRMED clause 1 TRIGGERS (margin 0.96 PPL... err: 0.96 eff_rank units)
  - block 7 eff_rank = 297.79 ≥ 100
  - mid-blocks {2, 3, 6, 8, 9, 10}: {2:1.05, 3:24.29, 6:290.97, 8:291.41, 9:195.50, 10:16.70}
  - CONFIRMED clause 2 (all mid ≥ 100) FAILS: blocks 2 (1.05), 3 (24.29), 10 (16.70) all well below 100
  - REFUTED clause (b4 AND b7 ≥ 100) FAILS: b4 = 49.04 < 100
  - → INCONCLUSIVE

## 12-block eff_rank profile

| Block | eff_rank | top-1 energy | c_proj sv1 |
|------:|---------:|-------------:|-----------:|
|  0 |  71.93 | 0.2771 | (see JSON) |
|  1 |   1.94 | 0.9294 | 28.71 |
|  2 |   1.05 | 0.9961 | (see JSON) |
|  3 |  24.29 | 0.5887 | (see JSON) |
|  4 |  49.04 | 0.4795 | (see JSON) |
|  5 | 179.19 | 0.2462 | 27.62 |
|  6 | 290.97 | 0.1421 | 29.64 |
|  7 | 297.79 | 0.1535 | 30.23 |
|  8 | 291.41 | 0.1452 | 31.08 |
|  9 | 195.50 | 0.2263 | 32.78 |
| 10 |  16.70 | 0.5889 | 41.59 |
| 11 |  22.49 | 0.4895 | 60.99 |

## Reading
- **Sanity check**: blocks {1, 5, 11} reproduce cycle 267 to 0 PPL (identical numerical values) — same calib slice, deterministic forward pass.
- **Pattern is NOT U-shape**. Pattern is W-shape (or M-shape on rank): low-rank "concentrators" at blocks {1, 2, 3, 4, 10, 11} and high-rank "diffusers" at {5, 6, 7, 8, 9}. Block 0 (eff_rank 71.93) sits intermediate between the two regimes.
- **Five low-rank blocks below 50**: {1: 1.94, 2: 1.05, 3: 24.29, 4: 49.04, 10: 16.70, 11: 22.49}. Block 2 is even lower than block 1 (top-1 energy 99.61% vs 92.94%) — block 2 is essentially a rank-1 operator.
- **Block 4 is on the boundary** (49.04 just under the 50 threshold). The CONFIRMED first clause triggers but block 4 is not as starkly low-rank as block 11 (22.49) or block 10 (16.70).
- **Block 7 is HIGH-rank** (297.79) — the early-mid block that the cycle 266 [1,4]→[1,4,7]→cycle 213 [1,4,7] cascade kept finding as a Pareto winner is NOT rank-concentrated. Block 7 is one of the 5 high-rank "diffusers" along with {5, 6, 8, 9}, all clustered in the 179-298 eff_rank range with 14-25% top-1 energy. The "rank-concentrated" story does NOT explain why block 7 is a useful compression target; W_out joint Adam at K=64-512 is exploiting structure that activation eff_rank does NOT see.
- **Block 10 is the surprise**: eff_rank 16.70 is LOWER than block 11 (22.49). Block 10 has not been individually tested as a compression target. **Block 10 is the most rank-concentrated late block in the model and is currently unexploited by any cycle.**
- **The Pareto-winning blocks {1, 4, 7} fall in DIFFERENT regimes**: block 1 (rank ~2, ultra-concentrated), block 4 (rank ~49, edge-case low), block 7 (rank ~298, high-rank). Their compressibility under W_out joint Adam (cycle 271 best_means 43.82 / 44.63 / 45.37) does NOT correlate strongly with c_proj output eff_rank. The 1-block cycle 271 ordering [1] < [4] < [7] is in the SAME direction as their eff_rank ordering (1.94 < 49.04 < 297.79), so a weak monotonic correlation exists, but spans a 150× rank range for a 1.55 PPL spread.
- **Cycle 267 + cycle 270 + cycle 278 combined**: c_proj output eff_rank is descriptive of activation energy concentration. It correctly identifies block 11 (22.49) and now also block 10 (16.70), block 2 (1.05), block 1 (1.94), block 3 (24.29), block 4 (49.04) as concentrated. Blocks 6/7/8 (~290) and 5/9 (~180) are diffusers. But cycle 273 already showed eff_rank-guided heterogeneous K is REFUTED at the cascade level — eff_rank is not prescriptive as a per-block K budget.
- **The U-shape hypothesis (low ends, high middle) is wrong**. The actual shape is: very low at blocks 1-4 (with block 0 intermediate), high at blocks 5-9, low again at blocks 10-11. This is roughly a **bathtub with a high middle and a slight bump at block 0** — closer to "sigmoid up then sharp drop" than a clean U.

## Open follow-ups (NOT in queue)
- Block 10 single-block weight-SVD rank-r patch (cycle 270 analogue): if block 10 rank-16 c_proj is no-retrain free, that's a 17.6K-param Pareto candidate parallel to block 11. Predicted by analogy: ~54-56 PPL no-retrain, ~52-54 PPL after retrain.
- Block 2 1-block compression at K=2-8 (block 2 is rank-1 in the activation energy sense; K=1-2 may suffice).
- Why does block 7 (eff_rank 298) work as a compression target at K=64? Activation rank doesn't predict it; the operator-side W_out Adam search is finding something the activation SVD misses.

## Artifacts
- `experiments/all_blocks_eff_rank_cycle278.py`
- `autoresearch/plans/cycle_278.md`
- `autoresearch/briefs/cycle_278.md`
- `autoresearch/logs/cycle_278_run.log`
- `autoresearch/results/cycle_278_all_blocks_eff_rank.json`
- `autoresearch/results/cycle_278_summary.md`
