# Cycle 275 — Item 285: Hybrid [1] K=512 + block 11 rank-22 c_proj

**Verdict: REFUTED**

- Criteria (item 285 verbatim): CONFIRMED if best_mean ≤ 43.5; REFUTED if best_mean ≥ 45.0; else INCONCLUSIVE
- Float PPL: 53.8949 (ref 53.895)
- Trainable: 411,648 = block 1 W_out (393,216) + b_out (768) + block 11 B (16,896) + bias (768)
- Config: K=512, RANK_11=22, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Wall time: 7.8 min

## Per-seed details

| seed | init | best | best_step | final@4000 | restored |
|---|---|---|---|---|---|
| 42 | 60.6639 | 48.2367 | 4000 | 48.2367 | 48.2367 |
| 137 | 60.6639 | 48.1320 | 4000 | 48.1320 | 48.1320 |
| 2024 | 60.6639 | 48.2325 | 4000 | 48.2325 | 48.2325 |

## Aggregate

- init_mean: **60.6639 ± 0.0000**
- best_mean: **48.2004 ± 0.0484**  (min 48.1320, max 48.2367)
- best_step_mean: 4000 ± 0
- final@4000_mean: 48.2004

## Verdict logic (item 285 verbatim)

- best_mean = 48.2004; CONFIRMED iff ≤ 43.5; REFUTED iff ≥ 45.0
- VERDICT: **REFUTED**

## Baselines and deltas

- cycle 271 [1] K=512: **43.821** (394K). Δ = +4.3794
- cycle 270 block 11 r=22 retrain: 54.469 (17.7K). Δ = -6.2686
- cycle 266 [1,4] K=512: 43.852 (788K). Δ = +4.3484
- cycle 273 hetero K_11=22 (channel-K, NOT weight-SVD): 52.637
- Float: 53.895. Δ = -5.6945
