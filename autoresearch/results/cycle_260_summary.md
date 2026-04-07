# Cycle 260 — Item 271: Jacobian delta ΔJ_l(x) = J_GELU − J_ReLU

**Verdict: INCONCLUSIVE**

- n_calib_tokens = 256, n_blocks = 12, trainable_params = 0
- CONFIRMED iff median rank_95(ΔJ_l) ≤ 100 in ≥6 blocks AND mode-1 top-5 cos sum with c_fc reader of block l+1 ≥ 2.0 in ≥6 pairs.
- REFUTED iff median rank_95(ΔJ_l) ≥ 400 in ≥6 blocks.

## Per-block rank_95(ΔJ) over 256 tokens

| Block | r95 p25 | r95 med | r95 p75 | r95 min | r95 max | avg ΔJ rank_95 | ||ΔJ||_F med |
|------:|--------:|--------:|--------:|--------:|--------:|---------------:|-------------:|
| 0 | 357 | 360 | 362 | 345 | 369 | 350 | 80.682 |
| 1 | 397 | 407 | 412 | 360 | 422 | 371 | 113.498 |
| 2 | 359 | 367 | 374 | 298 | 383 | 350 | 95.513 |
| 3 | 331 | 334 | 336 | 320 | 344 | 327 | 82.615 |
| 4 | 330 | 332 | 334 | 322 | 343 | 321 | 88.871 |
| 5 | 330 | 332 | 335 | 316 | 344 | 331 | 88.529 |
| 6 | 333 | 335 | 338 | 325 | 343 | 341 | 99.721 |
| 7 | 335 | 338 | 339 | 328 | 344 | 349 | 109.632 |
| 8 | 335 | 337 | 339 | 327 | 347 | 353 | 122.273 |
| 9 | 332 | 335 | 337 | 322 | 344 | 355 | 140.148 |
| 10 | 332 | 334 | 337 | 319 | 345 | 358 | 162.140 |
| 11 | 329 | 332 | 336 | 316 | 351 | 326 | 192.452 |

Blocks with median rank_95 ≤ 100: **0/12** (need ≥6 for CONFIRMED).
Blocks with median rank_95 ≥ 400: **1/12** (≥6 would REFUTE).

## Cross-block alignment: mode-1 ΔJ_l vs c_fc reader of block l+1

Random baseline (10 seeds): top-5 mean = 1.2557 ± 0.0448, top-20 mean = 2.7135 ± 0.0979.
Cycle 218 reference: random top-5 sum ≈ 1.27; centred Δ̃ top-5 sum ≈ 2.97.

| l → l+1 | top-5 cos sum | top-20 cos sum | ≥2.0 | sig 3σ |
|:-------:|--------------:|---------------:|:----:|:------:|
| 0→1 | 2.727 | 4.365 | ✓ | ✓ |
| 1→2 | 2.902 | 4.667 | ✓ | ✓ |
| 2→3 | 3.043 | 5.655 | ✓ | ✓ |
| 3→4 | 2.893 | 6.280 | ✓ | ✓ |
| 4→5 | 3.459 | 8.055 | ✓ | ✓ |
| 5→6 | 3.487 | 8.161 | ✓ | ✓ |
| 6→7 | 3.358 | 7.629 | ✓ | ✓ |
| 7→8 | 3.239 | 7.248 | ✓ | ✓ |
| 8→9 | 3.260 | 6.433 | ✓ | ✓ |
| 9→10 | 2.563 | 5.159 | ✓ | ✓ |
| 10→11 | 1.926 | 3.580 | · | ✓ |

Pairs with top-5 cos sum ≥ 2.0: **10/11** (need ≥6 for CONFIRMED).

## Verdict logic
- Clause 1 (rank_95 ≤ 100 in ≥6 blocks): **FAIL** (0/12)
- Clause 2 (mode-1 top-5 cos sum ≥ 2.0 in ≥6 pairs): **PASS** (10/11)
- REFUTED clause (rank_95 ≥ 400 in ≥6 blocks): **not triggered** (1/12)
- **Verdict: INCONCLUSIVE**
