# Cycle 309 — Item 319: Δy-aligned rank-r LASER at block 8

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 8, c_proj.weight (3072, 768)
- Cycle 306 ref: Δy top1_E=0.9141, stable rank=1.197, E_kept_top2 via W_proj=0.4722

## No-retrain PPL (eval slice, 100 seqs × 128 tok)

| basis | r | PPL | Δfloat | %above |
|---|---:|---:|---:|---:|
| dy | 1 | 61.6614 | +7.767 | +14.41% |
| dy | 2 | 61.1611 | +7.266 | +13.48% |
| wproj | 1 | 60.8573 | +6.962 | +12.92% |
| wproj | 2 | 60.4958 | +6.601 | +12.25% |

## Retrain best_mean (3 seeds, Adam LR=1e-4, 4000 steps, best-state restore)

| basis | r | n_train | best_mean | best_std | best_min | best_max | Δfloat |
|---|---:|---:|---:|---:|---:|---:|---:|
| dy | 1 | 1536 | 54.3185 | 0.0044 | 54.3142 | 54.3245 | +0.424 |
| dy | 2 | 2304 | 53.6040 | 0.0042 | 53.5998 | 53.6098 | -0.291 |
| wproj | 1 | 1536 | 53.3034 | 0.0071 | 53.2969 | 53.3133 | -0.591 |
| wproj | 2 | 2304 | 52.8538 | 0.0126 | 52.8433 | 52.8715 | -1.041 |

## Per-seed retrain detail

| cell | seed | init | best | best_step | final | restored |
|---|---:|---:|---:|---:|---:|---:|
| dy_r1 | 42 | 61.6614 | 54.3142 | 4000 | 54.3142 | 54.3142 |
| dy_r1 | 137 | 61.6614 | 54.3169 | 4000 | 54.3169 | 54.3169 |
| dy_r1 | 2024 | 61.6614 | 54.3245 | 4000 | 54.3245 | 54.3245 |
| dy_r2 | 42 | 61.1611 | 53.6025 | 4000 | 53.6025 | 53.6025 |
| dy_r2 | 137 | 61.1611 | 53.5998 | 4000 | 53.5998 | 53.5998 |
| dy_r2 | 2024 | 61.1611 | 53.6098 | 4000 | 53.6098 | 53.6098 |
| wproj_r1 | 42 | 60.8573 | 53.3000 | 4000 | 53.3000 | 53.3000 |
| wproj_r1 | 137 | 60.8573 | 53.2969 | 4000 | 53.2969 | 53.2969 |
| wproj_r1 | 2024 | 60.8573 | 53.3133 | 4000 | 53.3133 | 53.3133 |
| wproj_r2 | 42 | 60.4958 | 52.8433 | 4000 | 52.8433 | 52.8433 |
| wproj_r2 | 137 | 60.4958 | 52.8467 | 4000 | 52.8467 | 52.8467 |
| wproj_r2 | 2024 | 60.4958 | 52.8715 | 4000 | 52.8715 | 52.8715 |

## Verdict logic (item 319 verbatim)

- dy_r1 retrain best_mean = 54.3185
- CONFIRMED iff ≤ 60.0 (Δy basis is the right basis at b8)
- REFUTED iff ≥ 110.0 (b8 needs rank > 1 regardless of basis)
- **VERDICT: CONFIRMED**

## Geometric construction recap

- dy basis: A_y = W_proj @ Vh_y[:r].T frozen (3072, r); B_init = Vh_y[:r] (r, 768) trainable; bias trainable
- wproj basis (paired baseline, cycle 280-style): A_w = U_w[:, :r] * S_w[:r] frozen; B_w = Vh_w[:r] trainable; bias trainable
- W_new = A @ B replaces c_proj_8.weight in-place; rest of model frozen
