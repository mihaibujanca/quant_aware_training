# Cycle 316 — Item 326: (b10, b11) destructive pair deep dive

**Verdict: INCONCLUSIVE**

- Float PPL: 53.8949 (ref 53.895)
- Blocks: b10 r=5, b11 r=22; SEED=42; N_STEPS=2000 per phase; LR=0.0001
- Cycle 310 joint best (1 seed): 60.7161

## M1 — principal angle between Δy_10 and Δy_11 right singular subspaces

Two Δy variants:
- raw  : Δy_l = GELU(z_l) @ W_proj_l (full FFN residual write)
- curv : Δy_l = (GELU(z_l) − ReLU(z_l)) @ W_proj_l (cycle 302 form)

| variant | (5×22) mean | (5×22) min | (5×5) mean | (5×5) min | (22×22) mean |
|---|---|---|---|---|---|
| raw | 39.63° | 14.77° | 62.61° | 20.29° | 64.84° |
| curv | 56.32° | 36.29° | 67.49° | 37.73° | 71.39° |

- M1 min mean angle (smallest across variants × rank-pairings): **39.63°**
- Threshold: M1 supported iff <30.0°; REFUTE-unknown iff >60.0°

## M2 — sequential Adam (phase A: b10 only, phase B: b11 only)

- install init PPL (both LowRankCProj, no Adam): 76.6535 (cycle 310 init 76.6535)

| phase | trainable | init | best | best_step | final | restored |
|---|---|---|---|---|---|---|
| A (b10 r=5) | 4,608 | 76.6535 | 64.6726 | 2000 | 64.6726 | 64.6726 |
| B (b11 r=22) | 17,664 | 64.6726 | 61.2059 | 2000 | 61.2059 | 61.2059 |

- sequential_best_ppl = min(A,B) = **61.2059**
- joint best (cycle 310) = 60.7161
- delta vs joint = +0.4898
- Threshold: M3 (joint-Adam saddle) confirmed iff sequential_best_ppl <=55.0; destructive band starts at 60.0

## M3 — one-sided restores

| ablation | configuration | init | best | best_step | reference |
|---|---|---|---|---|---|
| A | b10 float + b11 r=22 fit | 59.6834 | 54.7724 | 2000 | cycle 270 b11 r=22 ref 54.4687 |
| B | b10 r=5 fit + b11 float | 59.9179 | 51.4783 | 1500 | cycle 282 b10 r=5 diag 51.4659 |

## Verdict logic (item 326 verbatim)

- sequential_best_ppl ≤ 55.0                                  → CONFIRMED-M3-pathology
- sequential_best_ppl ≥ 60.0 AND min mean angle < 30.0°          → CONFIRMED-M1-redundancy
- sequential_best_ppl ≥ 60.0 AND min mean angle > 60.0°          → REFUTED-mechanism-unknown
- else                                                                          → INCONCLUSIVE

- VERDICT: **INCONCLUSIVE**
