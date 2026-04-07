# Cycle 248 — Item 258: K=128 W_out+W_fc trainable at 6 blocks

**Verdict: REFUTED**

- Criteria: Confirmed ≤61.0, Refuted ≥66.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=128, LR=5e-05, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable set per block: W_fc(sliced to K), b_fc(sliced), W_out, b_out = 197,504
- Total trainable: 1,185,024 (~1.19M)
- Agenda item 258 formula `6 × (768×128 + 128 + 3072×768 + 3072) ≈ 14.77M` inflates W_fc to the **full** (3072, 768) matrix, contradicting K=128 compression: the forward path only uses the topk-sliced (768, K) c_fc. We train the sliced W_fc per cycle 213's FullCompressedMLP semantics. Documented in `autoresearch/plans/cycle_248.md`.
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 | ||ΔW_fc||/||W_fc|| (best) |
|---|---|---|---|---|---|
| 42 | 115.056 | 67.671 | 3500 | 67.751 | 7.359e-02 |
| 137 | 115.056 | 67.596 | 4000 | 67.596 | 8.099e-02 |
| 2024 | 115.056 | 67.559 | 4000 | 67.559 | 8.096e-02 |
| **mean** | 115.056 | **67.609** | 3833 | 67.635 | 7.851e-02 |
| **std**  | 0.000 | 0.047 | 236 | 0.083 | 3.481e-03 |

## Trajectory description

### Seed 42
- init=115.06, step1100=72.13, step2100=69.21, best=step3500@67.67, step3100=67.95, step4000=67.75
### Seed 137
- init=115.06, step1100=72.31, step2100=69.11, best=step4000@67.60, step3100=68.50, step4000=67.60
### Seed 2024
- init=115.06, step1100=71.88, step2100=69.28, best=step4000@67.56, step3100=68.01, step4000=67.56

## Comparison

- Cycle 231 K=128 W_out-only baseline: **67.562**
- Cycle 248 K=128 W_out+W_fc best_mean: **67.609 ± 0.047**
- Δ vs c231 (effect of unfreezing W_fc at K=128): **+0.047** PPL
- Cycle 221 K=512 ceiling: **59.515**
- Δ vs c221 (gap to K=512 quality ceiling): **+8.094** PPL
- Wall clock per seed (mean): 133.4 s
- Overfitting check: final@4000 mean = 67.635; W_fc drift moved off float with mean rel_fro=7.851e-02 at best (final: 8.097e-02).
