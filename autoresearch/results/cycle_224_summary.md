# Cycle 224 — Item 229: KD distillation baseline (W_out+W_fc, BLOCKS=[1,4,7] K=512)

**Verdict: REFUTED**

- Criteria (item 229): Confirmed ≤44.0, Refuted ≥47.0, else Inconclusive
- Config: BLOCKS=[1, 4, 7], K=512, LR=0.0001, N_STEPS=10000, EVAL_EVERY=500, SEEDS=[42, 137, 2024]
- KD: tau=2.0, KD_weight=0.5, CE_weight=0.5
- Float teacher PPL: 53.895

## Per-seed results

| seed | init | best | best_step |
|---|---|---|---|
| 42 | 61.746 | 61.075 | 2500 |
| 137 | 61.746 | 60.915 | 5000 |
| 2024 | 61.746 | 60.825 | 3000 |
| **mean** | 61.746 | **60.938** | 3500 |
| **std**  | 0.000 | 0.103 | 1080 |

## Trajectory

### Seed 42
- init=61.75, step3000=61.84, step5500=61.33, best=step2500@61.07, step8000=61.37, step10000=61.31
### Seed 137
- init=61.75, step3000=61.32, step5500=61.49, best=step5000@60.92, step8000=61.19, step10000=61.10
### Seed 2024
- init=61.75, step3000=60.82, step5500=60.99, best=step3000@60.82, step8000=61.46, step10000=61.56

## Comparison vs CE baselines

| training surface | loss | best PPL |
|---|---|---|
| W_out+b_out only | CE @1000 (cycle 206) | 48.111 |
| W_out+b_out only | CE @ best step (cycle 210) | 46.556 |
| W_out+W_fc       | CE (cycle 213) | 46.424 |
| W_out+W_fc       | KD+CE (this cycle) | **60.938 ± 0.103** |

- Δ vs cycle 213 (CE, same surface): **+14.514 PPL**
- Δ vs cycle 210 (CE, W_out only):  **+14.382 PPL**
- Float ceiling: 53.895
