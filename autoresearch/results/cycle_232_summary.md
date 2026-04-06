# Cycle 232 — Item 242: Sequential init + joint polish hybrid (3 blocks)

**Verdict: REFUTED**

- Criteria: Confirmed ≤45.8, Refuted ≥46.5, else Inconclusive
- Config: BLOCKS=[1, 4, 7], K=512, LR_PHASE1=0.0001, LR_PHASE2=5e-05, N_STEPS_PHASE1=2000, N_STEPS_PHASE2=1000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | post_sequential | polish_best | final | gain |
|---|---|---|---|---|---|
| 42 | 61.746 | 47.210 | 46.878 | 46.878 | +0.332 |
| 137 | 61.746 | 47.232 | 46.916 | 46.916 | +0.316 |
| 2024 | 61.746 | 47.318 | 46.941 | 46.941 | +0.378 |
| **mean** | 61.746 | 47.253 | 46.911 | **46.911** | +0.342 |
| **std**  | 0.000 | 0.047 | 0.026 | 0.026 | 0.026 |

## Comparison vs baselines

| approach | best_mean PPL |
|---|---|
| Float (GPT-2) | 53.895 |
| topk+lstsq init only | 61.746 |
| Sequential alone (c215) | 47.253 |
| Joint 1k W_out (c206) | 48.111 |
| Joint 4k W_out (c210) | 46.556 |
| Joint W_out+W_fc (c213) | 46.424 |
| **Hybrid sequential+polish (c232)** | **46.911 ± 0.026** |

- Δ vs sequential alone (c215): -0.342
- Δ vs joint 4k (c210):        +0.355
- Δ vs joint W_out+W_fc (c213): +0.487
- Gain from polish (post_seq − final): +0.342 ± 0.026
