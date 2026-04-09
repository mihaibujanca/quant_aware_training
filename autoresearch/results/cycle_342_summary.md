# Cycle 342 — LASER Retrain on pgolf b3 (Extended Budget)

**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`
**Target block**: b3 (output eff_rank=4.89 from cycle 318)
**Measured baseline BPB**: 1.362078

## Results (this cycle)

| config | rank | steps | lr | baseline | b_init | retrained | delta | params_B |
|--------|------|-------|----|----------|--------|-----------|-------|----------|
| r5_2000step_lr3e-5 | 5 | 2000 | 3e-05 | 1.362078 | 1.403734 | 1.399748 | +0.037669 | 9,600 |
| r10_2000step_lr3e-5 | 10 | 2000 | 3e-05 | 1.362078 | 1.401368 | 1.396926 | +0.034847 | 19,200 |
| r5_2000step_lr1e-4 | 5 | 2000 | 0.0001 | 1.362078 | 1.403734 | 1.399115 | +0.037037 | 9,600 |

## Comparison to cycle 327 (500 steps, lr=1e-4)

| rank | steps | lr | delta_327 | delta_342 | improvement |
|------|-------|----|-----------|-----------|-------------|
| 5 | 500→2000 | 0.0001 | +0.037780 | +0.037037 | +0.000743 |

## Verdict

**REFUTED**: best config (r10_2000step_lr3e-5) achieves BPB delta = +0.034847 > 0.01. Even 4x budget can't fix LASER on pgolf b3. eff_rank 4.89 is not concentrated enough for LASER.
