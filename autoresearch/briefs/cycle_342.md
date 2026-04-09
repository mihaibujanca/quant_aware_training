# Brief — Cycle 342

**Item**: 351 — LASER with output-covariance gating on pgolf — re-evaluate using the correct metric
**Type**: empirical
**Priority**: P1

## Context

Cycles 327-329 ran LASER on pgolf b3 and got +0.035-0.040 BPB degradation at ranks 2/5/10 with 500 Adam steps, lr=1e-4. This was declared REFUTED.

Cycle 338 later proved that weight SVD spectrum is the wrong gate for LASER — output-covariance concentration is what predicts success. Cycle 318 measured pgolf output covariance: b3 eff_rank=4.89, meaning output covariance IS concentrated (similar to GPT-2 b1=1.94 where LASER worked).

However, cycle 327 already ran LASER on b3 and it failed. Two hypotheses for the failure:
1. The retrain budget was too small (500 steps) and lr too high (1e-4) for a rank-5 factorization of a 640x1920 weight
2. pgolf b3's eff_rank=4.89 is not concentrated enough (GPT-2 b1 was ~1.94, essentially rank-1)

This cycle tests hypothesis 1 by extending retrain to 2000 steps with lr=3e-5.

## Key Prior Results

| Model | Block | Output Cov eff_rank | LASER rank | Retrain steps | lr | BPB delta |
|-------|-------|---------------------|------------|---------------|-----|-----------|
| GPT-2 | b1 | 1.94 (top-1=94.2%) | 2 | 500 | 1e-4 | ~0.000 (success) |
| pgolf | b3 | 4.89 | 2 | 500 | 1e-4 | +0.040 |
| pgolf | b3 | 4.89 | 5 | 500 | 1e-4 | +0.038 |
| pgolf | b3 | 4.89 | 10 | 500 | 1e-4 | +0.035 |

## Key Files

- `autoresearch/cross_project/parameter_golf.md`
- `experiments/pgolf_laser_retrain_cycle340.py` (cycle 327 qat wrapper, reusable)
- `parameter-golf/local_scripts/pgolf_laser_retrain.py` (pgolf-side script, already supports --n-train-steps and --lr)
- `autoresearch/results/cycle_327_pgolf_laser_retrain.json`

## Confirmed/Refuted/Inconclusive Criteria

From agenda item 351:
- **Confirmed**: BPB delta < 0.005 for at least one (rank, retrain) combo
- **Refuted**: BPB delta > 0.01 for all configs despite 4x retrain budget
- **Inconclusive**: output covariance is concentrated but LASER retrain underperforms — suggests retrain procedure is the bottleneck
