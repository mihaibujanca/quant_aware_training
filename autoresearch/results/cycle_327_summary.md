# Cycle 327 — LASER Rank-SVD + B-Retrain on pgolf b3

**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`
**Target block**: b3 (`mlp.proj.weight` shape ~640×1920)
**Baseline BPB** (from training summary): 1.3474
**Measured baseline BPB**: 1.362078
**Retrain**: 500 Adam steps, lr=0.0001, seq_len=1024

## Results

| rank | baseline_bpb | b_init_bpb | retrained_bpb | Δ_init | Δ_retrained | n_params_B | param_reduction% |
|------|-------------|------------|---------------|--------|-------------|------------|-----------------|
| 2 | 1.362078 | 1.406962 | 1.402559 | +0.044884 | +0.040481 | 3,840 | 99.7% |
| 5 | 1.362078 | 1.403734 | 1.399859 | +0.041656 | +0.037780 | 9,600 | 99.2% |
| 10 | 1.362078 | 1.401368 | 1.397063 | +0.039290 | +0.034984 | 19,200 | 98.4% |

## Verdict

**REFUTED-LASER-fails**: rank-5 delta = +0.037780 ≥ 0.01. b3's rank-5 capacity insufficient; LASER doesn't transfer.

## Comparison to QAT baselines

- Cycle 280 (GPT-2 LASER): rank-5 retrain recovered float BPB within 0.001
- Cycle 318 (pgolf eff_rank census): b3 output eff_rank = 4.89 (pre-scale), 4.01 (post-scale) — lowest in early stack
