# Cycle 330 Brief: Selective LeakyReLU² at b4/b5 only

## Item
343 — Selective LeakyReLU² at b4/b5 only (zero-shot inference test)

## Context
- Canonical checkpoint `13L_640d_322_7k_wd70_batch20k_best.npz` is lrelu2-trained (cycle 325)
- Cycle 319/324: curvature residual concentrated at b4 (rank_95=18) and b5 (rank_95=33)
- This experiment: keep lrelu² at b4/b5 only, switch all other blocks to relu². Compare BPB across 3 configs: global-lrelu², selective, global-relu².

## Hypothesis
Selective BPB ≈ global-lrelu² (within ±0.0005) AND selective < global-relu² by ≥0.005.

## Key Files
- `autoresearch/cross_project/parameter_golf.md` (REQUIRED READING)
- `parameter-golf/docs/EFFRANK_CURVATURE_FINDINGS.md` section 6.4
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` — has `load_model_v2` (reuse for variable mlp_mult)
