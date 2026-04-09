# Cycle 339 Brief

## Item
Item 348: mlp_scale-based block pruning — zero out b12's MLP contribution

## Context
- Cycle 323: b12 post-scale eff_rank = 1.03, b11 = 2.25 (near-zero effective contribution)
- b12 mean mlp_scale = 0.003, 577/640 channels |scale| < 0.1, 321 channels negative
- b12 has hidden=320 (smallest block in the 13L model)
- Hypothesis: model learned to suppress b12's MLP; zeroing it costs nothing in BPB

## Key Question
Does zeroing b12's mlp_scale (and possibly b11, b3) preserve BPB?

## Key Files
- `autoresearch/cross_project/parameter_golf.md` — subprocess invocation pattern
- `autoresearch/results/cycle_323_pgolf_postscale_eff_rank.json` — per-block eff_rank data
- `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_laser_retrain.py` — load_model_v2 reference
- `/Users/mihai/Projects/parameter-golf/local_scripts/analyze_checkpoint.py` — eval_bpb

## Verdict Thresholds
- CONFIRMED: BPB delta ≤ 0.005 (safe to prune)
- REFUTED: BPB delta > 0.02 (residual contribution matters)
- INCONCLUSIVE: BPB delta 0.005–0.02
