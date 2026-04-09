# Cycle 331 Brief: Eff_rank as per-block budget allocation oracle

## Item
345 — Eff_rank as per-block budget allocation oracle — train parameter-golf with mlp_mult shifted toward early blocks

## Context
- Item 345 depends on 335 and 336, both REFUTED.
- Cycle 322 (item 335, REFUTED): b3 collapse is checkpoint-specific; eff_rank profile varies by architecture/checkpoint, not a stable universal signal.
- Cycle 323 (item 336, REFUTED): post-mlp_scale eff_rank changes the picture entirely. Pre-scale eff_rank (which cycle 318 measured) is misleading — it doesn't reflect what reaches the residual stream.
- Cycles 327-329: LASER on pgolf REFUTED 3/3 blocks. Eff_rank does not predict weight compressibility; weight spectra are flat regardless of eff_rank.

## Pre-flight verdict
PLAN INVALID (P2). The item's premise — "allocate mlp_mult proportional to eff_rank/d_model" — assumes eff_rank is a stable, prescriptive measure of per-block capacity needs. All dependency items refuted this:
1. Eff_rank profiles are not stable across checkpoints (335).
2. Pre-scale eff_rank misrepresents the effective signal (336).
3. Eff_rank does not predict compressibility (327-329).
Allocating parameters based on a metric that is checkpoint-specific, misleading pre-scale, and uncorrelated with compressibility has no theoretical grounding left.

## Key Files
- `autoresearch/cross_project/parameter_golf.md`
- `parameter-golf/docs/EFFRANK_CURVATURE_FINDINGS.md` section 6.6
