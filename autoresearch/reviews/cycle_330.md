# Cycle 330 Review: Selective LeakyReLU² at b4/b5

## Checklist
- [x] Required reading listed in brief (`cross_project/parameter_golf.md`, EFFRANK_CURVATURE_FINDINGS 6.4)
- [x] Cross-project subprocess pattern followed (pgolf script + qat wrapper)
- [x] Reuses `load_model_v2` from `pgolf_eff_rank_census.py` (handles variable mlp_mult, bigram, skip_weights)
- [x] Verdict thresholds explicit and consistent with item 343 (±0.003 for localized, ≥0.003 for distributed)
- [x] Three-way comparison: global-lrelu2 / selective / global-relu2

## Critical detail caught
`load_model_v2` always sets `mlp_act="relu2"` (cannot detect lrelu2 from checkpoint weights — identical architecture). Plan correctly patches `block.mlp.act` after load. The `lrelu_slope` per block is loaded from checkpoint automatically, so slope flows through Block.__call__ -> MLP.__call__(slope=self.lrelu_slope).

## Risk assessment
LOW — zero-shot eval only (no training), 3 forward passes on FineWeb val. Script structure mirrors prior pgolf cycles (327-329). Only new code: per-block act patching (3 lines per config).

## Decision
APPROVE
