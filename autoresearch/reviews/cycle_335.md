# Cycle 335 Review: Selective LeakyReLU² at b4/b5 — RE-RUN (item 343b)

## Checklist

| Check | Status | Notes |
|---|---|---|
| Correct loader used | PASS | Plan specifies copying `load_model_v2` from `pgolf_laser_retrain.py` which reads summary JSON for `mlp_act`, `xsa_last_n`, `rope_dims` |
| Summary JSON discovery | PASS | Same discovery logic as cycle 327: tries `_summary.json`, `_batch20k_summary.json`, hardcoded fallback |
| Sanity check on baseline BPB | PASS | Explicit assert `global_lrelu2_bpb < 2.0` catches the cycle 330 failure mode (12.8 BPB) |
| Loader config reported in output | PASS | JSON includes `loader_config` with `mlp_act`, `xsa_last_n`, `rope_dims` for post-hoc verification |
| Same eval logic as cycle 330 | PASS | 3 configs (global-lrelu2, selective b4+b5, global-relu2) with `set_all_act` + per-block patching |
| Verdict thresholds match agenda | PASS | delta < 0.003 for CONFIRMED, >= 0.003 for REFUTED — matches item 343/343b |
| Cross-project invocation pattern | PASS | Subprocess via pgolf venv, JSON to stdout, qat wrapper parses and writes results |
| Time budget | PASS | 3 forward passes ~90s, timeout=600 is adequate |
| No stale code reuse | PASS | New script `_v2.py`, does not modify the broken `pgolf_selective_lrelu2.py` |

## Issues

None. This is a mechanical fix: replace the broken loader with the proven one from cycle 327. The eval logic was correct in cycle 330 — only the model construction was wrong. The sanity check assertion is a good addition to catch regressions.

## Decision
APPROVE
