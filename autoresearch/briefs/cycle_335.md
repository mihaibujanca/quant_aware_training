# Cycle 335 Brief

## Item
Item 343b: Selective LeakyReLU² at b4/b5 only — RE-RUN with correct loader

## Context
- Cycle 330 attempted this experiment but used a stale `load_model_v2` missing `xsa_last_n`, `rope_dims`, and `mlp_act` from the summary JSON. All BPB numbers were garbage (12.8 vs expected ~1.35).
- Root cause: the old loader in `pgolf_selective_lrelu2.py` hardcodes `mlp_act="relu2"`, `xsa_last_n=0`, `rope_dims=0`. The canonical checkpoint was trained with `mlp_act="lrelu2"`, `xsa_last_n=13`, `rope_dims=16`. Without XSA and RoPE, attention outputs are garbage.
- Cycle 327's `pgolf_laser_retrain.py` has the correct `load_model_v2` that reads the paired summary JSON for all hyperparameters. This loader also has the summary JSON discovery logic (fallback to `13L_640d_322_7k_wd70_batch20k_summary.json`).
- The canonical checkpoint IS lrelu2-trained (cycle 325 confirmed). Expected baseline BPB ~ 1.35.

## Dependencies
- Cycle 330 plan/results (broken): `autoresearch/plans/cycle_330.md`, `autoresearch/results/cycle_330_audit.txt`
- Cycle 327 correct loader: `parameter-golf/local_scripts/pgolf_laser_retrain.py` (lines 85-197)
- Cycle 330 pgolf script (to replace): `parameter-golf/local_scripts/pgolf_selective_lrelu2.py`

## Key Files
- `autoresearch/cross_project/parameter_golf.md` (cross-project invocation pattern)
- `parameter-golf/local_scripts/pgolf_laser_retrain.py` (correct `load_model_v2` to copy)
- `parameter-golf/local_scripts/pgolf_selective_lrelu2.py` (broken script to replace)
- Checkpoint: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`
- Summary JSON: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_summary.json`

## Constraints
- 3 forward passes (~90s total). Wall time ~5 min with overhead.
- Must sanity-check baseline BPB ~ 1.35 (not 12.8!) before trusting any deltas.
- Verdict thresholds (same as original item 343):
  - CONFIRMED-localized-mechanism: selective ~ global-lrelu2 (delta < 0.003) AND selective < global-relu2 by >= 0.003
  - REFUTED-distributed-mechanism: selective >= 0.003 worse than global-lrelu2
  - INCONCLUSIVE: deltas below thresholds in both directions

## Risk Assessment
- Low risk. The fix is mechanical (copy correct loader). The eval logic from cycle 330 is correct — only the loader was broken.

## Handoff
Write `pgolf_selective_lrelu2_v2.py` with cycle 327's `load_model_v2` + summary JSON discovery. Keep the same 3-config eval logic from the original. Write qat-side wrapper `experiments/pgolf_selective_lrelu2_v2_cycle335.py`.
