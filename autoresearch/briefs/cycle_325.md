# Cycle 325: LeakyReLU²-Trained Checkpoint Census

## Item
338: True LeakyReLU²-trained parameter-golf checkpoint — cycle 318/319 census (cross-project, caveat experiment).

## Context
Cycle 319 ran eff_rank and curvature residual measurements on a **ReLU²-trained** model, treating LeakyReLU² as a predictive perturbation. Found localization at blocks b4, b5, b12 (rank_95: 18, 33, 21). The question: is this localization real, or an artifact of measuring on the wrong activation distribution?

To answer: train a **true LeakyReLU²-trained** checkpoint and re-run cycle 318/319 measurements.

## Dependencies
- `autoresearch/cross_project/parameter_golf.md` (required reading: subprocess invocation pattern)
- parameter-golf venv: `/Users/mihai/Projects/parameter-golf/.venv/bin/python`
- MLX training infrastructure + MPS GPU available
- `pgolf_eff_rank_census.py`, `pgolf_curvature_residual.py` (in parameter-golf/local_scripts)

## Key Files
- `autoresearch/cross_project/parameter_golf.md`
- `parameter-golf/train_gpt_mlx.py` (MLP class supports `act="lrelu2"`)
- `parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz` (ReLU² baseline)
- `parameter-golf/logs/` (target: new lrelu2 checkpoint)

## Constraints
- No pre-existing lrelu2-trained NPZ checkpoint found in logs or records
- Must train one using the ReLU² recipe: 13L_640d, 7k steps, wd70, batch 20k (~30 min on MPS)
- Measurements on both checkpoints must be side-by-side in summary

## Handoff
**Step 0** (REQUIRED): `cat autoresearch/cross_project/parameter_golf.md`.
**Step 1**: Search for existing lrelu2 checkpoint in logs/ and records/track_10min_16mb/.
**Step 2** (if not found): Train 13L_640d_lrelu2 with 7k steps, save to `parameter-golf/logs/13L_640d_lrelu2_for_cycle338.npz`.
**Step 3**: Run cycle 318 (eff_rank census) and cycle 319 (curvature residual) on the lrelu2 checkpoint via subprocess.
**Step 4**: Build `cycle_338_summary.md` — side-by-side comparison of ReLU²-trained vs LeakyReLU²-trained profiles.

**Verdict threshold**: CONFIRMED if eff_rank within ±30% relative AND (lrelu², relu²) rank_95 localization at same blocks (b4, b5, b12). SHIFTED if rank_95 ≤ 50 blocks differ by ≥1. REFUTED if lrelu²-trained has rank_95 > 50 everywhere.
