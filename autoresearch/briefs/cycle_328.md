# Cycle 328 Brief: LASER on parameter-golf b12 (with spectrum preflight)

## Item
341 — rank-7 weight-SVD + B-factor retrain on `blocks.12.mlp.proj.weight` (13L_640d_322 checkpoint)

## Context
**Item 340 (cycle 327) was REFUTED-LASER-fails**: b3's weight SVD spectrum is FLAT — rank-5 captures only 2.6% energy. Retrained BPB was +0.038 above baseline (threshold +0.01). The audit proved that output-covariance eff_rank is NOT a proxy for weight compressibility via SVD.

**Key lesson**: eff_rank(b12)=6.94 (cycle 318) tells us nothing about b12's weight SVD spectrum. We must check the spectrum directly before committing to retrain.

**Why b12 might still work**: b12 has hidden=320, so `proj.weight` is (640, 320) in MLX — aspect ratio 2:1 vs b3's 1:3 (640x1920). Squarer matrices are structurally more likely to have concentrated singular spectra, but this is speculation not evidence. The spectrum check resolves this in seconds.

## Hypothesis
If b12's weight SVD top-7 energy fraction > 50%: rank-7 retrain BPP <= baseline + 0.001.
If b12's weight SVD top-7 energy fraction < 10%: skip retrain, declare REFUTED-flat-spectrum.

## Key Files
- `autoresearch/cross_project/parameter_golf.md` (REQUIRED READING)
- `parameter-golf/docs/EFFRANK_CURVATURE_FINDINGS.md` section 6.2
- `parameter-golf/local_scripts/pgolf_laser_retrain.py` (from cycle 327)
- `autoresearch/results/cycle_327_summary.md` (b3 REFUTED reference)

## Dependencies
- Item 340 (cycle 327): REFUTED-LASER-fails. Spectrum check recommended by audit.
- Cycle 318: eff_rank census (b12 output eff_rank=6.94, but this is NOT weight rank)
- Cycle 323: post-scale eff_rank b12=1.03 (also NOT weight rank)

## Constraints
- Cross-project subprocess: must invoke parameter-golf .venv (MLX) from qat (PyTorch).
- b12 `mlp.proj.weight` shape is (640, 320) — NOT (640, 1920) like b3. mlp_mult varies per block.
- Spectrum preflight is a GATE: if top-7 energy < 10%, skip retrain entirely.
