# Cycle 337 Brief

## Item
Item 346: Full 13-block weight spectrum census on pgolf — is ANY block LASER-viable?

## Context
- Cycles 327-329 measured 3 blocks: b3 (top-5 = 2.60%), b12 (top-7 = 7.07%), b11 (top-13 = 6.80%) — all flat spectra, all REFUTED
- 10 blocks remain unmeasured (b0-b2, b4-b10)
- pgolf checkpoint has variable mlp_mult: 8 blocks at 3x (1920 hidden), 4 blocks at 2x (1280 hidden), b12 at 0.5x (320 hidden)
- proj.weight shapes: (640, hidden_dim) where hidden_dim varies per block
- This is a census — no activations needed, pure weight SVD, no subprocess/MLX required

## Key Insight
The cycle 329 script already loads the checkpoint as raw numpy and computes SVD without MLX. This census can reuse that pattern and loop over all 13 blocks in one script. No cross-project subprocess needed.

## Key Files
- `autoresearch/cross_project/parameter_golf.md`
- `experiments/pgolf_laser_b11_spectrum_cycle329.py` (template — numpy-only SVD)
- Checkpoint: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`

## Verdict Thresholds (from agenda)
- CONFIRMED: >=1 block has top-5 SVD energy fraction >= 15%
- REFUTED: all 13 blocks have top-5 energy < 10%
- INCONCLUSIVE: 1+ block in 10-15% range
