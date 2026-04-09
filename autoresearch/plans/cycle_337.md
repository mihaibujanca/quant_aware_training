# Cycle 337 Plan

## Item
346: Full 13-block weight spectrum census on pgolf — is ANY block LASER-viable?

## Approach
Single numpy-only script (no MLX, no subprocess). Load checkpoint with `np.load`, loop all 13 blocks, SVD each `blocks.{i}.mlp.proj.weight`, report per-block spectrum stats.

## Steps

1. **Write `experiments/pgolf_weight_spectrum_census_cycle346.py`** (~80 lines)
   - Load checkpoint: `np.load(CHECKPOINT, allow_pickle=True)`
   - Loop `i in range(13)`:
     - Extract `blocks.{i}.mlp.proj.weight`, cast to float32
     - `U, S, Vh = np.linalg.svd(W, full_matrices=False)`
     - Compute: top-1, top-5, top-13 energy fractions; rank for 50%/90%/95% energy
   - Print JSON to stdout with per-block results
   - Write `autoresearch/results/cycle_337_spectrum_census.json`
   - Write `autoresearch/results/cycle_337_summary.md` — table of all 13 blocks
   - Write `autoresearch/results/cycle_337_manifest.txt`

2. **Run**: `python experiments/pgolf_weight_spectrum_census_cycle346.py`

3. **Verdict logic** (in script):
   - any block top-5 >= 15% → CONFIRMED (LASER candidate found)
   - all blocks top-5 < 10% → REFUTED (LASER dead for pgolf)
   - else → INCONCLUSIVE (borderline blocks need retrain test)

## Risks
- None significant. Pure numpy SVD on small matrices (max 640x1920). Runs in <5s.

## Expected outcome
REFUTED — the 3 blocks measured so far (spanning all 3 hidden dim sizes) are all flat. Uniform flat spectra across all 13 blocks would close R1-LASER permanently.
