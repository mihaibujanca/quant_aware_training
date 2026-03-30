# Item 146: Pullback-Metric Error Measurement — GPT-2 FFN

## Item
For each of 12 blocks, compress FFN at 6x via topk+lstsq. Measure FFN output
error e_l in two norms: Frobenius ||e_l||_F and G-weighted ||U e_l||_2 (U =
unembedding). Correlate each with single-block PPL deltas (from cycle 135).
- **Confirmed**: G-weighted Spearman |rho| exceeds Frobenius |rho| by >= 0.15, and G |rho| >= 0.6.
- **Refuted**: Frobenius |rho| >= G-weighted |rho|.

## Context
- **Cycle 135** (item 145): pullback metric confirmed on GPT-2. mean_ppl_deltas
  available per block at 6x. Block 0 dominates (PPL delta 9.59), blocks 1-8 are
  2-5 range, blocks 10-11 are 6-7. Salience and curvature anti-correlate
  (rho=-0.75, p=0.008). Curvature-damage rho=-0.65 (p=0.032).
- `aleph.pullback_geometry.pullback_norm(v, U)` computes ||Uv||_2. Already tested.
- PPL deltas from cycle 135 are reused directly (no re-measurement needed).

## Dependencies
- Cycle 135 results (hard): `autoresearch/results/cycle_135_curved_inference.json`.
- No training required. Forward passes + error measurement + correlation only.

## Key Files
- `aleph/pullback_geometry.py` — `pullback_norm`, `get_unembedding_matrix`
- `autoresearch/results/cycle_135_curved_inference.json` — PPL deltas to correlate against

## Constraints
- 3 seeds [42, 123, 7], 200 OWT sequences each. Report mean + per-seed correlations.
- Float PPL baseline: 57.65 (from cycle 115 manifest).
