# Cycle 130 Plan: Cross-block Procrustes Alignment (Item 87)

## Task
Compute pairwise Procrustes distances on polished W_down (c_proj_weight) matrices from k=4 compressed blocks [7,5,4,1]. If any pair < 10%, build shared-weight variant and measure PPL.

## Pre-flight
- P1 (duplicate): cycle 018 was inter-layer linearity (nuclear norm), not W_down Procrustes. No duplicate.
- P2 (deps): no hard deps. Compression pipeline (topk+lstsq+500 Adam) available from cycle 125.
- P3 (scope): 6 pairwise Procrustes (O(d^3) each, d=512, seconds). Shared-weight eval only if triggered. Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): independent k=4 PPL ~ 80 (from brief). Float PPL ~ 57.65.
- P6 (architecture): W_down = `CompressedFfnLayer.c_proj_weight`, shape (768, 512) after topk+lstsq. Extract AFTER 500-step Adam polish. `scipy.linalg.orthogonal_procrustes(W_i, W_j)` on (768, 512) matrices returns R of shape (512, 512). Residual = `||W_i - W_j @ R||_F / ||W_i||_F`.
- P7 (seeds): seed 42 for compression/polish. Procrustes is deterministic given weights. 3-seed clause below.

## Steps
1. Copy boilerplate from `experiments/placement_sweep_k4_cycle125.py`. New file: `experiments/procrustes_alignment_cycle130.py`.
2. Build k=4 compressed model with per-block polish (same `_compress_blocks` pattern as cycle 125). Extract polished `c_proj_weight` for blocks [7,5,4,1].
3. For all 6 pairs (i,j): `R, _ = scipy.linalg.orthogonal_procrustes(W_i, W_j)` where W_i, W_j are (768, 512). R is (512, 512). Residual = `||W_i - W_j @ R||_F / ||W_i||_F`. Print 6×6 distance matrix.
4. **3-seed clause**: if any residual falls in [5%, 15%] or [25%, 35%], re-run compression+polish with seeds {42, 123, 7} and report mean ± std of all pairwise residuals. Use the mean for threshold decisions.
5. **Gate**: if ALL pairs > 30%, REFUTE and stop. If any pair < 10%, proceed to step 6.
6. Shared-weight variant: pick pair (i*,j*) with smallest residual. Use W_{i*} as shared W_down. For block j*, set c_proj_weight = W_{i*} @ R_{i*j*}^T (rotate shared basis into j*'s column space). Rebuild stacked model, evaluate PPL on 200 OWT sequences.
7. Extreme variant (optional, only if residual < 5%): ONE shared W_down for all 4 blocks, per-block R_i^T. Evaluate PPL.
8. Save: distance matrix JSON, PPL results (if applicable), heatmap plot.

## Validation
- Procrustes residuals must be in [0, 1]. Sanity-check: self-distance = 0.
- If shared-weight test runs: PPL must be > float (57.65) and comparable to independent (~80).
- Param savings: 1 shared W_down (768×512) replaces 2. R is 512×512 orthogonal, baked in at inference.

## Criteria (from agenda)
- **Confirmed**: >= 1 pair residual < 10% AND shared-weight PPL within 2pp of independent.
- **Refuted**: all pairwise residuals > 30%.
- **Inconclusive**: low residual but shared-weight PPL degrades significantly.

## Artifacts
- `experiments/procrustes_alignment_cycle130.py`
- `autoresearch/results/cycle_130_procrustes.json`
- `autoresearch/results/cycle_130_heatmap.png`
