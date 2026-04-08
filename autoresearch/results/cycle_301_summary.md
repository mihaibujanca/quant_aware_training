# Cycle 301 — Item 311: Curvature residual decomposition vs cycle 280 col(A)

**Verdict: REFUTED**

E_kept(block 1) = 0.7463 >= 0.30 -- curvature residual concentrates inside the rank-2 kept subspace; LASER preserves rather than denoises it.

## Setup
- Calib: 200 wikitext-2 val seqs × 128 tokens (seed 42); N_TOKENS=25600.
- Float baseline PPL on 100-seq eval slice = 53.8949 (ref 53.895).
- Blocks measured: 0, 1, 2, 11.
- For each block: capture LN2(x) via forward pre-hook on c_fc; compute z = LN2(x) @ W_fc + b_fc; delta = GELU(z) - ReLU(z); Dy = delta @ W_proj  (W_proj is HF Conv1D (3072, 768)).
- SVD W_proj = U diag(S) Vh; kept-2D output subspace = span(Vh[:2, :]) ⊂ R^768.
- E_kept = ||Dy @ Vh[:2,:]^T||_F^2 / ||Dy||_F^2.
- Random null: 100 random 2D orthonormal subspaces of R^768 (per-block seed 20260406+l). Trainable params: 0.

## Per-block results

| block | ||Dy||_F | E_top1 | **E_top2** | null mean | null std | cos median | cos q95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.482e+03 | 0.5218 | **0.5902** | 0.0026 | 0.00257 | 0.7689 | 0.8069 |
| 1 | 2.809e+03 | 0.6901 | **0.7463** | 0.0031 | 0.00288 | 0.8673 | 0.9027 |
| 2 | 6.073e+03 | 0.8919 | **0.9190** | 0.0026 | 0.00240 | 0.9585 | 0.9644 |
| 11 | 2.456e+04 | 0.9096 | **0.9803** | 0.0027 | 0.00260 | 0.9901 | 0.9916 |

Isotropic reference (random 2D in R^768): 2/768 = 0.00260.

## Verdict logic (item 311 verbatim)
- CONFIRMED iff E_kept(block 1) ≤ 0.05.
- REFUTED iff E_kept(block 1) ≥ 0.30.
- INCONCLUSIVE otherwise.

E_kept(block 1) = **0.7463** → **REFUTED**.

## Artifacts
- experiments/curvature_residual_A_proj_cycle301.py
- autoresearch/results/cycle_301_decomp.json
- autoresearch/results/cycle_301_cos_histograms.png
- autoresearch/results/cycle_301_summary.md
