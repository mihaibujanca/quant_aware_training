# Cycle 136 Plan: Pullback-Metric Error Measurement (Item 146)

## Task
Measure per-block FFN compression error in Frobenius and G-weighted norms at 6x,
then correlate each with the PPL deltas already recorded in cycle 135.

## Hypothesis
G-weighted error (||U e_l||_2) captures semantic damage better than Frobenius
(||e_l||_F) because not all directions in activation space matter equally for
the output distribution. Blocks whose error aligns with high-curvature output
directions should show disproportionate PPL damage.

## Pre-flight
- P1 (duplicate): no prior cycle measured G-weighted vs Frobenius correlation.
- P2 (deps): cycle 135 completed, JSON available with mean_ppl_deltas for 12 blocks.
- P3 (scope): 12 blocks x 3 seeds x 200 seqs = forward passes only. No training. Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): Frobenius error is the null predictor. G-weighted must beat it.
- P6 (architecture): `pullback_norm(v, U)` takes v=(batch, seq, 768), U=(50257, 768).
  Must verify shape handling for batched error tensors.
- P7 (seeds): 3 seeds mandatory per agenda. Report per-seed rho and mean rho.

## Steps
1. Create `experiments/pullback_error_cycle136.py`.
2. Load GPT-2 (124M) and OWT eval set (200 sequences, 128 tokens).
3. Get unembedding matrix U via `aleph.pullback_geometry.get_unembedding_matrix(model)`.
4. For each seed in [42, 123, 7]:
   a. For each block l in 0..11:
      - Compress block l FFN at 6x using standard topk+lstsq pipeline.
      - Run 200 sequences through the model with ONLY block l compressed.
      - Collect per-token FFN output error: e_l = FFN_compressed(x) - FFN_float(x).
      - Compute Frobenius norm: mean over tokens of ||e_l||_2 (per-token L2).
      - Compute G-weighted norm: mean over tokens of pullback_norm(e_l, U).
      - Record both norms.
   b. Load PPL deltas from cycle 135 JSON (mean_ppl_deltas, 12 values).
   c. Compute Spearman(frob_errors, ppl_deltas) and Spearman(g_errors, ppl_deltas).
   d. Compute per-block ratio: g_error / frob_error (semantic alignment coefficient).
5. Aggregate across seeds: mean rho_frob, mean rho_g, per-seed table.
6. Save results JSON. Generate scatter plot (2 panels: Frob vs PPL, G vs PPL).
7. Apply verdict criteria, update agenda.md, commit.

## Error Collection Detail
Hook `transformer.h[l].mlp` with a forward hook that stores both input and output.
Run teacher (float) forward pass, store mlp outputs. Replace mlp weights with
compressed version (topk+lstsq), run again, store compressed outputs. Error =
compressed_output - teacher_output. This avoids needing to swap mid-forward-pass.

Two full forward passes per block per seed: one float, one with block l compressed.
Total: 2 x 12 x 3 = 72 forward passes. At ~0.5s each on GPU, ~36 seconds total.

## Validation
- Sanity: Frobenius error must be > 0 for all blocks (compression is lossy at 6x).
- Sanity: G-weighted error must be > 0 and generally different from Frobenius.
- PPL deltas must match cycle 135 values exactly (loaded from JSON, not re-measured).
- Spearman rho must be in [-1, 1]. If |rho_g| < 0.3, the metric has no predictive
  power — flag as potential problem with the pullback_norm implementation.

## Criteria (verbatim from agenda)
- **Confirmed** if: G-weighted |rho| exceeds Frobenius |rho| by >= 0.15 AND G |rho| >= 0.6.
- **Refuted** if: Frobenius |rho| >= G-weighted |rho|.
- **Inconclusive** otherwise (G marginally better but gap < 0.15 or |rho| < 0.6).

## Artifacts
- `experiments/pullback_error_cycle136.py`
- `autoresearch/results/cycle_136_pullback_error.json`
- `autoresearch/results/cycle_136_scatter.png`

## Fallbacks
- If pullback_norm is slow on (batch, seq, 768) tensors, compute per-token in a loop
  or reduce to per-sequence mean error before applying pullback_norm.
- If Frobenius rho is already very high (>0.9), the margin for G-weighted improvement
  is small. In that case, even rho_g - rho_frob < 0.15 may be meaningful. Report raw
  numbers and let the agenda criteria decide.
