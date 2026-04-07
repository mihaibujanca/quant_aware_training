# Cycle 267 Plan — Item 277: Block 11 c_proj output rank decomposition

## Hypothesis (verbatim)
block 11 c_proj output has eff_rank ≤ 50 (concentration story) OR top-5 SVs of c_proj_11 align ≥0.5 cosine median with the top-200 lm_head reading directions (alignment story) — at least one of the two clean explanations holds.

## Verdict thresholds
- CONFIRMED if block 11 eff_rank ≤ 50 OR median top-5 cosine with top-200 lm_head rows ≥ 0.5.
- REFUTED if block 11 eff_rank ≥ 200 AND median top-5 cosine with top-200 lm_head rows ≤ 0.2.
- INCONCLUSIVE otherwise.

## Setup
Reuse cycle 252 harness pattern. BLOCKS=[1,5,11]. Calib 200 wikitext-2 val × 128, seed 42, MPS. Hook `model.transformer.h[b].mlp.c_proj` forward OUTPUT (post-c_proj residual contribution), shape (25600, 768). Trainable params = 0.

## Metrics per block
1. `cov = mlp_out.T @ mlp_out / N` (768×768). `svdvals(cov)` → eigenvalues λ. Normalize p = λ/Σλ; `eff_rank = exp(−Σ p log p)`.
2. Top-k energy fractions Σ(top k)/Σ for k ∈ {1, 5, 20, 50, 100, 200}.
3. Top-20 right singular vectors V_mlp (eigenvectors of cov = U @ diag(s) @ U^T → take first 20 columns of U).
4. SVD of `c_proj.weight` (3072, 768) → top-20 right singular vectors V_wproj ∈ R^768.
5. Cosines between top-20 V_mlp and rows of `model.lm_head.weight` (50257, 768) normalized → (20, 50257). Per top-5 V_mlp, max cosine over top-200 lm_head rows (top-200 by row L2 norm). Median of those 5 maxes = `med_top5_cos_top200`.
6. Also report: median over top-5 V_mlp of max cosine over ALL 50257 lm_head rows (`med_top5_cos_all`), and the same quantity for V_wproj (operator-level).

## Steps
1. Create `experiments/block11_cproj_decomp_cycle267.py` from cycle 252 template: swap BLOCKS=[1,5,11]; hook OUTPUT not input; drop cohort z-score stuff; add SVD of cov, top-k energy, V_mlp, V_wproj, lm_head cosines (top-200 rows selected by L2 norm of lm_head weight rows).
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/block11_cproj_decomp_cycle267.py`, log to `autoresearch/logs/cycle_267_run.log`.
3. Save JSON to `autoresearch/results/cycle_267_block11_cproj_decomp.json` with per-block eff_rank, top-k energy, top-20 V_mlp singular values (not vectors — scalars only for JSON), top-20 W_proj singular values, and cosine stats (med_top5_cos_top200, med_top5_cos_all, per-V_mlp max cosines).
4. Write `autoresearch/results/cycle_267_summary.md`.

## Bookkeeping
Delete item 277 from NEXT/P0. Add `### Cycle 267` entry at top of PAST EXPERIMENTS. Append TSV row. Stage cycle files only. Commit with verdict. Manifest + bookkeeping commit. Audit.

## Expected outcome
If cycle 252 post-GELU eff_rank ~358 transfers to c_proj output, block 5 likely eff_rank 100-300. Block 11 unknown — agenda predicts ≤50 OR alignment ≥0.5. Most probable: middling eff_rank (50-200) plus some alignment boost → INCONCLUSIVE unless either threshold lands clean.
