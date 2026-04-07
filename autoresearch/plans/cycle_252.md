# Cycle 252 — Item 262: Block 5 geometric deep-dive (analysis-only)

## Hypothesis (verbatim from item 262)
Block 5 will look an outlier on at least 2 of the 6 metrics — most likely effective rank (higher) or topk concentration (lower), explaining why K=512 topk leaves more residual at block 5 even when raw MSE is mid-tier (the residual is in directions that downstream attention reads more than the lower-MSE blocks' residuals do).

## Prior
Cycle 245 (CONFIRMED) → cycle 250 (REFUTED) downgraded urgency: the cycle-245 "drop block 5" win was a COUNT transition (6→5), not block-5-specific. The prior is that block 5 is NOT geometrically special. This cycle closes the direction cleanly.

## Script
`experiments/block5_geometric_cycle252.py` (new). Reuses data loader pattern from `experiments/adam_wout_correction_cycle206.py`.

## Setup
- Model: HuggingFace `gpt2` (12 blocks), eval mode, CPU or CUDA auto.
- Calib: wikitext-2 validation, `all_ids[:200]`, SEQ_LEN=128 → 25,600 tokens.
- BLOCKS = {1,2,3,4,5,6,7,8}. Cohort for z-score = {1,2,4,7,8}; block 5 is target; blocks {3,6} reported separately.
- Hooks on each target block's `mlp.c_fc` OUTPUT (preact, input to GELU) AND `mlp.c_proj` INPUT (post-GELU, shape (25600, 3072)).
- Skip metric (f) Jacobian residual (brief's explicit allowance).

## Metrics per block (on post_gelu, shape (25600, 3072))
(a) per-channel std: distribution + mean/std/percentiles (p10/p50/p90/p99).
(b) per-channel kurtosis: `torch` implementation (E[(x-μ)^4]/σ^4 − 3).
(c) effective rank: SVD → s² normalized → entropy → `exp(H)`.
(d) topk coverage: sort channels by `std_i · ‖c_proj[i,:]‖`, sum top-128 and top-512 squared weights / total.
(e) c_proj row cosine: row-normalize `c_proj.weight` (3072, 768); mean and max off-diagonal cosine.

Total = 6 scalars per block (std_mean, kurt_mean, eff_rank, cov128, cov512, mean_cos) + (max_cos bonus).

## Outlier test
Cohort C = {1,2,4,7,8}. For each metric m, `z_5 = (m_5 − mean(C_m)) / std(C_m)`. Flag if |z| ≥ 1.5 AND sign matches "harder for K=512 topk" (higher eff_rank, lower cov128/cov512, higher mean_cos, lower std_mean tail → flag any directional anomaly per metric).

## Verdict
- CONFIRMED: block 5 is ≥1.5σ outlier on ≥2 metrics in the harder direction.
- REFUTED: block 5 within 1σ on ALL 6 metrics.
- INCONCLUSIVE: outlier on exactly 1 metric.

## Outputs
- `autoresearch/results/cycle_252_block5_geometry.json` — per-block metrics dict + z-scores + verdict.
- `autoresearch/results/cycle_252_summary.md` — table (rows=blocks, cols=metrics), z-score row for block 5, verdict.
- Run cap: 15 min. SVD of (25600, 3072) × 8 blocks ≈ 30s each on CPU.

## Verification
- Assert activation shapes == (25600, 3072) per block.
- Assert `eff_rank ≤ 3072` and `cov512 ≥ cov128` per block.
- Assert z-score denominator `std(C) > 0`.
