# Cycle 015 Plan — Item 26: Intrinsic Dimension Trajectory (GPT-2)

## Pre-flight

- P1 dependency: item 9 (benchmark lock-in) complete since cycle 004.
- P2 data: cycle 005 per-block relative perplexity increase available in `runs/nanochat_single_block_localization_overnight_main/results.jsonl` (fields: `layer_id`, `relative_perplexity_increase`, filter `method=="geometric"`, `hidden_fraction==0.0625`).
- P3 infra: `load_nanochat_teacher`, `build_probe_corpus`, `_extract_ffn_handles` all available in `aleph/lm_geometric_compression.py`. Loading pattern in `experiments/jacobian_variance_atlas.py:95-108`.
- P4 compute: TwoNN is O(N²) pairwise distance; N=32×128=4096 tokens in 768-d is ~seconds on MPS.

## Script: `experiments/intrinsic_dim_trajectory.py`

1. Load GPT-2 + Shakespeare probe corpus (32 holdout seqs, 128 tokens). Device=MPS.
2. Register forward hooks on all 12 transformer blocks to capture residual-stream output (shape `(N_tokens, 768)`). Also store the embedding output (layer 0) for 13 total snapshots.
3. Run a single forward pass over the holdout set with `torch.no_grad()`.
4. For each of the 13 layers, compute TwoNN intrinsic dimension:
   - `D = torch.cdist(X, X)` — pairwise L2 distances.
   - For each point, sort distances, take `d1, d2` = 1st and 2nd nearest (skip self at index 0).
   - `mu = d2 / d1`, clamp `mu >= 1+eps`.
   - `ID = N / sum(log(mu))` (MLE from Facco et al. 2017).
5. Compute per-block ID change: `delta_ID[i] = |ID[i] - ID[i-1]|` for blocks 1–12.
6. Load cycle 005 damage: from results JSONL, extract best geometric 16× swap per layer (field `relative_perplexity_increase`, `hidden_fraction==0.0625`). Take mean across repeats per `layer_id`.
7. Compute Spearman correlation between `delta_ID` (blocks 1–12, aligning block i with layer i) and cycle 005 damage vector.
8. Check final-third drop: compare `mean(ID[layers 9-12])` vs `mean(ID[layers 0-4])`. Fraction decrease = `1 - final/initial`.

## Outputs

- `autoresearch/results/cycle_015_intrinsic_dim.json`: `{layer_ids, intrinsic_dims, delta_ids, cycle005_damage, spearman_rho, spearman_p, final_third_drop_frac}`.
- `autoresearch/results/cycle_015_intrinsic_dim.png`: two-panel plot (left: ID vs layer, right: scatter delta_ID vs damage with Spearman annotation).

## Criteria (verbatim from agenda)

- Confirmed if: intrinsic dimension decreases by ≥ 30% over the final third of layers AND Spearman correlation between (per-block intrinsic dimension change) and (compression difficulty from cycle 005) is ≥ 0.4.
- Refuted if: intrinsic dimension is approximately constant across all layers (< 10% variation), indicating that all layers operate on the same-complexity manifold.
- Inconclusive if: intrinsic dimension varies but the trend is non-monotonic with no correlation to compression difficulty.
