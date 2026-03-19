# Cycle 052 — Item 68: Corrected lstsq refit residual rank analysis — GPT-2 FFN

## Pre-flight
- P1 (duplicate): Retry of cycle 051 (design-review-rejected). Projection formula corrected per review. Novel analysis.
- P2 (dependencies): None. Uses existing pipeline from cycles 044/045.
- P3 (concrete): Three numeric measures with explicit thresholds (rank-95 ratio, energy fraction).
- P4 (feasible): Single-GPU, ~10 min. No training. SVD on (tokens, 768) matrices.
- P5 (null baseline): Cycle 044 established the corpus gap. This explains the mechanism.
- P6 (alignment): Same pipeline, same blocks/ratios as cycle 044.
- P7 (power): 4 blocks x 2 corpora = 8 conditions; threshold is majority (>=3/4).

## Hypothesis
The corpus gap (Shakespeare +6% vs WikiText +120% PPL) is caused by discarded neurons projecting into output-space directions that carry corpus-dependent information. The orthogonal residual (irreducible loss from channel removal) will be low-rank on Shakespeare but high-rank on WikiText-2.

## Method
1. Load nanochat teacher (GPT-2 124M Shakespeare fine-tuned) via `load_nanochat_teacher`.
2. Build two probe corpora via `build_probe_corpus`: Shakespeare (`data/shakespeare.txt`, 200 train seqs) and WikiText-2 (`data/wikitext2_valid.txt`, 200 train seqs).
3. For each corpus, run `probe_ffn_geometry` on blocks [7, 5, 4, 1] to get `FfnGeometryAtlas`.
4. For each block, run `derive_ffn_compression` at 6x (`hidden_fraction=1/6`, `selection_mode="topk"`).
5. Compute residual: `E = layer.output_sample - (W_refit @ h_selected.T).T - b_refit` where `h_selected = layer.hidden_sample[:, selected]`, `W_refit = compressed.c_proj_weight`, `b_refit = compressed.c_proj_bias`.
6. **Measure (a) — Total residual SVD**: `U, S, Vh = torch.linalg.svd(E, full_matrices=False)`. Compute `rank_95 = min k s.t. cumsum(S[:k]**2) >= 0.95 * sum(S**2)`.
7. **Measure (b) — Discarded-subspace projection** (corrected per cycle 051 review): Get original c_proj weight columns for discarded neurons: `W_orig = _standard_weight(layer.handle.c_proj)` shape `(768, 3072)`, `W_discard = W_orig[:, ~selected]` shape `(768, num_discarded)`. QR decompose: `Q, _ = torch.linalg.qr(W_discard)`. Project residual: `E_orth = E - E @ Q @ Q.T`. This is the component of E NOT in the discarded subspace. The in-discarded component: `E_in = E @ Q @ Q.T`. Report `discarded_energy = ||E_in||_F^2 / ||E||_F^2`.
8. **Measure (c) — Orthogonal residual rank**: SVD on `E_orth`; compute `orth_rank_95`.

## Key parameters
- `target_blocks = [7, 5, 4, 1]`, `hidden_fraction = 1/6`, `selection_mode = "topk"`
- `probe_train_sequences = 200`, `sequence_length = 128`, seed = 42
- Shakespeare probes used for channel selection (matching cycle 044 condition C)

## Implementation
- New file: `experiments/residual_rank_analysis.py`
- Reuse: `load_nanochat_teacher`, `build_probe_corpus`, `probe_ffn_geometry`, `derive_ffn_compression`, `_standard_weight` from `aleph/lm_geometric_compression.py`
- Output: `autoresearch/results/cycle_052_residual_rank.json`

## Outputs
- Per block, per corpus: total_rank_95, discarded_energy_fraction, orth_rank_95
- CONFIRMED if orth_rank_95(WikiText) >= 2x orth_rank_95(Shakespeare) for >=3/4 blocks
- REFUTED if ratio < 1.3 for all blocks
- INCONCLUSIVE if rank differs but discarded_energy < 0.20 on both corpora
