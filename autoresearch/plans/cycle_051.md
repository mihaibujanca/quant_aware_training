# Cycle 051 — Item 67: Lstsq refit residual analysis — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle decomposed the lstsq residual itself. Cycle 039 measured rank-95 of block-8 residual (392/768) but not per-corpus or per-block. Novel.
- P2 (dependencies): None. Uses existing pipeline and data paths from cycles 044/045.
- P3 (concrete): Yes — three measurements (SVD rank, correlation, subspace alignment) with numeric thresholds.
- P5 (null baseline): Cycle 044 established the corpus gap (+6.29% Shakespeare vs +120.57% WikiText-2). This cycle explains why.
- P6 (alignment): Same probe + lstsq pipeline. Only adds post-hoc analysis of residual E.
- P7 (power): 3 blocks × 2 corpora = 6 conditions. Threshold is majority (≥2/3 blocks).

## Hypothesis
The lstsq residual E = f_block(x) − W_refit @ h_selected is low-rank on Shakespeare (narrow distribution, correctable) and full-rank on WikiText-2 (broad distribution, chaotic). This rank gap is the mechanical cause of corpus-specific compression.

## Method
1. Load nanochat teacher (GPT-2 124M Shakespeare fine-tuned).
2. Build two probe corpora: Shakespeare (`data/shakespeare.txt`, 200 train seqs) and WikiText-2 (`data/wikitext2_valid.txt`, 200 train seqs).
3. For each corpus, run `probe_ffn_geometry` on blocks [7, 5, 4] to get `FfnGeometryAtlas`.
4. For each block, run `derive_ffn_compression` at 6× (`hidden_fraction=1/6`, `selection_mode="topk"`).
5. Compute residual explicitly: `E = layer.output_sample - (W_refit @ h_selected.T).T - b_refit` where `h_selected = layer.hidden_sample[:, selected_indices]`, `W_refit = compressed_layer.c_proj_weight`, and `b_refit = compressed_layer.c_proj_bias`.
6. **Measure (a) — SVD rank**: `rank_90 = min k s.t. sum(s[:k]²) ≥ 0.9 * sum(s²)`.
7. **Measure (b) — Token correlation**: Pearson correlation matrix of E across tokens; report mean off-diagonal |r| (high = structured, low = noise).
8. **Measure (c) — Discarded subspace alignment**: Let `H_discard = layer.hidden_sample[:, ~selected]` (tokens × discarded neurons). Compute `Q, _ = torch.linalg.qr(H_discard.T)` so columns of Q are an orthonormal basis for the discarded-neuron column space. Project: `E_proj = E @ Q @ Q.T`. Report fraction `‖E_proj‖_F² / ‖E‖_F²`.

## Key parameters
- `target_blocks = [7, 5, 4]`, `hidden_fraction = 1/6` (6×), `selection_mode = "topk"`
- `probe_train_sequences = 200`, `sequence_length = 128`
- Seed: 42 (single seed — this is analysis, not stochastic)

## Implementation
- New file: `experiments/lstsq_residual_analysis.py`
- Reuse: `load_nanochat_teacher`, `build_probe_corpus`, `probe_ffn_geometry`, `derive_ffn_compression` from `aleph/lm_geometric_compression.py`
- Output: `autoresearch/results/cycle_051_residual_analysis.json` — per-block, per-corpus: rank_90, mean_offdiag_corr, discarded_subspace_fraction

## Outputs
- CONFIRMED if rank_90 ≤ 50 on Shakespeare AND ≥ 200 on WikiText-2 for ≥2/3 blocks.
- REFUTED if rank_90 similar on both corpora (corpus specificity not rank-driven).
- INCONCLUSIVE if rank differs but gap doesn't predict PPL delta across blocks.
