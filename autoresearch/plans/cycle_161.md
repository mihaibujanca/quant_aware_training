# Task
Cycle 161 / Item 171: Centroid-affinity channel selection vs activation-norm topk — GPT-2 FFN k=4. For each block, compute per-channel centroid embeddings in hidden-state space (d_model-dimensional), then select k=4 channels via farthest-point sampling to maximize coverage. Compare single-block and stacked PPL against activation-norm topk baseline.

# Hypothesis
- **Confirmed if:** centroid-selected channels achieve lower stacked PPL than activation-norm topk by at least 1.5 PPL points at k=4.
- **Refuted if:** centroid selection is worse than or within 0.5 PPL of activation-norm topk, OR Jaccard overlap > 0.75 on all blocks (methods equivalent, centroid adds no value).
- **Inconclusive if:** better on some blocks, worse on others, with net stacked PPL within 0.5-1.5 of baseline, and Jaccard ≤ 0.75.

# Premise note
Cycle 27's centroid similarity (rho=0.853) validated centroid affinity as an inter-block merge predictor. Applying it to intra-block channel selection is exploratory — different granularity, different question. No strong prior backing; this tests whether the geometric signal transfers.

# Pre-flight Results
- P1: PASS — not a duplicate. Cycle 143 used eigenstructure whitening; existing kmeans mode uses 5-feature descriptors. This uses d_model centroid embeddings.
- P2: PASS — no dependencies. Uses existing calibration infra.
- P3: PASS — single experiment, clear metric.
- P5: topk baseline PPL well-established: single-block ~85, stacked ~70.7 (with polish).
- P7: Deterministic selection. Eval on 200 OWT seqs, 3 seeds for PPL variance.

# Steps
1. **New file**: `experiments/centroid_affinity_selection_cycle161.py`. Scaffold from existing topk experiment.
2. **Centroid embedding computation**: For each block, hook post-GELU intermediate activations `a` (shape [n_tokens, 3072]) and hidden states `h` (shape [n_tokens, d_model]). Per-channel centroid: `c_j = mean_over_tokens(a[:, j].unsqueeze(-1) * h)` — the activation-weighted mean hidden state for channel j (shape [d_model]). This gives 3072 vectors in R^{768}.
3. **Farthest-point sampling**: From 3072 centroid vectors, greedily select k=4: (a) seed = channel with largest centroid norm, (b) iteratively pick channel maximizing min-distance to selected set (Euclidean). O(3072 * k).
4. **Lstsq refit**: For each block, fit output projection on selected channels using same lstsq as topk baseline.
5. **Eval — single-block**: Blocks [1, 4, 7, 11], swap compressed FFN, eval PPL on 200 OWT seqs. Record per-block PPL for both methods.
6. **Eval — stacked**: Compress all 4 blocks, eval stacked PPL. Run with 500 Adam polish steps (matching baseline protocol).
7. **Diagnostic**: Report selected channels and Jaccard overlap per block. If Jaccard > 0.75 on all blocks → REFUTED (methods equivalent).
8. **Save**: `autoresearch/results/cycle_161_centroid_affinity.json` and `.png`.
9. **Agenda update**: Move item 171 from NEXT to PAST with outcome.
10. **Commit**: `[autoresearch] cycle 161: centroid-affinity channel selection — {OUTCOME}`

# Validation
- Topk baseline PPL must match known values within 3%. If not, abort.
- Both methods use identical lstsq refit and polish — only channel selection differs.

# Artifacts
- `experiments/centroid_affinity_selection_cycle161.py`
- `autoresearch/results/cycle_161_centroid_affinity.json`
- `autoresearch/results/cycle_161_centroid_affinity.png`

# Fallbacks
- If calibration set too small (n_tokens < 512), increase to 2048.
- If farthest-point selects dead channels: filter via `dead_mask` from geometry atlas first.
