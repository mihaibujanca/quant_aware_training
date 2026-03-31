# Plan — Cycle 165 / Item 175 (rev 1)

## Goal
Decompose why z_mean ≈ -1 in GPT-2 FFN pre-GELU activations. Is the negativity learned (W_up anti-aligned with data) or a geometric accident of high-D projection?

## Pre-flights
- P1 scope: single script, 5 analyses, 1 JSON, 1 plot. ≤60 lines plan. ✓
- P2 falsifiable: CONFIRMED if trained Wx_mean shift > 0.5 vs random init; REFUTED if random Wx_mean ≈ trained.
- P3 no prior result: cycle 109 decomposed neg/pos channels but never compared random init or measured alignment.
- P4 compute: ~5 min (4096 tokens, 12 blocks, CPU).

## Script: `experiments/weight_data_alignment_cycle165.py`

### Setup
- GPT-2 from transformers, wikitext-2-raw-v1, 4096 tokens (cycle 109 pattern).
- Hook `model.transformer.h[i].mlp.c_fc` to capture input x and output z = W_up @ x + b.

### Analysis (a): E[W_up @ x] vs bias
- Per block: z_mean = mean(z over tokens), b_mean = mean(bias).
- Wx_mean = z_mean - b_mean. This isolates weight-data contribution from bias.

### Analysis (b): Per-channel cosine alignment
- Per block j: cos_j = cosine(W_up[j], E[x]) for all 3072 channels.
- Note: x is post-LayerNorm (zero-mean per token, but E[x] across tokens is nonzero).
- Report: mean, median, fraction negative.

### Analysis (c): PCA on inputs
- Per block: PCA on x (4096×768), project W_up rows onto top-10 PCs.
- Report: fraction of W_up variance in top-10, sign of mean projection onto PC1.

### Analysis (d): Random-init baseline (≥3 draws)
- Per block: draw 3 random W_up (same shape, same init std as GPT-2 c_fc).
- Compute Wx_rand = W_rand @ x (no bias) for each draw. Report mean and std of Wx_rand_mean across draws.
- Key comparison: |Wx_mean_trained - mean(Wx_rand_mean)|.

### Analysis (e): Per-block comparison table
- 12-row table: block, z_mean, Wx_mean, b_mean, cos_mean, Wx_rand_mean (±std), shift.

### Output
- JSON: `autoresearch/results/cycle_165_alignment.json` — full table + per-block arrays.
- Plot: 2-panel. Left: Wx_mean (trained) vs Wx_rand_mean (±std) per block. Right: cosine alignment histogram (all blocks pooled).

### Verdict logic
```
shift = mean(|Wx_mean_trained - mean(Wx_rand_mean)|) across blocks
if shift > 0.5: CONFIRMED
elif shift < 0.1: REFUTED
else: INCONCLUSIVE
```
