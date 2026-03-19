# Cycle 046 Plan — Lstsq Refit Quality Degradation Analysis

## Goal
Diagnose whether the lstsq W_down refit becomes ill-conditioned at high compression ratios, and whether this explains the Shakespeare-works / WikiText-fails domain gap from cycle 044.

## Experiment design

### Setup
- Model: GPT-2-small (same as cycles 037-045)
- Blocks: [7, 5, 4] (the compressible blocks from cycle 005/033)
- Ratios: [2, 4, 6, 8, 10]
- Corpora: Shakespeare AND WikiText-2
- Probe: 200 sequences, seq_len=128 (same as prior cycles)
- Seeds: 3 (42, 137, 2024)

### Step 1 — Extract refit system per ratio
For each (block, ratio, corpus, seed):
1. Run topk channel selection on probe tokens (same pipeline as cycles 026/040)
2. Collect the lstsq design matrix A (probe activations through selected channels, shape [n_tokens, k_channels])
3. Collect the target matrix B (original W_down output on full activations)
4. Split probe tokens 80/20 into train/held-out
5. Solve lstsq on train split: W_refit = lstsq(A_train, B_train)
6. Record: cond(A_train), train_MSE = ||A_train @ W_refit - B_train||^2 / n_train, held_out_MSE = ||A_heldout @ W_refit - B_heldout||^2 / n_heldout

### Step 2 — Compute diagnostics
For each (block, ratio):
- **Condition number**: np.linalg.cond(A_train) — averaged over seeds
- **Overfit ratio**: held_out_MSE / train_MSE — averaged over seeds
- **Cross-corpus gap**: cond(A_shakespeare) vs cond(A_wikitext) at same ratio

### Step 3 — Verdict
- **CONFIRMED**: cond# increases >10x from 4x to 8x AND overfit ratio >2.0 at 8x for >=2/3 blocks
- **REFUTED**: cond# <1e4 at all ratios AND overfit ratio <1.5 everywhere
- **INCONCLUSIVE**: cond# rises but overfit ratio stays <1.5

If WikiText cond# is systematically higher than Shakespeare at the same ratio, this directly explains the domain gap: diverse tokens create a higher-rank activation space that the reduced channel set cannot span.

## New file
`experiments/lstsq_refit_quality_gpt2.py` — adapted from `experiments/compression_ceiling_scan_gpt2.py`

## Outputs
- `autoresearch/results/cycle_046_refit_quality.json` — all measurements
- `autoresearch/results/cycle_046_refit_quality.png` — 2-panel: cond# vs ratio, overfit ratio vs ratio (Shakespeare vs WikiText lines)

## Estimated cost
~15 min: 3 blocks x 5 ratios x 2 corpora x 3 seeds = 90 configs. Each config is one lstsq solve (fast). The bottleneck is loading GPT-2 and running forward passes for probe collection (shared across ratios within a block/corpus/seed).
