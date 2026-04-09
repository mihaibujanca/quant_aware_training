# Cycle 371 Plan — Residual Stream Geometry Across Compression (REVISED)

**Item**: 379
**Hypothesis**: The composition barrier (3-block PPL 47 vs 6-block PPL 65) is visible in the residual stream geometry of *trained* models: 6-block trained models show superlinear effective-rank degradation compared to 3-block, indicating the representation subspace collapses faster than adapter training can recover.

## Pre-flight

- **P1 (duplicate)**: PASS. No prior cycle measures eff_rank or participation ratio at block boundaries across trained compression configs.
- **P2 (dependencies)**: PASS. c370 confirmed composition barrier is count-driven.
- **P3 (one cycle)**: PASS. Train 3-block (~8 min) and 6-block (~8 min), then 2 forward passes with hooks. ~20 min total.
- **P5 (null baseline)**: Float model geometry is the baseline. If trained compression causes no geometric degradation, all configs match float.

## Design

### What we measure

At each of 12 block boundaries (`transformer.h[i]` output), capture residual stream H in R^{T x 768}. Compute:

1. **Effective rank**: `exp(entropy(sigma/sum(sigma)))` from SVD of centered H.
2. **Participation ratio**: `(sum(lambda))^2 / sum(lambda^2)` from covariance eigenvalues. More robust than lambda_min/lambda_max.
3. **Top-5 PC concentration**: fraction of variance in top 5 PCs.
4. **Mean pairwise cosine**: sample 1000 pairs, mean |cos(h_i, h_j)|.

### Three configurations (all WITH adapters, trained)

| Config | Blocks | Training | Expected PPL |
|--------|--------|----------|--------------|
| Float  | 0      | none     | ~53.89       |
| 3-block | [2, 1, 10] | topk(k=512)+lstsq + LoRA r=8, 500 Adam steps | ~47 |
| 6-block | [2, 1, 10, 4, 3, 11] | same recipe | ~65 |

This measures the geometry of models that actually exhibit the composition barrier.

### Protocol

1. Load GPT-2, tokenize 200 wikitext-2 val sequences (128 tokens).
2. For each config (float, 3-block, 6-block):
   a. Load fresh GPT-2. Compress N blocks with topk+lstsq (k=512).
   b. Train adapters: 500 Adam steps, lr=3e-4, same recipe as cycle 374.
   c. Eval PPL to confirm we reproduced the barrier (3-block < 50, 6-block > 60).
   d. Run forward pass with hooks at all 12 block outputs. Compute 4 geometry metrics.
3. Compare geometry profiles. Key question: does geometry degrade superlinearly from 3-block to 6-block?

### Caveat (from review R4)

These metrics measure representation geometry (how many dims tokens span), not functional rank of the compressed map. This is a necessary but not sufficient condition for composition — if eff_rank drops, the compressed blocks project residuals into a lower-dim subspace, which IS composition-relevant, but stable eff_rank doesn't guarantee functional composition.

## Verdict criteria (trained configs)

- **CONFIRMED**: 6-block shows >30% eff_rank drop from float at >=2 layers AND 3-block shows <15% drop. Clear separation between the PPL-47 and PPL-65 regimes.
- **REFUTED**: Both trained configs show <10% eff_rank deviation from float (geometry is intact despite PPL gap — barrier is not geometric).
- **INCONCLUSIVE**: Both show similar degradation, or metrics are noisy.

## Output

- Script: `experiments/gpt2_residual_geometry_cycle371.py`
- Results: `autoresearch/results/cycle_371_residual_geometry.json`
- Plot: `autoresearch/results/cycle_371_geometry_profiles.png`
- Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_residual_geometry_cycle371.py`

## Template

Base on `experiments/gpt2_adaptive_k_cycle374.py`. Keep compress_blocks, CompressedGPT2MLPWithAdapter, train_joint. Add compute_geometry() with participation ratio instead of isotropy. Add hooks at block outputs. Run 3 configs sequentially.

## Runtime estimate

- Data loading: ~30s. Compression + training (2 configs): ~16 min. Forward passes + geometry: ~4 min. **Total: ~21 min.**
