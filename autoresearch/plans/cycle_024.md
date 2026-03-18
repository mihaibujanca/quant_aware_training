# Cycle 024 Plan — CKA Representational Similarity Matrix (GPT-2)

## Pre-flight

- P1: GPT-2 available via HuggingFace (no download needed, cached)
- P2: Shakespeare corpus at `data/shakespeare.txt`
- P3: Cycle 005 damage data readable from
  `runs/nanochat_single_block_localization_overnight_main/results.jsonl`
- P4: `.venv` has torch, transformers, scipy installed

## Script

`experiments/cka_gpt2_layers.py`

### Steps

1. Load `gpt2` model + tokenizer. Load Shakespeare, tokenize, split into
   non-overlapping 128-token chunks (up to 512 chunks = ~65K tokens).

2. Register forward hooks on:
   - embedding output (after token+position embed, before block 0)
   - each of 12 block outputs (after residual add)
   Total: 13 checkpoints, indexed 0..12.

3. Run forward pass with no_grad on all chunks. Accumulate representations
   as (N_tokens, 768) tensors for each checkpoint. Use first 4096 tokens max
   for CKA (subsample if needed — CKA scales as O(n²)).

4. Compute linear CKA for all 13×13 pairs:
   ```
   def linear_cka(X, Y):
       # X, Y: (n, d) — already centered
       X = X - X.mean(0)
       Y = Y - Y.mean(0)
       hsic_xy = (X @ Y.T).pow(2).sum()
       hsic_xx = (X @ X.T).pow(2).sum()
       hsic_yy = (Y @ Y.T).pow(2).sum()
       return (hsic_xy / (hsic_xx * hsic_yy).sqrt()).item()
   ```

5. Save 13×13 matrix as JSON.

6. Plot heatmap with matplotlib (imshow, annotate values, layer labels).
   Save to `autoresearch/results/cycle_024_cka_heatmap.png`.

7. Detect plateaus: find maximal groups of consecutive layers where ALL
   pairwise CKA ≥ 0.80 and group size ≥ 3. Report groups.

8. Compute CKA drop between adjacent layers: drop[i] = 1 - CKA[i, i+1]
   for i in 0..11 (12 values, one per inter-layer transition).

9. Load cycle 005 damage (8x geometric), align by layer_id 0..11.
   Compute Spearman(drop[0..11], damage[0..11]).

10. Save results JSON: matrix, plateaus, spearman_rho, spearman_p.

## Output files

- `autoresearch/results/cycle_024_cka_matrix.json`
- `autoresearch/results/cycle_024_cka_heatmap.png`
- `autoresearch/results/cycle_024_summary.md`

## Evaluation

Apply agenda criteria:
- Confirmed: plateau ≥3 layers with CKA ≥ 0.80 AND Spearman ≥ 0.40
- Refuted: all non-adjacent CKA < 0.50
- Inconclusive: otherwise
