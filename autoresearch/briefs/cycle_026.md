# Brief — Cycle 026

## Item
14 — Rank-only baseline versus geometry versus distillation

## Motivation
The geometric method (channel selection by activation statistics + least-squares output refit) may
or may not outperform a simple SVD low-rank factorization at the same FFN parameter budget.
This is the cleanest test of whether "geometry" adds signal beyond matrix rank structure.

## Methods compared (equal param budget at each compression ratio)
1. **SVD baseline**: W_product = W_down @ W_up; truncated-SVD to rank k; reconstruct
   W_up' = S_k^(1/2) V_k^T, W_down' = U_k S_k^(1/2) for equal params. No data needed.
2. **Geometric** (existing): top-k channels by channel_scores (activation × output-norm),
   keep W_up rows, refit W_down via least-squares on probe activations.
3. **Distillation**: init compressed FFN randomly (same k hidden units), train 200 steps
   Adam with KL + CE + layer-MSE loss on probe corpus.

## Geometry method summary
`derive_ffn_compression`: select k neurons by `channel_scores`, keep W_fc rows, refit W_proj
via lstsq on `hidden_sample` → `output_sample`. Data-dependent channel selection + refitting.

## Criteria
- Confirmed: geometric beats SVD in > 50% of (block, budget) pairs on full-model ppl
- Refuted: SVD beats geometric in > 80% of pairs
- Inconclusive: wins trade back and forth

## Design
- GPT-2 + Shakespeare corpus; blocks 2, 5, 8, 11; compression ratios 4×, 8×, 16× (hidden_fraction 0.25, 0.125, 0.0625)
- Per-block single-swap perplexity (replace one FFN, measure full-model ppl)
