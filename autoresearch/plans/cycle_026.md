# Plan — Cycle 026: Rank-only baseline vs geometry vs distillation

## Pre-flight
- P1: `aleph/lm_geometric_compression.py` — confirmed: geometric method = channel_scores topk + lstsq refit
- P2: `run_ffn_geometric_compression`, `run_ffn_distillation_baseline` both callable from existing API
- P3: No SVD baseline exists yet — must implement in experiment script
- P4: Shakespeare corpus loads via `text_path`; GPT-2 from HuggingFace; device auto-detected

## Design
- GPT-2 (hf_pretrained), blocks 2, 5, 8, 11 (early→late), compression ratios 4×/8×/16×
- hidden_fraction: 0.25 (4×), 0.125 (8×), 0.0625 (16×) → k = 768, 384, 192 hidden units
- Single-block swaps: replace one FFN at a time, measure full-model ppl on holdout
- 3 methods × 4 blocks × 3 budgets = 36 (block, budget, method) data points

## SVD baseline implementation
For each block layer_id at compression ratio r (hidden k = 3072 / r):
  1. Get W_up = c_fc.weight [3072, 768], W_down = c_proj.weight [768, 3072]
  2. Compute W_product = W_down @ W_up  [768, 768]  (composed linear map)
  3. SVD: U, S, Vh = torch.linalg.svd(W_product, full_matrices=False)  → shapes [768,768],[768],[768,768]
  4. Truncate to rank k: U_k [768, k], S_k [k], Vh_k [k, 768]
  5. Factorize into FFN shapes: W_up' = diag(sqrt(S_k)) @ Vh_k → [k, 768]; W_down' = U_k @ diag(sqrt(S_k)) → [768, k]
  6. Build CompressedFfnLayer with W_up' as c_fc_weight (no bias), W_down' as c_proj_weight + original c_proj bias
  7. No data or refit — pure analytic

Note: SVD on composed map is mathematically equivalent to SVD on the full FFN Jacobian at mean activation
(when activation ≈ linear). This ignores GELU nonlinearity but is the canonical rank baseline.

## Script
File: `experiments/rank_baseline_vs_geometry_vs_distillation.py`

Pseudocode:
```
load teacher + corpus
for block in [2, 5, 8, 11]:
    run probe_ffn_geometry for this block (reuse atlas across methods)
    for hidden_fraction in [0.25, 0.125, 0.0625]:
        k = int(3072 * hidden_fraction)
        svd_pack = make_svd_pack(atlas, k)           # pure analytic
        geo_pack = derive_ffn_compression(atlas, budget)  # existing
        # per-block single swap for SVD and geo (no SGD needed)
        for method, pack in [("svd", svd_pack), ("geometric", geo_pack)]:
            student = swap_compressed_ffns(teacher, pack, inplace=False)
            ppl = evaluate(teacher, student, corpus)
            record result
        # distillation: run 200 steps
        distill_result = run_ffn_distillation_baseline(distill_spec with steps=200)
        record distill result
save all results to autoresearch/results/cycle_026_results.json
print summary table: block × budget × method → ppl, block_mse
```

## Output
- `autoresearch/results/cycle_026_results.json` — raw per-cell results
- `autoresearch/results/cycle_026_summary.md` — win/loss table, conclusion

## Verification
- Baseline (uncompressed) ppl should be ~120–130 (consistent with prior cycles)
- SVD at 4× should cause modest ppl increase; at 16× should be large
- Geometric should beat random init (check via distillation result at step 0 vs 200)
- If SVD ppl < geometric ppl on majority → refuted; if geometric < SVD on majority → confirmed
