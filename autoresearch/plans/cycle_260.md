# Cycle 260 Plan — Item 271

## Item (verbatim)
271 (P1): Jacobian delta ΔJ_l(x) = J_GELU − J_ReLU per token, per block. For 256 calib tokens × 12 blocks: J_l(x) = W_down @ diag(σ'(z(x))) @ W_up for σ ∈ {GELU, ReLU}. Save ΔJ. Per token: SVD, rank_95. Per block: stack along token, mode-1 SVD, alignment with cycle 218 c_fc reading basis of next block.

## Hypothesis
per-token median rank_95(ΔJ_l) ≤ 100 in ≥6 blocks (low-D operator) AND the mode-1 (stacked-across-tokens) top-20 subspace of ΔJ_l aligns with the c_fc reading basis of block l+1 at top-5 principal cosine sum ≥ 2.0 in ≥6 blocks (well above cycle 218's random 1.27).

## Baselines
Cycle 217: rank_95(Δ̃) ≤ 64 in 9/12 blocks (zeroth-order). Cycle 218: centred Δ̃ top-5 cos sum 2.97 (random 1.27). Cycle 195: J_GELU rank_95 ~472/768 (full Jacobian is dense). The operator-level difference (diagonal σ'_GELU − σ'_ReLU) is concentrated near the GELU knee (z near 0).

## Setup
Create `experiments/jacobian_delta_cycle260.py`. Load HF gpt2 + tokenizer. Build calib slice as in cycles 218/258/259: 200 wikitext-2 val texts ≥50 chars × SEQ_LEN=128, seed 42 → `calib_ids` list. Take the FIRST 256 tokens across the concatenated (seq, pos) grid: run a single forward over the first 2 sequences (2×128=256 tokens) with a `c_fc` forward hook on every block to capture `ln2_out` of shape [B, T, 768]. Flatten to [256, 768]. For each block l ∈ {0..11}: extract `W_fc = block.mlp.c_fc.weight` (768, 3072), `b_fc = block.mlp.c_fc.bias` (3072,), `W_proj = block.mlp.c_proj.weight` (3072, 768). Compute `preact = ln2_out_l @ W_fc + b_fc` shape [256, 3072]. Compute `sigma_gelu_prime` via torch.autograd.grad on `F.gelu(approximate='tanh')` with a scalar sum trick (`z.requires_grad_(True); y=F.gelu(z,approximate='tanh').sum(); g=torch.autograd.grad(y, z)[0]`). Compute `sigma_relu_prime = (preact > 0).float()`. `delta_diag = sigma_gelu_prime - sigma_relu_prime` shape [256, 3072]. Per-token ΔJ: `DJ_t = (W_proj.T * delta_diag[t].unsqueeze(0)) @ W_fc.T` → (768, 768). Vectorized: build all 256 DJ at once via `DJ_all = torch.einsum('cd,tc,ec->tde', W_proj, delta_diag, W_fc)` or per-token loop (256 small matmuls). Per-token rank_95 via `torch.linalg.svdvals(DJ_t)` (singular values only, no U/V). Collect per-token left-SV top-3 via `torch.linalg.svd(..., full_matrices=False)` on sampled 16 tokens only (keep cost bounded). Per-block aggregate: two paths — (i) average DJ across 256 tokens → DJ_avg (768, 768), SVD → top-20 LEFT SVs U_avg ∈ R^{768×20}; (ii) stack-mode-1 (flatten per-token DJ to [256, 768*768] → SVD → top right SV reshaped) as secondary diagnostic. Use (i) as the primary mode-1 subspace for the alignment test. Build c_fc reading basis of block l+1 as in cycle 218: `U_fc_next = top_left_svs(W_fc_{l+1}, 20)` (top-20 left singular vectors of the next block's `c_fc.weight` as columns, (768, 20)). Principal cosines: `cos = svdvals(U_avg.T @ U_fc_next)` (20,). Report `top5_cos_sum = cos[:5].sum()` and `top20_cos_sum = cos.sum()`. Random baseline: 10 seeds of Gaussian orthonormal (768, 20) bases matched against the same reader; mean ± std of top-5 sum (cycle 218 value 1.27 is the reference). Do the alignment over blocks 0..10 (paired with readers at 1..11); block 11 has no l+1 reader. trainable_params=0.

## Verdict
**CONFIRMED** if median rank_95(ΔJ_l) ≤ 100 in ≥6 blocks AND mode-1 top-5 cos sum ≥ 2.0 in ≥6 block-pairs.
**REFUTED** if median rank_95 ≥ 400 in ≥6 blocks.
**INCONCLUSIVE** otherwise.

## Execute
`cd /Users/mihai/Projects/quant_aware_training && MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/jacobian_delta_cycle260.py 2>&1 | tee autoresearch/logs/cycle_260_run.log`. Output JSON: `autoresearch/results/cycle_260_jacobian_delta.json` (per-block median/p25/p75 rank_95, per-token rank_95 array, top-5/top-20 cos sums vs c_fc reader, random baseline mean±std, verdict logic, config). Summary md: `autoresearch/results/cycle_260_summary.md`.

## Bookkeeping
Stage cycle 260 files ONLY (NOT pre-existing dirty files like program.md, aleph/pullback_geometry.py, prior manifests). Commit `[autoresearch] cycle 260: Jacobian delta ΔJ — <verdict>`. Update agenda.md: delete item 271 from NEXT/P1; prepend `### Cycle 260` PAST entry above `### Cycle 259`. Append TSV row. Manifest, bookkeeping commit, exec_result.

## Risks
Memory: 256 per-token (768, 768) float32 DJ = 256·768·768·4 = 604 MB peak if built all at once. Mitigation: per-token loop with `torch.linalg.svdvals` (only SVs), freeing DJ each iteration; keep running sum for DJ_avg. SVD cost: 256 × 768³ ≈ 1.2e11 ops on CPU, few minutes on MPS. Mitigation: use CPU for SVDs (float32 linalg on MPS is slower than CPU for small matrices). c_fc.weight layout: HF Conv1D stores as (in=768, out=3072), so `preact = x @ W_fc + b_fc` is correct (same as cycle 218 line 138). σ'_GELU(0) = 0.5 (finite; autograd handles this cleanly). ReLU at z=0 is 0 by convention (`(z>0).float()`), fine because the GELU - ReLU difference at z=0 is 0.5. Run cap: ~15 minutes on MPS+CPU mix.
