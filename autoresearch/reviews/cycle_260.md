# Cycle 260 Review — Item 271

**Decision: APPROVE**

## Summary
Pure analysis cycle (0 trainable params). For 256 calib tokens × 12 blocks, compute per-token operator-level Jacobian delta ΔJ_l(x) = W_proj.T @ diag(σ'_GELU(z) − σ'_ReLU(z)) @ W_fc.T ∈ R^{768×768}, report per-token rank_95, and test whether the mode-1 subspace of ΔJ (averaged over tokens, top-20 left singular vectors) aligns with the next block's c_fc reading basis (cycle 218 convention) at top-5 principal cosine sum ≥ 2.0 in ≥6 block-pairs.

## Checks
- **Brief alignment**: matches item 271 verbatim (verdict thresholds: median rank_95 ≤ 100 in ≥6 blocks AND top-5 cos sum ≥ 2.0 in ≥6 blocks for CONFIRMED; median rank_95 ≥ 400 in ≥6 blocks for REFUTED). PASS.
- **Tokenization & calib slice**: 256 tokens = first 2 sequences of the standard cycle 218/258/259 calib loader (200 wikitext-2 val texts ≥50 chars × SEQ_LEN=128, seed 42). 2×128=256 tokens — deterministic, matches the brief's "256 calib tokens" requirement. PASS.
- **c_fc input hook**: registered on every block's `mlp.c_fc` forward hook (same pattern as cycle 218 `make_cfc_hook` and cycle 258 `make_hook`), captures `input[0]` = LN2 output (768-dim). PASS.
- **HF Conv1D layout**: `c_fc.weight` shape is (768, 3072) so `preact = ln2_out @ W_fc + b_fc` is correct (cycle 218 line 138). `c_proj.weight` shape is (3072, 768). ΔJ construction uses `W_proj.T` (768, 3072) and `W_fc.T` (3072, 768). Shape chain: `(W_proj.T * delta_diag[t].unsqueeze(0)) @ W_fc.T` = (768, 3072) * (1, 3072) broadcast → (768, 3072) @ (3072, 768) = (768, 768). PASS.
- **σ'_GELU via autograd**: `z.requires_grad_(True); F.gelu(z, approximate='tanh').sum().backward()` or `torch.autograd.grad` gives exact σ'_GELU. Since c_fc.weight and hooked inputs are frozen, need a leaf tensor on the preact; use `preact = preact.detach().requires_grad_(True)` then autograd.grad. PASS.
- **σ'_ReLU**: `(preact > 0).float()` — correct (Heaviside, with convention at 0 = 0). At z=0 GELU' = 0.5 and ReLU' = 0 so delta_diag[t, c] = 0.5, finite and small — no boundary pathology. PASS.
- **Per-token rank_95**: `torch.linalg.svdvals(DJ_t)` returns only singular values, much cheaper than full SVD. Cumulative squared energy → smallest k with cum ≥ 0.95. PASS.
- **Mode-1 aggregation**: averaging DJ over 256 tokens → DJ_avg (768, 768), SVD → top-20 left singular vectors (`U[:, :20]`) is the primary mode-1 subspace used for the alignment test. Secondary diagnostic: stack-mode-1 flatten-then-SVD on (256, 768*768). The primary path matches the brief's "average ΔJ over tokens, then SVD" simpler option. PASS.
- **c_fc reading basis of block l+1**: `top_left_svs(W_fc_{l+1}, 20)` returns the top-20 left singular vectors of the next block's c_fc.weight as columns (768, 20), exactly as cycle 218 line 216. PASS.
- **Principal cosines**: `svdvals(U_avg.T @ U_fc_next)` — this is the standard top-k principal-cosine computation over two 20-D subspaces in R^768. cycle 218 used `V20 @ B` where `V20` was (20, 768) and `B` was (768, 20); the shape convention here (both `U_avg` and `U_fc_next` are (768, 20)) gives `U_avg.T @ U_fc_next` shape (20, 20), same quantity. PASS.
- **Random baseline**: 10 seeds of Gaussian orthonormal (768, 20) bases matched against the same reader, mean ± std. The cycle 218 top-5 random baseline value 1.27 is the reference number for the CONFIRMED threshold (2.0 >> 1.27). PASS.
- **Block coverage**: alignment tested for l ∈ {0..10} paired with reader of block l+1 ∈ {1..11} (11 pairs). Block 11 has no downstream reader and is only used for rank_95. PASS.
- **Output namespacing**: `cycle_260_*` paths, no collision. PASS.
- **Run cap**: ~15 min on MPS+CPU mix. Rough cost: 12 blocks × 256 tokens × (build DJ 768×768 + svdvals 768³ ≈ 4.5e8 ops) ≈ 1.4e12 ops total. On CPU float32 that's ~30 s per block, ~6 min total. PASS.

## Risks
- **Memory blow-up on per-token DJ**: 256 × 768 × 768 × 4 bytes = 604 MB if all DJ built at once as a [256, 768, 768] tensor. **Mitigation**: per-token loop, free DJ_t after svdvals, accumulate running sum for DJ_avg. Plan acknowledges this. PASS.
- **SVD on MPS**: float32 linalg.svd on MPS is sometimes slow or unavailable for certain shapes. Mitigation: move DJ to CPU for linalg.svdvals and linalg.svd (384 KB per matrix, cheap transfer). Plan acknowledges this. PASS.
- **σ'_GELU autograd leaf**: preact is not a leaf because it was computed from `ln2_out @ W_fc + b_fc`. Must rebuild it on a fresh leaf: `z = preact.detach().clone().requires_grad_(True); y = F.gelu(z, approximate='tanh').sum(); g = torch.autograd.grad(y, z)[0]`. Script should use this pattern. PASS.
- **σ'_ReLU at z=0 convention**: `(z > 0).float()` (not `>=`). The GELU − ReLU difference is +0.5 at z=0 exactly; using `>=` would give −0.5. The brief says "σ'_ReLU = (z > 0).float()". PASS.
- **Rank_95 convention**: smallest k such that cumulative squared-SV energy ≥ 0.95, counted as `(cum < 0.95).sum() + 1` (cycle 218 line 153). PASS.
- **`F.linear` vs Conv1D**: HF GPT-2 Conv1D forward is `x @ W + b` not `F.linear(x, W.T) + b`. Cycle 218 confirmed this by computing `preact = xc @ W_fc + b_fc` directly. The script must NOT use `F.linear(x, W_fc)` (which would compute `x @ W_fc.T`). Plan explicitly avoids this. PASS.
- **Median rank_95 per block**: compute over the 256-token per-block rank_95 array, report p25/p50/p75 for robustness. Plan has this. PASS.
- **Cycle 218 random baseline reuse**: the CONFIRMED threshold of 2.0 is set against cycle 218's measured random value 1.27. The script also recomputes the random baseline locally (10 seeds) to verify the threshold is still well above random in the current measurement context. PASS.

## Decision
APPROVE — execute as planned. The per-token loop for DJ construction (avoid 604 MB peak), the fresh-leaf autograd pattern for σ'_GELU, and the CPU-side SVD are the three must-haves the script needs to get right. All other details match cycle 218 / 258 / 259 conventions.
