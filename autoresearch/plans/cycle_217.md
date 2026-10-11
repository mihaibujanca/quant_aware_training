# Task
Item 231 (P0): per-token Δ_l(x) = GELU_FFN(x) − ReLU_FFN(x) at each of GPT-2's 12 blocks. SVD Δ_l, project top right SVs onto lm_head (vocab interpretability) and onto next block's c_fc left SVs (cross-block reader alignment). ReLU = pure relu(z), no bias correction. Single fixed OWT batch (200 × 128 = 25600 tokens).

# Hypothesis
Claim: "Δ(x) lives in a low-rank subspace (K << 768). The bump is not a full-dimensional correction; it's a structured token-dependent message in K specific directions, and those directions are semantically interpretable via unembedding projection."
- Confirmed if: rank_95(Δ) ≤ 64 in ≥ 6 of 12 blocks, AND top singular vectors produce interpretable token shifts (visible clusters of related vocabulary items).
- Refuted if: rank_95(Δ) > 400 in all blocks (Δ is essentially full-rank noise).
- Inconclusive if: low rank but no semantic structure in vocabulary projection, or vice versa.

# Pre-flight Results
- P1 (duplicate): cycles 184/188/190/193 measured GELU vs ReLU at OUTPUT level (PPL, residual stats, rank). None did per-token Δ SVD + vocabulary projection. PASS.
- P2 (dependency): none. PASS.
- P3 (≤30 min): one fwd pass + 12 × SVD((25600, 768)) sub-second each. PASS.
- P4 (modify program/prompts): no. PASS.

# Steps
1. Create `experiments/delta_per_token_cycle217.py`.
2. Load HF `gpt2`, `.eval()`, `torch.no_grad()`; load tokenizer.
3. Reuse OWT data loader from `experiments/adam_wout_correction_cycle206.py`: N=200, SEQ_LEN=128, seed=42. Build token tensor (200, 128).
4. Register forward pre-hook on each `model.transformer.h[l].mlp.c_fc` to capture its input (post-LN2 residual stream). Store `stored[l]: (N, T, 768)`. Run one forward pass, then remove hooks.
5. For each block `l` in 0..11:
   a. `x = stored[l].reshape(-1, 768)` → (25600, 768).
   b. `preact = block.mlp.c_fc(x)` → (25600, 3072).
   c. `gelu_post = block.mlp.act(preact)`; `relu_post = F.relu(preact)`.
   d. `delta_post = gelu_post - relu_post`.
   e. `delta_out = block.mlp.c_proj(delta_post) - block.mlp.c_proj.bias` (bias cancels; subtract for clean zero-bias form) → (25600, 768). (Equivalently: `delta_post @ c_proj.weight`.)
   f. `U, S, Vh = torch.linalg.svd(delta_out, full_matrices=False)`.
   g. `cum = torch.cumsum(S**2, 0) / (S**2).sum()`; `rank_95 = (cum < 0.95).sum().item() + 1`; same for 99.
   h. `mean_delta = delta_out.mean(0)`; `ti_td_ratio = (mean_delta.norm()**2 / (delta_out.norm(dim=1)**2).mean()).item()`.
   i. For k in [0,1,2]: `v = Vh[k]`; `logit_impact = model.lm_head.weight @ v`; top-10 boost/suppress via `torch.topk(logit_impact,10)` and `torch.topk(-logit_impact,10)`; decode with tokenizer.
   j. If l < 11: `U_next, _, _ = torch.linalg.svd(model.transformer.h[l+1].mlp.c_fc.weight, full_matrices=False)`; `U_next20 = U_next[:, :20]` (shape (768,20)). `principal_cos = torch.linalg.svdvals(Vh[:3] @ U_next20)` → (3,). Random baseline: 3 Gaussian unit vectors in R^768, same product, report cosines.
   k. If memory tight, chunk token axis in 4096-token blocks for steps b-e; accumulate delta_out.
6. Write per-block JSON to `autoresearch/results/cycle_217_delta_per_token.json`: {block, rank_95, rank_99, ti_td_ratio, delta_norm_mean, sv_energy (first 20), top3_sv_vocab: [{sv, top_boost:[tokens], top_supp:[tokens]}], cross_block_cos, random_baseline_cos}.
7. Write `autoresearch/results/cycle_217_summary.md`: headline table per block (rank_95, ti/td, top-1 boost SV0), verdict vs criteria, 3-5 most striking blocks with their top-10 boost/suppress token lists, 1-paragraph plain-language interpretation.

Verdict logic:
- If rank_95 ≤ 64 in ≥6/12 blocks → Confirmed pending qualitative interpretability check (print lists for audit).
- If rank_95 > 400 in ALL blocks → Refuted.
- Else → Inconclusive. If rank criterion met but executor can't verify semantic clusters programmatically, return Inconclusive and flag for audit upgrade.

# Validation
- Assert captured tensor shape `(N, T, 768)` per block.
- Sanity: `torch.allclose(block.mlp.c_fc(stored[l]), preact)` (up to reshape).
- `delta_out.norm(dim=1).mean()` should be O(1-10) in fp32 (typical residual update scale).
- rank_95 must satisfy 1 ≤ rank_95 ≤ 768.

# Artifacts
- `experiments/delta_per_token_cycle217.py`
- `autoresearch/results/cycle_217_delta_per_token.json`
- `autoresearch/results/cycle_217_summary.md`
- `autoresearch/results/cycle_217_manifest.txt` (execute stage)

# Fallbacks
- OOM on (25600, 3072): chunk tokens in 4096-sized slices; accumulate delta_out on CPU.
- Slow SVD: unlikely at (25600, 768); if so, switch to `torch.linalg.svd(..., driver='gesdd')` or SVD of delta_out^T delta_out (768×768).
- Wrong hook capture (shape mismatch): fall back to capturing post-LN2 residual via hook on `block.ln_2` output.

# Update bookkeeping
- Move item 231 from NEXT/P0 → PAST/Cycle 217 in `autoresearch/agenda.md` (delete from NEXT, add under PAST with outcome line).
- Append TSV row to `autoresearch/results/cycles.tsv`.
- Commit: `[autoresearch] cycle 217: Δ per-token rank/subspace/logit impact`.
- Refresh `autoresearch/STATUS.md`.
