# Cycle 092 Plan: Adam Polish Dissection — GPT-2 FFN

**Item 107** | File: `experiments/adam_polish_dissection_gpt2.py`

## Pre-flights

**P1 Data.** OWT corpus via `build_probe_corpus`; same splits as cycle 061 (200 train / 200 holdout / 200 eval, seq_len=128). Exists at `data/openwebtext_sample.txt`.

**P2 Baseline.** `load_nanochat_teacher("gpt2")` → teacher PPL on OWT eval split. Compress blocks [7,5,4,1] at 6× via topk+lstsq (`derive_ffn_compression`). Eval unpolished student PPL (step-0 checkpoint).

**P3 Snapshots.** Joint polish (single Adam, lr=1e-4, all 4 blocks) for 500 steps. **Before step 0**, clone `c_proj.weight` for each block → `W_before[block_id]`. At steps {50, 100, 200, 500}, clone `c_proj.weight` → `W_after[block_id][step]`. Eval stacked PPL at each checkpoint.

**P4 Delta analysis.** For each block and snapshot:
```
delta = W_after - W_before                          # [d_model, k]
U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
rank95 = (S.cumsum(0) / S.sum() < 0.95).sum() + 1  # rank at 95% energy
frob_ratio = delta.norm() / W_before.norm()         # relative magnitude
```

**P4b Null baseline for rank95.** For each block, build a random-matrix control:
```
# same shape and Frobenius norm as the real delta
null_delta = torch.randn_like(delta)
null_delta *= delta.norm() / null_delta.norm()
U_n, S_n, _ = torch.linalg.svd(null_delta, full_matrices=False)
null_rank95 = (S_n.cumsum(0) / S_n.sum() < 0.95).sum() + 1
```
Repeat 10 times, report mean ± std of `null_rank95`. This establishes what rank95 a generic (structureless) matrix of the same shape produces. The real delta's rank95 is meaningful only relative to this null. Adjust the confirmation/refutation thresholds in P6 accordingly: "low rank" means rank95(delta) << null_rank95, not rank95 < 16 in absolute terms.

**P5 Residual projection.** Compute lstsq residual subspace per block using **8192 tokens** (overriding the default `max_tokens_per_layer`; N/d ≈ 10.7 for d_model=768):
```
hidden = atlas.layer_geometries[block_idx].hidden_sample[:, selected]  # [N, k]
outputs = atlas.layer_geometries[block_idx].output_sample              # [N, d_model]
predicted = hidden @ W_lstsq.t()        # lstsq reconstruction
residual = outputs - predicted           # [N, d_model]
U_res, _, _ = torch.linalg.svd(residual.t(), full_matrices=False)
U_res_k = U_res[:, :rank95]             # top rank95 directions of residual
proj_energy = (U_res_k.t() @ delta).norm()**2
total_energy = delta.norm()**2
residual_frac = proj_energy / total_energy
```
Pass `max_tokens_per_layer=8192` to atlas construction to ensure the residual covariance is well-estimated.

**P6 Criterion** (calibrated against null baseline):
- **Confirmed**: rank₉₅(delta) < 0.5 × null_rank95 AND residual_frac > 0.70 → low-rank correction in lstsq residual subspace
- **Refuted**: rank₉₅(delta) ≥ null_rank95 OR residual_frac < 0.50 → full-rank / orthogonal discovery
- **Inconclusive**: intermediate rank ratio (0.5–1.0×) or mixed energy split

**P7 Outputs.** JSON results → `autoresearch/results/cycle_092_polish_dissection.json`. Contains per-block per-step: rank95, null_rank95 (mean/std), top-5 singular values, residual_frac, frob_ratio, PPL. 3 seeds. Convergence plot (PPL vs step) → `autoresearch/results/cycle_092_convergence.png`.

## Deferred

**GELU sparsity stratification** (agenda item 107 part c): per-token error stratified by GELU activation sparsity. Deferred to a follow-up cycle — the SVD + residual projection analysis answers the core "lstsq-correction vs new-structure" question; GELU stratification would add mechanistic color (does polish disproportionately help rare activation patterns?) but is not needed for the binary verdict.
