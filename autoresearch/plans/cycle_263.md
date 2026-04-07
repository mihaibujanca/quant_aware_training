# Cycle 263 Plan — Item 273

## Item (verbatim)
273 (P2): Local linearity defect ε_l(x, h) = ‖f(x+h) − f(x) − J(x)h‖_G / ‖h‖²_G — sectional second-derivative of GELU vs ReLU FFN block. Sample 64 calib tokens, 32 isotropic Gaussian directions, 4 ε scales × ‖x‖, 12 blocks × 2 activations.

## Hypothesis (verbatim)
GELU defect is monotonically quadratic in ‖h‖ at all blocks; ReLU defect is bimodal — ~0 for most directions, large for directions that cross a hyperplane. Median ε_l^G > median ε_l^R at every block (GELU is everywhere-curved), but p99 ε_l^R > p99 ε_l^G in early blocks 0–4 (where ReLU's sparse hyperplane-crossings dominate the tail).

## Setup
Create `experiments/local_linearity_defect_cycle263.py`. Load HF gpt2 (mps). Wikitext-2 val 200×128 loader (cycles 218/258/260) → take first 64 tokens (token positions 0..63 of the first sequence). For each block l ∈ {0..11}: compute b_leak per block via cycle 258 recipe (ln2_out hooked across all 25,600 calib tokens, then `b_leak_l = E[gelu(z) − relu(z)]` on the 3072-dim preactivation). For each token position: capture x = ln_2(residual)[token] (768,). Define block MLP function f(x) = c_proj(σ(c_fc(x) + b_fc) [+ b_leak if ReLU]) + c_proj.bias. Jacobian: J(x) = (W_fc · diag(σ'(z))) · W_proj where z = x @ W_fc + b_fc. For each (l, σ_type, token): compute J once, sample 32 unit Gaussian directions h ∼ N(0, I_768) seed 42, normalize, scale by ε ∈ {0.01, 0.03, 0.1, 0.3} × pullback_norm(x, U), batch all 128 perturbed inputs in ONE forward pass through the MLP block, compute residuals = f(x+δ) − f(x) − J@δ. Defect ε_defect = pullback_norm(residual, U) / pullback_norm(δ, U)². Aggregate per (l, σ_type, scale) median, p90, p99.

## Verdict
**CONFIRMED** iff median(ε_G/ε_R) ≥ 5 at the smallest scale (ε=0.01·‖x‖) AND p99(ε_R) ≥ p99(ε_G) (i.e. p99 ratio ε_R/ε_G ≥ 1) in ≥3 of blocks 0..4.
**REFUTED** iff both median ratios AND p99 ratios ∈ [0.5, 2] at all 12 blocks.
**INCONCLUSIVE** otherwise. Trainable params: 0.

## Vectorisation
Per (token, block, σ): J computed once (O(768·3072)). f(x) computed once. f(x+δ) computed batched: stack the 128 perturbed inputs as (128, 768) tensor and run a single c_fc → activation → c_proj. J@δ as (768, 768) @ (768, 128).T. Total forward passes ≈ 64 × 12 × 2 × 1 batch = 1536 batched forwards (each of size 128) → ~3 minutes wall.

## Execute
Working dir `/Users/mihai/Projects/quant_aware_training`. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/local_linearity_defect_cycle263.py 2>&1 | tee autoresearch/logs/cycle_263_run.log`. Output JSON `autoresearch/results/cycle_263_local_linearity_defect.json` (per-block per-σ per-scale median/p90/p99, ratios, verdict). Summary md `autoresearch/results/cycle_263_summary.md`.

## Bookkeeping
Stage cycle 263 files ONLY (NOT pre-existing dirty files). Commit `[autoresearch] cycle 263: local linearity defect — <verdict>`. Update agenda.md: delete item 273 from NEXT/P2; prepend `### Cycle 263` PAST entry above `### Cycle 262`. Append TSV row 263. Manifest, bookkeeping commit, exec_result. Audit at end.
