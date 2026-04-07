# Cycle 270 — Item 280

## Goal
Targeted block 11 low-rank c_proj compression — does the rank-22 SVD subspace pay for itself? For r ∈ {5, 22, 50, 100}: replace c_proj_11.weight with rank-r SVD approximation, measure PPL. For r=22: also retrain W_out (the trainable factor) via Adam.

## Hypothesis
*(verbatim from item 280)* Block 11 is structurally cheap to compress on its rank-22 subspace → rank-22 no-retrain PPL ≤ 55.0 (within 1 PPL of float 53.895), and rank-22 with W_out retrain PPL ≤ 54.5.

## Setup
Load HF GPT-2 small, MPS, wikitext-2 val. Single calib/eval split: `all_ids[:200]` for retrain calib, `all_ids[200:300]` for eval (SEQ_LEN=128, disjoint, matches cycle 266/268). HF Conv1D has `c_proj.weight` shape `(3072, 768)`, applied as `x @ weight + bias`. SVD: `U, Σ, Vh = torch.linalg.svd(W, full_matrices=False)` gives `U: (3072, 768), Σ: (768,), Vh: (768, 768)`. Rank-r approx: `W_r = U[:, :r] @ diag(Σ[:r]) @ Vh[:r, :]`, same shape `(3072, 768)`. **Note**: this is the literal "rank-r SVD of the weight matrix" approach in item 280's setup wording. Cycle 267 measured the eff_rank of the c_proj OUTPUT (post-projection activations in R^768) at 22.49; the weight matrix itself has full numerical rank 768. The rank-22 weight approximation captures the top-22 input-output coupling directions of the operator, which is **not identical** to projecting onto the activation top-22 subspace cycle 267 found, but the brief and item 280 both phrase the experiment in terms of the weight SVD — so we use that literal interpretation.

## Procedure
1. Float baseline: load fresh GPT-2, eval PPL on `eval_ids = all_ids[200:300]`.
2. SVD of `model.transformer.h[11].mlp.c_proj.weight` → U, Σ, Vh.
3. **No-retrain sweep** for r ∈ {5, 22, 50, 100}: snapshot original weight, replace with `(U[:, :r] * Σ[:r]) @ Vh[:r, :]`, eval PPL, restore. Bias unchanged. Record per-r PPL.
4. **Retrain at r=22**: 3 seeds {42, 137, 2024}.
   - Build a `LowRankCProj` `nn.Module` wrapping the c_proj forward: stores frozen `A22 = U[:, :22] * Σ[:22]` (3072, 22) as buffer, trainable `B22 = Vh[:22, :]` (22, 768) as `nn.Parameter` (initialised from SVD), trainable bias `b` (768,) initialised from `c_proj.bias`. Forward: `x @ (A22 @ B22) + b` (recompute the weight each forward). Replace `model.transformer.h[11].mlp.c_proj` with this module.
   - Freeze ALL other params; only `B22` and `b` trainable. Trainable count = 22*768 + 768 = 17,664 (matches brief).
   - Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, calib=`all_ids[:200]` shuffled per seed. Snapshot best `(B22, b)`, restore at end, assert restored eval PPL == recorded best to within 1e-3.
   - Per seed: record init PPL, best PPL, best step, final PPL @ 4000.
5. Aggregate retrain `best_mean ± std` across seeds.

## Verdict thresholds (item 280 verbatim)
- **CONFIRMED** if rank-22 no-retrain ≤ 55.0 OR rank-22 retrain best_mean ≤ 54.5.
- **REFUTED** if rank-22 no-retrain ≥ 60.0 AND rank-22 retrain best_mean ≥ 58.0.
- **INCONCLUSIVE** otherwise.

## Baselines
float 53.895; cycle 266 K=512 [1,4]=43.85; cycle 267 c_proj_11 eff_rank=22.49 / top-5 energy 68.18% / top-1 energy 48.95%; cycle 173 block 11 single-block PPL +4.02%; cycle 264 block 11 ‖Δκ‖₁=2.48e-6 (rank 11/12).

## Run cap
~12 minutes (5 forward-only PPL evals @ ~30s + 3 retrain fits @ ~3 min).

## Artifacts
`experiments/block11_lowrank_cycle270.py`, `autoresearch/results/cycle_270_block11_lowrank.json`, `autoresearch/results/cycle_270_summary.md`, `autoresearch/logs/cycle_270_run.log`, `autoresearch/results/cycle_270_manifest.txt`.

## Verification
JSON+log present; per-r no-retrain PPLs recorded; per-seed retrain history recorded; restore matches recorded best (script-internal assert); verdict matches threshold logic.
