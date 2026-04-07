# Cycle 261 Plan — Item 272

## Item (verbatim)
272 (P1): W_down column geometry vs per-channel sign frequency σ_c — the rescue-note experiment. For each block, compute σ_c = E_x[1{z_c > 0}], split channels into σ ∈ [0,0.1] / [0.45,0.55] / [0.9,1.0], compute PCA spectrum of each subspace, principal angles between subspaces, special check at block 2 for rank-1 positive subspace (cycle 174 finding).

## Hypothesis
In blocks 1–3 the mostly-positive (σ ≥ 0.9) and mostly-negative (σ ≤ 0.1) column subspaces are partially orthogonal (median principal angle ≥ 60°) AND the mostly-positive subspace at block 2 is approximately rank-1 (top-1 SV captures ≥ 70% of squared Frobenius). In late blocks (8–11) the two subspaces merge (median angle ≤ 30°).

## Baselines
- Cycle 174: positive-regime token-dependent signal at block 2 is rank-1.
- Cycle 216: anti-podal hypothesis on W_down columns refuted.
- Cycle 218: W_down → c_fc reader cos sum baseline 0.649; random baseline ~1.27 (top-5).
- Cycle 222: act_std × ‖w_c‖ channel-selection refuted.

## Setup
Create `experiments/wdown_sign_geometry_cycle261.py`. Reuse cycle 260 calib loader: `transformers GPT2LMHeadModel/Tokenizer`, `datasets wikitext wikitext-2-raw-v1 validation`, filter texts with len(strip)>50, tokenize/truncate to SEQ_LEN=128, take first 200 sequences from `all_ids` list, seed=42, model on MPS+linalg on CPU. For each block l ∈ {0..11}: register c_fc forward hook capturing `input[0].detach().float().cpu()` (shape (B,T,768)) for ALL 200 calib sequences (single forward pass, then remove hooks). Concatenate to (200·128, 768) = (25600, 768). Compute `preact = ln2_out @ W_fc + b_fc` shape (25600, 3072) on CPU. `sigma_c = (preact > 0).float().mean(dim=0)` shape (3072,). Group masks: `mostly_neg = (sigma_c >= 0.0) & (sigma_c <= 0.1)`, `balanced = (sigma_c >= 0.45) & (sigma_c <= 0.55)`, `mostly_pos = (sigma_c >= 0.9) & (sigma_c <= 1.0)`. `W_down_rows = c_proj.weight` (3072, 768) — each row is a column of W_down in R^768. For each group `g`: `Wg = W_down_rows[mask_g]` shape (n_g, 768); compute SVD `svdvals = torch.linalg.svdvals(Wg)`; report top-20 singular values, `rank_95 = (cumsum(svdvals**2)/sum(svdvals**2) < 0.95).sum() + 1`, top-1 SV energy fraction = `svdvals[0]**2 / svdvals.pow(2).sum()`. For top-20 right SVs: `_, _, Vh = torch.linalg.svd(Wg, full_matrices=False); V_g = Vh[:20].T` shape (768, 20). Block 2: explicitly report mostly_pos top-1 SV energy fraction. Principal angles: `M = V_pos.T @ V_neg` (20, 20); `cos = torch.linalg.svdvals(M)` (20,); `angles_rad = torch.arccos(cos.clamp(-1, 1))`; `angles_deg = angles_rad * 180.0 / torch.pi`; `median_angle_deg = float(angles_deg.median())`. Null distribution per block: 100 shuffles, each shuffle generates a random permutation of `sigma_c` (preserving group sizes n_pos and n_neg from this block), recompute `V_pos_perm`, `V_neg_perm`, `M_perm`, `cos_perm`, `angles_perm_deg`, `median_perm`. Report null `mean ± std` of `median_perm` per block, plus z-score `(median_real − null_mean) / null_std`. Use seeds 42..141 for the 100 shuffles. trainable_params=0.

## Verdict
**CONFIRMED** if (a) block-2 mostly_pos top-1 SV energy ≥ 0.70, AND (b) median principal angle (pos vs neg) decreases monotonically from block 1 to block 11 by ≥ 30° (i.e. angle[1] − angle[11] ≥ 30°), AND (c) each of those angles is outside null mean ± 2σ.
**REFUTED** if all 12 blocks have median angle within ±2σ of the null mean.
**INCONCLUSIVE** otherwise.

## Execute
Working dir: `/Users/mihai/Projects/quant_aware_training`. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/wdown_sign_geometry_cycle261.py 2>&1 | tee autoresearch/logs/cycle_261_run.log`. Output JSON: `autoresearch/results/cycle_261_wdown_sign_geometry.json` (per-block group sizes, top20 svdvals, rank_95, top1 energy, median principal angle pos↔neg, null mean±std, z-score, verdict logic, config). Summary md: `autoresearch/results/cycle_261_summary.md`.

## Bookkeeping
Stage cycle 261 files ONLY (NOT pre-existing dirty files like program.md, aleph/pullback_geometry.py, prior cycle artifacts). Commit `[autoresearch] cycle 261: W_down sign-frequency geometry — <verdict>`. Update agenda.md: delete item 272 from NEXT/P1; prepend `### Cycle 261` PAST entry above `### Cycle 260`. Append TSV row. Manifest, bookkeeping commit, exec_result.

## Risks
SVD on potentially small groups: if `n_g < 20` for some group/block, `V_g` should use `min(n_g, 20)` columns; the principal-angle SVD then operates on a smaller M. Mitigate by reporting `k_used = min(n_pos, n_neg, 20)` per block. The σ thresholds are narrow ([0,0.1], [0.45,0.55], [0.9,1.0]) so the "balanced" group may be small in some blocks; that group is informational only — the verdict only uses pos↔neg. Random null shuffle preserves the per-block n_pos and n_neg by sampling without replacement from the full 3072 indices (uses a fresh `torch.Generator().manual_seed(42 + s)` per shuffle s). Linalg on CPU avoids MPS float32 instability for the (3072, 768) SVDs. Cost: 12 blocks × (2 group SVDs + 100 shuffle SVDs of 768×768 row subsets) — under a minute on CPU. Total run cap: ~5 minutes.
