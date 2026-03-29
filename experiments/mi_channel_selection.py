"""Cycle 091: MI bottleneck channel selection vs topk — GPT-2 FFN.

For blocks [1, 4, 7, 11] at 6x compression (k=512 of 3072), compares:
- MI-greedy forward selection (column subset selection minimizing det(Sigma_residual))
- Topk selection (existing heuristic)
- Random selection (null baseline)
Each method uses lstsq reconstruction. Single-block swap PPL, 3 seeds.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _load_compressed_layer,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
    CompressionBudgetSpec,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [1, 4, 7, 11]
RATIO = 6
K = 3072 // RATIO  # 512
SEEDS = [42, 123, 7]
# 8192 tokens for covariance estimation (N/D ≈ 2.67 for 3072-dim)
# probe_train_sequences * sequence_length ≥ 8192 → 64 * 128 = 8192
COV_SEQUENCES = 64   # 64 * 128 = 8192 tokens for covariance
LSTSQ_SEQUENCES = 32  # 32 * 128 = 4096 tokens for lstsq
EVAL_SEQUENCES = 200
BATCH_SIZE = 4
MI_RECOMPUTE_EVERY = 50  # Full Schur recomputation every 50 steps
REG_LAMBDA = 1e-6        # Regularization for PD

print(f"K = {K} (6x compression of 3072)")


# ── MI greedy selection ────────────────────────────────────────────────────────

def mi_greedy_select(h: torch.Tensor, y: torch.Tensor, k: int, reg: float = REG_LAMBDA) -> torch.Tensor:
    """
    Greedy forward selection of k columns from h to minimize det(Sigma_{y|S}).
    Equivalent to maximizing MI under Gaussian approximation.
    h: [N, D]  hidden activations (D=3072)
    y: [N, d]  output activations (d=768)
    Returns: sorted indices [k]
    """
    N, D = h.shape
    d = y.shape[1]
    device = h.device
    dtype = torch.float64  # Use float64 for numerical stability

    h = h.to(dtype)
    y = y.to(dtype)

    # Center
    h = h - h.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)

    # Compute covariance matrices (full)
    # Sigma_yy: [d, d]
    Sigma_yy = (y.T @ y) / (N - 1) + reg * torch.eye(d, dtype=dtype, device=device)
    # Sigma_hy: [D, d]
    Sigma_hy = (h.T @ y) / (N - 1)  # [D, d]
    # Sigma_hh diagonal entries (variances) + cross-correlations needed for Schur
    # We need Sigma_hh: [D, D] — too large to store fully (3072^2 * 8 bytes = 75MB, OK)
    Sigma_hh = (h.T @ h) / (N - 1) + reg * torch.eye(D, dtype=dtype, device=device)

    selected = []
    remaining = list(range(D))

    # Current Schur complement: Sigma_yy - Sigma_y,S @ Sigma_S,S^-1 @ Sigma_S,y
    # Start with no channels selected → Schur = Sigma_yy
    # We track: Sigma_sel_sel_inv (cholesky of selected block), and Sigma_hy[selected]
    # Use log-det for numerical stability: log det(Sigma_residual) = sum of log diagonal of cholesky

    # Full Schur complement from scratch
    def compute_schur_full(sel: list) -> torch.Tensor:
        """Return Sigma_yy - Sigma_y,sel @ inv(Sigma_sel,sel) @ Sigma_sel,y"""
        if len(sel) == 0:
            return Sigma_yy.clone()
        sel_t = torch.tensor(sel, dtype=torch.long, device=device)
        Ssy = Sigma_hy[sel_t, :]   # [k_sel, d]
        Sss = Sigma_hh[sel_t][:, sel_t]  # [k_sel, k_sel]
        # Solve: Sss @ X = Ssy  → X = Sss^-1 @ Ssy
        try:
            L = torch.linalg.cholesky(Sss)
            X = torch.cholesky_solve(Ssy, L)  # [k_sel, d]
        except Exception:
            X = torch.linalg.lstsq(Sss, Ssy).solution
        return Sigma_yy - Ssy.T @ X  # [d, d]

    # logdet of a PD matrix
    def logdet_pd(M: torch.Tensor) -> float:
        try:
            L = torch.linalg.cholesky(M + reg * torch.eye(M.shape[0], dtype=dtype, device=device))
            return 2.0 * L.diagonal().log().sum().item()
        except Exception:
            eigvals = torch.linalg.eigvalsh(M)
            return eigvals.clamp(min=1e-30).log().sum().item()

    print(f"    Starting MI greedy selection (k={k})...")
    # Schur complement of current selected set
    S_cur = compute_schur_full([])  # [d, d] = Sigma_yy

    for step in range(k):
        # Recompute full Schur every MI_RECOMPUTE_EVERY steps
        if step > 0 and step % MI_RECOMPUTE_EVERY == 0:
            S_cur = compute_schur_full(selected)

        # For each candidate, rank-1 update to Schur
        # Adding channel c: new Schur = S_cur - S_cur_sc @ S_cur_cs / sigma_cc
        # where sigma_cc = Sigma_hh[c,c] - Sigma_hs[c] @ Sigma_ss^-1 @ Sigma_sh[c]
        # = S_cur contribution of c to self
        # More precisely: conditional variance of h_c given selected:
        # sigma_c|S = Sigma_hh[c,c] - Sigma_h_S[c,:] @ Sigma_SS^-1 @ Sigma_Sh[:,c]
        # And cross-cov: Sigma_y_c|S = Sigma_hy[c] - Sigma_yS @ Sigma_SS^-1 @ Sigma_Sh[:,c]

        # Efficient: for rank-1 Schur update, the new residual Schur after adding c is:
        # S_new = S_cur - (Sigma_hy[c] - Sigma_hS @ X_c) outer / sigma_c|S
        # where X_c = Sigma_SS^-1 @ Sigma_Sh[:,c]

        # We compute the partial correlation of each candidate w.r.t. selected
        # Sigma_hy: [D, d]
        # After conditioning on selected, cov(h_c, y | S) = Sigma_hy[c] - Sigma_hS[c] @ Sigma_SS^-1 @ Sigma_Sy
        # = Sigma_hy[c] - [row c of (Sigma_hh[C,S] @ Sigma_SS^-1)] @ Sigma_Sy

        if len(selected) == 0:
            # No conditioning needed
            cand_cov_hy = Sigma_hy  # [D, d]
            cand_var = Sigma_hh.diagonal()  # [D]
        else:
            sel_t = torch.tensor(selected, dtype=torch.long, device=device)
            Sss = Sigma_hh[sel_t][:, sel_t]  # [n_sel, n_sel]
            try:
                L_ss = torch.linalg.cholesky(Sss)
                # Sigma_hS for ALL candidates: [D, n_sel]
                Sigma_hS_all = Sigma_hh[:, sel_t]  # [D, n_sel]
                # X = Sigma_SS^-1 @ Sigma_Sh_all = [n_sel, D]
                Sigma_Sy = Sigma_hy[sel_t, :]  # [n_sel, d]
                # cand_cov_hy[c] = Sigma_hy[c] - Sigma_hS_all[c] @ Sigma_SS^-1 @ Sigma_Sy
                # = Sigma_hy[c] - (Sigma_hS_all[c] @ L_ss^-T @ L_ss^-1) @ Sigma_Sy
                # Let V = L_ss^-1 @ Sigma_Sy: [n_sel, d]
                V = torch.cholesky_solve(Sigma_Sy, L_ss)  # [n_sel, d]
                # cand contribution to y: [D, d]
                cand_cov_hy = Sigma_hy - Sigma_hS_all @ V  # [D, d]
                # cand conditional variance: [D]
                # sigma_c|S = Sigma_hh[c,c] - Sigma_hS_all[c] @ Sigma_SS^-1 @ Sigma_Sh_all[c]
                # = diag(Sigma_hh) - row-wise: (Sigma_hS_all @ L_ss^-T) * (L_ss^-1 @ Sigma_Sh_all.T).T
                # = diag(Sigma_hh) - sum_j (L_ss^-1 @ Sigma_hS_all.T)_ji^2
                U = torch.linalg.solve_triangular(L_ss, Sigma_hS_all.T, upper=False)  # [n_sel, D]
                cand_var = Sigma_hh.diagonal() - (U ** 2).sum(0)  # [D]
            except Exception:
                # Fallback: full recompute S_cur and use diagonal
                S_cur = compute_schur_full(selected)
                cand_cov_hy = Sigma_hy.clone()
                cand_var = Sigma_hh.diagonal().clone()

        # Rank-1 Schur update: det(S_new) = det(S_cur) * (1 - g_c^T S_cur^-1 g_c / sigma_c|S)
        # where g_c = cand_cov_hy[c] (the cross-cov vector)
        # Minimize det(S_new) = maximize the reduction factor:
        # reduction[c] = g_c^T S_cur^-1 g_c / sigma_c|S
        # We want the c with max reduction.
        # g_c^T S_cur^-1 g_c = ||L_cur^-1 g_c||^2 where S_cur = L_cur L_cur^T

        rem_t = torch.tensor(remaining, dtype=torch.long, device=device)
        g = cand_cov_hy[rem_t, :]  # [n_rem, d]
        v = cand_var[rem_t].clamp(min=1e-12)  # [n_rem]

        try:
            L_cur = torch.linalg.cholesky(S_cur + reg * torch.eye(d, dtype=dtype, device=device))
            # Solve L_cur @ z = g^T → z: [d, n_rem]
            z = torch.linalg.solve_triangular(L_cur, g.T, upper=False)  # [d, n_rem]
            reduction = (z ** 2).sum(0) / v  # [n_rem]
        except Exception:
            # Fallback: use trace proxy
            reduction = (g ** 2).sum(1) / v

        best_idx = reduction.argmax().item()
        best_c = remaining[best_idx]

        # Update S_cur with rank-1 downdate
        g_best = cand_cov_hy[best_c, :].unsqueeze(1)  # [d, 1]
        v_best = float(cand_var[best_c].clamp(min=1e-12).item())
        try:
            L_cur_check = torch.linalg.cholesky(S_cur + reg * torch.eye(d, dtype=dtype, device=device))
            z_best = torch.linalg.solve_triangular(L_cur_check, g_best, upper=False)  # [d, 1]
            # S_new = S_cur - g g^T / sigma_c|S
            S_cur = S_cur - (g_best @ g_best.T) / v_best
        except Exception:
            S_cur = compute_schur_full(selected + [best_c])

        selected.append(best_c)
        remaining.pop(best_idx)

        if (step + 1) % 100 == 0:
            print(f"      step {step+1}/{k}, logdet(S_cur)={logdet_pd(S_cur):.4f}")

    return torch.tensor(sorted(selected), dtype=torch.long)


def build_compressed_layer(
    layer_id: int,
    atlas: FfnGeometryAtlas,
    selected: torch.Tensor,
    method: str,
    seed: int,
) -> CompressedFfnLayer:
    """Build a CompressedFfnLayer from a selection of channels."""
    layer = next(lg for lg in atlas.layer_geometries if lg.layer_id == layer_id)

    source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
    source_b_in = _standard_bias(layer.handle.c_fc)
    use_output_bias = _standard_bias(layer.handle.c_proj) is not None

    student_w_in = source_w_in[selected].contiguous()
    student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

    hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
    student_w_out, student_b_out = _fit_output_projection(
        hidden, layer.output_sample, use_bias=use_output_bias
    )

    source_param_count = (
        source_w_in.numel() + (source_b_in.numel() if source_b_in is not None else 0)
        + _standard_weight(layer.handle.c_proj).numel()
        + (_standard_bias(layer.handle.c_proj).numel() if _standard_bias(layer.handle.c_proj) is not None else 0)
    )
    compressed_param_count = (
        student_w_in.numel() + (student_b_in.numel() if student_b_in is not None else 0)
        + student_w_out.numel() + (student_b_out.numel() if student_b_out is not None else 0)
    )

    return CompressedFfnLayer(
        layer_id=layer_id,
        activation_kind=layer.activation_kind,
        source_hidden_dim=layer.hidden_dim,
        compressed_hidden_dim=int(selected.shape[0]),
        c_fc_weight=student_w_in,
        c_fc_bias=student_b_in,
        c_proj_weight=student_w_out,
        c_proj_bias=student_b_out,
        selected_indices=selected,
        source_param_count=source_param_count,
        compressed_param_count=compressed_param_count,
        notes={"selection_mode": method, "seed": seed},
    )


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Main ───────────────────────────────────────────────────────────────────────

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
print(f"  device: {teacher.device}")

all_results = []
mi_selections = {}  # cache MI selections per block (deterministic given cov)

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    # Build corpus for PPL evaluation
    eval_corpus_spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=SHAKESPEARE_PATH,
        sequence_length=128,
        probe_train_sequences=LSTSQ_SEQUENCES,
        probe_holdout_sequences=EVAL_SEQUENCES,
        eval_val_sequences=EVAL_SEQUENCES,
        dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    eval_corpus = build_probe_corpus(eval_corpus_spec)

    print("Computing baseline perplexity...")
    baseline_ppl = eval_ppl(teacher.model, teacher, eval_corpus, BATCH_SIZE)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Probe geometry for lstsq (4096 tokens = LSTSQ_SEQUENCES * 128)
    print("Probing FFN geometry (lstsq pass, 4096 tokens)...")
    lstsq_atlas = probe_ffn_geometry(
        teacher,
        eval_corpus,
        layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=LSTSQ_SEQUENCES * 128,
        batch_size=BATCH_SIZE,
        jacobian_samples=1,
    )

    # Build a separate corpus for covariance estimation (8192 tokens)
    # We use a different shuffle seed offset to get different sequences
    cov_corpus_spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=SHAKESPEARE_PATH,
        sequence_length=128,
        probe_train_sequences=COV_SEQUENCES,
        probe_holdout_sequences=50,
        eval_val_sequences=50,
        dataset_name="proxy_shakespeare",
        shuffle_seed=seed + 1000,  # Different data from lstsq pass
    )
    cov_corpus = build_probe_corpus(cov_corpus_spec)

    print("Probing FFN geometry (covariance pass, 8192 tokens)...")
    cov_atlas = probe_ffn_geometry(
        teacher,
        cov_corpus,
        layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=COV_SEQUENCES * 128,
        batch_size=BATCH_SIZE,
        jacobian_samples=1,
    )

    print(f"\n{'block':>6} {'method':>10} {'PPL':>8} {'rel%':>8}")
    print("-" * 45)

    for block_id in TARGET_BLOCKS:
        lstsq_layer = next(lg for lg in lstsq_atlas.layer_geometries if lg.layer_id == block_id)
        cov_layer = next(lg for lg in cov_atlas.layer_geometries if lg.layer_id == block_id)

        # Check condition number
        h_cov = cov_layer.hidden_sample.float()
        y_cov = cov_layer.output_sample.float()
        N_cov, D_h = h_cov.shape
        print(f"\n  Block {block_id}: cov_N={N_cov}, D_h={D_h}, D_y={y_cov.shape[1]}")

        # ── Topk selection ─────────────────────────────────────────────────
        budget = CompressionBudgetSpec(hidden_fraction=1.0 / RATIO, selection_mode="topk")
        topk_indices = _select_hidden_units(lstsq_layer, budget, K)
        topk_layer_obj = build_compressed_layer(block_id, lstsq_atlas, topk_indices, "topk", seed)
        topk_pack = CompressedFfnPack(
            method="topk_lstsq",
            model_family="gpt2",
            dataset_name="proxy_shakespeare",
            layers=(topk_layer_obj,),
            metadata={},
        )
        student = swap_compressed_ffns(teacher, topk_pack, inplace=False)
        topk_ppl = eval_ppl(student, teacher, eval_corpus, BATCH_SIZE)
        topk_rel = (topk_ppl - baseline_ppl) / baseline_ppl * 100
        del student
        print(f"  {block_id:>4}      topk  {topk_ppl:8.2f}  {topk_rel:+7.2f}%")

        # ── MI-greedy selection ────────────────────────────────────────────
        # MI selection is deterministic given the covariance — cache per block
        cache_key = (seed, block_id)
        if cache_key not in mi_selections:
            print(f"  Computing MI greedy selection for block {block_id}, seed {seed}...")
            mi_indices = mi_greedy_select(h_cov, y_cov, K)
            mi_selections[cache_key] = mi_indices
        else:
            mi_indices = mi_selections[cache_key]

        mi_layer_obj = build_compressed_layer(block_id, lstsq_atlas, mi_indices, "mi_greedy", seed)
        mi_pack = CompressedFfnPack(
            method="mi_greedy_lstsq",
            model_family="gpt2",
            dataset_name="proxy_shakespeare",
            layers=(mi_layer_obj,),
            metadata={},
        )
        student = swap_compressed_ffns(teacher, mi_pack, inplace=False)
        mi_ppl = eval_ppl(student, teacher, eval_corpus, BATCH_SIZE)
        mi_rel = (mi_ppl - baseline_ppl) / baseline_ppl * 100
        del student
        print(f"  {block_id:>4}  mi_greedy  {mi_ppl:8.2f}  {mi_rel:+7.2f}%")

        # ── Random selection ───────────────────────────────────────────────
        rng = torch.Generator().manual_seed(seed * 1000 + block_id)
        perm = torch.randperm(3072, generator=rng)
        rand_indices = perm[:K].sort().values
        rand_layer_obj = build_compressed_layer(block_id, lstsq_atlas, rand_indices, "random", seed)
        rand_pack = CompressedFfnPack(
            method="random_lstsq",
            model_family="gpt2",
            dataset_name="proxy_shakespeare",
            layers=(rand_layer_obj,),
            metadata={},
        )
        student = swap_compressed_ffns(teacher, rand_pack, inplace=False)
        rand_ppl = eval_ppl(student, teacher, eval_corpus, BATCH_SIZE)
        rand_rel = (rand_ppl - baseline_ppl) / baseline_ppl * 100
        del student
        print(f"  {block_id:>4}    random  {rand_ppl:8.2f}  {rand_rel:+7.2f}%")

        # ── Channel overlap ────────────────────────────────────────────────
        topk_set = set(topk_indices.tolist())
        mi_set = set(mi_indices.tolist())
        overlap = len(topk_set & mi_set) / K
        print(f"  Block {block_id}: overlap topk∩MI = {overlap:.3f} ({len(topk_set & mi_set)}/{K})")

        all_results.append({
            "seed": seed,
            "block": block_id,
            "baseline_ppl": float(baseline_ppl),
            "topk_ppl": float(topk_ppl),
            "topk_rel_pct": float(topk_rel),
            "mi_ppl": float(mi_ppl),
            "mi_rel_pct": float(mi_rel),
            "random_ppl": float(rand_ppl),
            "random_rel_pct": float(rand_rel),
            "channel_overlap_fraction": float(overlap),
            "k": K,
        })

# ── Aggregate ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")

summary_rows = []
for block_id in TARGET_BLOCKS:
    block_results = [r for r in all_results if r["block"] == block_id]
    topk_ppls = [r["topk_ppl"] for r in block_results]
    mi_ppls = [r["mi_ppl"] for r in block_results]
    rand_ppls = [r["random_ppl"] for r in block_results]
    overlaps = [r["channel_overlap_fraction"] for r in block_results]
    baselines = [r["baseline_ppl"] for r in block_results]

    mean_topk_rel = np.mean([(t - b) / b * 100 for t, b in zip(topk_ppls, baselines)])
    mean_mi_rel = np.mean([(m - b) / b * 100 for m, b in zip(mi_ppls, baselines)])
    mean_rand_rel = np.mean([(r - b) / b * 100 for r, b in zip(rand_ppls, baselines)])
    mean_overlap = np.mean(overlaps)

    # Relative improvement of MI over topk (how much better MI is)
    mi_vs_topk_rel = np.mean([(t - m) / t * 100 for t, m in zip(topk_ppls, mi_ppls)])

    print(f"\n  Block {block_id}:")
    print(f"    topk mean PPL:  {np.mean(topk_ppls):.2f} ({mean_topk_rel:+.2f}% vs baseline)")
    print(f"    MI   mean PPL:  {np.mean(mi_ppls):.2f} ({mean_mi_rel:+.2f}% vs baseline)")
    print(f"    rand mean PPL:  {np.mean(rand_ppls):.2f} ({mean_rand_rel:+.2f}% vs baseline)")
    print(f"    MI vs topk:     {mi_vs_topk_rel:+.2f}% (positive = MI better)")
    print(f"    channel overlap:{mean_overlap:.3f}")

    summary_rows.append({
        "block": block_id,
        "mean_topk_ppl": float(np.mean(topk_ppls)),
        "mean_mi_ppl": float(np.mean(mi_ppls)),
        "mean_rand_ppl": float(np.mean(rand_ppls)),
        "mean_topk_rel_pct": float(mean_topk_rel),
        "mean_mi_rel_pct": float(mean_mi_rel),
        "mean_rand_rel_pct": float(mean_rand_rel),
        "mi_vs_topk_rel_pct": float(mi_vs_topk_rel),
        "mean_channel_overlap": float(mean_overlap),
    })

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

# Confirmed: MI-greedy outperforms topk by ≥5% relative PPL in ≥2/4 blocks
blocks_mi_better_5pct = sum(1 for r in summary_rows if r["mi_vs_topk_rel_pct"] >= 5.0)
# Refuted: ≥90% channel overlap
blocks_high_overlap = sum(1 for r in summary_rows if r["mean_channel_overlap"] >= 0.90)
# Inconclusive: different channels but <3% improvement
blocks_mi_better_3pct = sum(1 for r in summary_rows if r["mi_vs_topk_rel_pct"] >= 3.0)

print(f"  Blocks where MI-greedy ≥5% better PPL than topk: {blocks_mi_better_5pct}/4")
print(f"  Blocks with ≥90% channel overlap (topk≈MI):       {blocks_high_overlap}/4")
print(f"  Blocks where MI-greedy ≥3% better PPL than topk: {blocks_mi_better_3pct}/4")

if blocks_mi_better_5pct >= 2:
    verdict = "CONFIRMED"
    verdict_reason = f"MI-greedy outperforms topk by ≥5% relative PPL in {blocks_mi_better_5pct}/4 blocks"
elif blocks_high_overlap >= 3:
    verdict = "REFUTED"
    verdict_reason = f"MI-greedy selects ≥90% same channels as topk in {blocks_high_overlap}/4 blocks (heuristic near-optimal)"
elif blocks_mi_better_3pct < 2:
    verdict = "REFUTED"
    verdict_reason = f"MI-greedy channels differ from topk but PPL improvement <3% in most blocks (column subset selection adds no practical value over topk heuristic)"
else:
    verdict = "INCONCLUSIVE"
    verdict_reason = f"MI-greedy selects different channels but PPL improvement <5% (max: {max(r['mi_vs_topk_rel_pct'] for r in summary_rows):.1f}%)"

print(f"\n  VERDICT: {verdict}")
print(f"  Reason: {verdict_reason}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "experiment": "mi_channel_selection",
    "cycle": 91,
    "k": K,
    "ratio": RATIO,
    "target_blocks": TARGET_BLOCKS,
    "seeds": SEEDS,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "summary": summary_rows,
    "per_seed_results": all_results,
}

out_path = RESULTS_DIR / "cycle_091_mi_channel.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

blocks = [r["block"] for r in summary_rows]
x = np.arange(len(blocks))
width = 0.25

ax = axes[0]
ax.bar(x - width, [r["mean_topk_rel_pct"] for r in summary_rows], width, label="topk+lstsq", color="steelblue")
ax.bar(x, [r["mean_mi_rel_pct"] for r in summary_rows], width, label="MI-greedy+lstsq", color="coral")
ax.bar(x + width, [r["mean_rand_rel_pct"] for r in summary_rows], width, label="random+lstsq", color="gray", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"Block {b}" for b in blocks])
ax.set_ylabel("Mean relative PPL change (%)")
ax.set_title("Single-block 6x compression: PPL degradation by method")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)

ax = axes[1]
ax.bar(x, [r["mean_channel_overlap"] for r in summary_rows], color="purple", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f"Block {b}" for b in blocks])
ax.set_ylabel("Channel overlap fraction (topk ∩ MI-greedy) / 512")
ax.set_title("Channel overlap: topk vs MI-greedy")
ax.set_ylim(0, 1)
ax.axhline(0.9, color="red", linewidth=1, linestyle="--", label="90% threshold")
ax.legend()

plt.tight_layout()
plot_path = RESULTS_DIR / "cycle_091_mi_channel.png"
plt.savefig(plot_path, dpi=120)
print(f"Plot saved to {plot_path}")
print("\nDone.")
