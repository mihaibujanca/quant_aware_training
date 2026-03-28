"""Cycle 081: Analytic Gauss-Newton correction on stacked compressed GPT-2 FFN blocks.

After topk+lstsq compression of blocks [7,5,4,1] at 6x compression, compute a
single Newton step (Gauss-Newton) on each block's W_down using the full stacked
model loss. Compare: (a) unpolished, (b) steepest descent, (c) single Gauss-Newton
step (primary), (d) 500 Adam steps.

Confirmed if: Newton step achieves <= +8% PPL (within 2x of Adam's +4.9%).
Refuted if: Newton step diverges or gives > +20% PPL.
Inconclusive if: Newton step helps (+15-20%) but doesn't match Adam.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200   # Newton / steepest-descent / Adam data
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 8              # halved to 4 on OOM (auto-guarded)
ADAM_STEPS = 500
ADAM_LR = 1e-4
CG_MAX_ITER = 10            # single Newton step; tight convergence not needed
CG_TOL = 1e-3
HV_SEQS = 32                # sequences used for Hv (subset of holdout, for speed)

DATA_DIR = Path(__file__).parent.parent / "data"
OWT_PATH = DATA_DIR / "openwebtext_sample.txt"

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        return model.transformer.h[block_id].mlp
    except (AttributeError, IndexError):
        return None


def _compute_gradient(
    model: nn.Module,
    param: torch.Tensor,
    holdout_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute dL/d(param) over the full holdout set, accumulating batches."""
    model.eval()
    n_seqs = holdout_tokens.shape[0]
    grad_acc = torch.zeros_like(param, dtype=torch.float32)
    n_batches = 0

    for start in range(0, n_seqs, batch_size):
        batch = holdout_tokens[start:start + batch_size].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        if param.grad is not None:
            param.grad.zero_()

        logits = _forward_logits(model, inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        if param.grad is not None:
            grad_acc += param.grad.detach().float()
        n_batches += 1

    # Average over batches
    if n_batches > 0:
        grad_acc /= n_batches

    return grad_acc.detach()


def _gnhv(
    model: nn.Module,
    param: torch.Tensor,
    v: torch.Tensor,
    holdout_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    n_seqs_limit: int | None = None,
) -> torch.Tensor:
    """Compute Hessian-vector product H @ v via finite differences.

    Hv ≈ (g(w + eps*v) - g(w - eps*v)) / (2*eps)
    Avoids create_graph (no double backward). Works on CPU with flash attention.
    n_seqs_limit: if set, use only first n_seqs_limit sequences (faster for CG).
    """
    model.eval()
    n_seqs = holdout_tokens.shape[0]
    if n_seqs_limit is not None:
        n_seqs = min(n_seqs, n_seqs_limit)
    hv_acc = torch.zeros_like(param, dtype=torch.float32)
    n_batches = 0

    v_flat = v.reshape(-1).to(param.dtype).detach()
    v_norm = float(v_flat.norm().item())
    w_norm = float(param.data.norm().item())
    # eps chosen so eps * ||v|| ~ 1e-3 * ||w||
    eps = 1e-3 * (w_norm / (v_norm + 1e-10))
    eps = max(eps, 1e-7)

    for start in range(0, n_seqs, batch_size):
        batch = holdout_tokens[start:start + batch_size].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        # g(w + eps*v)
        with torch.no_grad():
            param.data.add_((eps * v_flat).reshape(param.shape).to(param.dtype))
        if param.grad is not None:
            param.grad.zero_()
        logits_p = _forward_logits(model, inputs)
        loss_p = F.cross_entropy(logits_p.reshape(-1, logits_p.shape[-1]), targets.reshape(-1))
        g_plus = torch.autograd.grad(loss_p, param)[0].detach().float()

        # g(w - eps*v)
        with torch.no_grad():
            param.data.add_((-2 * eps * v_flat).reshape(param.shape).to(param.dtype))
        if param.grad is not None:
            param.grad.zero_()
        logits_m = _forward_logits(model, inputs)
        loss_m = F.cross_entropy(logits_m.reshape(-1, logits_m.shape[-1]), targets.reshape(-1))
        g_minus = torch.autograd.grad(loss_m, param)[0].detach().float()

        # Restore w
        with torch.no_grad():
            param.data.add_((eps * v_flat).reshape(param.shape).to(param.dtype))

        hv_acc += (g_plus - g_minus) / (2 * eps)
        n_batches += 1

    if n_batches > 0:
        hv_acc /= n_batches

    return hv_acc.detach()


def _conjugate_gradient(
    hv_fn,
    b: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, bool, int, float]:
    """Solve H x = b via conjugate gradient.

    hv_fn: callable(v) -> H @ v
    Returns (x, converged, n_iter, residual_norm)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rsold = (r * r).sum()

    converged = False
    final_iter = max_iter
    final_residual = float(rsold.item() ** 0.5)

    for i in range(max_iter):
        hp = hv_fn(p)
        alpha = rsold / ((p * hp).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * hp
        rsnew = (r * r).sum()
        residual = float(rsnew.item() ** 0.5)
        final_iter = i + 1
        final_residual = residual

        if residual < tol:
            converged = True
            break

        beta = rsnew / (rsold + 1e-12)
        p = r + beta * p
        rsold = rsnew

    return x, converged, final_iter, final_residual


def _ppl_from_model(
    teacher: TeacherBundle,
    model: nn.Module,
    owt_corpus,
    batch_size: int,
) -> float:
    metrics = _evaluate_model(teacher, model, owt_corpus, batch_size=batch_size)
    return metrics["perplexity"]


def _adam_polish_all_blocks(
    model: nn.Module,
    block_ids: list[int],
    holdout_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    batch_size: int,
) -> None:
    """Run n_steps Adam on all W_down (c_proj.weight) jointly."""
    model.requires_grad_(False)
    params_to_opt = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        mlp.c_proj.weight.requires_grad_(True)
        params_to_opt.append(mlp.c_proj.weight)
        if mlp.c_proj.bias is not None:
            mlp.c_proj.bias.requires_grad_(True)
            params_to_opt.append(mlp.c_proj.bias)

    if not params_to_opt:
        return

    optimizer = torch.optim.Adam(params_to_opt, lr=lr)
    model.train()
    n_seqs = holdout_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:batch_size]
        batch = holdout_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        raw = model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

    model.eval()
    model.requires_grad_(False)


def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="cpu",
        )
    )
    device = teacher.device
    print(f"Teacher loaded: {teacher.model_name}, device={device}")

    print("Building OWT corpus for baseline PPL...")
    owt_eval_corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQUENCE_LENGTH,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_ppl = _ppl_from_model(teacher, teacher.model, owt_eval_corpus, batch_size=BATCH_SIZE)
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    seed_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        _set_seed(seed)

        owt_corpus = build_probe_corpus(
            ProbeCorpusSpec(
                teacher=teacher,
                text_path=str(OWT_PATH),
                sequence_length=SEQUENCE_LENGTH,
                probe_train_sequences=PROBE_TRAIN_SEQS,
                probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
                eval_val_sequences=EVAL_VAL_SEQS,
                shuffle_seed=seed,
            )
        )

        holdout_tokens = owt_corpus.probe_holdout  # [n_seqs, seq_len]
        print(f"  holdout shape: {holdout_tokens.shape}")

        # Probe geometry
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=4096,
            batch_size=BATCH_SIZE,
        )

        # ---- Build unpolished student ----
        _set_seed(seed)
        pack_all = derive_ffn_compression(atlas, budget)
        student_unpolished = swap_compressed_ffns(teacher, pack_all, inplace=False)
        student_unpolished.eval()

        unpolished_ppl = _ppl_from_model(teacher, student_unpolished, owt_corpus, batch_size=BATCH_SIZE)
        unpolished_delta = 100.0 * (unpolished_ppl - baseline_ppl) / baseline_ppl
        print(f"  Unpolished PPL: {unpolished_ppl:.3f} ({unpolished_delta:+.1f}%)")

        # Sanity check
        if not (20.0 <= unpolished_delta <= 40.0):
            print(f"  WARNING: Unpolished delta {unpolished_delta:.1f}% outside [+20%, +40%]. Proceeding anyway.")

        # ---- Gauss-Newton correction (primary) ----
        print("\n  -- Gauss-Newton Newton step correction --")
        _set_seed(seed)
        pack_all2 = derive_ffn_compression(atlas, budget)
        student_newton = swap_compressed_ffns(teacher, pack_all2, inplace=False)
        student_newton.eval()

        per_block_ppl_newton = []  # PPL after correcting each block
        newton_cg_info = []        # CG convergence info per block
        batch_size_used = BATCH_SIZE

        for block_id in TARGET_BLOCKS:
            mlp = _find_compressed_mlp(student_newton, block_id)
            if mlp is None or not hasattr(mlp, "c_proj"):
                print(f"  WARNING: no c_proj for block {block_id}")
                continue

            w_down = mlp.c_proj.weight  # [out_dim, k]

            # Freeze all, unfreeze w_down
            student_newton.requires_grad_(False)
            w_down.requires_grad_(True)

            print(f"  Block {block_id}: w_down shape={tuple(w_down.shape)}, n_params={w_down.numel()}")

            # Compute gradient
            g = _compute_gradient(
                student_newton, w_down, holdout_tokens, batch_size_used, device
            )
            grad_norm = float(g.norm().item())
            print(f"    grad_norm = {grad_norm:.4e}")

            if grad_norm < 1e-10:
                print(f"    Gradient is zero -- skipping Newton step for block {block_id}")
                per_block_ppl_newton.append(
                    _ppl_from_model(teacher, student_newton, owt_corpus, batch_size=BATCH_SIZE)
                )
                newton_cg_info.append({
                    "block_id": block_id, "converged": True, "n_iter": 0,
                    "residual": 0.0, "grad_norm": grad_norm,
                })
                continue

            # Define Hv callable for CG (with OOM guard)
            # Use only HV_SEQS sequences for Hessian-vector products (faster CG)
            def make_hv_fn(model, param, tokens, bs, dev, hv_seqs=HV_SEQS):
                def hv_fn(v):
                    nonlocal bs
                    while True:
                        try:
                            return _gnhv(model, param, v, tokens, bs, dev, n_seqs_limit=hv_seqs).reshape(-1)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower() and bs > 1:
                                bs = max(1, bs // 2)
                                print(f"    OOM: halving batch_size to {bs}")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                raise
                return hv_fn

            hv_fn = make_hv_fn(student_newton, w_down, holdout_tokens, batch_size_used, device)

            # Run CG: solve H dw = -g
            b_cg = -g.reshape(-1)

            # Add small Tikhonov damping for numerical stability
            lambda_damp = 1e-4 * grad_norm / w_down.numel()
            def hv_damped(v):
                return hv_fn(v) + lambda_damp * v

            try:
                dw, converged, n_iter, residual = _conjugate_gradient(
                    hv_damped, b_cg, CG_MAX_ITER, CG_TOL
                )
                print(f"    CG: converged={converged}, n_iter={n_iter}, residual={residual:.4e}")
            except RuntimeError as e:
                print(f"    CG failed: {e}. Falling back to simple gradient step.")
                # Fallback: fixed small step size to avoid needing Hv
                dw = b_cg * 1e-3  # conservative step
                converged = False
                n_iter = -1
                residual = float("nan")

            # Check if Newton step would diverge (NaN/Inf in dw)
            if not torch.isfinite(dw).all():
                print(f"    Newton step has NaN/Inf! Using steepest descent fallback.")
                # Fallback to steepest descent
                optimal_lr = (g.reshape(-1) ** 2).sum() / ((g.reshape(-1) * hv_fn(g.reshape(-1))).sum() + 1e-12)
                dw = -g.reshape(-1) * float(optimal_lr.item())
                converged = False
                n_iter = -1
                residual = float("nan")

            # Line-search damping if step is too large
            step_norm = float(dw.norm().item())
            w_norm = float(w_down.data.norm().item())
            if step_norm > 10.0 * w_norm:
                alpha_damp = 10.0 * w_norm / step_norm
                print(f"    Step too large (norm={step_norm:.2e}), damping by alpha={alpha_damp:.3f}")
                dw = dw * alpha_damp

            # Apply Newton step
            with torch.no_grad():
                w_down.data += dw.reshape(w_down.shape).to(w_down.dtype)

            # Re-freeze
            student_newton.requires_grad_(False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Eval PPL after this block's correction
            block_ppl = _ppl_from_model(teacher, student_newton, owt_corpus, batch_size=BATCH_SIZE)
            per_block_ppl_newton.append(block_ppl)
            newton_cg_info.append({
                "block_id": block_id,
                "converged": converged,
                "n_iter": n_iter,
                "residual": residual,
                "grad_norm": grad_norm,
                "dw_norm": float(dw.norm().item()),
            })
            print(f"    After block {block_id} correction: PPL={block_ppl:.3f}")

        newton_ppl = per_block_ppl_newton[-1] if per_block_ppl_newton else float("nan")
        newton_delta = 100.0 * (newton_ppl - baseline_ppl) / baseline_ppl
        print(f"  Newton final PPL: {newton_ppl:.3f} ({newton_delta:+.1f}%)")

        # ---- Second pass check (block 7 gradient re-evaluation) ----
        newton_second_pass_ppl = None
        second_pass_note = ""
        # Re-enable grad for block 7
        mlp7 = _find_compressed_mlp(student_newton, 7)
        if mlp7 is not None and hasattr(mlp7, "c_proj"):
            student_newton.requires_grad_(False)
            mlp7.c_proj.weight.requires_grad_(True)
            orig_cg_info_b7 = next((x for x in newton_cg_info if x["block_id"] == 7), None)
            if orig_cg_info_b7 is not None:
                g2 = _compute_gradient(
                    student_newton, mlp7.c_proj.weight, holdout_tokens, batch_size_used, device
                )
                new_grad_norm = float(g2.norm().item())
                orig_grad_norm = orig_cg_info_b7["grad_norm"]
                ratio = new_grad_norm / (orig_grad_norm + 1e-12)
                print(f"  Second-pass check: block 7 grad_norm ratio = {ratio:.3f}")
                if ratio > 0.1:
                    print(f"  Grad ratio {ratio:.3f} > 0.1 -- running second pass...")
                    # Full second pass
                    for block_id in TARGET_BLOCKS:
                        mlp = _find_compressed_mlp(student_newton, block_id)
                        if mlp is None or not hasattr(mlp, "c_proj"):
                            continue
                        student_newton.requires_grad_(False)
                        w = mlp.c_proj.weight
                        w.requires_grad_(True)
                        g3 = _compute_gradient(student_newton, w, holdout_tokens, batch_size_used, device)
                        gn = float(g3.norm().item())
                        lam2 = 1e-4 * gn / w.numel()
                        hv_fn2 = make_hv_fn(student_newton, w, holdout_tokens, batch_size_used, device)
                        def hv_d2(v):
                            return hv_fn2(v) + lam2 * v
                        dw2, _, _, _ = _conjugate_gradient(hv_d2, -g3.reshape(-1), CG_MAX_ITER, CG_TOL)
                        if torch.isfinite(dw2).all():
                            with torch.no_grad():
                                w.data += dw2.reshape(w.shape).to(w.dtype)
                        student_newton.requires_grad_(False)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    newton_second_pass_ppl = _ppl_from_model(teacher, student_newton, owt_corpus, batch_size=BATCH_SIZE)
                    second_pass_delta = 100.0 * (newton_second_pass_ppl - baseline_ppl) / baseline_ppl
                    second_pass_note = f"Second pass PPL: {newton_second_pass_ppl:.3f} ({second_pass_delta:+.1f}%)"
                    print(f"  {second_pass_note}")
            student_newton.requires_grad_(False)

        # ---- Steepest descent baseline ----
        print("\n  -- Steepest descent baseline --")
        _set_seed(seed)
        pack_all3 = derive_ffn_compression(atlas, budget)
        student_sd = swap_compressed_ffns(teacher, pack_all3, inplace=False)
        student_sd.eval()
        sd_batch_size = batch_size_used

        for block_id in TARGET_BLOCKS:
            mlp = _find_compressed_mlp(student_sd, block_id)
            if mlp is None or not hasattr(mlp, "c_proj"):
                continue
            student_sd.requires_grad_(False)
            w = mlp.c_proj.weight
            w.requires_grad_(True)

            g_sd = _compute_gradient(student_sd, w, holdout_tokens, sd_batch_size, device)
            g_flat = g_sd.reshape(-1)
            g_norm = float(g_flat.norm().item())

            # Optimal LR for steepest descent: lr* = ||g||^2 / (g^T H g)
            hv_fn_sd = make_hv_fn(student_sd, w, holdout_tokens, sd_batch_size, device)
            hg = hv_fn_sd(g_flat)
            ghg = float((g_flat * hg).sum().item())
            gg = float((g_flat * g_flat).sum().item())
            if abs(ghg) > 1e-12:
                lr_opt = gg / ghg
            else:
                lr_opt = 1e-4
            lr_opt = min(lr_opt, 10.0)  # clip to prevent divergence
            print(f"    Block {block_id}: optimal_lr={lr_opt:.4e}, grad_norm={g_norm:.4e}")

            with torch.no_grad():
                w.data -= (lr_opt * g_flat).reshape(w.shape).to(w.dtype)
            student_sd.requires_grad_(False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sd_ppl = _ppl_from_model(teacher, student_sd, owt_corpus, batch_size=BATCH_SIZE)
        sd_delta = 100.0 * (sd_ppl - baseline_ppl) / baseline_ppl
        print(f"  Steepest descent PPL: {sd_ppl:.3f} ({sd_delta:+.1f}%)")

        # ---- Adam 500 baseline ----
        print("\n  -- Adam 500 steps baseline --")
        _set_seed(seed)
        pack_all4 = derive_ffn_compression(atlas, budget)
        student_adam = swap_compressed_ffns(teacher, pack_all4, inplace=False)
        student_adam.eval()

        _adam_polish_all_blocks(
            student_adam, TARGET_BLOCKS, holdout_tokens,
            n_steps=ADAM_STEPS, lr=ADAM_LR, device=device,
            batch_size=BATCH_SIZE,
        )
        adam_ppl = _ppl_from_model(teacher, student_adam, owt_corpus, batch_size=BATCH_SIZE)
        adam_delta = 100.0 * (adam_ppl - baseline_ppl) / baseline_ppl
        print(f"  Adam 500 PPL: {adam_ppl:.3f} ({adam_delta:+.1f}%)")

        # Adam sanity check
        if adam_delta > 10.0:
            print(f"  WARNING: Adam 500 delta {adam_delta:.1f}% > +10%. Implementation may be wrong.")

        seed_results[seed] = {
            "unpolished_ppl": unpolished_ppl,
            "unpolished_delta_pct": unpolished_delta,
            "per_block_ppl_newton": per_block_ppl_newton,
            "newton_cg_info": newton_cg_info,
            "newton_ppl": newton_ppl,
            "newton_delta_pct": newton_delta,
            "newton_second_pass_ppl": newton_second_pass_ppl,
            "steepest_descent_ppl": sd_ppl,
            "steepest_descent_delta_pct": sd_delta,
            "adam_500_ppl": adam_ppl,
            "adam_500_delta_pct": adam_delta,
        }

    # ---- Aggregate ----
    def agg(vals):
        t = torch.tensor([v for v in vals if v is not None and not math.isnan(v)], dtype=torch.float32)
        if len(t) == 0:
            return float("nan"), float("nan")
        return float(t.mean().item()), float(t.std().item()) if len(t) > 1 else 0.0

    conditions = ["unpolished", "steepest_descent", "newton", "adam_500"]
    agg_results = {}
    for cond in conditions:
        ppls = [seed_results[s][f"{cond}_ppl"] for s in SEEDS]
        deltas = [seed_results[s][f"{cond}_delta_pct"] for s in SEEDS]
        ppl_mean, ppl_std = agg(ppls)
        delta_mean, delta_std = agg(deltas)
        agg_results[cond] = {
            "ppl_mean": ppl_mean, "ppl_std": ppl_std,
            "delta_pct_mean": delta_mean, "delta_pct_std": delta_std,
            "per_seed": {s: seed_results[s][f"{cond}_ppl"] for s in SEEDS},
        }

    # ---- Verdict ----
    newton_delta_mean = agg_results["newton"]["delta_pct_mean"]
    adam_delta_mean = agg_results["adam_500"]["delta_pct_mean"]
    cg_converged_count = sum(
        1 for s in SEEDS
        for info in seed_results[s]["newton_cg_info"]
        if info.get("converged", False)
    )
    total_cg_runs = sum(len(seed_results[s]["newton_cg_info"]) for s in SEEDS)

    if newton_delta_mean <= 8.0:
        verdict = "CONFIRMED"
        verdict_reason = f"Newton step achieves {newton_delta_mean:.1f}% PPL degradation, within +8% threshold."
    elif math.isnan(newton_delta_mean) or newton_delta_mean > 20.0:
        verdict = "REFUTED"
        verdict_reason = f"Newton step gives {newton_delta_mean:.1f}% PPL degradation (> +20% or diverged)."
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = f"Newton step gives {newton_delta_mean:.1f}% PPL degradation (between +8% and +20%)."

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  {verdict_reason}")
    print(f"  Newton: {newton_delta_mean:+.1f}% PPL | Adam 500: {adam_delta_mean:+.1f}% PPL")
    print(f"  CG convergence: {cg_converged_count}/{total_cg_runs} blocks converged")
    print(f"{'='*60}")

    # ---- Build final results dict ----
    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "agg": agg_results,
        "per_seed": {str(s): seed_results[s] for s in SEEDS},
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "cg_convergence": f"{cg_converged_count}/{total_cg_runs}",
        "reference_cycle_061_adam500_delta": 4.863,
        "reference_cycle_061_unpolished_delta": 30.196,
    }

    # Save JSON
    json_path = RESULTS_DIR / "cycle_081_newton_correction.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: PPL comparison across conditions
    ax = axes[0]
    cond_labels = ["Unpolished", "Steepest\nDescent", "Newton\n(GN)", "Adam\n500"]
    means = [agg_results[c]["delta_pct_mean"] for c in conditions]
    stds = [agg_results[c]["delta_pct_std"] for c in conditions]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    x_pos = list(range(len(cond_labels)))
    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(x_pos, means, yerr=stds, fmt="none", color="black", capsize=5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(8.0, color="purple", linewidth=1.5, linestyle="--", label="+8% threshold (confirmed)")
    ax.axhline(20.0, color="red", linewidth=1.5, linestyle="--", label="+20% threshold (refuted)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel("PPL degradation (%)")
    ax.set_title("PPL Degradation by Correction Method\n(3 seeds, blocks [7,5,4,1], 6x compression)")
    ax.legend(fontsize=8)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.5, f"{mean:.1f}±{std:.1f}%",
                ha="center", va="bottom", fontsize=8)

    # Right: Per-block PPL evolution for Newton correction
    ax2 = axes[1]
    for seed in SEEDS:
        seq = seed_results[seed]["per_block_ppl_newton"]
        if seq:
            pct_seq = [100.0 * (p - baseline_ppl) / baseline_ppl for p in seq]
            ax2.plot(range(1, len(pct_seq)+1), pct_seq, "o-", label=f"seed {seed}", alpha=0.7)

    unpolished_vals = [100.0 * (seed_results[s]["unpolished_ppl"] - baseline_ppl) / baseline_ppl for s in SEEDS]
    ax2.axhline(sum(unpolished_vals)/len(unpolished_vals), color="red", linestyle="--", alpha=0.6, label="unpolished avg")
    ax2.set_xlabel("Blocks corrected (cumulative)")
    ax2.set_ylabel("PPL degradation (%)")
    ax2.set_title("Per-Block Newton Correction Progress")
    ax2.set_xticks(range(1, len(TARGET_BLOCKS)+1))
    ax2.set_xticklabels([f"Block {b}" for b in TARGET_BLOCKS], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Cycle 081: Gauss-Newton Correction on Stacked Compressed GPT-2 FFN\nVerdict: {verdict} | Newton: {newton_delta_mean:+.1f}% PPL | Adam: {adam_delta_mean:+.1f}% PPL", fontsize=11)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_081_newton_correction.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\nDone.")
