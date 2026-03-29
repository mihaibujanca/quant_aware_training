"""Cycle 099: Inter-block residual-stream projection correction — GPT-2 FFN (item 112).

After topk+lstsq compression of blocks [7,5,4,1] at 6x, collect per-block
residual-stream error vectors on OWT probe data, compute SVD to find error
subspace, build orthogonal projection P_i = I - U_r @ U_r.T, insert after
each compressed block's FFN output. Test r in {4,8,16,32}. Refit cascade rate c.

Hypothesis:
  CONFIRMED    if c_best <= 0.25 at r <= 16, enabling k=8 at <=+30% PPL.
  REFUTED      if c_best >= 0.35 for all r tested.
  INCONCLUSIVE otherwise (c drops modestly, 0.25-0.35).
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _iter_lm_batches,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_BLOCKS_K4 = [7, 5, 4, 1]   # k=4 blocks
TARGET_BLOCKS_K8 = [7, 5, 4, 3, 2, 1, 0, 6]  # k=8 blocks
HIDDEN_FRACTION = 1 / 6  # 6× compression
SEEDS = [42, 123, 7]
SEQ_LEN = 128
PROBE_SEQS = 200   # for error collection
EVAL_SEQS = 200
BATCH_SIZE = 4

PROJECTION_RANKS = [4, 8, 16, 32]

ADAM_STEPS = 500
ADAM_LR = 1e-4

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Residual-stream error collection
# ---------------------------------------------------------------------------

def _collect_residual_errors(
    teacher_model: nn.Module,
    student_model: nn.Module,
    block_ids: list[int],
    tokens: torch.Tensor,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Collect E_i = residual_student - residual_float at block boundary.

    GPT-2 pre-norm: residual = x + FFN(LN(x)).
    We hook the output of the transformer block (after the residual add),
    which is the input to the NEXT block (and next LN).
    """
    float_acts: dict[int, list[torch.Tensor]] = {b: [] for b in block_ids}
    comp_acts: dict[int, list[torch.Tensor]] = {b: [] for b in block_ids}

    def make_hook(store: dict[int, list[torch.Tensor]], bid: int):
        def hook(module, inp, out):
            # GPT-2 block output is a tuple; first element is hidden states
            h = out[0] if isinstance(out, tuple) else out
            store[bid].append(h.detach().reshape(-1, h.shape[-1]))  # (T, 768)
        return hook

    def attach_hooks(model, store):
        hooks = []
        for bid in block_ids:
            h = model.transformer.h[bid].register_forward_hook(make_hook(store, bid))
            hooks.append(h)
        return hooks

    n_seqs = tokens.shape[0]
    teacher_model.eval()
    student_model.eval()

    with torch.no_grad():
        for start in range(0, n_seqs, batch_size):
            batch = tokens[start:start + batch_size, :-1].contiguous()
            # Teacher pass
            hooks = attach_hooks(teacher_model, float_acts)
            teacher_model(batch)
            for h in hooks:
                h.remove()
            # Student pass
            hooks = attach_hooks(student_model, comp_acts)
            student_model(batch)
            for h in hooks:
                h.remove()

    errors: dict[int, torch.Tensor] = {}
    for b in block_ids:
        f = torch.cat(float_acts[b], dim=0)   # (N_tokens, 768)
        s = torch.cat(comp_acts[b], dim=0)
        errors[b] = s - f  # E_i
    return errors


# ---------------------------------------------------------------------------
# SVD analysis
# ---------------------------------------------------------------------------

def _energy_rank(S: torch.Tensor, threshold: float) -> int:
    cum = S.pow(2).cumsum(0) / S.pow(2).sum()
    r = int((cum < threshold).sum().item()) + 1
    return min(r, len(S))


def _svd_error_analysis(E: torch.Tensor) -> dict:
    """E: (N, D). Return SVD info + energy ranks."""
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    r50 = _energy_rank(S, 0.50)
    r80 = _energy_rank(S, 0.80)
    r90 = _energy_rank(S, 0.90)
    r95 = _energy_rank(S, 0.95)
    return {
        "U": U,         # (N, min(N,D))
        "S": S,         # singular values
        "Vh": Vh,       # (min(N,D), D) — row vectors = right singular vectors
        "rank_50": r50,
        "rank_80": r80,
        "rank_90": r90,
        "rank_95": r95,
        "singular_values_top32": S[:32].tolist(),
        "total_energy": float(S.pow(2).sum().item()),
    }


# ---------------------------------------------------------------------------
# Projection correction layers
# ---------------------------------------------------------------------------

class ProjectionLayer(nn.Module):
    """P = I - U_r @ U_r.T — projects OUT the top-r error directions.

    U_r: (D, r) orthonormal columns (right singular vectors of E).
    Applied to residual stream vector x: P @ x = x - U_r @ (U_r.T @ x).
    """
    def __init__(self, U_r: torch.Tensor):
        super().__init__()
        # U_r: (D, r) — right singular vectors (columns of Vh.T)
        self.register_buffer("U_r", U_r.float())  # (D, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        coeff = x @ self.U_r           # (..., r)
        return x - coeff @ self.U_r.t()  # (..., D)


class CorrectedGPT2Block(nn.Module):
    """Wraps a GPT-2 block and appends a projection correction after it."""
    def __init__(self, block: nn.Module, proj: ProjectionLayer):
        super().__init__()
        self.block = block
        self.proj = proj

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        # GPT-2 block returns tuple (hidden_state, present, ...)
        if isinstance(out, tuple):
            h = out[0]
            h = self.proj(h)
            return (h,) + out[1:]
        return self.proj(out)


def _insert_projections(
    model: nn.Module,
    block_ids: list[int],
    svd_info: dict[int, dict],
    r: int,
) -> nn.Module:
    """Return a copy of model with ProjectionLayer inserted after each block in block_ids."""
    model = copy.deepcopy(model)
    for bid in block_ids:
        info = svd_info[bid]
        # Right singular vectors: Vh is (min(N,D), D), so Vh[:r].T gives (D, r)
        U_r = info["Vh"][:r].t()  # (D, r)
        proj = ProjectionLayer(U_r)
        original_block = model.transformer.h[bid]
        model.transformer.h[bid] = CorrectedGPT2Block(original_block, proj)
    return model


# ---------------------------------------------------------------------------
# Cascade rate fitting
# ---------------------------------------------------------------------------

def _fit_multiplicative(ks: list[int], ppls: list[float]) -> tuple[float, float, float, float]:
    """Fit PPL(k) = a * (1+c)^k. Returns (a, c, r2, rmse)."""
    ks_t = torch.tensor(ks, dtype=torch.float64)
    ppls_t = torch.tensor(ppls, dtype=torch.float64)
    log_ppls = torch.log(ppls_t)
    X = torch.stack([torch.ones_like(ks_t), ks_t], dim=1)
    coeffs, _, _, _ = torch.linalg.lstsq(X, log_ppls.unsqueeze(1), rcond=None)
    log_a = float(coeffs[0, 0].item())
    log_1pc = float(coeffs[1, 0].item())
    a = math.exp(log_a)
    c = math.exp(log_1pc) - 1.0
    pred = torch.exp(log_a + log_1pc * ks_t)
    ss_res = float(((ppls_t - pred) ** 2).sum().item())
    ss_tot = float(((ppls_t - ppls_t.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = math.sqrt(ss_res / len(ks))
    return a, c, r2, rmse


# ---------------------------------------------------------------------------
# Adam polish (500-step, only W_down of compressed blocks)
# ---------------------------------------------------------------------------

def _adam_polish(
    student: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    teacher: TeacherBundle,
    owt_corpus,
    device: torch.device,
    n_steps: int = ADAM_STEPS,
    lr: float = ADAM_LR,
) -> float:
    """Run Adam polish on W_down params. Returns final PPL."""
    student.requires_grad_(False)
    params = []
    for bid in block_ids:
        block = student.transformer.h[bid]
        # unwrap CorrectedGPT2Block if present
        actual_block = block.block if hasattr(block, "block") else block
        mlp = getattr(actual_block, "mlp", None)
        if mlp is None:
            continue
        w = getattr(mlp, "c_proj", None)
        if w is None:
            continue
        w.weight.requires_grad_(True)
        params.append(w.weight)
        if w.bias is not None:
            w.bias.requires_grad_(True)
            params.append(w.bias)

    if not params:
        print("  [adam] no trainable params found")
        return float("nan")

    opt = torch.optim.Adam(params, lr=lr)
    student.train()
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        opt.zero_grad()
        raw = student(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        opt.step()

    student.eval()
    student.requires_grad_(False)
    m = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
    return m["perplexity"]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu")
    )
    device = teacher.device
    print(f"Teacher loaded on {device}")

    # Build eval corpus (shared)
    owt_corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_SEQS,
            probe_holdout_sequences=PROBE_SEQS,
            eval_val_sequences=EVAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Teacher baseline PPL: {baseline_ppl:.3f}")

    # Abort condition
    if not (25 <= baseline_ppl <= 65):
        raise RuntimeError(f"Baseline PPL {baseline_ppl:.3f} outside [25, 65]")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    all_results: dict = {
        "teacher_ppl": baseline_ppl,
        "seeds": {},
    }

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        _set_seed(seed)

        seed_corpus = build_probe_corpus(
            ProbeCorpusSpec(
                teacher=teacher,
                text_path=str(OWT_PATH),
                sequence_length=SEQ_LEN,
                probe_train_sequences=PROBE_SEQS,
                probe_holdout_sequences=PROBE_SEQS,
                eval_val_sequences=EVAL_SEQS,
                shuffle_seed=seed,
            )
        )

        seed_res: dict = {
            "k4": {},
            "k8": {},
        }

        for config_name, block_ids in [("k4", TARGET_BLOCKS_K4), ("k8", TARGET_BLOCKS_K8)]:
            print(f"\n  [{config_name}] Compressing blocks {block_ids}...")
            _set_seed(seed)

            # Probe + compress
            atlas = probe_ffn_geometry(
                teacher,
                seed_corpus,
                layer_ids=tuple(block_ids),
                max_tokens_per_layer=PROBE_SEQS * SEQ_LEN,
                batch_size=BATCH_SIZE,
            )
            _set_seed(seed)
            pack = derive_ffn_compression(atlas, budget)
            student = swap_compressed_ffns(teacher, pack, inplace=False)
            student.eval()

            # Uncorrected PPL
            m_base = _evaluate_model(teacher, student, seed_corpus, batch_size=BATCH_SIZE)
            ppl_uncorrected = m_base["perplexity"]
            print(f"  [{config_name}] Uncorrected PPL: {ppl_uncorrected:.3f} "
                  f"(+{100*(ppl_uncorrected/baseline_ppl-1):.1f}%)")

            # Collect error vectors on probe_train tokens
            probe_tokens = seed_corpus.probe_train
            print(f"  [{config_name}] Collecting residual errors on {probe_tokens.shape[0]} seqs...")
            error_map = _collect_residual_errors(
                teacher.model, student, block_ids, probe_tokens, batch_size=BATCH_SIZE
            )

            # SVD analysis per block
            svd_info: dict[int, dict] = {}
            svd_serializable: dict = {}
            for bid in block_ids:
                E = error_map[bid]
                print(f"    block {bid}: E shape={tuple(E.shape)}, "
                      f"frobenius={E.norm():.3f}")
                info = _svd_error_analysis(E)
                svd_info[bid] = info
                svd_serializable[str(bid)] = {
                    "rank_50": info["rank_50"],
                    "rank_80": info["rank_80"],
                    "rank_90": info["rank_90"],
                    "rank_95": info["rank_95"],
                    "singular_values_top32": info["singular_values_top32"],
                    "total_energy": info["total_energy"],
                }
                print(f"      rank_50={info['rank_50']}, rank_80={info['rank_80']}, "
                      f"rank_90={info['rank_90']}, rank_95={info['rank_95']}")

            # Projection evaluations for each r
            ppl_by_r: dict[int, float] = {}
            for r in PROJECTION_RANKS:
                print(f"  [{config_name}] r={r}: inserting projections...")
                corrected = _insert_projections(student, block_ids, svd_info, r)
                corrected.eval()
                m = _evaluate_model(teacher, corrected, seed_corpus, batch_size=BATCH_SIZE)
                ppl_r = m["perplexity"]
                delta_pct = 100 * (ppl_r / baseline_ppl - 1)
                print(f"    r={r}: PPL={ppl_r:.3f} (+{delta_pct:.1f}%)")
                ppl_by_r[r] = ppl_r
                # Abort condition: projection worse than uncorrected
                if ppl_r > ppl_uncorrected * 1.5:
                    print(f"    WARNING: r={r} projection makes PPL much worse (signal destruction)")

            seed_res[config_name] = {
                "uncorrected_ppl": ppl_uncorrected,
                "ppl_by_r": {str(k): v for k, v in ppl_by_r.items()},
                "svd_by_block": svd_serializable,
            }

        # Cascade rate refit: compress blocks one-at-a-time for k=4 config,
        # using best r from k4 ppl_by_r
        print(f"\n  [cascade] Fitting cascade rate c...")
        best_r_k4 = min(PROJECTION_RANKS, key=lambda r: seed_res["k4"]["ppl_by_r"].get(str(r), float("inf")))
        print(f"  Best r for k4: {best_r_k4} "
              f"(PPL={seed_res['k4']['ppl_by_r'].get(str(best_r_k4), float('nan')):.3f})")

        # Build accumulation curve: k=0..len(TARGET_BLOCKS_K4) with projections at best_r
        acc_ppls = [baseline_ppl]
        acc_ks = [0]
        for n_blocks in range(1, len(TARGET_BLOCKS_K4) + 1):
            partial_blocks = TARGET_BLOCKS_K4[:n_blocks]
            _set_seed(seed)
            atlas_p = probe_ffn_geometry(
                teacher, seed_corpus,
                layer_ids=tuple(partial_blocks),
                max_tokens_per_layer=PROBE_SEQS * SEQ_LEN,
                batch_size=BATCH_SIZE,
            )
            _set_seed(seed)
            pack_p = derive_ffn_compression(atlas_p, budget)
            student_p = swap_compressed_ffns(teacher, pack_p, inplace=False)
            student_p.eval()

            # Collect errors for these blocks
            err_p = _collect_residual_errors(
                teacher.model, student_p, partial_blocks, probe_tokens, batch_size=BATCH_SIZE
            )
            svd_p = {}
            for bid in partial_blocks:
                svd_p[bid] = _svd_error_analysis(err_p[bid])

            corrected_p = _insert_projections(student_p, partial_blocks, svd_p, best_r_k4)
            corrected_p.eval()
            m_p = _evaluate_model(teacher, corrected_p, seed_corpus, batch_size=BATCH_SIZE)
            ppl_p = m_p["perplexity"]
            print(f"    k={n_blocks} (blocks={partial_blocks}): PPL={ppl_p:.3f}")
            acc_ppls.append(ppl_p)
            acc_ks.append(n_blocks)

        a_fit, c_fit, r2_fit, rmse_fit = _fit_multiplicative(acc_ks, acc_ppls)
        print(f"  Cascade fit: a={a_fit:.3f}, c={c_fit:.4f}, R2={r2_fit:.4f}")

        # Also uncorrected accumulation curve (k=0..4)
        print("  [cascade] Uncorrected accumulation curve...")
        acc_ppls_unc = [baseline_ppl]
        for n_blocks in range(1, len(TARGET_BLOCKS_K4) + 1):
            partial_blocks = TARGET_BLOCKS_K4[:n_blocks]
            _set_seed(seed)
            atlas_u = probe_ffn_geometry(
                teacher, seed_corpus,
                layer_ids=tuple(partial_blocks),
                max_tokens_per_layer=PROBE_SEQS * SEQ_LEN,
                batch_size=BATCH_SIZE,
            )
            _set_seed(seed)
            pack_u = derive_ffn_compression(atlas_u, budget)
            student_u = swap_compressed_ffns(teacher, pack_u, inplace=False)
            student_u.eval()
            m_u = _evaluate_model(teacher, student_u, seed_corpus, batch_size=BATCH_SIZE)
            acc_ppls_unc.append(m_u["perplexity"])
        a_unc, c_unc, r2_unc, _ = _fit_multiplicative(list(range(len(acc_ppls_unc))), acc_ppls_unc)
        print(f"  Uncorrected cascade: c={c_unc:.4f}")

        # Adam polish (on k=4 with best projection)
        print("  [adam] Polishing k=4 corrected model...")
        _set_seed(seed)
        atlas_adam = probe_ffn_geometry(
            teacher, seed_corpus,
            layer_ids=tuple(TARGET_BLOCKS_K4),
            max_tokens_per_layer=PROBE_SEQS * SEQ_LEN,
            batch_size=BATCH_SIZE,
        )
        _set_seed(seed)
        pack_adam = derive_ffn_compression(atlas_adam, budget)
        student_adam = swap_compressed_ffns(teacher, pack_adam, inplace=False)

        # Collect errors, insert projections
        err_adam = _collect_residual_errors(
            teacher.model, student_adam, TARGET_BLOCKS_K4, probe_tokens, batch_size=BATCH_SIZE
        )
        svd_adam = {bid: _svd_error_analysis(err_adam[bid]) for bid in TARGET_BLOCKS_K4}
        corrected_adam = _insert_projections(student_adam, TARGET_BLOCKS_K4, svd_adam, best_r_k4)

        ppl_adam = _adam_polish(
            corrected_adam,
            TARGET_BLOCKS_K4,
            seed_corpus.probe_holdout,
            teacher,
            seed_corpus,
            device,
        )
        print(f"  Adam polish PPL (projection+polish): {ppl_adam:.3f}")

        seed_res["cascade"] = {
            "best_r": best_r_k4,
            "ks": acc_ks,
            "ppls_corrected": acc_ppls,
            "ppls_uncorrected": acc_ppls_unc,
            "c_corrected": c_fit,
            "c_uncorrected": c_unc,
            "a_corrected": a_fit,
            "r2_corrected": r2_fit,
        }
        seed_res["adam_ppl"] = ppl_adam

        all_results["seeds"][str(seed)] = seed_res

    # ---------------------------------------------------------------------------
    # Aggregate and verdict
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")

    # Average c across seeds and r values
    c_by_r: dict[int, list[float]] = {r: [] for r in PROJECTION_RANKS}
    for s in SEEDS:
        s_key = str(s)
        c_corr = all_results["seeds"][s_key]["cascade"]["c_corrected"]
        best_r = all_results["seeds"][s_key]["cascade"]["best_r"]
        c_by_r[best_r].append(c_corr)

    # For verdict, use c from cascade fit (corrected, best_r_k4) per seed
    c_vals = [all_results["seeds"][str(s)]["cascade"]["c_corrected"] for s in SEEDS]
    c_unc_vals = [all_results["seeds"][str(s)]["cascade"]["c_uncorrected"] for s in SEEDS]
    c_mean = float(np.mean(c_vals))
    c_unc_mean = float(np.mean(c_unc_vals))

    # PPL at k=8 with best projection (per seed)
    ppl_k8_vals = []
    for s in SEEDS:
        s_key = str(s)
        best_r = all_results["seeds"][s_key]["cascade"]["best_r"]
        ppl_k8 = all_results["seeds"][s_key]["k8"]["ppl_by_r"].get(str(best_r), float("nan"))
        ppl_k8_vals.append(ppl_k8)
    ppl_k8_mean = float(np.nanmean(ppl_k8_vals))
    delta_k8 = 100 * (ppl_k8_mean / baseline_ppl - 1)

    # Best r across seeds
    best_r_vals = [all_results["seeds"][str(s)]["cascade"]["best_r"] for s in SEEDS]
    best_r_overall = int(round(float(np.mean(best_r_vals))))

    if c_mean <= 0.25 and best_r_overall <= 16 and delta_k8 <= 30.0:
        verdict = "CONFIRMED"
    elif c_mean >= 0.35:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"c_corrected (mean): {c_mean:.4f}")
    print(f"c_uncorrected (mean): {c_unc_mean:.4f}")
    print(f"best_r: {best_r_overall}")
    print(f"PPL k=8 corrected: {ppl_k8_mean:.3f} (+{delta_k8:.1f}%)")
    print(f"VERDICT: {verdict}")

    adam_ppls = [all_results["seeds"][str(s)]["adam_ppl"] for s in SEEDS]
    adam_ppl_mean = float(np.nanmean(adam_ppls))
    print(f"Adam polish (projection+polish) PPL mean: {adam_ppl_mean:.3f}")

    summary = {
        "verdict": verdict,
        "c_corrected_mean": c_mean,
        "c_uncorrected_mean": c_unc_mean,
        "c_corrected_per_seed": {str(s): all_results["seeds"][str(s)]["cascade"]["c_corrected"] for s in SEEDS},
        "c_uncorrected_per_seed": {str(s): all_results["seeds"][str(s)]["cascade"]["c_uncorrected"] for s in SEEDS},
        "best_r_per_seed": {str(s): all_results["seeds"][str(s)]["cascade"]["best_r"] for s in SEEDS},
        "best_r_overall": best_r_overall,
        "ppl_k8_corrected_per_seed": {str(s): ppl_k8_vals[i] for i, s in enumerate(SEEDS)},
        "ppl_k8_corrected_mean": ppl_k8_mean,
        "delta_k8_pct": delta_k8,
        "adam_ppl_mean": adam_ppl_mean,
        "adam_ppl_per_seed": {str(s): all_results["seeds"][str(s)]["adam_ppl"] for s in SEEDS},
    }
    all_results["summary"] = summary

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    json_path = RESULTS_DIR / "cycle_099_residual_projection.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------

    # Plot 1: Singular value spectra per block (seed=42, k4)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    seed0 = str(SEEDS[0])
    if seed0 in all_results["seeds"]:
        svd_data = all_results["seeds"][seed0]["k4"]["svd_by_block"]
        for bid_str, info in svd_data.items():
            sv = info["singular_values_top32"]
            ax1.semilogy(range(1, len(sv) + 1), sv, marker="o", markersize=3,
                         label=f"block {bid_str}")
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value")
    ax1.set_title(f"Residual error singular value spectra (seed={SEEDS[0]}, k=4 blocks)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plot1_path = RESULTS_DIR / "cycle_099_residual_projection_spectra.png"
    fig1.savefig(plot1_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved plot 1: {plot1_path}")

    # Plot 2: PPL vs r for k=4 and k=8 (averaged over seeds)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    r_vals = PROJECTION_RANKS
    for config_name in ["k4", "k8"]:
        ppl_means = []
        for r in r_vals:
            vals = [all_results["seeds"][str(s)][config_name]["ppl_by_r"].get(str(r), float("nan"))
                    for s in SEEDS]
            ppl_means.append(float(np.nanmean(vals)))
        ax2.plot(r_vals, ppl_means, marker="o", label=config_name)

    ax2.axhline(baseline_ppl, color="gray", linestyle="--", label=f"Float baseline ({baseline_ppl:.1f})")
    ax2.set_xlabel("Projection rank r")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("PPL vs projection rank r (k=4 and k=8 blocks, mean over 3 seeds)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plot2_path = RESULTS_DIR / "cycle_099_residual_projection.png"
    fig2.savefig(plot2_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved plot 2: {plot2_path}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
    summary = results.get("summary", {})
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {summary.get('verdict', 'N/A')}")
    print(f"c_corrected={summary.get('c_corrected_mean', float('nan')):.4f}, "
          f"c_uncorrected={summary.get('c_uncorrected_mean', float('nan')):.4f}")
    print(f"best_r={summary.get('best_r_overall')}, "
          f"PPL_k8={summary.get('ppl_k8_corrected_mean', float('nan')):.3f} "
          f"(+{summary.get('delta_k8_pct', float('nan')):.1f}%)")
    print(f"Adam polish PPL={summary.get('adam_ppl_mean', float('nan')):.3f}")
