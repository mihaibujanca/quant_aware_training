"""Cycle 151: Correction subspace analysis on UNPOLISHED model (item 161).

Fixes cycle 142 method failure: corrections on polished model worsened PPL.
This version trains corrections on unpolished topk+lstsq model (~PPL 85).
Decomposes corrections into local-error vs cascade-compensation subspaces.
Seed 42 only.

Verdict:
  CONFIRMED    if single subspace accounts for >=60% of PPL recovery
  REFUTED      if both contribute roughly equally
  INCONCLUSIVE if dominant subspace varies by block
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEED = 42
BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
RANK = 8
HIDDEN_FRACTION = 1 / 6
SEQ_LEN = 128
BATCH_SIZE = 4
CORRECTION_STEPS = 2000
LR = 1e-4
POLISH_STEPS = 500
MAX_TOKENS = 4096

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class LowRankCorrection(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.randn(r, d) * 0.01)

    def forward(self, x):
        return x + (x @ self.V.t()) @ self.U.t()


def build_compressed_model(teacher, atlas, budget, blocks, seed, corpus):
    """Build k=4 compressed model WITHOUT polish (unpolished for correction training)."""
    from aleph.lm_geometric_compression import _layer_target_hidden, _select_hidden_units
    layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in blocks:
            continue
        k = _layer_target_hidden(layer, budget)
        sel = _select_hidden_units(layer, budget, k)
        sw = _standard_weight(layer.handle.c_fc).float()
        sb = _standard_bias(layer.handle.c_fc)
        use_ob = _standard_bias(layer.handle.c_proj) is not None
        stw = sw[sel].contiguous()
        stb = None if sb is None else sb[sel].contiguous()
        h = layer.hidden_sample[:, sel].float()
        o = layer.output_sample.float()
        swo, sbo = _fit_output_projection(h, o, use_bias=use_ob)
        sp = sw.numel(); cp = stw.numel() + swo.numel()
        layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id, activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim, compressed_hidden_dim=int(k),
            c_fc_weight=stw, c_fc_bias=stb, c_proj_weight=swo, c_proj_bias=sbo,
            selected_indices=sel, source_param_count=sp, compressed_param_count=cp,
            notes={}))
    pack = CompressedFfnPack(method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(layers),
        metadata={"sequence_length": atlas.sequence_length})
    model = swap_compressed_ffns(teacher, pack, inplace=False)
    model.eval()
    # NO POLISH — corrections will be trained on the raw compressed model
    return model


def train_corrections(model, blocks, tokens, n_steps, lr, device, seed):
    """Train LoRA corrections, return corrections dict."""
    corrections = nn.ModuleDict({str(b): LowRankCorrection(D_MODEL, RANK) for b in blocks})
    hooks = []
    for bid in blocks:
        corr = corrections[str(bid)]
        block = model.transformer.h[bid]
        def make_hook(c):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (c(out[0]),) + out[1:]
                return c(out)
            return hook
        hooks.append(block.register_forward_hook(make_hook(corr)))

    model.requires_grad_(False)
    corr_params = list(corrections.parameters())
    for p in corr_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(corr_params, lr=lr)
    rng = torch.Generator(); rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        opt.zero_grad()
        raw = model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)

    for h in hooks:
        h.remove()
    model.requires_grad_(False)
    return corrections


@torch.no_grad()
def collect_block_errors(float_model, comp_model, tokens, blocks, device):
    """Collect per-block residual stream error (post-FFN)."""
    def collect_acts(model, bids):
        acts = {b: [] for b in bids}
        hooks = []
        for bid in bids:
            def make_hook(b):
                def hook(mod, inp, out):
                    acts[b].append(out[0].detach().cpu() if isinstance(out, tuple) else out.detach().cpu())
                return hook
            hooks.append(model.transformer.h[bid].register_forward_hook(make_hook(bid)))
        for i in range(0, tokens.shape[0], BATCH_SIZE):
            batch = tokens[i:i+BATCH_SIZE].to(device)
            model(batch[:, :-1])
        for h in hooks:
            h.remove()
        return {b: torch.cat(acts[b]).reshape(-1, D_MODEL) for b in bids}

    float_acts = collect_acts(float_model, blocks)
    comp_acts = collect_acts(comp_model, blocks)
    return {b: float_acts[b] - comp_acts[b] for b in blocks}


def project_onto_subspace(C, basis):
    """Project matrix C onto column space of basis (d, k). Returns projected C."""
    if basis.shape[1] == 0:
        return torch.zeros_like(C)
    Q, _ = torch.linalg.qr(basis)  # orthonormalize
    return Q @ (Q.T @ C)


if __name__ == "__main__":
    _set_seed(SEED)
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=200, probe_holdout_sequences=200,
        eval_val_sequences=200, dataset_name="openwebtext", shuffle_seed=SEED,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS, batch_size=BATCH_SIZE)

    # Build compressed model
    print(f"Building compressed model...", flush=True)
    comp_model = build_compressed_model(teacher, atlas, budget, BLOCKS, SEED, corpus)
    unc = _evaluate_model(teacher, comp_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Uncorrected PPL: {unc['perplexity']:.3f}", flush=True)

    # Train corrections
    print(f"\nTraining corrections ({CORRECTION_STEPS} steps)...", flush=True)
    corrections = train_corrections(comp_model, BLOCKS, corpus.probe_holdout,
                                     CORRECTION_STEPS, LR, device, SEED)

    # Evaluate corrected
    hooks = []
    for bid in BLOCKS:
        corr = corrections[str(bid)]
        block = comp_model.transformer.h[bid]
        def make_hook(c):
            def hook(mod, inp, out):
                return (c(out[0]),) + out[1:] if isinstance(out, tuple) else c(out)
            return hook
        hooks.append(block.register_forward_hook(make_hook(corr)))
    corr_m = _evaluate_model(teacher, comp_model, corpus, batch_size=BATCH_SIZE)
    for h in hooks:
        h.remove()
    print(f"  Corrected PPL: {corr_m['perplexity']:.3f}", flush=True)
    recovery = unc['perplexity'] - corr_m['perplexity']
    print(f"  Recovery: {recovery:.3f} PPL points", flush=True)

    # Collect errors for reference subspaces
    print(f"\nCollecting reference subspaces...", flush=True)
    errors = collect_block_errors(teacher.model, comp_model, corpus.eval_val, BLOCKS, device)

    # Decomposition per block
    results_per_block = {}
    for bid in sorted(BLOCKS):
        print(f"\n  Block {bid}:", flush=True)
        U = corrections[str(bid)].U.data  # (768, 8)
        V = corrections[str(bid)].V.data  # (8, 768)
        C = U @ V  # (768, 768)

        # Error subspace: top-8 PCs of error matrix
        E = errors[bid].T.float()  # (768, n_tokens)
        U_e, _, _ = torch.linalg.svd(E, full_matrices=False)
        error_basis = U_e[:, :8]  # (768, 8)

        # Correction SVD
        U_c, S_c, Vt_c = torch.linalg.svd(C, full_matrices=False)
        corr_basis = U_c[:, :8]  # (768, 8)

        # Project correction onto error subspace
        C_error = project_onto_subspace(C, error_basis)
        C_residual = C - C_error

        # Novel = everything else (since attention subspace was weak in cycle 127, merge with novel)
        C_novel = C_residual

        # Frobenius fractions
        frac_error = float(C_error.norm() / (C.norm() + 1e-12))
        frac_novel = float(C_novel.norm() / (C.norm() + 1e-12))

        print(f"    ||C||_F = {C.norm():.4f}", flush=True)
        print(f"    Error fraction: {frac_error:.4f} ({frac_error*100:.1f}%)", flush=True)
        print(f"    Novel fraction: {frac_novel:.4f} ({frac_novel*100:.1f}%)", flush=True)

        # Ablation: replace correction with component-only
        # Error-only correction
        U_eo, S_eo, Vt_eo = torch.linalg.svd(C_error, full_matrices=False)
        # Novel-only correction
        U_no, S_no, Vt_no = torch.linalg.svd(C_novel, full_matrices=False)

        results_per_block[bid] = {
            "C_fro": float(C.norm()),
            "frac_error": frac_error,
            "frac_novel": frac_novel,
            "singular_values": S_c[:8].tolist(),
        }

    # Ablation PPL measurements
    print(f"\n{'='*50}\nABLATION\n{'='*50}", flush=True)

    for bid in sorted(BLOCKS):
        C = corrections[str(bid)].U.data @ corrections[str(bid)].V.data
        E = errors[bid].T.float()
        U_e, _, _ = torch.linalg.svd(E, full_matrices=False)
        error_basis = U_e[:, :8]
        C_error = project_onto_subspace(C, error_basis)
        C_novel = C - C_error

        for component_name, C_comp in [("error_only", C_error), ("novel_only", C_novel), ("full", C)]:
            # Set this block's correction to component
            U_new, S_new, Vt_new = torch.linalg.svd(C_comp, full_matrices=False)
            # Reconstruct as rank-8
            C_r8 = U_new[:, :RANK] @ torch.diag(S_new[:RANK]) @ Vt_new[:RANK, :]

            # Install hooks with modified correction for this block
            hooks = []
            for b in BLOCKS:
                if b == bid:
                    # Modified correction
                    def make_mod_hook(c_mat):
                        def hook(mod, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            corrected = h + h @ c_mat.T
                            return (corrected,) + out[1:] if isinstance(out, tuple) else corrected
                        return hook
                    hooks.append(comp_model.transformer.h[b].register_forward_hook(make_mod_hook(C_r8)))
                else:
                    # Original correction
                    corr = corrections[str(b)]
                    def make_hook(c):
                        def hook(mod, inp, out):
                            return (c(out[0]),) + out[1:] if isinstance(out, tuple) else c(out)
                        return hook
                    hooks.append(comp_model.transformer.h[b].register_forward_hook(make_hook(corr)))

            m = _evaluate_model(teacher, comp_model, corpus, batch_size=BATCH_SIZE)
            for h in hooks:
                h.remove()

            results_per_block[bid][f"ppl_{component_name}"] = m["perplexity"]
            print(f"  Block {bid} {component_name:12s}: PPL={m['perplexity']:.3f}", flush=True)

    # Verdict
    print(f"\n{'='*50}\nVERDICT\n{'='*50}", flush=True)

    # Check if novel component dominates across blocks
    novel_dominates = 0
    error_dominates = 0
    for bid in sorted(BLOCKS):
        r = results_per_block[bid]
        full_ppl = r["ppl_full"]
        err_ppl = r["ppl_error_only"]
        nov_ppl = r["ppl_novel_only"]
        # Recovery from each component (lower is better)
        err_recovery = unc['perplexity'] - err_ppl
        nov_recovery = unc['perplexity'] - nov_ppl
        full_recovery = unc['perplexity'] - full_ppl
        r["error_recovery_frac"] = err_recovery / (full_recovery + 1e-12)
        r["novel_recovery_frac"] = nov_recovery / (full_recovery + 1e-12)
        print(f"  B{bid}: error_recovery={err_recovery:.2f}/{full_recovery:.2f} ({r['error_recovery_frac']:.1%}), "
              f"novel_recovery={nov_recovery:.2f}/{full_recovery:.2f} ({r['novel_recovery_frac']:.1%})", flush=True)
        if r["novel_recovery_frac"] >= 0.6:
            novel_dominates += 1
        if r["error_recovery_frac"] >= 0.6:
            error_dominates += 1

    if novel_dominates >= 3 or error_dominates >= 3:
        verdict = "CONFIRMED"
        dominant = "novel" if novel_dominates >= 3 else "error"
        detail = f"{dominant} subspace dominates in {max(novel_dominates, error_dominates)}/4 blocks"
    elif novel_dominates + error_dominates == 0:
        verdict = "REFUTED"
        detail = "No subspace dominates (diffuse correction)"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Novel dominates in {novel_dominates}/4, error in {error_dominates}/4"

    print(f"\n  VERDICT: {verdict}\n  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 151, "item": 161, "seed": SEED,
        "float_ppl": bl["perplexity"],
        "uncorrected_ppl": unc["perplexity"],
        "corrected_ppl": corr_m["perplexity"],
        "recovery": recovery,
        "per_block": {str(b): results_per_block[b] for b in sorted(BLOCKS)},
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_151_correction_subspace.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved JSON\nDone.", flush=True)
