"""Cycle 080: Retained-spectrum denoising via analytic projection — GPT-2 FFN (item 92).

After topk+lstsq compression of blocks [7,5,4,1] at 6x, compute per-block error
on 4096 probe tokens. Project error onto retained SVD basis of block output
covariance. Re-solve lstsq for W_down targeting corrected outputs.
Compare single-step analytic correction vs 500-step Adam polish on stacked PPL.

Hypothesis:
  CONFIRMED  if analytic correction achieves <= +10% PPL degradation
             (within 2x of Adam's ~+4.9%).
  REFUTED    if analytic correction gives > +25% PPL or diverges.
  INCONCLUSIVE otherwise (helps, +10-25%, but doesn't match Adam closely).
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
SEQ_LEN = 128
PROBE_TRAIN_SEQS = 32    # 32x128 = 4096 probe tokens
PROBE_HOLDOUT_SEQS = 32  # 32x128 = 4096 holdout (for Adam polish)
EVAL_SEQS = 32           # 32x128 = 4096 eval tokens
BATCH_SIZE = 4
N_CORRECTION_PASSES = 3

ADAM_STEPS = 500
ADAM_LR = 1e-4

DATA_PATH = ROOT / "data" / "shakespeare.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _std_weight(module: nn.Module) -> torch.Tensor:
    """Return (out_features, in_features) weight, handling Conv1D."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _find_compressed_mlp(model: nn.Module, block_id: int):
    """Return the CompressedMLP module for a given block_id, or None."""
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _collect_mlp_outputs_and_hidden(
    float_model: nn.Module,
    comp_model: nn.Module,
    block_idx: int,
    token_ids: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect float MLP output, compressed MLP output, and compressed hidden
    activations (input to c_proj) for a given block.

    Returns:
        y_float: (N*T, 768) float MLP outputs
        y_comp:  (N*T, 768) compressed MLP outputs
        h_comp:  (N*T, k_b) compressed hidden activations (pre-c_proj)
    """
    float_outs = []
    comp_outs = []
    comp_hidden = []

    def _hook_float(module, inp, out):
        float_outs.append(out.detach().float().reshape(-1, out.shape[-1]))

    def _hook_comp_out(module, inp, out):
        comp_outs.append(out.detach().float().reshape(-1, out.shape[-1]))

    def _hook_comp_hidden(module, inp, out):
        # inp[0] is the input to c_proj — the hidden activations
        comp_hidden.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))

    hf = float_model.transformer.h[block_idx].mlp.register_forward_hook(_hook_float)

    comp_mlp = comp_model.transformer.h[block_idx].mlp
    hc_out = comp_mlp.register_forward_hook(_hook_comp_out)
    hc_hidden = comp_mlp.c_proj.register_forward_hook(_hook_comp_hidden)

    try:
        N = token_ids.shape[0]
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = token_ids[i:i + batch_size]
                float_model(batch, use_cache=False)
                comp_model(batch, use_cache=False)
    finally:
        hf.remove()
        hc_out.remove()
        hc_hidden.remove()

    y_float = torch.cat(float_outs, dim=0)
    y_comp = torch.cat(comp_outs, dim=0)
    h_comp = torch.cat(comp_hidden, dim=0)
    return y_float, y_comp, h_comp


def _output_covariance_svd(y_float: torch.Tensor, energy_thresh: float = 0.99):
    """
    SVD of float output covariance.
    Returns V_retained (768, r) where r captures energy_thresh variance.
    """
    Y_c = y_float - y_float.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Y_c, full_matrices=False)  # Vh: (min(N,768), 768)
    S2 = S ** 2
    cumS2 = torch.cumsum(S2, dim=0)
    total_S2 = S2.sum()
    rank = int((cumS2 / total_S2 < energy_thresh).sum().item()) + 1
    rank = min(rank, Vh.shape[0])
    V_retained = Vh[:rank, :].T  # (768, rank)
    return V_retained, rank, S


def _resolv_w_down(
    h_comp: torch.Tensor,   # (N, k) compressed hidden activations
    y_target: torch.Tensor,  # (N, 768) corrected targets
    old_cprojb: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Re-solve lstsq: W_down_new = argmin_W || h_comp @ W.T - y_target ||_F
    i.e. y_target.T = W_down @ h_comp.T  =>  y_target = h_comp @ W_down.T

    Returns new_weight (768, k) in standard (out_features, in_features) format.
    """
    # Subtract bias from targets if bias present
    if old_cprojb is not None:
        y_no_bias = y_target - old_cprojb.unsqueeze(0)
    else:
        y_no_bias = y_target

    # lstsq: solve h_comp @ X = y_no_bias for X of shape (k, 768)
    result = torch.linalg.lstsq(h_comp.double(), y_no_bias.double())
    X = result.solution  # (k, 768)
    W_new = X.T.float()  # (768, k)
    return W_new, old_cprojb


def _apply_new_w_down(comp_model: nn.Module, block_idx: int, W_new: torch.Tensor) -> None:
    """Replace c_proj weight in compressed MLP with W_new."""
    mlp = comp_model.transformer.h[block_idx].mlp
    c_proj = mlp.c_proj
    if c_proj.__class__.__name__ == "Conv1D":
        # Conv1D stores weight transposed: (in_features, out_features)
        c_proj.weight.data.copy_(W_new.T)
    else:
        c_proj.weight.data.copy_(W_new)


def _collect_polish_params(model: nn.Module, block_ids: list[int]) -> list[torch.Tensor]:
    params = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)
    return params


def _adam_polish(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
) -> None:
    """Run n_steps of Adam on W_down (c_proj) for all blocks jointly."""
    if n_steps == 0:
        return

    student_model.requires_grad_(False)
    params = _collect_polish_params(student_model, block_ids)
    if not params:
        return

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx]
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (
            raw.logits if hasattr(raw, "logits") else raw
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"      Adam step {step+1}/{n_steps}, loss={loss.item():.4f}")

    student_model.eval()
    student_model.requires_grad_(False)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int, teacher_bundle: TeacherBundle) -> dict:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            text_path=str(DATA_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_SEQS,
            shuffle_seed=seed,
            teacher=teacher_bundle,
        )
    )
    print(f"  Corpus: probe_train={corpus.probe_train.shape}, "
          f"holdout={corpus.probe_holdout.shape}, eval={corpus.eval_val.shape}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )

    float_model = teacher_bundle.model
    float_model.eval()

    # ------------------------------------------------------------------
    # Step 1: Baseline compression (topk+lstsq)
    # ------------------------------------------------------------------
    print("  Probing FFN geometry...")
    atlas = probe_ffn_geometry(teacher_bundle, corpus, layer_ids=tuple(TARGET_BLOCKS))
    pack = derive_ffn_compression(atlas, budget)
    model_comp = swap_compressed_ffns(teacher_bundle, pack, inplace=False)
    model_comp.eval()

    baseline_metrics = _evaluate_model(teacher_bundle, float_model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]

    unpolished_metrics = _evaluate_model(teacher_bundle, model_comp, corpus, batch_size=BATCH_SIZE)
    unpolished_ppl = unpolished_metrics["perplexity"]
    unpolished_delta = (unpolished_ppl / baseline_ppl - 1.0) * 100.0

    print(f"  Baseline PPL: {baseline_ppl:.3f}")
    print(f"  Unpolished PPL: {unpolished_ppl:.3f} ({unpolished_delta:+.1f}%)")

    # Validation: should be ~+30%
    if unpolished_delta < 10 or unpolished_delta > 80:
        print(f"  WARNING: unpolished delta {unpolished_delta:.1f}% outside expected 10-80%")

    # ------------------------------------------------------------------
    # Steps 3-6: Iterative analytic correction (3 passes)
    # ------------------------------------------------------------------
    probe_tokens = corpus.probe_train  # (32, 128)

    model_analytic = copy.deepcopy(model_comp)
    model_analytic.eval()

    pass_ppls = []
    error_energy_fracs = []

    for pass_idx in range(N_CORRECTION_PASSES):
        print(f"\n  --- Analytic Correction Pass {pass_idx+1}/{N_CORRECTION_PASSES} ---")
        block_info = []

        for block_idx in TARGET_BLOCKS:
            y_float, y_comp, h_comp = _collect_mlp_outputs_and_hidden(
                float_model, model_analytic, block_idx, probe_tokens
            )

            # Error
            e_b = y_comp - y_float  # (N, 768)
            e_norm_total = float(e_b.norm().item())

            # Output covariance SVD on float outputs
            V_retained, rank_b, S_b = _output_covariance_svd(y_float, energy_thresh=0.99)

            # Project error onto retained subspace
            e_retained = e_b @ V_retained @ V_retained.T

            e_norm_retained = float(e_retained.norm().item())
            energy_frac = float((e_retained.norm() ** 2 / (e_b.norm() ** 2 + 1e-12)).item())

            print(f"    Block {block_idx}: rank={rank_b}, "
                  f"error_norm={e_norm_total:.4f}, "
                  f"retained_frac={energy_frac:.3f}")

            # Corrected targets
            y_corrected = y_float - e_retained

            # Get current c_proj bias
            mlp = model_analytic.transformer.h[block_idx].mlp
            b_old = mlp.c_proj.bias.detach().float() if (
                hasattr(mlp, "c_proj") and mlp.c_proj.bias is not None
            ) else None

            # Re-solve lstsq
            W_new, _ = _resolv_w_down(h_comp, y_corrected, b_old)

            # Apply new W_down
            _apply_new_w_down(model_analytic, block_idx, W_new)

            block_info.append({
                "block_idx": block_idx,
                "rank_b": rank_b,
                "error_norm_total": e_norm_total,
                "error_norm_retained": e_norm_retained,
                "energy_frac_retained": energy_frac,
            })
            error_energy_fracs.append(energy_frac)

        # Evaluate after this pass
        pass_metrics = _evaluate_model(teacher_bundle, model_analytic, corpus, batch_size=BATCH_SIZE)
        pass_ppl = pass_metrics["perplexity"]
        pass_delta = (pass_ppl / baseline_ppl - 1.0) * 100.0
        print(f"  Pass {pass_idx+1} PPL: {pass_ppl:.3f} ({pass_delta:+.1f}%)")

        # Sanity check: if pass 1 is WORSE than unpolished, try flipped sign
        if pass_idx == 0 and pass_ppl > unpolished_ppl * 1.1:
            print(f"  WARNING: Pass 1 PPL {pass_ppl:.3f} > unpolished {unpolished_ppl:.3f} * 1.1")
            print(f"  Trying flipped sign (y_corrected = y_float + e_retained)...")

            # Retry with flipped sign on a fresh copy
            model_analytic_flip = copy.deepcopy(model_comp)
            model_analytic_flip.eval()
            for block_idx in TARGET_BLOCKS:
                y_float2, y_comp2, h_comp2 = _collect_mlp_outputs_and_hidden(
                    float_model, model_analytic_flip, block_idx, probe_tokens
                )
                e_b2 = y_comp2 - y_float2
                V_ret2, _, _ = _output_covariance_svd(y_float2, energy_thresh=0.99)
                e_ret2 = e_b2 @ V_ret2 @ V_ret2.T
                y_corr2 = y_float2 + e_ret2  # flipped
                mlp2 = model_analytic_flip.transformer.h[block_idx].mlp
                b2 = mlp2.c_proj.bias.detach().float() if (
                    hasattr(mlp2, "c_proj") and mlp2.c_proj.bias is not None
                ) else None
                W_new2, _ = _resolv_w_down(h_comp2, y_corr2, b2)
                _apply_new_w_down(model_analytic_flip, block_idx, W_new2)

            flip_metrics = _evaluate_model(teacher_bundle, model_analytic_flip, corpus, batch_size=BATCH_SIZE)
            flip_ppl = flip_metrics["perplexity"]
            flip_delta = (flip_ppl / baseline_ppl - 1.0) * 100.0
            print(f"  Flipped sign PPL: {flip_ppl:.3f} ({flip_delta:+.1f}%)")

            if flip_ppl < pass_ppl:
                print(f"  Flipped sign is BETTER — switching to flipped model")
                model_analytic = model_analytic_flip
                pass_ppl = flip_ppl
                pass_delta = flip_delta

        pass_ppls.append({"pass": pass_idx + 1, "ppl": pass_ppl, "delta_pct": pass_delta})

    analytic_1pass_ppl = pass_ppls[0]["ppl"]
    analytic_2pass_ppl = pass_ppls[1]["ppl"]
    analytic_3pass_ppl = pass_ppls[2]["ppl"]

    # ------------------------------------------------------------------
    # Step 8: Adam polish (on separate holdout)
    # ------------------------------------------------------------------
    print("\n  --- Adam Polish (500 steps, holdout data) ---")
    model_adam = copy.deepcopy(model_comp)
    model_adam.eval()

    _set_seed(seed)
    _adam_polish(model_adam, TARGET_BLOCKS, corpus.probe_holdout, ADAM_STEPS, ADAM_LR)

    adam_metrics = _evaluate_model(teacher_bundle, model_adam, corpus, batch_size=BATCH_SIZE)
    adam_ppl = adam_metrics["perplexity"]
    adam_delta = (adam_ppl / baseline_ppl - 1.0) * 100.0
    print(f"  Adam 500-step PPL: {adam_ppl:.3f} ({adam_delta:+.1f}%)")

    return {
        "seed": seed,
        "baseline_ppl": baseline_ppl,
        "unpolished_ppl": unpolished_ppl,
        "unpolished_delta_pct": unpolished_delta,
        "analytic_1pass_ppl": analytic_1pass_ppl,
        "analytic_1pass_delta_pct": (analytic_1pass_ppl / baseline_ppl - 1.0) * 100.0,
        "analytic_2pass_ppl": analytic_2pass_ppl,
        "analytic_2pass_delta_pct": (analytic_2pass_ppl / baseline_ppl - 1.0) * 100.0,
        "analytic_3pass_ppl": analytic_3pass_ppl,
        "analytic_3pass_delta_pct": (analytic_3pass_ppl / baseline_ppl - 1.0) * 100.0,
        "adam_500_ppl": adam_ppl,
        "adam_500_delta_pct": adam_delta,
        "mean_energy_frac_retained": float(np.mean(error_energy_fracs)),
    }


def main():
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device=DEVICE,
        )
    )
    print(f"Teacher loaded: {teacher.model_name}")

    seed_results = []
    for seed in SEEDS:
        result = run_seed(seed, teacher)
        seed_results.append(result)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def _agg(key: str) -> dict:
        vals = [r[key] for r in seed_results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals}

    aggregated = {
        "baseline_ppl": _agg("baseline_ppl"),
        "unpolished_ppl": _agg("unpolished_ppl"),
        "unpolished_delta_pct": _agg("unpolished_delta_pct"),
        "analytic_1pass_ppl": _agg("analytic_1pass_ppl"),
        "analytic_1pass_delta_pct": _agg("analytic_1pass_delta_pct"),
        "analytic_2pass_ppl": _agg("analytic_2pass_ppl"),
        "analytic_2pass_delta_pct": _agg("analytic_2pass_delta_pct"),
        "analytic_3pass_ppl": _agg("analytic_3pass_ppl"),
        "analytic_3pass_delta_pct": _agg("analytic_3pass_delta_pct"),
        "adam_500_ppl": _agg("adam_500_ppl"),
        "adam_500_delta_pct": _agg("adam_500_delta_pct"),
        "mean_energy_frac_retained": _agg("mean_energy_frac_retained"),
    }

    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    print(f"Baseline PPL:        {aggregated['baseline_ppl']['mean']:.3f} ± {aggregated['baseline_ppl']['std']:.3f}")
    print(f"Unpolished PPL:      {aggregated['unpolished_ppl']['mean']:.3f} ± {aggregated['unpolished_ppl']['std']:.3f}  ({aggregated['unpolished_delta_pct']['mean']:+.1f}%)")
    print(f"Analytic 1pass PPL:  {aggregated['analytic_1pass_ppl']['mean']:.3f} ± {aggregated['analytic_1pass_ppl']['std']:.3f}  ({aggregated['analytic_1pass_delta_pct']['mean']:+.1f}%)")
    print(f"Analytic 2pass PPL:  {aggregated['analytic_2pass_ppl']['mean']:.3f} ± {aggregated['analytic_2pass_ppl']['std']:.3f}  ({aggregated['analytic_2pass_delta_pct']['mean']:+.1f}%)")
    print(f"Analytic 3pass PPL:  {aggregated['analytic_3pass_ppl']['mean']:.3f} ± {aggregated['analytic_3pass_ppl']['std']:.3f}  ({aggregated['analytic_3pass_delta_pct']['mean']:+.1f}%)")
    print(f"Adam 500-step PPL:   {aggregated['adam_500_ppl']['mean']:.3f} ± {aggregated['adam_500_ppl']['std']:.3f}  ({aggregated['adam_500_delta_pct']['mean']:+.1f}%)")
    print(f"Mean retained frac:  {aggregated['mean_energy_frac_retained']['mean']:.3f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    best_analytic_delta = min(
        aggregated["analytic_1pass_delta_pct"]["mean"],
        aggregated["analytic_2pass_delta_pct"]["mean"],
        aggregated["analytic_3pass_delta_pct"]["mean"],
    )
    if best_analytic_delta <= 10.0:
        verdict = "CONFIRMED"
        verdict_reason = f"Best analytic delta {best_analytic_delta:+.1f}% <= +10%"
    elif best_analytic_delta > 25.0:
        verdict = "REFUTED"
        verdict_reason = f"Best analytic delta {best_analytic_delta:+.1f}% > +25%"
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = f"Best analytic delta {best_analytic_delta:+.1f}% in (10-25%) range"

    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_reason}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "experiment": "cycle_080_retained_spectrum_denoising",
        "config": {
            "target_blocks": TARGET_BLOCKS,
            "hidden_fraction": HIDDEN_FRACTION,
            "seeds": SEEDS,
            "probe_train_seqs": PROBE_TRAIN_SEQS,
            "probe_holdout_seqs": PROBE_HOLDOUT_SEQS,
            "eval_seqs": EVAL_SEQS,
            "adam_steps": ADAM_STEPS,
            "adam_lr": ADAM_LR,
            "n_correction_passes": N_CORRECTION_PASSES,
        },
        "seed_results": seed_results,
        "aggregated": aggregated,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    json_path = RESULTS_DIR / "cycle_080_analytic_projection.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    methods = ["Unpolished", "Analytic\n1pass", "Analytic\n2pass", "Analytic\n3pass", "Adam\n500-step"]
    keys = ["unpolished_delta_pct", "analytic_1pass_delta_pct", "analytic_2pass_delta_pct",
            "analytic_3pass_delta_pct", "adam_500_delta_pct"]

    means = [aggregated[k]["mean"] for k in keys]
    stds = [aggregated[k]["std"] for k in keys]

    colors = ["#d62728", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#2ca02c"]
    alpha_vals = [1.0, 0.5, 0.75, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=5,
                  color=[c for c in colors],
                  alpha=0.85)
    for bar, alpha in zip(bars, alpha_vals):
        bar.set_alpha(alpha)

    ax.axhline(y=10.0, color="green", linestyle="--", alpha=0.5, label="Confirmed threshold (+10%)")
    ax.axhline(y=25.0, color="red", linestyle="--", alpha=0.5, label="Refuted threshold (+25%)")
    ax.set_ylabel("PPL degradation (%)")
    ax.set_title(f"Cycle 080: Retained-spectrum denoising vs Adam polish\n(GPT-2, blocks {TARGET_BLOCKS}, 6x compression, n={len(SEEDS)} seeds)")
    ax.legend()
    ax.set_ylim(bottom=0)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f"{m:+.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_080_analytic_projection.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    print(f"\nVERDICT: {verdict}")
    return output


if __name__ == "__main__":
    main()
