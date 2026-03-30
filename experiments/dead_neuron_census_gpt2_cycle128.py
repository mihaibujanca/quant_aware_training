"""Cycle 128: Dead/always-active neuron census + exact pruning — GPT-2 FFN (item 134).

For each of GPT-2's 12 FFN blocks, compute per-neuron pre-GELU activation
statistics over 256 OWT sequences (seq_len=256) per seed. Three seeds: [42, 123, 7].

Categories (strict):
  dead:          max(z) < -3 for ALL tokens  (GELU(-3) = -0.004)
  always_active: min(z) > 4 for ALL tokens   (GELU(4) ≈ 3.9998)
  near_linear:   max|GELU(z)-z| < 0.01       (z always in linear GELU regime)

Soft census: per-channel token fractions (frac_dead, frac_active, frac_nearlinear).

Verdict:
  CONFIRMED    if >=5% strict-removable/block AND exact pruning <0.1% PPL change
  REFUTED      if <1% strict-removable AND no channels with >95% soft fraction
  INCONCLUSIVE if strict <1% but some channels >95% soft fraction
"""
from __future__ import annotations

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
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    load_nanochat_teacher,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
N_SEQS = 256
SEQ_LEN = 256
BATCH_SIZE = 8

N_BLOCKS = 12
D_FFN = 3072
D_MODEL = 768

# GELU thresholds
DEAD_THRESHOLD = -3.0       # z < -3: GELU(-3) = -0.004
ACTIVE_THRESHOLD = 4.0      # z > 4: GELU(4) ≈ 3.9998, effectively identity
NEARLINEAR_TOL = 0.01       # |GELU(z) - z| < 0.01

# Soft census thresholds
SOFT_95 = 0.95
SOFT_99 = 0.99

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gelu_approx(z: float) -> float:
    """GELU(z) using PyTorch formula."""
    return float(torch.nn.functional.gelu(torch.tensor(z)))


@torch.no_grad()
def census_one_seed(model: nn.Module, tokens: torch.Tensor, device: torch.device) -> dict:
    """Run census for all 12 blocks on given tokens.
    Returns per-block: min_z, max_z, frac_dead, frac_active, frac_nearlinear per channel.
    """
    n_tokens_total = tokens.shape[0] * (tokens.shape[1] - 1)  # -1 for input vs target split

    results = {}
    for block_id in range(N_BLOCKS):
        fc = model.transformer.h[block_id].mlp.c_fc
        # Collect all pre-GELU activations
        pre_acts = []

        def make_hook():
            def hook(module, input, output):
                pre_acts.append(output.detach().cpu())
            return hook

        h = fc.register_forward_hook(make_hook())

        n_seqs = tokens.shape[0]
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = tokens[i:i+BATCH_SIZE].to(device)
            inputs = batch[:, :-1].contiguous()
            model(inputs)

        h.remove()

        # Stack: (total_tokens, D_FFN)
        z_all = torch.cat(pre_acts, dim=0).reshape(-1, D_FFN)
        n_tokens = z_all.shape[0]

        # Per-channel statistics
        min_z = z_all.min(dim=0).values  # (D_FFN,)
        max_z = z_all.max(dim=0).values
        mean_z = z_all.mean(dim=0)
        std_z = z_all.std(dim=0)

        # Strict census
        strict_dead = (max_z < DEAD_THRESHOLD).sum().item()
        strict_active = (min_z > ACTIVE_THRESHOLD).sum().item()

        # Near-linear: max|GELU(z) - z| < tol for all tokens per channel
        gelu_z = F.gelu(z_all)
        diff = (gelu_z - z_all).abs()
        max_diff = diff.max(dim=0).values
        strict_nearlinear = (max_diff < NEARLINEAR_TOL).sum().item()

        # Soft census: fraction of tokens per channel
        frac_dead = (z_all < DEAD_THRESHOLD).float().mean(dim=0)      # (D_FFN,)
        frac_active = (z_all > ACTIVE_THRESHOLD).float().mean(dim=0)
        frac_nearlinear = (diff < NEARLINEAR_TOL).float().mean(dim=0)

        # Count channels with >95% and >99% soft fractions
        soft95_dead = (frac_dead > SOFT_95).sum().item()
        soft99_dead = (frac_dead > SOFT_99).sum().item()
        soft95_active = (frac_active > SOFT_95).sum().item()
        soft99_active = (frac_active > SOFT_99).sum().item()
        soft95_nearlinear = (frac_nearlinear > SOFT_95).sum().item()
        soft99_nearlinear = (frac_nearlinear > SOFT_99).sum().item()

        results[block_id] = {
            "strict_dead": strict_dead,
            "strict_active": strict_active,
            "strict_nearlinear": strict_nearlinear,
            "strict_removable": strict_dead + strict_active,
            "soft95_dead": soft95_dead,
            "soft99_dead": soft99_dead,
            "soft95_active": soft95_active,
            "soft99_active": soft99_active,
            "soft95_nearlinear": soft95_nearlinear,
            "soft99_nearlinear": soft99_nearlinear,
            "n_tokens": n_tokens,
            "mean_z_stats": {
                "mean": float(mean_z.mean()),
                "std": float(mean_z.std()),
                "min": float(mean_z.min()),
                "max": float(mean_z.max()),
            },
            "frac_dead_hist": _histogram(frac_dead),
            "frac_active_hist": _histogram(frac_active),
            # Store raw fractions for cross-seed analysis
            "_frac_dead": frac_dead.numpy().tolist(),
            "_frac_active": frac_active.numpy().tolist(),
        }

        print(f"  Block {block_id:2d}: dead={strict_dead:4d} active={strict_active:4d} "
              f"nearlinear={strict_nearlinear:4d} | "
              f"soft95_dead={soft95_dead:4d} soft95_active={soft95_active:4d} "
              f"soft95_nearlinear={soft95_nearlinear:4d}", flush=True)

    return results


def _histogram(fracs: torch.Tensor) -> dict:
    """Bin fractions into histogram."""
    bins = {
        "0-1%": int((fracs < 0.01).sum()),
        "1-50%": int(((fracs >= 0.01) & (fracs < 0.50)).sum()),
        "50-95%": int(((fracs >= 0.50) & (fracs < 0.95)).sum()),
        "95-99%": int(((fracs >= 0.95) & (fracs < 0.99)).sum()),
        "99-100%": int((fracs >= 0.99).sum()),
    }
    return bins


# ===========================================================================
if __name__ == "__main__":
    # Sanity checks
    print("GELU sanity checks:", flush=True)
    g_neg3 = gelu_approx(-3.0)
    g_pos4 = gelu_approx(4.0)
    print(f"  GELU(-3) = {g_neg3:.6f} (expect ~-0.004)", flush=True)
    print(f"  GELU(4)  = {g_pos4:.6f} (expect ~3.9999)", flush=True)
    assert abs(g_neg3 - (-0.004)) < 0.002, f"GELU(-3) sanity failed: {g_neg3}"
    assert abs(g_pos4 - 4.0) < 0.001, f"GELU(4) sanity failed: {g_pos4}"

    # Load model
    print("\nLoading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    model = teacher.model
    model.eval()
    device = teacher.device
    print(f"  Device: {device}", flush=True)

    # Census per seed
    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}: Loading {N_SEQS} sequences (seq_len={SEQ_LEN})", flush=True)
        print(f"{'='*60}", flush=True)

        _set_seed(seed)
        corpus = build_probe_corpus(ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=N_SEQS,
            probe_holdout_sequences=50,
            eval_val_sequences=50,
            dataset_name="openwebtext",
            shuffle_seed=seed,
        ))
        tokens = corpus.probe_train  # (N_SEQS, SEQ_LEN)
        print(f"  Tokens shape: {tokens.shape}", flush=True)

        results = census_one_seed(model, tokens, device)
        all_seed_results[seed] = results

    # Cross-seed analysis
    print(f"\n{'='*60}", flush=True)
    print("CROSS-SEED SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {}
    any_soft95 = False
    for block_id in range(N_BLOCKS):
        strict_dead_vals = [all_seed_results[s][block_id]["strict_dead"] for s in SEEDS]
        strict_active_vals = [all_seed_results[s][block_id]["strict_active"] for s in SEEDS]
        strict_removable_vals = [all_seed_results[s][block_id]["strict_removable"] for s in SEEDS]
        soft95_dead_vals = [all_seed_results[s][block_id]["soft95_dead"] for s in SEEDS]
        soft95_active_vals = [all_seed_results[s][block_id]["soft95_active"] for s in SEEDS]

        # Cross-seed std for soft fractions
        frac_dead_arrays = np.array([all_seed_results[s][block_id]["_frac_dead"] for s in SEEDS])  # (3, 3072)
        frac_active_arrays = np.array([all_seed_results[s][block_id]["_frac_active"] for s in SEEDS])
        frac_dead_std = frac_dead_arrays.std(axis=0)
        frac_active_std = frac_active_arrays.std(axis=0)

        block_summary = {
            "strict_dead": {"min": min(strict_dead_vals), "max": max(strict_dead_vals), "mean": np.mean(strict_dead_vals)},
            "strict_active": {"min": min(strict_active_vals), "max": max(strict_active_vals), "mean": np.mean(strict_active_vals)},
            "strict_removable": {"min": min(strict_removable_vals), "max": max(strict_removable_vals), "mean": np.mean(strict_removable_vals)},
            "soft95_dead": {"min": min(soft95_dead_vals), "max": max(soft95_dead_vals), "mean": np.mean(soft95_dead_vals)},
            "soft95_active": {"min": min(soft95_active_vals), "max": max(soft95_active_vals), "mean": np.mean(soft95_active_vals)},
            "frac_dead_std_median": float(np.median(frac_dead_std)),
            "frac_dead_std_p95": float(np.percentile(frac_dead_std, 95)),
            "frac_active_std_median": float(np.median(frac_active_std)),
            "frac_active_std_p95": float(np.percentile(frac_active_std, 95)),
        }
        summary[block_id] = block_summary

        if max(soft95_dead_vals) > 0 or max(soft95_active_vals) > 0:
            any_soft95 = True

        pct_removable = np.mean(strict_removable_vals) / D_FFN * 100
        print(f"  Block {block_id:2d}: strict_removable={np.mean(strict_removable_vals):.0f}±{np.std(strict_removable_vals):.0f} "
              f"({pct_removable:.2f}%) | "
              f"soft95_dead={np.mean(soft95_dead_vals):.0f} "
              f"soft95_active={np.mean(soft95_active_vals):.0f}", flush=True)

    # Verdict
    total_strict = sum(summary[b]["strict_removable"]["mean"] for b in range(N_BLOCKS))
    total_neurons = N_BLOCKS * D_FFN
    pct_strict_total = total_strict / total_neurons * 100
    max_pct_block = max(summary[b]["strict_removable"]["mean"] / D_FFN * 100 for b in range(N_BLOCKS))

    print(f"\n  Total strict-removable: {total_strict:.0f}/{total_neurons} ({pct_strict_total:.2f}%)", flush=True)
    print(f"  Max per-block: {max_pct_block:.2f}%", flush=True)
    print(f"  Any soft95 channels: {any_soft95}", flush=True)

    if pct_strict_total >= 5.0:
        verdict = "CONFIRMED_PENDING_PPL"
        detail = f">=5% strict-removable ({pct_strict_total:.1f}%), need PPL check"
    elif pct_strict_total < 1.0 and not any_soft95:
        verdict = "REFUTED"
        detail = f"<1% strict-removable ({pct_strict_total:.2f}%) AND no channels with >95% soft fraction"
    elif pct_strict_total < 1.0 and any_soft95:
        verdict = "INCONCLUSIVE"
        detail = f"<1% strict-removable but soft census reveals approximate pruning candidates"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"1-5% strict-removable ({pct_strict_total:.2f}%), needs PPL verification"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Clean up _frac arrays (too large for JSON)
    for seed in SEEDS:
        for block_id in range(N_BLOCKS):
            del all_seed_results[seed][block_id]["_frac_dead"]
            del all_seed_results[seed][block_id]["_frac_active"]

    # Save JSON
    output = {
        "cycle": 128,
        "item": 134,
        "seeds": SEEDS,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "dead_threshold": DEAD_THRESHOLD,
        "active_threshold": ACTIVE_THRESHOLD,
        "nearlinear_tol": NEARLINEAR_TOL,
        "total_strict_removable": total_strict,
        "total_neurons": total_neurons,
        "pct_strict_total": pct_strict_total,
        "any_soft95": any_soft95,
        "per_seed": {str(s): {str(b): all_seed_results[s][b] for b in range(N_BLOCKS)} for s in SEEDS},
        "cross_seed_summary": {str(b): summary[b] for b in range(N_BLOCKS)},
        "verdict": verdict,
        "verdict_detail": detail,
    }
    json_path = RESULTS_DIR / "cycle_128_census.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}", flush=True)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 1. Per-block strict counts (mean across seeds)
    ax = axes[0]
    blocks = range(N_BLOCKS)
    dead_counts = [summary[b]["strict_dead"]["mean"] for b in blocks]
    active_counts = [summary[b]["strict_active"]["mean"] for b in blocks]
    x = np.arange(N_BLOCKS)
    w = 0.35
    ax.bar(x - w/2, dead_counts, w, label="Strict dead", color="#e74c3c")
    ax.bar(x + w/2, active_counts, w, label="Strict always-active", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i}" for i in range(N_BLOCKS)])
    ax.set_ylabel("Channel count (mean across 3 seeds)")
    ax.set_title("Strict Dead/Always-Active Census — GPT-2 FFN")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 2. Soft95 counts
    ax2 = axes[1]
    soft95_dead = [summary[b]["soft95_dead"]["mean"] for b in blocks]
    soft95_active = [summary[b]["soft95_active"]["mean"] for b in blocks]
    ax2.bar(x - w/2, soft95_dead, w, label="Soft >95% dead", color="#e74c3c", alpha=0.6)
    ax2.bar(x + w/2, soft95_active, w, label="Soft >95% active", color="#3498db", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"B{i}" for i in range(N_BLOCKS)])
    ax2.set_ylabel("Channel count (mean across 3 seeds)")
    ax2.set_title(f"Soft Census (>95% tokens in zone) — Verdict: {verdict}")
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_128_census.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved plot: {plot_path}", flush=True)

    print("\nDone.", flush=True)
