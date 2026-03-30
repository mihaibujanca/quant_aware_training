"""Cycle 134: Data-conditioned local affine map extraction — GPT-2 FFN (item 135).

Compute per-token Jacobians of FFN blocks {4, 7} on 128 OWT tokens
(from >=8 sequences). Measure pairwise cosine similarity.

Verdict:
  CONFIRMED    if <=50 groups at tau=0.95 cover >=90% tokens (+ compression test, deferred)
  REFUTED      if >200 groups needed
  INCONCLUSIVE otherwise (strongest possible outcome without compression test)
"""
from __future__ import annotations

import json
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
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
)

SEED = 42
BLOCKS = [4, 7]
N_TOKENS = 128
D_MODEL = 768
D_FFN = 3072
SEQ_LEN = 128
BATCH_SIZE = 8
THRESHOLDS = [0.90, 0.95, 0.99]

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def collect_ffn_inputs(model, tokens, block_ids, device):
    """Collect FFN input activations (post-LayerNorm) for target blocks."""
    acts = {bid: [] for bid in block_ids}
    hooks = []

    for bid in block_ids:
        block = model.transformer.h[bid]
        # Hook on mlp forward pre-hook to get input (which is ln_2 output)
        def make_hook(block_id):
            def hook(mod, inp):
                acts[block_id].append(inp[0].detach().cpu())
            return hook
        hooks.append(block.mlp.register_forward_pre_hook(make_hook(bid)))

    for i in range(0, tokens.shape[0], BATCH_SIZE):
        batch = tokens[i:i+BATCH_SIZE].to(device)
        model(batch[:, :-1])

    for h in hooks:
        h.remove()

    result = {}
    for bid in block_ids:
        stacked = torch.cat(acts[bid], dim=0).reshape(-1, D_MODEL)
        result[bid] = stacked
    return result


def compute_jacobians(model, block_id, inputs, device, max_tokens=128):
    """Compute per-token Jacobians of FFN block."""
    block = model.transformer.h[block_id]

    def ffn_fn(x):
        """FFN: ln_2 → c_fc → GELU → c_proj."""
        h = block.ln_2(x.unsqueeze(0))  # (1, 768)
        h = block.mlp.c_fc(h)
        h = block.mlp.act(h)
        h = block.mlp.c_proj(h)
        return h.squeeze(0)  # (768,)

    jacobians = []
    n = min(max_tokens, inputs.shape[0])
    for i in range(n):
        x = inputs[i].to(device).requires_grad_(False)
        J = torch.func.jacrev(ffn_fn)(x)  # (768, 768)
        jacobians.append(J.detach().cpu())
        if (i + 1) % 32 == 0:
            print(f"      Jacobian {i+1}/{n}", flush=True)
    return torch.stack(jacobians)  # (n, 768, 768)


def cosine_similarity_matrix(jacobians):
    """Pairwise cosine similarity of flattened Jacobians."""
    n = jacobians.shape[0]
    flat = jacobians.reshape(n, -1).float()  # (n, 768*768)
    norms = flat.norm(dim=1, keepdim=True) + 1e-12
    flat_normed = flat / norms
    S = flat_normed @ flat_normed.T  # (n, n)
    return S


def greedy_cluster(S, tau, n_runs=3):
    """Greedy threshold clustering. Run n_runs with different orderings, report mean."""
    n = S.shape[0]
    group_counts = []
    for run in range(n_runs):
        if run == 0:
            order = list(range(n))  # natural order
        else:
            order = list(range(n))
            random.shuffle(order)

        groups = []
        assigned = [False] * n
        for idx in order:
            if assigned[idx]:
                continue
            # Start new group
            group = [idx]
            assigned[idx] = True
            for other in range(n):
                if assigned[other]:
                    continue
                if S[idx, other] > tau:
                    group.append(other)
                    assigned[other] = True
            groups.append(group)
        group_counts.append(len(groups))

    return {
        "n_groups_mean": float(np.mean(group_counts)),
        "n_groups_std": float(np.std(group_counts)),
        "n_groups_per_run": group_counts,
    }


def spectral_analysis(S):
    """Eigendecompose similarity matrix, find effective number of clusters."""
    eigvals = torch.linalg.eigvalsh(S.float())
    eigvals = eigvals.flip(0)  # descending
    n = S.shape[0]
    # Effective rank: number of eigenvalues above 1/N
    threshold = 1.0 / n
    n_above = int((eigvals > threshold).sum())
    # Spectral gap: largest ratio of consecutive eigenvalues
    ratios = eigvals[:-1] / (eigvals[1:] + 1e-12)
    max_gap_idx = int(ratios.argmax()) + 1  # +1 because gap is between idx and idx+1
    return {
        "n_eigenvalues_above_threshold": n_above,
        "top_5_eigenvalues": eigvals[:5].tolist(),
        "spectral_gap_at": max_gap_idx,
        "spectral_gap_ratio": float(ratios.max()),
    }


@torch.no_grad()
def check_interchangeability(model, block_id, inputs, jacobians, S, tau=0.95):
    """For high-similarity pairs, check functional interchangeability."""
    block = model.transformer.h[block_id]
    n = inputs.shape[0]

    # Compute FFN outputs and biases for each token
    ffn_outputs = []
    for i in range(n):
        x = inputs[i].unsqueeze(0)
        h = block.ln_2(x)
        h = block.mlp.c_fc(h)
        h = block.mlp.act(h)
        h = block.mlp.c_proj(h)
        ffn_outputs.append(h.squeeze(0).cpu())
    ffn_outputs = torch.stack(ffn_outputs)  # (n, 768)

    # For each high-similarity pair (i, j), check J_i @ x_j + b_i ≈ FFN(x_j)
    errors = []
    n_pairs = 0
    for i in range(n):
        b_i = ffn_outputs[i] - jacobians[i] @ inputs[i]  # bias term
        for j in range(n):
            if i == j or S[i, j] <= tau:
                continue
            y_approx = jacobians[i] @ inputs[j] + b_i
            y_true = ffn_outputs[j]
            rel_err = float((y_approx - y_true).norm() / (y_true.norm() + 1e-12))
            errors.append(rel_err)
            n_pairs += 1

    if not errors:
        return {"n_pairs": 0, "mean_error": None, "median_error": None, "max_error": None}

    errors = np.array(errors)
    return {
        "n_pairs": n_pairs,
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "max_error": float(errors.max()),
        "pct_below_5pct": float((errors < 0.05).mean() * 100),
    }


if __name__ == "__main__":
    _set_seed(SEED)
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    model = teacher.model
    model.eval()
    device = teacher.device

    # Build corpus — sample from multiple sequences
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=16,  # 16 sequences × ~128 tokens
        probe_holdout_sequences=50,
        eval_val_sequences=50,
        dataset_name="openwebtext",
        shuffle_seed=SEED,
    ))
    tokens = corpus.probe_train  # (16, SEQ_LEN)

    # Collect FFN inputs
    print(f"\nCollecting FFN inputs...", flush=True)
    ffn_inputs = collect_ffn_inputs(model, tokens, BLOCKS, device)

    results = {}
    for bid in BLOCKS:
        print(f"\n{'='*50}", flush=True)
        print(f"BLOCK {bid}", flush=True)
        print(f"{'='*50}", flush=True)

        # Sample 128 tokens from available (spread across sequences)
        all_inputs = ffn_inputs[bid]
        n_available = all_inputs.shape[0]
        indices = torch.randperm(n_available)[:N_TOKENS]
        selected_inputs = all_inputs[indices]
        print(f"  Selected {N_TOKENS} tokens from {n_available} available", flush=True)

        # Compute Jacobians
        print(f"  Computing Jacobians...", flush=True)
        import time
        t0 = time.time()
        J = compute_jacobians(model, bid, selected_inputs, device, max_tokens=N_TOKENS)
        t1 = time.time()
        print(f"  Jacobians computed in {t1-t0:.1f}s ({(t1-t0)/N_TOKENS:.2f}s per token)", flush=True)

        if t1 - t0 > 1200:  # 20 min abort
            print("  ABORT: Jacobian computation too slow", flush=True)
            results[bid] = {"status": "ABORTED", "time": t1-t0}
            continue

        # Sanity: check diagonal non-zero
        diag_norms = torch.diagonal(J, dim1=1, dim2=2).norm(dim=1)
        print(f"  Jacobian diag norms: mean={diag_norms.mean():.4f}, min={diag_norms.min():.4f}", flush=True)

        # Cosine similarity
        print(f"  Computing cosine similarity...", flush=True)
        S = cosine_similarity_matrix(J)

        # Sanity: self-similarity = 1
        self_sim = S.diag().mean()
        print(f"  Self-similarity: {self_sim:.6f} (should be 1.0)", flush=True)

        # Statistics
        mask = torch.ones_like(S, dtype=torch.bool)
        mask.fill_diagonal_(False)
        off_diag = S[mask]
        print(f"  Off-diagonal cosine: mean={off_diag.mean():.4f}, "
              f"median={off_diag.median():.4f}, "
              f"min={off_diag.min():.4f}, max={off_diag.max():.4f}", flush=True)

        # Spectral analysis
        spec = spectral_analysis(S)
        print(f"  Spectral: {spec['n_eigenvalues_above_threshold']} eigenvalues above 1/N, "
              f"gap at {spec['spectral_gap_at']}, ratio={spec['spectral_gap_ratio']:.2f}", flush=True)

        # Threshold clustering
        cluster_results = {}
        for tau in THRESHOLDS:
            cr = greedy_cluster(S.numpy(), tau, n_runs=3)
            cluster_results[str(tau)] = cr
            print(f"  tau={tau}: {cr['n_groups_mean']:.1f}±{cr['n_groups_std']:.1f} groups", flush=True)

        # Interchangeability check
        print(f"  Checking interchangeability (tau=0.95)...", flush=True)
        interchange = check_interchangeability(model, bid, selected_inputs, J, S, tau=0.95)
        if interchange["n_pairs"] > 0:
            print(f"  Interchangeability: {interchange['n_pairs']} pairs, "
                  f"mean_err={interchange['mean_error']:.4f}, "
                  f"pct_below_5%={interchange['pct_below_5pct']:.1f}%", flush=True)
        else:
            print(f"  No pairs with cosine > 0.95", flush=True)

        results[bid] = {
            "n_tokens": N_TOKENS,
            "jacobian_time_s": t1 - t0,
            "cosine_stats": {
                "mean": float(off_diag.mean()),
                "median": float(off_diag.median()),
                "min": float(off_diag.min()),
                "max": float(off_diag.max()),
            },
            "spectral": spec,
            "clusters": cluster_results,
            "interchangeability": interchange,
        }

    # Verdict
    print(f"\n{'='*50}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'='*50}", flush=True)

    all_refuted = True
    any_confirmed_potential = False
    for bid in BLOCKS:
        if isinstance(results.get(bid), dict) and "clusters" in results[bid]:
            n_groups_95 = results[bid]["clusters"]["0.95"]["n_groups_mean"]
            if n_groups_95 <= 50:
                any_confirmed_potential = True
            if n_groups_95 <= 200:
                all_refuted = False
            print(f"  Block {bid}: {n_groups_95:.0f} groups at tau=0.95", flush=True)

    if any_confirmed_potential:
        verdict = "INCONCLUSIVE"
        detail = "<=50 groups found (CONFIRMED-pending), but centroid compression test deferred"
    elif all_refuted:
        verdict = "REFUTED"
        detail = ">200 groups in both blocks at tau=0.95"
    else:
        verdict = "INCONCLUSIVE"
        detail = "50-200 groups, moderate clustering"

    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save
    output = {
        "cycle": 134, "item": 135, "n_tokens": N_TOKENS, "blocks": BLOCKS,
        "per_block": {str(b): results[b] for b in BLOCKS if b in results},
        "verdict": verdict, "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_134_affine_maps.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)

    # Plot: similarity heatmap for each block
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(12, 5))
    for i, bid in enumerate(BLOCKS):
        ax = axes[i] if len(BLOCKS) > 1 else axes
        if isinstance(results.get(bid), dict) and "cosine_stats" in results[bid]:
            # Re-compute S for plot (or save it)
            ax.set_title(f"Block {bid}\nmean cos={results[bid]['cosine_stats']['mean']:.3f}")
            ax.text(0.5, 0.5, f"Groups@0.95:\n{results[bid]['clusters']['0.95']['n_groups_mean']:.0f}",
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_xlabel("Token")
        ax.set_ylabel("Token")
    fig.suptitle(f"Affine Map Analysis — Verdict: {verdict}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cycle_134_similarity_heatmap.png", dpi=120)
    plt.close()
    print("Saved plot", flush=True)
    print("\nDone.", flush=True)
