"""Cycle 129: Is FFN isotropy architectural or learned? — GPT-2 FFN (item 100).

Test 1 (critical): random-init GPT-2 (3 seeds) vs trained — same isotropy metric.
Test 2 (qualitative): post-hoc GELU→ReLU swap in block 7.

Isotropy metric: lambda_512 / lambda_1 from covariance eigenspectrum of post-GELU
hidden activations.

Decision (Test 1 only):
  REFUTED      if mean random-init ratio within 2x of trained (architectural)
  CONFIRMED    if mean random-init ratio >10x trained (learned)
  INCONCLUSIVE if 2-10x
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

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:
    print("ERROR: transformers not available", flush=True)
    sys.exit(1)

# Config
SEEDS = [42, 123, 7]
ANALYSIS_BLOCKS = [1, 4, 5, 7]
SEQ_LEN = 128
N_SEQS = 128  # 128 seqs * 128 tokens = 16384 tokens
BATCH_SIZE = 8
D_FFN = 3072
RANK_IDX = 512  # lambda_512 / lambda_1

# Cycle 086 trained baselines
TRAINED_BASELINES = {1: 0.0043, 4: 0.0089, 5: 0.0102, 7: 0.0151}

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def collect_post_gelu_activations(model: nn.Module, tokens: torch.Tensor,
                                   block_ids: list[int], device: torch.device) -> dict:
    """Collect post-GELU hidden activations at specified blocks."""
    activations = {bid: [] for bid in block_ids}
    hooks = []

    for bid in block_ids:
        block = model.transformer.h[bid]
        # Hook on c_fc output (pre-GELU) then apply GELU manually?
        # Actually, we want post-GELU = the hidden activation after GELU.
        # In GPT-2, mlp.act is the GELU activation applied after c_fc.
        # We can hook mlp.c_proj input, which is the post-GELU activation.
        c_proj = block.mlp.c_proj

        def make_hook(block_id):
            def hook(module, input, output):
                # input[0] is the post-GELU hidden activation
                h = input[0]
                activations[block_id].append(h.detach().cpu())
            return hook

        hooks.append(c_proj.register_forward_hook(make_hook(bid)))

    n_seqs = tokens.shape[0]
    for i in range(0, n_seqs, BATCH_SIZE):
        batch = tokens[i:i+BATCH_SIZE].to(device)
        inputs = batch[:, :-1].contiguous()
        model(inputs)

    for h in hooks:
        h.remove()

    result = {}
    for bid in block_ids:
        stacked = torch.cat(activations[bid], dim=0).reshape(-1, D_FFN)
        result[bid] = stacked
    return result


def compute_isotropy_ratio(activations: torch.Tensor, rank_idx: int = RANK_IDX) -> dict:
    """Compute isotropy ratio = lambda_rank_idx / lambda_1 from covariance eigenspectrum."""
    # Center
    h = activations - activations.mean(dim=0, keepdim=True)
    # Covariance
    n = h.shape[0]
    cov = (h.T @ h) / (n - 1)
    # Eigenvalues (sorted descending)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.flip(0)  # descending
    lambda_1 = float(eigvals[0])
    lambda_k = float(eigvals[min(rank_idx - 1, len(eigvals) - 1)])
    ratio = lambda_k / lambda_1 if lambda_1 > 1e-12 else 0.0
    eff_rank = float((eigvals / eigvals.sum()).exp().sum()) if eigvals.sum() > 0 else 0.0  # Shannon effective rank
    return {
        "lambda_1": lambda_1,
        "lambda_k": lambda_k,
        "ratio": ratio,
        "effective_rank": eff_rank,
        "n_tokens": activations.shape[0],
    }


if __name__ == "__main__":
    print("Loading trained GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    trained_model = teacher.model
    trained_model.eval()
    device = teacher.device

    # Build corpus (seed 42 for consistency)
    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=N_SEQS,
        probe_holdout_sequences=50,
        eval_val_sequences=50,
        dataset_name="openwebtext",
        shuffle_seed=42,
    ))
    tokens = corpus.probe_train

    # Test 1a: Trained model isotropy
    print(f"\n{'='*50}", flush=True)
    print("TRAINED MODEL ISOTROPY", flush=True)
    print(f"{'='*50}", flush=True)
    trained_acts = collect_post_gelu_activations(trained_model, tokens, ANALYSIS_BLOCKS, device)
    trained_results = {}
    for bid in ANALYSIS_BLOCKS:
        r = compute_isotropy_ratio(trained_acts[bid])
        trained_results[bid] = r
        print(f"  Block {bid}: ratio={r['ratio']:.6f} (lambda_1={r['lambda_1']:.2f}, "
              f"lambda_{RANK_IDX}={r['lambda_k']:.4f}, eff_rank={r['effective_rank']:.1f})", flush=True)

    # Test 1b: Random-init GPT-2 (3 seeds)
    print(f"\n{'='*50}", flush=True)
    print("RANDOM-INIT GPT-2 (3 seeds)", flush=True)
    print(f"{'='*50}", flush=True)

    config = GPT2Config()  # Default GPT-2 config (same architecture)
    random_results_per_seed = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        _set_seed(seed)
        random_model = GPT2LMHeadModel(config)
        random_model.eval()
        random_model.to(device)

        random_acts = collect_post_gelu_activations(random_model, tokens, ANALYSIS_BLOCKS, device)
        seed_results = {}
        for bid in ANALYSIS_BLOCKS:
            r = compute_isotropy_ratio(random_acts[bid])
            seed_results[bid] = r
            print(f"    Block {bid}: ratio={r['ratio']:.6f} (eff_rank={r['effective_rank']:.1f})", flush=True)
        random_results_per_seed[seed] = seed_results
        del random_model

    # Cross-seed summary
    print(f"\n  Cross-seed summary:", flush=True)
    random_mean = {}
    random_std = {}
    for bid in ANALYSIS_BLOCKS:
        ratios = [random_results_per_seed[s][bid]["ratio"] for s in SEEDS]
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        random_mean[bid] = mean_r
        random_std[bid] = std_r
        print(f"    Block {bid}: {mean_r:.6f} ± {std_r:.6f}", flush=True)

    # Test 2: GELU→ReLU swap (qualitative only)
    print(f"\n{'='*50}", flush=True)
    print("GELU→ReLU SWAP (qualitative only, block 7)", flush=True)
    print(f"{'='*50}", flush=True)

    relu_model = GPT2LMHeadModel.from_pretrained("gpt2")
    relu_model.eval()
    relu_model.to(device)

    # Replace GELU with ReLU in block 7
    original_act = relu_model.transformer.h[7].mlp.act
    relu_model.transformer.h[7].mlp.act = nn.ReLU()

    relu_acts = collect_post_gelu_activations(relu_model, tokens, [7], device)
    relu_result = compute_isotropy_ratio(relu_acts[7])
    print(f"  Block 7 (ReLU swap): ratio={relu_result['ratio']:.6f} "
          f"(eff_rank={relu_result['effective_rank']:.1f})", flush=True)
    print(f"  Block 7 (trained GELU): ratio={trained_results[7]['ratio']:.6f}", flush=True)
    print(f"  NOTE: ReLU hard zeros mechanically reduce ~50% of activation variance. "
          "Direct comparison unreliable.", flush=True)
    del relu_model

    # Decision
    print(f"\n{'='*50}", flush=True)
    print("DECISION", flush=True)
    print(f"{'='*50}", flush=True)

    ratios_comparison = {}
    for bid in ANALYSIS_BLOCKS:
        t = trained_results[bid]["ratio"]
        r = random_mean[bid]
        factor = r / t if t > 1e-12 else float('inf')
        ratios_comparison[bid] = {
            "trained": t,
            "random_mean": float(r),
            "random_std": float(random_std[bid]),
            "factor": factor,
        }
        print(f"  Block {bid}: trained={t:.6f}, random_mean={r:.6f} ± {random_std[bid]:.6f}, "
              f"factor={factor:.2f}x", flush=True)

    # Average factor across blocks
    factors = [ratios_comparison[bid]["factor"] for bid in ANALYSIS_BLOCKS]
    mean_factor = np.mean(factors)
    print(f"\n  Mean factor (random/trained): {mean_factor:.2f}x", flush=True)

    if mean_factor <= 2.0:
        verdict = "REFUTED"
        detail = f"Random-init isotropy within 2x of trained (mean factor={mean_factor:.2f}x). Isotropy is architectural."
    elif mean_factor >= 10.0:
        verdict = "CONFIRMED"
        detail = f"Random-init isotropy >10x different from trained (mean factor={mean_factor:.2f}x). Isotropy is learned."
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Random-init isotropy 2-10x different from trained (mean factor={mean_factor:.2f}x)."

    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save JSON
    output = {
        "cycle": 129,
        "item": 100,
        "seeds": SEEDS,
        "rank_idx": RANK_IDX,
        "trained_isotropy": {str(bid): trained_results[bid] for bid in ANALYSIS_BLOCKS},
        "random_init_per_seed": {str(s): {str(bid): random_results_per_seed[s][bid] for bid in ANALYSIS_BLOCKS} for s in SEEDS},
        "random_init_mean": {str(bid): {"ratio": float(random_mean[bid]), "std": float(random_std[bid])} for bid in ANALYSIS_BLOCKS},
        "relu_swap_block7": relu_result,
        "factor_comparison": {str(bid): ratios_comparison[bid] for bid in ANALYSIS_BLOCKS},
        "mean_factor": mean_factor,
        "verdict": verdict,
        "verdict_detail": detail,
    }
    json_path = RESULTS_DIR / "cycle_129_isotropy_architectural.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ANALYSIS_BLOCKS))
    w = 0.25
    trained_vals = [trained_results[bid]["ratio"] for bid in ANALYSIS_BLOCKS]
    random_vals = [float(random_mean[bid]) for bid in ANALYSIS_BLOCKS]
    random_errs = [float(random_std[bid]) for bid in ANALYSIS_BLOCKS]

    ax.bar(x - w/2, trained_vals, w, label="Trained GPT-2", color="#3498db")
    ax.bar(x + w/2, random_vals, w, yerr=random_errs, label="Random-init (3 seeds)", color="#e74c3c", alpha=0.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {bid}" for bid in ANALYSIS_BLOCKS])
    ax.set_ylabel(f"Isotropy ratio (λ_{RANK_IDX}/λ_1)")
    ax.set_title(f"FFN Isotropy: Architectural vs Learned — Verdict: {verdict}")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    for i, (tv, rv) in enumerate(zip(trained_vals, random_vals)):
        ax.text(i - w/2, tv + 0.001, f"{tv:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, rv + 0.001, f"{rv:.4f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_129_isotropy_comparison.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved plot: {plot_path}", flush=True)

    print("\nDone.", flush=True)
