"""
Cycle 110: Channel-Level Bump Anatomy — Compression Relevance (GPT-2 FFN)

Tests whether bump fraction (fraction of tokens with |z_j| < 1.0) correlates
with compression importance across three variants:
  A (full): std(GELU(z_j)) * ||w_out_j|| * mean(|GELU'(z_j)|)
  B (weight-only): ||w_out_j|| — no tautological coupling
  C (activation-weight): std(GELU(z_j)) * ||w_out_j||

Verdict applied to variant B (weight-only):
  CONFIRMED if Spearman(bump_frac, importance_B) > 0.3 in >=8/12 blocks
  REFUTED   if |Spearman| < 0.1 in >=8/12 blocks
  INCONCLUSIVE otherwise

GPT-2 Conv1D: c_proj.weight is (3072, 768), so use .weight.T for (768, 3072).
Bump zone threshold: |z_j| < 1.0 (per Key Design in execution spec).
"""

import json
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_TOKENS = 4096
SEQ_LEN = 128
N_BLOCKS = 12
BUMP_THRESHOLD = 1.0  # |z_j| < 1.0 defines bump zone (per spec)
RESULTS_DIR = "autoresearch/results"
ABORT_PPL_LO, ABORT_PPL_HI = 20.0, 65.0

print(f"Device: {DEVICE}")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_owt_tokens(tokenizer, n_tokens=N_TOKENS, seq_len=SEQ_LEN):
    """Load from local OWT sample file (deterministic, no seed needed)."""
    with open("data/openwebtext_sample.txt", "r") as f:
        text = f.read()
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = enc["input_ids"][0]
    seqs = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(all_ids[i : i + seq_len])
        if len(seqs) * seq_len >= n_tokens:
            break
    print(f"Loaded {len(seqs)} sequences x {seq_len} tokens = {len(seqs)*seq_len} tokens")
    return torch.stack(seqs)  # [N_seqs, seq_len]


# ── Baseline PPL ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_ppl(model, seqs, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for seq in seqs:
        seq = seq.unsqueeze(0).to(device)
        out = model(seq, labels=seq)
        n = seq.shape[1] - 1
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


# ── Pre-activation collection ─────────────────────────────────────────────────
def collect_preactivations(model, seqs, device, n_blocks=N_BLOCKS):
    """Collect c_fc outputs (pre-GELU) for all blocks. Shape: [N_tokens, 3072]."""
    all_z = defaultdict(list)

    def make_hook(b):
        def hook(module, input, output):
            # output of c_fc is z (pre-activation), shape [1, L, 3072]
            all_z[b].append(output.detach().cpu().squeeze(0))
        return hook

    hooks = []
    for b in range(n_blocks):
        h = model.transformer.h[b].mlp.c_fc.register_forward_hook(make_hook(b))
        hooks.append(h)

    with torch.no_grad():
        for seq in seqs:
            model(seq.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    result = {}
    for b in range(n_blocks):
        result[b] = torch.cat(all_z[b], dim=0)  # [N_tokens, 3072]
    return result


# ── GELU helpers ──────────────────────────────────────────────────────────────
def gelu_approx(z):
    """GELU with tanh approximation."""
    return F.gelu(z, approximate='tanh')

def gelu_prime(z):
    """Derivative of tanh-approx GELU."""
    # d/dz GELU_tanh(z) using autograd
    z_ = z.requires_grad_(True)
    g = F.gelu(z_, approximate='tanh')
    g.sum().backward()
    return z_.grad.detach()


# ── Main analysis ─────────────────────────────────────────────────────────────
def analyze_block(block_idx, z, w_out):
    """
    z:     [N_tokens, 3072] pre-activation
    w_out: [3072, 768] weight matrix (c_proj.weight.T)

    Returns dict with per-block metrics.
    """
    N, C = z.shape  # N=4096, C=3072

    # Bump fraction per channel
    bump_frac = (z.abs() < BUMP_THRESHOLD).float().mean(dim=0)  # [3072]

    # GELU activations and derivative
    with torch.no_grad():
        h = gelu_approx(z)  # [N, 3072]

    # GELU derivative: use finite difference (faster than autograd for large tensors)
    eps = 1e-4
    gelu_deriv_mean = ((gelu_approx(z + eps) - gelu_approx(z - eps)) / (2 * eps)).abs().mean(dim=0)  # [3072]

    # Output norms: ||w_out_j|| = norm of row j of w_out (shape [3072, 768])
    w_out_norms = w_out.norm(dim=1)  # [3072]

    # Importance variants
    h_std = h.std(dim=0)  # [3072]

    importance_A = h_std * w_out_norms * (gelu_deriv_mean + 1e-6)
    importance_B = w_out_norms  # weight-only, zero tautology
    importance_C = h_std * w_out_norms

    # Spearman correlations
    bf_np = bump_frac.numpy()
    rho_A, _ = spearmanr(bf_np, importance_A.numpy())
    rho_B, _ = spearmanr(bf_np, importance_B.numpy())
    rho_C, _ = spearmanr(bf_np, importance_C.numpy())

    # Bump-critical channels: top quartile bump_frac AND top quartile importance_B
    q75_bump = np.percentile(bf_np, 75)
    q75_imp_B = np.percentile(importance_B.numpy(), 75)
    bump_crit_mask = (bump_frac.numpy() >= q75_bump) & (importance_B.numpy() >= q75_imp_B)
    n_bump_crit = int(bump_crit_mask.sum())

    # Aggregate contribution of bump-critical channels
    # ||h_j * w_out_j|| summed over bump-critical j / total
    # h: [N, 3072], w_out: [3072, 768]
    # contribution_j = ||h[:, j:j+1] * w_out[j, :]|| summed over tokens → use column-wise
    # Simpler: scale each channel's output by its activation magnitude
    # contribution_j = mean_over_tokens(||h_j * w_out[j, :]||) = mean(|h_j|) * ||w_out_j||
    h_mean_abs = h.abs().mean(dim=0)  # [3072]
    contrib = h_mean_abs * w_out_norms  # [3072]
    agg_contrib_crit = float(contrib[bump_crit_mask].sum() / (contrib.sum() + 1e-12))

    # Per-token output variance for bump-critical channels
    h_var = h.var(dim=0)  # [3072]
    if n_bump_crit > 0:
        median_var_crit = float(h_var[bump_crit_mask].median())
    else:
        median_var_crit = float('nan')

    return {
        "rho_A": float(rho_A),
        "rho_B": float(rho_B),
        "rho_C": float(rho_C),
        "n_bump_crit": n_bump_crit,
        "agg_contrib_crit": agg_contrib_crit,
        "median_var_crit": median_var_crit,
        "bump_frac_mean": float(bump_frac.mean()),
        "bump_frac_std": float(bump_frac.std()),
    }


def main():
    # Load model
    print("Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval().to(DEVICE)

    # Load data
    seqs = load_owt_tokens(tokenizer)

    # Baseline PPL
    print("Computing baseline PPL...")
    ppl = compute_ppl(model, seqs, DEVICE)
    print(f"Baseline PPL: {ppl:.2f}")
    if not (ABORT_PPL_LO <= ppl <= ABORT_PPL_HI):
        raise RuntimeError(f"Baseline PPL {ppl:.2f} outside [{ABORT_PPL_LO}, {ABORT_PPL_HI}]. Aborting.")

    # Collect pre-activations
    print("Collecting pre-activations...")
    z_dict = collect_preactivations(model, seqs, DEVICE)
    print(f"Pre-activations collected. Shape example: {z_dict[0].shape}")

    # Check for NaN
    for b in range(N_BLOCKS):
        if torch.isnan(z_dict[b]).any():
            raise RuntimeError(f"NaN in pre-activations at block {b}. Aborting.")

    # Extract c_proj weights for each block
    # Conv1D in GPT-2: c_proj.weight is (3072, 768), so .T gives (768, 3072)
    # We need w_out as [3072, 768] (each row = output projection for channel j)
    w_out_dict = {}
    for b in range(N_BLOCKS):
        # c_proj.weight shape: (3072, 768) in Conv1D — each column is the output weights for one input channel
        # .weight is already [out_features, in_features] = [768, 3072] after .T? Let's check
        w_raw = model.transformer.h[b].mlp.c_proj.weight  # Conv1D: (nf=768, nx=3072) → weight is (nx, nf) = (3072, 768)
        # So w_raw.shape = (3072, 768). Row j = output weights for channel j. That's what we want.
        w_out_dict[b] = w_raw.detach().cpu()  # [3072, 768]

    print(f"c_proj weight shape: {w_out_dict[0].shape}")

    # Per-block analysis
    print("Analyzing blocks...")
    block_results = {}
    for b in range(N_BLOCKS):
        res = analyze_block(b, z_dict[b], w_out_dict[b])
        block_results[b] = res
        print(f"  Block {b:2d}: rho_B={res['rho_B']:+.3f}  rho_A={res['rho_A']:+.3f}  "
              f"rho_C={res['rho_C']:+.3f}  n_crit={res['n_bump_crit']}  "
              f"bump_mean={res['bump_frac_mean']:.3f}")

    # Verdict
    rho_B_vals = [block_results[b]["rho_B"] for b in range(N_BLOCKS)]
    blocks_above = sum(1 for r in rho_B_vals if r > 0.3)
    blocks_near_zero = sum(1 for r in rho_B_vals if abs(r) < 0.1)

    if blocks_above >= 8:
        verdict = "CONFIRMED"
    elif blocks_near_zero >= 8:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVerdict: {verdict}")
    print(f"  blocks_above_0.3: {blocks_above}/12")
    print(f"  blocks_near_zero: {blocks_near_zero}/12")

    # Save JSON
    output = {
        "cycle": 110,
        "item": 121,
        "verdict": verdict,
        "baseline_ppl": ppl,
        "bump_threshold": BUMP_THRESHOLD,
        "blocks_above_03": blocks_above,
        "blocks_near_zero": blocks_near_zero,
        "blocks": {str(b): block_results[b] for b in range(N_BLOCKS)},
        "summary": {
            "rho_B_mean": float(np.mean(rho_B_vals)),
            "rho_B_std": float(np.std(rho_B_vals)),
            "rho_B_min": float(min(rho_B_vals)),
            "rho_B_max": float(max(rho_B_vals)),
        }
    }

    json_path = f"{RESULTS_DIR}/cycle_110_bump_relevance.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    # Plot: 3-panel bar chart (rho_A, rho_B, rho_C per block)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    blocks = list(range(N_BLOCKS))

    rho_A_vals = [block_results[b]["rho_A"] for b in range(N_BLOCKS)]
    rho_C_vals = [block_results[b]["rho_C"] for b in range(N_BLOCKS)]

    def plot_rho(ax, vals, label, color, threshold=None):
        bars = ax.bar(blocks, vals, color=color, alpha=0.75, label=label)
        ax.axhline(0, color='k', lw=0.8)
        if threshold is not None:
            ax.axhline(threshold, color='red', lw=1.2, ls='--', label=f'threshold={threshold}')
            ax.axhline(-threshold, color='red', lw=1.2, ls='--')
        ax.set_ylabel("Spearman ρ")
        ax.set_title(label)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=8)
        ax.set_xticks(blocks)

    plot_rho(axes[0], rho_A_vals, "A: Full importance (std*||w||*|GELU'|)", "#2196F3", threshold=0.1)
    plot_rho(axes[1], rho_B_vals, "B: Weight-only ||w_out_j|| (verdict metric)", "#FF5722", threshold=0.3)
    plot_rho(axes[2], rho_C_vals, "C: Activation-weight std(GELU(z))*||w||", "#4CAF50", threshold=0.1)

    axes[2].set_xlabel("Block")
    fig.suptitle(f"Spearman(bump_frac, importance) per block — GPT-2 FFN\n"
                 f"Verdict: {verdict}  (B>0.3: {blocks_above}/12, |B|<0.1: {blocks_near_zero}/12)",
                 fontsize=11)
    plt.tight_layout()

    plot_path = f"{RESULTS_DIR}/cycle_110_bump_relevance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
