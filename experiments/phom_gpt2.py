"""
Cycle 074 — Persistent homology of GPT-2 hidden states through depth.
Tests whether Betti complexity predicts compression difficulty.
"""

import json
import os
import sys
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from ripser import ripser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "autoresearch/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 7]
N_TOKENS = 512
N_BLOCKS = 12
PCA_DIM = 30
DEVICE = "cpu"

# Ground truth damage from cycle 048
DAMAGE_8X = [0.17684, 0.002476, 0.024471, 0.01871, 0.018526, 0.022925,
             0.021623, 0.024599, 0.029085, 0.026125, 0.03068, 0.067325]
DAMAGE_16X = [0.33116, 0.007408, 0.011434, 0.009973, 0.006393, 0.001226,
              0.010001, -0.006285, 0.019531, 0.016346, 0.018099, 0.065159]


def load_model_and_tokenizer():
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


def get_token_windows(text, tokenizer, seed, n_tokens=512):
    """Get a 512-token window from the text at a position determined by seed."""
    tokens = tokenizer.encode(text)
    rng = np.random.RandomState(seed)
    # Use three non-overlapping offsets based on seed
    n = len(tokens)
    offsets = [n // 4, n // 2, 3 * n // 4]
    # Vary slightly per seed
    seed_idx = SEEDS.index(seed)
    offset = offsets[seed_idx]
    offset = min(offset, n - n_tokens - 1)
    window = tokens[offset: offset + n_tokens]
    return torch.tensor(window).unsqueeze(0)  # (1, 512)


def collect_hidden_states(model, input_ids):
    """Run forward pass and collect residual stream at each block output."""
    hidden_states_list = []
    hooks = []

    def make_hook(block_idx):
        def hook(module, input, output):
            # output is a tuple; index 0 is the hidden states (residual stream output)
            hs = output[0]  # (1, seq_len, 768)
            hidden_states_list.append((block_idx, hs.detach().squeeze(0)))  # (512, 768)
        return hook

    for i, block in enumerate(model.h):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Sort by block index
    hidden_states_list.sort(key=lambda x: x[0])
    return [hs for _, hs in hidden_states_list]  # list of 12 tensors (512, 768)


def verify_pre_ln(hs_list):
    """Verify hooks captured pre-LN residual stream (std should be > 0.5)."""
    norms = [hs.norm(dim=-1).std().item() for hs in hs_list]
    # Post-LN would have std ~1.0 but very uniform; pre-LN has large variance
    return norms


def pca_project(hs_np, n_components):
    """PCA project and return projected data + variance explained."""
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(hs_np)
    var_explained = np.sum(pca.explained_variance_ratio_)
    return projected, var_explained


def compute_tp(projected, max_r_percentile=95):
    """Compute total persistence for H0 and H1."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(projected, metric="euclidean")
    max_r = np.percentile(D, max_r_percentile)
    thresh = 0.01 * max_r  # 1% of max_r lifetime threshold

    result = ripser(D, maxdim=1, thresh=max_r, distance_matrix=True)
    dgms = result["dgms"]

    def total_pers(dgm, max_r):
        finite = dgm[dgm[:, 1] < np.inf]
        lifetimes = finite[:, 1] - finite[:, 0]
        return float(np.sum(lifetimes[lifetimes > 0.01 * max_r]))

    tp_h0 = total_pers(dgms[0], max_r)
    tp_h1 = total_pers(dgms[1], max_r) if len(dgms) > 1 else 0.0
    return tp_h0, tp_h1, max_r


def compute_null_baseline(seed, block6_hs_np, n_points=512, n_dim=30):
    """Isotropic Gaussian matched to block-6 mean norm."""
    rng = np.random.RandomState(seed + 1000)
    mean_norm = np.linalg.norm(block6_hs_np, axis=1).mean()
    # Project block 6 to 30D for norm reference
    pca = PCA(n_components=n_dim)
    proj = pca.fit_transform(block6_hs_np)
    ref_norm = np.linalg.norm(proj, axis=1).mean()
    # Generate Gaussian points
    pts = rng.randn(n_points, n_dim)
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8) * ref_norm
    return pts


def main():
    print("=== Cycle 074: Persistent Homology GPT-2 ===\n")
    model, tokenizer = load_model_and_tokenizer()

    with open("data/shakespeare.txt", "r") as f:
        text = f.read()

    results = {
        "cycle": 74,
        "item": 28,
        "seeds": SEEDS,
        "n_tokens": N_TOKENS,
        "pca_dim": PCA_DIM,
        "pca_variance": {},  # seed -> block -> var
        "tp_h0": {},         # seed -> block -> float
        "tp_h1": {},         # seed -> block -> float
        "tp_combined": {},   # seed -> block -> float
        "norm_std": {},      # seed -> block -> float (pre-LN verify)
        "null_tp_h0": {},
        "null_tp_h1": {},
        "pca_sensitivity_spearman": None,
        "correlations": {},
        "monotonicity": {},
        "damage_8x": DAMAGE_8X,
        "damage_16x": DAMAGE_16X,
        "outcome": None,
        "verdict": None,
    }

    # Collect per-seed data
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        input_ids = get_token_windows(text, tokenizer, seed, N_TOKENS)
        hs_list = collect_hidden_states(model, input_ids)

        # Verify pre-LN
        norm_stds = verify_pre_ln(hs_list)
        results["norm_std"][str(seed)] = norm_stds
        print(f"  Norm std across blocks: {[f'{s:.3f}' for s in norm_stds]}")
        min_std = min(norm_stds)
        if min_std < 0.5:
            print(f"  WARNING: min norm std = {min_std:.3f} < 0.5, may be post-LN")

        tp_h0_blocks = []
        tp_h1_blocks = []
        var_blocks = []

        for block_idx in range(N_BLOCKS):
            hs_np = hs_list[block_idx].numpy()  # (512, 768)

            # PCA to 30D
            proj_30d, var_30 = pca_project(hs_np, PCA_DIM)
            var_blocks.append(float(var_30))

            # PCA sensitivity for seed=42, block=6
            if seed == 42 and block_idx == 6:
                proj_20d, _ = pca_project(hs_np, 20)
                proj_40d, _ = pca_project(hs_np, 40)

            # Compute TP
            tp_h0, tp_h1, max_r = compute_tp(proj_30d)
            tp_h0_blocks.append(tp_h0)
            tp_h1_blocks.append(tp_h1)

            print(f"  Block {block_idx:2d}: var={var_30:.3f} tp_h0={tp_h0:.2f} tp_h1={tp_h1:.2f} max_r={max_r:.3f}")

        results["pca_variance"][str(seed)] = var_blocks
        results["tp_h0"][str(seed)] = tp_h0_blocks
        results["tp_h1"][str(seed)] = tp_h1_blocks
        results["tp_combined"][str(seed)] = [h0 + h1 for h0, h1 in zip(tp_h0_blocks, tp_h1_blocks)]

        # Null baseline (use block-6 hidden states)
        block6_np = hs_list[6].numpy()
        null_pts = compute_null_baseline(seed, block6_np, n_points=N_TOKENS, n_dim=PCA_DIM)
        null_h0, null_h1, _ = compute_tp(null_pts)
        results["null_tp_h0"][str(seed)] = null_h0
        results["null_tp_h1"][str(seed)] = null_h1
        print(f"  Null baseline: tp_h0={null_h0:.2f} tp_h1={null_h1:.2f}")

    # PCA sensitivity: Spearman(TP_20D, TP_40D) for seed=42
    print("\n--- PCA sensitivity (seed=42) ---")
    seed42_hs = collect_hidden_states(model, get_token_windows(text, tokenizer, 42, N_TOKENS))
    tp_20d_list = []
    tp_40d_list = []
    for block_idx in range(N_BLOCKS):
        hs_np = seed42_hs[block_idx].numpy()
        proj_20d, _ = pca_project(hs_np, 20)
        proj_40d, _ = pca_project(hs_np, 40)
        h0_20, h1_20, _ = compute_tp(proj_20d)
        h0_40, h1_40, _ = compute_tp(proj_40d)
        tp_20d_list.append(h0_20 + h1_20)
        tp_40d_list.append(h0_40 + h1_40)

    sens_rho, sens_p = spearmanr(tp_20d_list, tp_40d_list)
    results["pca_sensitivity_spearman"] = float(sens_rho)
    print(f"  Spearman(TP_20D, TP_40D) = {sens_rho:.3f} (p={sens_p:.3f})")

    # Compute correlations
    print("\n--- Correlations ---")
    # Mean TP across seeds
    mean_tp = np.array([
        np.mean([results["tp_combined"][str(s)][b] for s in SEEDS])
        for b in range(N_BLOCKS)
    ])
    mean_tp_h0 = np.array([
        np.mean([results["tp_h0"][str(s)][b] for s in SEEDS])
        for b in range(N_BLOCKS)
    ])
    mean_tp_h1 = np.array([
        np.mean([results["tp_h1"][str(s)][b] for s in SEEDS])
        for b in range(N_BLOCKS)
    ])
    results["mean_tp_combined"] = mean_tp.tolist()
    results["mean_tp_h0"] = mean_tp_h0.tolist()
    results["mean_tp_h1"] = mean_tp_h1.tolist()

    damage_8x = np.array(DAMAGE_8X)
    damage_16x = np.array(DAMAGE_16X)

    rho_8x, p_8x = spearmanr(mean_tp, damage_8x, alternative="greater")
    rho_16x, p_16x = spearmanr(mean_tp, damage_16x, alternative="greater")
    rho_h0_8x, p_h0_8x = spearmanr(mean_tp_h0, damage_8x, alternative="greater")
    rho_h1_8x, p_h1_8x = spearmanr(mean_tp_h1, damage_8x, alternative="greater")

    # Without block 0 (leverage)
    rho_8x_no0, p_8x_no0 = spearmanr(mean_tp[1:], damage_8x[1:], alternative="greater")

    print(f"  Spearman(TP_combined, damage_8x)  = {rho_8x:.3f} (p={p_8x:.3f})")
    print(f"  Spearman(TP_combined, damage_16x) = {rho_16x:.3f} (p={p_16x:.3f})")
    print(f"  Spearman(TP_H0, damage_8x)        = {rho_h0_8x:.3f} (p={p_h0_8x:.3f})")
    print(f"  Spearman(TP_H1, damage_8x)        = {rho_h1_8x:.3f} (p={p_h1_8x:.3f})")
    print(f"  Spearman(TP_combined, damage_8x) [no L0] = {rho_8x_no0:.3f} (p={p_8x_no0:.3f})")

    results["correlations"] = {
        "rho_combined_8x": float(rho_8x), "p_combined_8x": float(p_8x),
        "rho_combined_16x": float(rho_16x), "p_combined_16x": float(p_16x),
        "rho_h0_8x": float(rho_h0_8x), "p_h0_8x": float(p_h0_8x),
        "rho_h1_8x": float(rho_h1_8x), "p_h1_8x": float(p_h1_8x),
        "rho_combined_8x_no_L0": float(rho_8x_no0), "p_combined_8x_no_L0": float(p_8x_no0),
    }

    # Monotonicity: count adjacent pairs where TP[i] > TP[i+1]
    decreasing_pairs = sum(1 for i in range(N_BLOCKS - 1) if mean_tp[i] > mean_tp[i + 1])
    monotonicity_frac = decreasing_pairs / (N_BLOCKS - 1)
    results["monotonicity"] = {
        "decreasing_pairs": decreasing_pairs,
        "total_pairs": N_BLOCKS - 1,
        "fraction": float(monotonicity_frac),
    }
    print(f"\n  Monotonicity (decreasing): {decreasing_pairs}/{N_BLOCKS-1} = {monotonicity_frac:.2f}")

    # Validation checks
    min_pca_var = min(
        results["pca_variance"][str(s)][b]
        for s in SEEDS for b in range(N_BLOCKS)
    )
    print(f"\n  Min PCA variance @ 30D: {min_pca_var:.3f}")

    cross_seed_stds = [
        np.std([results["tp_combined"][str(s)][b] for s in SEEDS]) /
        (np.mean([results["tp_combined"][str(s)][b] for s in SEEDS]) + 1e-8)
        for b in range(N_BLOCKS)
    ]
    max_cv = max(cross_seed_stds)
    print(f"  Max cross-seed CV: {max_cv:.3f}")

    null_mean = np.mean([results["null_tp_h0"][str(s)] + results["null_tp_h1"][str(s)] for s in SEEDS])
    blocks_above_null = sum(1 for tp in mean_tp if tp > null_mean)
    print(f"  Null TP mean = {null_mean:.2f}, blocks above null: {blocks_above_null}/12")

    # Verdict
    pca_ok = min_pca_var >= 0.80
    pre_ln_ok = all(
        results["norm_std"][str(s)][b] >= 0.5
        for s in SEEDS for b in range(N_BLOCKS)
    )
    sensitivity_ok = results["pca_sensitivity_spearman"] >= 0.7

    confirmed = (rho_8x >= 0.35 and monotonicity_frac >= 0.60 and pca_ok and pre_ln_ok)
    # Betti approximately constant: std/mean < 0.15
    betti_std_over_mean = np.std(mean_tp) / (np.mean(mean_tp) + 1e-8)
    refuted = (betti_std_over_mean < 0.15 and pca_ok and pre_ln_ok)
    inconclusive = not confirmed and not refuted

    print(f"\n  pca_ok={pca_ok} ({min_pca_var:.3f})")
    print(f"  pre_ln_ok={pre_ln_ok}")
    print(f"  sensitivity_ok={sensitivity_ok} ({sens_rho:.3f})")
    print(f"  rho_8x={rho_8x:.3f} (need >= 0.35 for confirmed)")
    print(f"  monotonicity={monotonicity_frac:.2f} (need >= 0.60 for confirmed)")
    print(f"  betti_std/mean={betti_std_over_mean:.3f} (< 0.15 -> refuted)")

    if not pca_ok or not sensitivity_ok:
        verdict = "INCONCLUSIVE"
        reason = f"pca_ok={pca_ok} ({min_pca_var:.3f}), sensitivity_ok={sensitivity_ok} ({sens_rho:.3f})"
    elif confirmed:
        verdict = "CONFIRMED"
        reason = f"rho_8x={rho_8x:.3f}>= 0.35, monotonicity={monotonicity_frac:.2f}>= 0.60"
    elif refuted:
        verdict = "REFUTED"
        reason = f"Betti std/mean={betti_std_over_mean:.3f} < 0.15 (approximately constant)"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"rho_8x={rho_8x:.3f}, monotonicity={monotonicity_frac:.2f}, betti_std/mean={betti_std_over_mean:.3f}"

    results["outcome"] = verdict
    results["verdict"] = reason
    results["validation"] = {
        "pca_ok": pca_ok,
        "pre_ln_ok": pre_ln_ok,
        "sensitivity_ok": sensitivity_ok,
        "min_pca_var": float(min_pca_var),
        "max_cross_seed_cv": float(max_cv),
        "null_mean_tp": float(null_mean),
        "blocks_above_null": int(blocks_above_null),
        "betti_std_over_mean": float(betti_std_over_mean),
    }

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  {reason}")

    # Save JSON
    json_path = f"{RESULTS_DIR}/cycle_074_phom_gpt2.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: TP vs block for all seeds + null baseline
    ax = axes[0]
    colors = ["steelblue", "darkorange", "green"]
    for si, seed in enumerate(SEEDS):
        tp_s = results["tp_combined"][str(seed)]
        ax.plot(range(N_BLOCKS), tp_s, marker="o", alpha=0.6,
                color=colors[si], label=f"seed={seed}")
    ax.plot(range(N_BLOCKS), mean_tp, marker="D", linewidth=2,
            color="black", label="mean")
    null_vals = [results["null_tp_h0"][str(s)] + results["null_tp_h1"][str(s)] for s in SEEDS]
    null_mean_line = np.mean(null_vals)
    ax.axhline(null_mean_line, linestyle="--", color="red", alpha=0.7, label=f"null ({null_mean_line:.1f})")
    ax.set_xlabel("GPT-2 Block")
    ax.set_ylabel("Total Persistence (H0+H1)")
    ax.set_title("TP through depth")
    ax.legend(fontsize=8)
    ax.set_xticks(range(N_BLOCKS))

    # Panel 2: scatter TP vs damage_8x
    ax2 = axes[1]
    ax2.scatter(mean_tp, damage_8x, color="steelblue", s=60, zorder=3)
    for b in range(N_BLOCKS):
        ax2.annotate(str(b), (mean_tp[b], damage_8x[b]), textcoords="offset points",
                     xytext=(3, 3), fontsize=8)
    ax2.set_xlabel("Mean Total Persistence")
    ax2.set_ylabel("Compression Damage (8x)")
    ax2.set_title(f"TP vs Damage (rho={rho_8x:.2f}, p={p_8x:.3f})")

    plt.suptitle(f"GPT-2 Persistent Homology — {verdict}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    png_path = f"{RESULTS_DIR}/cycle_074_phom_gpt2.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

    return results


if __name__ == "__main__":
    main()
