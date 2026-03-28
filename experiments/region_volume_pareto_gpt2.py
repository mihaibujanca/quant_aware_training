"""
Cycle 075: Region volume Pareto — GPT-2 FFN analog.
K-means on per-token Jacobian singular value profiles as proxy for "linear regions."
Cumulative reconstruction error as clusters added in descending size order.
Test for Pareto structure (top 20% covers >= 70% variance) and correlation with
compression difficulty (damage_8x from cycle_048).
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"
CYCLE_048_PATH = OUTPUT_DIR / "cycle_048_local_complexity_gpt2.json"

SEEDS = [42, 123, 7]
SEQ_LEN = 128
N_WINDOWS = 4          # 4 × 128 = 512 tokens per seed
K_CLUSTERS = 50
N_BLOCKS = 12
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from ffn_jacobian_svd_compression.py)
# ---------------------------------------------------------------------------

def _standard_weight(module):
    """Return (out_features, in_features) weight regardless of Conv1D vs Linear."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _gelu_grad_autograd(z: torch.Tensor) -> torch.Tensor:
    """Element-wise GELU'(z) via autograd — GPT-2 tanh approximation."""
    z_req = z.detach().requires_grad_(True)
    out = F.gelu(z_req, approximate="tanh")
    out.sum().backward()
    return z_req.grad.detach()


def load_model_and_tokenizer():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def load_probe_tokens(tokenizer, seed: int) -> torch.Tensor:
    """Load shakespeare.txt, tokenize, take 4 non-overlapping 128-token windows."""
    text = DATA_PATH.read_text()
    tokens = tokenizer.encode(text)
    rng = np.random.default_rng(seed)
    n_total = len(tokens)
    # Pick starting position deterministically per seed
    start = int(rng.integers(0, n_total - N_WINDOWS * SEQ_LEN))
    windows = []
    for i in range(N_WINDOWS):
        chunk = tokens[start + i * SEQ_LEN: start + (i + 1) * SEQ_LEN]
        windows.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return torch.cat(windows, dim=0)  # (N_WINDOWS, SEQ_LEN)


def collect_c_fc_activations(block_idx: int, model, token_ids: torch.Tensor) -> torch.Tensor:
    """Hook mlp.c_fc output to get pre-GELU activations z. Returns (T, 3072)."""
    z_list = []

    def hook_fn(module, inputs, output):
        z_list.append(output.detach().float().reshape(-1, output.shape[-1]))

    mlp = model.transformer.h[block_idx].mlp
    handle = mlp.c_fc.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for i in range(token_ids.shape[0]):
                model(token_ids[i:i+1].to(DEVICE), use_cache=False)
    finally:
        handle.remove()

    return torch.cat(z_list, dim=0)  # (T, 3072)


def compute_sv_profiles(block_idx: int, model, z_tokens: torch.Tensor) -> torch.Tensor:
    """
    For each token, compute singular value profile of J(x) = W_out @ diag(GELU'(z)) @ W_in.
    Returns (T, 768) normalized SV profiles.
    """
    mlp = model.transformer.h[block_idx].mlp
    W_in = _standard_weight(mlp.c_fc)    # (3072, 768)
    W_out = _standard_weight(mlp.c_proj)  # (768, 3072)

    T = z_tokens.shape[0]
    profiles = []

    for t in range(T):
        z_t = z_tokens[t]  # (3072,)
        g_t = _gelu_grad_autograd(z_t)  # (3072,)
        # J_t = W_out @ diag(g_t) @ W_in — shape (768, 768)
        J_t = (W_out * g_t.unsqueeze(0)) @ W_in
        sv = torch.linalg.svdvals(J_t)  # (768,)
        # Normalize to unit L2
        norm = sv.norm()
        if norm > 1e-12:
            sv = sv / norm
        profiles.append(sv)

    return torch.stack(profiles, dim=0)  # (T, 768)


def gini_coefficient(sizes: np.ndarray) -> float:
    """Gini coefficient of a discrete distribution (sizes = counts per cluster)."""
    sizes = np.sort(sizes.astype(float))
    n = len(sizes)
    if n == 0 or sizes.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * sizes).sum()) / (n * sizes.sum()) - (n + 1) / n


def cumulative_coverage(profiles_np: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """
    Compute cumulative coverage fraction as clusters added in descending volume order.
    coverage[i] = fraction of total_SS explained by top i+1 clusters (by size).

    Uses within-cluster SS reduction:
    total_SS = sum_t ||p_t - global_mean||^2
    within_SS_c = sum_{t in c} ||p_t - c_mean||^2
    coverage_c = (total_SS - within_SS_c) / total_SS (per-cluster contribution)
    """
    global_mean = profiles_np.mean(axis=0)
    total_SS = ((profiles_np - global_mean) ** 2).sum()

    cluster_sizes = np.bincount(labels, minlength=K)
    # Per-cluster within-SS reduction = (between-cluster SS for this cluster)
    cluster_SS_reduction = np.zeros(K)
    for c in range(K):
        mask = labels == c
        if mask.sum() == 0:
            continue
        pts = profiles_np[mask]
        c_mean = pts.mean(axis=0)
        # Between-cluster SS contribution = ||c_mean - global_mean||^2 * n_c
        cluster_SS_reduction[c] = ((c_mean - global_mean) ** 2).sum() * mask.sum()

    # Sort clusters descending by size
    order = np.argsort(cluster_sizes)[::-1]
    cumulative = np.cumsum(cluster_SS_reduction[order])
    if total_SS > 1e-12:
        coverage = cumulative / total_SS
    else:
        coverage = np.zeros(K)
    return coverage  # (K,) — coverage after adding top-i clusters


def lorenz_curve(sizes: np.ndarray):
    """Return (x, y) for Lorenz curve."""
    sizes_sorted = np.sort(sizes)
    n = len(sizes_sorted)
    x = np.linspace(0, 1, n + 1)
    y = np.concatenate([[0], np.cumsum(sizes_sorted) / (sizes_sorted.sum() + 1e-12)])
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()

    # Load damage from cycle_048
    with open(CYCLE_048_PATH) as f:
        c048 = json.load(f)
    damage_8x = np.array(c048["damage_8x"])   # (12,)
    damage_16x = np.array(c048["damage_16x"])  # (12,)

    # Per seed, per block: SV profiles
    # seed_profiles[seed_idx][block_idx] = np.ndarray (T, 768)
    all_profiles = []  # list of len(SEEDS), each is list of len(N_BLOCKS) arrays

    for seed in SEEDS:
        print(f"  Seed {seed}: loading tokens...", flush=True)
        token_ids = load_probe_tokens(tokenizer, seed)  # (4, 128)

        seed_profiles = []
        for block_idx in range(N_BLOCKS):
            print(f"    Block {block_idx}...", end=" ", flush=True)
            z_tokens = collect_c_fc_activations(block_idx, model, token_ids)  # (512, 3072)
            profiles = compute_sv_profiles(block_idx, model, z_tokens)  # (512, 768)
            seed_profiles.append(profiles.numpy())
            print(f"done ({profiles.shape[0]} tokens)", flush=True)
        all_profiles.append(seed_profiles)

    print(f"Profile extraction done in {time.time() - t0:.1f}s", flush=True)

    # K-means + analysis per block, per seed
    results_by_block = []
    for block_idx in range(N_BLOCKS):
        block_ginis = []
        block_null_ginis = []
        block_top20_coverages = []
        block_null_top20 = []
        block_lorenz = []  # for first seed only
        block_coverage_curves = []  # for first seed only

        for s_idx, seed in enumerate(SEEDS):
            profiles_np = all_profiles[s_idx][block_idx]  # (512, 768)
            T = profiles_np.shape[0]

            # Real K-means
            km = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=5)
            labels = km.fit_predict(profiles_np)
            sizes = np.bincount(labels, minlength=K_CLUSTERS).astype(float)

            gini = gini_coefficient(sizes)
            block_ginis.append(float(gini))

            coverage = cumulative_coverage(profiles_np, labels, K_CLUSTERS)
            block_coverage_curves.append(coverage.tolist()) if s_idx == 0 else None

            # Top 20% of clusters (10 clusters out of 50)
            n_top20 = max(1, K_CLUSTERS // 5)
            top20_cov = float(coverage[n_top20 - 1]) if len(coverage) >= n_top20 else float(coverage[-1])
            block_top20_coverages.append(top20_cov)

            if s_idx == 0:
                block_lorenz.append(lorenz_curve(sizes))

            # Null baseline: random assignment
            rng = np.random.default_rng(seed + 1000)
            null_labels = rng.integers(0, K_CLUSTERS, size=T)
            null_sizes = np.bincount(null_labels, minlength=K_CLUSTERS).astype(float)
            null_gini = gini_coefficient(null_sizes)
            block_null_ginis.append(float(null_gini))

            null_cov = cumulative_coverage(profiles_np, null_labels, K_CLUSTERS)
            null_top20 = float(null_cov[n_top20 - 1]) if len(null_cov) >= n_top20 else float(null_cov[-1])
            block_null_top20.append(null_top20)

        results_by_block.append({
            "block": block_idx,
            "gini_mean": float(np.mean(block_ginis)),
            "gini_std": float(np.std(block_ginis)),
            "gini_per_seed": block_ginis,
            "null_gini_mean": float(np.mean(block_null_ginis)),
            "null_gini_std": float(np.std(block_null_ginis)),
            "null_gini_per_seed": block_null_ginis,
            "top20_coverage_mean": float(np.mean(block_top20_coverages)),
            "top20_coverage_std": float(np.std(block_top20_coverages)),
            "top20_per_seed": block_top20_coverages,
            "null_top20_mean": float(np.mean(block_null_top20)),
            "coverage_curve_seed0": block_coverage_curves[0] if block_coverage_curves else [],
        })

    print("K-means done. Computing correlations...", flush=True)

    # Spearman correlation: Gini vs (1 - damage)
    from scipy.stats import spearmanr
    ginis = np.array([r["gini_mean"] for r in results_by_block])
    spear_8x = spearmanr(ginis, 1.0 - damage_8x)
    spear_16x = spearmanr(ginis, 1.0 - damage_16x)

    # Pareto check: majority of blocks have top-20% coverage >= 70%
    top20_coverages = np.array([r["top20_coverage_mean"] for r in results_by_block])
    pareto_check = bool((top20_coverages >= 0.70).sum() >= N_BLOCKS // 2)

    # Gini > null for majority
    gini_above_null = sum(
        1 for r in results_by_block if r["gini_mean"] > r["null_gini_mean"]
    )

    # Cross-seed stability: std < 50% of mean
    gini_stable = sum(
        1 for r in results_by_block
        if r["gini_mean"] > 0 and r["gini_std"] < 0.5 * r["gini_mean"]
    )

    summary = {
        "cycle": 75,
        "item": 35,
        "K_clusters": K_CLUSTERS,
        "n_seeds": len(SEEDS),
        "n_tokens_per_seed": N_WINDOWS * SEQ_LEN,
        "ginis_by_block": [r["gini_mean"] for r in results_by_block],
        "null_ginis_by_block": [r["null_gini_mean"] for r in results_by_block],
        "top20_coverages_by_block": top20_coverages.tolist(),
        "null_top20_by_block": [r["null_top20_mean"] for r in results_by_block],
        "pareto_check_70pct": pareto_check,
        "n_blocks_pareto": int((top20_coverages >= 0.70).sum()),
        "gini_above_null_count": gini_above_null,
        "gini_stable_count": gini_stable,
        "spearman_gini_vs_1minus_damage8x": {
            "rho": float(spear_8x.statistic),
            "pvalue": float(spear_8x.pvalue),
        },
        "spearman_gini_vs_1minus_damage16x": {
            "rho": float(spear_16x.statistic),
            "pvalue": float(spear_16x.pvalue),
        },
        "damage_8x": damage_8x.tolist(),
        "damage_16x": damage_16x.tolist(),
        "blocks": results_by_block,
    }

    # Determine outcome
    if pareto_check and spear_8x.statistic >= 0.35:
        outcome = "confirmed"
    elif np.mean(ginis) < 0.2:
        outcome = "refuted_uniform"
    else:
        outcome = "inconclusive"
    summary["outcome"] = outcome

    json_path = OUTPUT_DIR / "cycle_075_pareto_gpt2.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON: {json_path}", flush=True)

    # ---------------------------------------------------------------------------
    # Plot (2×2)
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GPT-2 FFN Region Volume Pareto (Cycle 075)", fontsize=13)

    blocks_x = np.arange(N_BLOCKS)

    # (a) Gini per block vs null
    ax = axes[0, 0]
    ax.bar(blocks_x - 0.2, [r["gini_mean"] for r in results_by_block], 0.4,
           label="K-means Gini", color="steelblue",
           yerr=[r["gini_std"] for r in results_by_block], capsize=3)
    ax.bar(blocks_x + 0.2, [r["null_gini_mean"] for r in results_by_block], 0.4,
           label="Null Gini", color="orange",
           yerr=[r["null_gini_std"] for r in results_by_block], capsize=3)
    ax.axhline(0.2, color="red", linestyle="--", label="Uniform threshold (0.2)")
    ax.set_xlabel("Block")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("(a) Gini per block vs null")
    ax.legend(fontsize=8)
    ax.set_xticks(blocks_x)

    # (b) Cumulative variance curves for 3 representative blocks
    ax = axes[0, 1]
    rep_blocks = [0, 5, 11]
    colors = ["steelblue", "green", "red"]
    x_frac = np.linspace(0, 1, K_CLUSTERS)
    for b, c in zip(rep_blocks, colors):
        curve = np.array(results_by_block[b]["coverage_curve_seed0"])
        ax.plot(x_frac, curve, color=c, label=f"Block {b}")
    ax.axhline(0.70, color="black", linestyle="--", alpha=0.5, label="70% threshold")
    ax.axvline(0.20, color="gray", linestyle=":", alpha=0.5, label="20% clusters")
    ax.set_xlabel("Fraction of clusters (descending size)")
    ax.set_ylabel("Cumulative coverage")
    ax.set_title("(b) Cumulative coverage curves")
    ax.legend(fontsize=8)

    # (c) Lorenz curves for same representative blocks
    ax = axes[1, 0]
    for b, c in zip(rep_blocks, colors):
        # Compute Lorenz from cluster sizes (use all seeds merged for seed 0)
        profiles_np = all_profiles[0][b]
        km = KMeans(n_clusters=K_CLUSTERS, random_state=SEEDS[0], n_init=5)
        labels = km.fit_predict(profiles_np)
        sizes = np.bincount(labels, minlength=K_CLUSTERS).astype(float)
        lx, ly = lorenz_curve(sizes)
        ax.plot(lx, ly, color=c, label=f"Block {b}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Equality")
    ax.set_xlabel("Cumulative cluster fraction")
    ax.set_ylabel("Cumulative token fraction")
    ax.set_title("(c) Lorenz curves (seed 42)")
    ax.legend(fontsize=8)

    # (d) Gini vs damage_8x scatter
    ax = axes[1, 1]
    ax.scatter(ginis, damage_8x, color="steelblue", s=60, zorder=3)
    for i, b in enumerate(blocks_x):
        ax.annotate(str(b), (ginis[i], damage_8x[i]), fontsize=7, ha="left")
    rho = spear_8x.statistic
    pval = spear_8x.pvalue
    ax.set_xlabel("Gini coefficient")
    ax.set_ylabel("Damage (8x compression)")
    ax.set_title(f"(d) Gini vs damage_8x\nSpearman ρ={rho:.3f}, p={pval:.3f}")

    plt.tight_layout()
    png_path = OUTPUT_DIR / "cycle_075_pareto_gpt2.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {png_path}", flush=True)

    # Final summary
    print(f"\n=== RESULTS ===")
    print(f"Outcome: {outcome}")
    print(f"Gini range: [{min(ginis):.3f}, {max(ginis):.3f}], mean={np.mean(ginis):.3f}")
    print(f"Null Gini mean: {np.mean([r['null_gini_mean'] for r in results_by_block]):.3f}")
    print(f"Gini > null: {gini_above_null}/{N_BLOCKS} blocks")
    print(f"Gini stable (std < 50% mean): {gini_stable}/{N_BLOCKS} blocks")
    print(f"Pareto (top20 >= 70%): {(top20_coverages >= 0.70).sum()}/{N_BLOCKS} blocks")
    print(f"Top-20% coverage range: [{top20_coverages.min():.3f}, {top20_coverages.max():.3f}]")
    print(f"Spearman(Gini, 1-damage_8x): rho={spear_8x.statistic:.3f}, p={spear_8x.pvalue:.3f}")
    print(f"Spearman(Gini, 1-damage_16x): rho={spear_16x.statistic:.3f}, p={spear_16x.pvalue:.3f}")
    print(f"Total time: {time.time() - t0:.1f}s")

    return summary


if __name__ == "__main__":
    main()
