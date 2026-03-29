"""
Cycle 086 — FFN output manifold geometry — GPT-2 FFN (item 99).
For blocks [7, 5, 4, 1], measure hidden activation eigenspectrum,
compare to Marchenko-Pastur null, test JL distance distortion vs bound.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from aleph.lm_geometric_compression import (
    TeacherLoadSpec,
    ProbeCorpusSpec,
    load_nanochat_teacher,
    build_probe_corpus,
)

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [7, 5, 4, 1]
SEQ_LEN = 128
# 128 sequences x 128 tokens = 16384 tokens total (4 sources x 32 sequences)
SEQS_PER_SOURCE = 32        # 32 * 128 = 4096 tokens per source
N_TOKENS = 16384
K = 512                     # compression index (6x from 3072)
N_JL_PAIRS = 10000
N_PROJ_DRAWS = 10
DEVICE = "cpu"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

print("=== Loading GPT-2 ===")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()

# ── Build 16384-token probe corpus (4 sources x 4096 tokens) ─────────────────
print("\n=== Building probe corpus (16384 tokens) ===")
corpus_parts = []
for fname in ["shakespeare.txt", "wikitext2_valid.txt", "openwebtext_sample.txt", "c4_sample.txt"]:
    spec = ProbeCorpusSpec(
        text_path=os.path.join(DATA, fname),
        probe_train_sequences=SEQS_PER_SOURCE,
        sequence_length=SEQ_LEN,
        teacher=teacher,
    )
    part = build_probe_corpus(spec)
    corpus_parts.append(part.probe_train)
    print(f"  {fname}: {part.probe_train.shape} -> {part.probe_train.numel()} tokens")

all_tokens = torch.cat(corpus_parts, dim=0)  # (128, 128)
print(f"  Combined: {all_tokens.shape}  -> {all_tokens.numel()} tokens")
assert all_tokens.numel() >= N_TOKENS, f"Need {N_TOKENS} tokens, got {all_tokens.numel()}"
token_batches = all_tokens[:N_TOKENS // SEQ_LEN]  # (128, 128)
print(f"  Using {token_batches.shape[0]} sequences x {SEQ_LEN} tokens = {token_batches.numel()}")


# ── Collect activations per block ─────────────────────────────────────────────
print("\n=== Collecting activations ===")

# Storage: per block
block_xln = {b: [] for b in BLOCKS}   # post-ln_2 (768D)
block_h = {b: [] for b in BLOCKS}     # post-GELU (3072D)
block_y = {b: [] for b in BLOCKS}     # post-W_down (768D)

hooks = []


def make_xln_hook(block_idx):
    """Hook on ln_2 output (768D FFN input)."""
    def hook_fn(module, input, output):
        block_xln[block_idx].append(output.detach().cpu().float())
    return hook_fn


def make_h_hook(block_idx):
    """Hook on mlp.c_fc output post-GELU (3072D hidden)."""
    def hook_fn(module, input, output):
        # output is pre-GELU (c_fc linear output); apply GELU here to get h
        h = F.gelu(output.detach().float(), approximate="tanh")
        block_h[block_idx].append(h.cpu())
    return hook_fn


def make_y_hook(block_idx):
    """Hook on mlp.c_proj output (768D FFN output)."""
    def hook_fn(module, input, output):
        block_y[block_idx].append(output.detach().cpu().float())
    return hook_fn


for b in BLOCKS:
    block = model.transformer.h[b]
    hooks.append(block.ln_2.register_forward_hook(make_xln_hook(b)))
    hooks.append(block.mlp.c_fc.register_forward_hook(make_h_hook(b)))
    hooks.append(block.mlp.c_proj.register_forward_hook(make_y_hook(b)))

with torch.no_grad():
    for i in range(0, token_batches.shape[0], 4):
        batch = token_batches[i:i+4].to(DEVICE)
        model(batch)
        if (i // 4) % 8 == 0:
            print(f"  Processed {(i + 4) * SEQ_LEN}/{N_TOKENS} tokens")

for h in hooks:
    h.remove()

# Concatenate: (N_TOKENS, dim)
xln_data = {b: torch.cat(block_xln[b], dim=0).view(-1, 768) for b in BLOCKS}
h_data = {b: torch.cat(block_h[b], dim=0).view(-1, 3072) for b in BLOCKS}
y_data = {b: torch.cat(block_y[b], dim=0).view(-1, 768) for b in BLOCKS}

for b in BLOCKS:
    print(f"  Block {b}: xln={xln_data[b].shape}, h={h_data[b].shape}, y={y_data[b].shape}")


# ── Marchenko-Pastur density ──────────────────────────────────────────────────
def mp_density(lambdas, gamma, sigma2):
    """Marchenko-Pastur pdf values at eigenvalue array lambdas."""
    lp = sigma2 * (1 + np.sqrt(gamma))**2
    lm = sigma2 * (1 - np.sqrt(gamma))**2
    l = np.clip(lambdas, lm + 1e-12, lp - 1e-12)
    density = np.sqrt(np.maximum((lp - l) * (l - lm), 0)) / (2 * np.pi * gamma * sigma2 * l)
    density[lambdas < lm] = 0
    density[lambdas > lp] = 0
    return density


def compute_eigenspectrum(X):
    """Compute sample covariance eigenvalues (sorted descending) of centered X (n, d)."""
    X_c = X - X.mean(dim=0, keepdim=True)
    C = (X_c.T @ X_c) / (X_c.shape[0] - 1)
    # Ensure symmetry
    C = (C + C.T) / 2
    eigvals = torch.linalg.eigh(C).eigenvalues  # ascending
    eigvals = eigvals.flip(0)  # descending
    eigvals = torch.clamp(eigvals, min=0)
    return eigvals


# ── Main analysis ─────────────────────────────────────────────────────────────
print("\n=== Computing eigenspectra ===")
results = {}
gamma_h = 3072 / N_TOKENS   # = 0.1875
gamma_xln = 768 / N_TOKENS  # = 0.046875

for b in BLOCKS:
    print(f"\nBlock {b}:")
    res = {}

    # 1. LayerNorm isotropy (xln, 768D)
    xln_eigs = compute_eigenspectrum(xln_data[b])
    res["xln_eigs"] = xln_eigs.tolist()
    # isotropy ratio at k=192 (6x compression from 768: 768/6=128... use k=256 for 3x)
    # plan says k=512 for x_ln at 768D: use k=256 (768/3) but k=512 > 768 is invalid
    # Plan says "isotropy ratio = lambda_512/lambda_1" for 768D — but 768D only has 768 eigvals.
    # Use k=128 (= 768/6, same 6x ratio as k=512/3072) for comparability.
    k_xln = 128
    res["xln_isotropy_ratio"] = float(xln_eigs[k_xln - 1] / (xln_eigs[0] + 1e-30))
    print(f"  xln eigs: max={xln_eigs[0]:.4f}, k={k_xln} -> {xln_eigs[k_xln-1]:.4f}, "
          f"isotropy={res['xln_isotropy_ratio']:.4f}")

    # 2. Hidden activation eigenspectrum (h, 3072D)
    h_eigs = compute_eigenspectrum(h_data[b])
    res["h_eigs"] = h_eigs.tolist()
    res["h_isotropy_ratio"] = float(h_eigs[K - 1] / (h_eigs[0] + 1e-30))
    print(f"  h eigs: max={h_eigs[0]:.4f}, k={K} -> {h_eigs[K-1]:.4f}, "
          f"isotropy={res['h_isotropy_ratio']:.4f}")
    print(f"  PSD check: min_eig={h_eigs[-1]:.6f} (>=0 required)")
    assert h_eigs[-1] >= -1e-6, f"Non-PSD covariance, min_eig={h_eigs[-1]}"
    assert res["h_isotropy_ratio"] <= 1.0 + 1e-6, f"Isotropy > 1: {res['h_isotropy_ratio']}"

    # 3. Marchenko-Pastur comparison (for h)
    h_eigs_np = h_eigs.numpy()
    sigma2 = float(h_eigs.mean())
    lp = sigma2 * (1 + np.sqrt(gamma_h))**2
    lm = sigma2 * (1 - np.sqrt(gamma_h))**2
    res["mp_sigma2"] = sigma2
    res["mp_support"] = [float(lm), float(lp)]
    # Fraction of eigenvalues outside MP support
    frac_outside = float(np.mean((h_eigs_np < lm) | (h_eigs_np > lp)))
    res["h_frac_outside_mp"] = frac_outside
    print(f"  MP support: [{lm:.4f}, {lp:.4f}], sigma2={sigma2:.4f}")
    print(f"  Frac outside MP: {frac_outside:.3f}")

    # KS statistic: compare empirical CDF to MP CDF
    # Generate MP samples by inverse CDF sampling
    lambda_grid = np.linspace(lm + 1e-8, lp - 1e-8, 5000)
    mp_pdf = mp_density(lambda_grid, gamma_h, sigma2)
    mp_cdf = np.cumsum(mp_pdf) * (lambda_grid[1] - lambda_grid[0])
    mp_cdf = mp_cdf / (mp_cdf[-1] + 1e-30)
    # Sample from MP via inverse CDF
    rng = np.random.default_rng(42)
    u_samples = rng.uniform(0, 1, len(h_eigs_np))
    mp_samples = np.interp(u_samples, mp_cdf, lambda_grid)
    ks_stat, ks_pval = ks_2samp(h_eigs_np, mp_samples)
    res["ks_stat_vs_mp"] = float(ks_stat)
    res["ks_pval_vs_mp"] = float(ks_pval)
    print(f"  KS vs MP: stat={ks_stat:.4f}, p={ks_pval:.4g}")

    # 4. Projection condition numbers
    # (a) Top-K eigenvectors (from h covariance)
    h_centered = (h_data[b] - h_data[b].mean(dim=0, keepdim=True)).float()

    # Build projection matrices
    # Top-K eigenvectors
    C_h = (h_centered.T @ h_centered) / (h_centered.shape[0] - 1)
    C_h = (C_h + C_h.T) / 2
    eig_result = torch.linalg.eigh(C_h)
    # Top-K: last K eigenvectors (eigh returns ascending)
    V_topk = eig_result.eigenvectors[:, -K:]  # (3072, K)

    # (b) Top-K right singular vectors of h (= top-K eigenvectors of h^T h, same as above)
    # For h (n, d) with n >> d, V from SVD = V from eig of h^T h
    V_svd = V_topk  # identical for full covariance

    # (c) Random Gaussian projections (QR orthonormalized)
    torch.manual_seed(42)

    def proj_condition_number(P, H):
        """Condition number of P^T @ H^T: P is (d, k), H is (n, d)."""
        PH = (P.T @ H.T)  # (k, n)
        s = torch.linalg.svdvals(PH)
        return float(s[0] / (s[-1] + 1e-30))

    cond_topk = proj_condition_number(V_topk, h_centered)
    res["cond_topk"] = cond_topk

    cond_random_list = []
    for seed_i in range(N_PROJ_DRAWS):
        torch.manual_seed(seed_i * 7 + 13)
        G = torch.randn(3072, K)
        Q, _ = torch.linalg.qr(G)  # (3072, K) orthonormal
        cond_rand = proj_condition_number(Q, h_centered)
        cond_random_list.append(cond_rand)

    res["cond_random_mean"] = float(np.mean(cond_random_list))
    res["cond_random_std"] = float(np.std(cond_random_list))
    print(f"  Cond(topk): {cond_topk:.2f}")
    print(f"  Cond(random): {res['cond_random_mean']:.2f} +/- {res['cond_random_std']:.2f}")

    # 5. JL distortion
    rng2 = np.random.default_rng(123)
    pair_idx = rng2.integers(0, N_TOKENS, size=(N_JL_PAIRS, 2))
    # Ensure no self-pairs
    same = pair_idx[:, 0] == pair_idx[:, 1]
    pair_idx[same, 1] = (pair_idx[same, 1] + 1) % N_TOKENS

    hi = h_centered[pair_idx[:, 0]]  # (N_JL_PAIRS, 3072)
    hj = h_centered[pair_idx[:, 1]]
    diff = hi - hj
    d_orig = torch.norm(diff, dim=1)  # (N_JL_PAIRS,)

    def jl_distortion(P, diff_vecs, d_orig):
        """Max, 95th, mean relative distance distortion for projection P (3072, k)."""
        proj_diff = diff_vecs @ P  # (N_JL_PAIRS, K)
        # Scale by sqrt(d/k) to preserve distances in expectation
        d_proj = torch.norm(proj_diff, dim=1) * np.sqrt(3072 / K)
        dist_ratio = d_proj / (d_orig + 1e-30)
        distortion = torch.abs(dist_ratio - 1).numpy()
        return float(distortion.max()), float(np.percentile(distortion, 95)), float(distortion.mean())

    # JL bound for k=512, n=16384
    eps_jl = np.sqrt(8 * np.log(N_TOKENS) / K)
    res["eps_jl_bound"] = float(eps_jl)

    # topk
    d_topk = jl_distortion(V_topk, diff, d_orig)
    res["jl_topk"] = {"max": d_topk[0], "p95": d_topk[1], "mean": d_topk[2]}

    # random (mean over draws)
    rand_max_list, rand_p95_list, rand_mean_list = [], [], []
    for seed_i in range(N_PROJ_DRAWS):
        torch.manual_seed(seed_i * 7 + 13)
        G = torch.randn(3072, K)
        Q, _ = torch.linalg.qr(G)
        dm, dp, dmn = jl_distortion(Q, diff, d_orig)
        rand_max_list.append(dm)
        rand_p95_list.append(dp)
        rand_mean_list.append(dmn)
    res["jl_random"] = {
        "max_mean": float(np.mean(rand_max_list)),
        "max_std": float(np.std(rand_max_list)),
        "p95_mean": float(np.mean(rand_p95_list)),
        "mean_mean": float(np.mean(rand_mean_list)),
    }

    print(f"  JL bound eps={eps_jl:.3f}")
    print(f"  JL topk:   max={d_topk[0]:.3f}, p95={d_topk[1]:.3f}, mean={d_topk[2]:.3f}")
    print(f"  JL random: max={res['jl_random']['max_mean']:.3f}±{res['jl_random']['max_std']:.3f}, "
          f"p95={res['jl_random']['p95_mean']:.3f}, mean={res['jl_random']['mean_mean']:.3f}")

    results[str(b)] = res

# ── Verdict ───────────────────────────────────────────────────────────────────
iso_ratios = [results[str(b)]["h_isotropy_ratio"] for b in BLOCKS]
n_above_01 = sum(1 for r in iso_ratios if r > 0.1)
n_below_001 = sum(1 for r in iso_ratios if r < 0.01)

print("\n=== Verdict ===")
print(f"h isotropy ratios: {[f'{r:.4f}' for r in iso_ratios]}")
print(f"Blocks with ratio > 0.1: {n_above_01}/4")
print(f"Blocks with ratio < 0.01: {n_below_001}/4")

if n_above_01 >= 3:
    verdict = "CONFIRMED"
    verdict_str = "near-spherical (isotropy > 0.1 for >= 3/4 blocks); JL explains lstsq dominance"
elif n_below_001 >= 3:
    verdict = "REFUTED"
    verdict_str = "steep spectrum (isotropy < 0.01 for >= 3/4 blocks)"
else:
    verdict = "INCONCLUSIVE"
    verdict_str = "moderate isotropy (0.01-0.1 range)"

print(f"Verdict: {verdict} — {verdict_str}")
results["verdict"] = verdict
results["verdict_detail"] = verdict_str
results["iso_ratios"] = {str(b): results[str(b)]["h_isotropy_ratio"] for b in BLOCKS}

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "cycle_086_manifold_geometry.json")
# Don't serialize full eigenvalue lists (3072 per block = too large); save summary only
results_summary = {}
for b in BLOCKS:
    r = results[str(b)]
    results_summary[str(b)] = {
        k: v for k, v in r.items() if k not in ("h_eigs", "xln_eigs")
    }
    results_summary[str(b)]["h_eigs_top10"] = results[str(b)]["h_eigs"][:10]
    results_summary[str(b)]["h_eigs_last5"] = results[str(b)]["h_eigs"][-5:]
    results_summary[str(b)]["xln_eigs_top5"] = results[str(b)]["xln_eigs"][:5]

results_summary["verdict"] = verdict
results_summary["verdict_detail"] = verdict_str
results_summary["iso_ratios"] = results["iso_ratios"]
results_summary["n_blocks_above_01"] = n_above_01
results_summary["n_blocks_below_001"] = n_below_001

with open(json_path, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nJSON saved: {json_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n=== Plotting ===")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(f"FFN Hidden Activation Eigenspectrum — GPT-2 (blocks {BLOCKS})", fontsize=11)

for plot_i, b in enumerate(BLOCKS):
    ax = axes[plot_i]
    h_eigs_np = np.array(results[str(b)]["h_eigs"])
    sigma2 = results[str(b)]["mp_sigma2"]
    lm, lp = results[str(b)]["mp_support"]

    # Normalized spectrum
    x = np.arange(1, len(h_eigs_np) + 1) / len(h_eigs_np)
    ax.semilogy(x, h_eigs_np / h_eigs_np[0], color="steelblue", lw=1.2, label="Empirical")

    # MP support overlay
    ax.axvline(K / 3072, color="red", ls="--", lw=1, label=f"k={K} (6x)")

    # Mark isotropy ratio
    ir = results[str(b)]["h_isotropy_ratio"]
    ax.semilogy([K / 3072], [h_eigs_np[K-1] / h_eigs_np[0]], "r*", ms=8)
    ax.set_title(f"Block {b}\niso={ir:.4f}, KS={results[str(b)]['ks_stat_vs_mp']:.3f}", fontsize=9)
    ax.set_xlabel("Eigenvalue rank (normalized)")
    ax.set_ylabel("Eigenvalue / max")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Panel 5: isotropy comparison (xln vs h) across blocks
ax5 = axes[4]
xln_isos = [results[str(b)]["xln_isotropy_ratio"] for b in BLOCKS]
h_isos = [results[str(b)]["h_isotropy_ratio"] for b in BLOCKS]
x_pos = np.arange(len(BLOCKS))
ax5.bar(x_pos - 0.2, xln_isos, 0.4, label="xln (768D, k=128)", color="orange", alpha=0.8)
ax5.bar(x_pos + 0.2, h_isos, 0.4, label="h (3072D, k=512)", color="steelblue", alpha=0.8)
ax5.axhline(0.1, color="red", ls="--", lw=1, label="0.1 threshold")
ax5.axhline(0.01, color="darkred", ls=":", lw=1, label="0.01 threshold")
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f"B{b}" for b in BLOCKS])
ax5.set_ylabel("Isotropy ratio (lambda_k / lambda_1)")
ax5.set_title(f"Isotropy: xln vs h\nVerdict: {verdict}", fontsize=9)
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "cycle_086_manifold_geometry.png")
plt.savefig(png_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Plot saved: {png_path}")

print("\n=== Done ===")
print(f"Verdict: {verdict}")
print(f"Isotropy ratios: {dict(zip(BLOCKS, [f'{r:.4f}' for r in h_isos]))}")
