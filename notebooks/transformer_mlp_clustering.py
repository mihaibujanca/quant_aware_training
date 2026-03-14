# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: quant-aware-training (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiment 1: Direction Clustering in GPT-2 MLP Blocks
#
# **Thesis:** Do the up-projection direction vectors in trained transformer MLPs cluster into
# significantly fewer effective directions? If so, replacing neuron clusters with compact
# splines (KANs) offers a path to >6× compression per MLP block.
#
# **What we measure:**
# - Pairwise cosine similarity vs. random baseline
# - Hierarchical clustering effective direction counts
# - PCA effective dimensionality
# - Joint input+output direction clustering
# - Within-cluster bias spread and GeLU collective response

# %% Configuration
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pathlib import Path

# Lazy import to avoid slow startup if not needed
from transformers import GPT2Model

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = Path("plots/transformer_mlp_clustering")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")

# %% Load GPT-2 and verify weight shapes
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Verify Conv1D convention for layer 0
# New transformers: layers are model.h[i], not model.transformer.h[i]
layer0 = model.h[0]
c_fc_w = layer0.mlp.c_fc.weight    # Conv1D: shape (d_model, d_ff) = (768, 3072)
c_proj_w = layer0.mlp.c_proj.weight  # Conv1D: shape (d_ff, d_model) = (3072, 768)

print(f"c_fc.weight shape:   {c_fc_w.shape}  → up-projection columns are neurons in ℝ^768")
print(f"c_proj.weight shape: {c_proj_w.shape} → down-projection rows are output directions")
print()
# In GPT-2 Conv1D: weight shape is (in_features, out_features), so
# c_fc: (768, 3072) → each COLUMN is a direction vector in ℝ^768
# c_proj: (3072, 768) → each ROW is an output direction in ℝ^768

n_layers = len(model.h)
d_model = c_fc_w.shape[0]   # 768
d_ff = c_fc_w.shape[1]      # 3072
print(f"Architecture: {n_layers} layers, d_model={d_model}, d_ff={d_ff}")


# %% Step 1: Extract and normalize direction vectors for all layers
def extract_directions(model):
    """
    Returns:
        up_dirs[layer]:   (d_ff, d_model) — normalized up-projection directions
        up_norms[layer]:  (d_ff,)         — original norms (importance proxy)
        down_dirs[layer]: (d_ff, d_model) — normalized down-projection directions
        biases[layer]:    (d_ff,)         — c_fc bias
    """
    up_dirs, up_norms, down_dirs, biases = [], [], [], []
    with torch.no_grad():
        for layer in model.h:
            # c_fc.weight: (768, 3072) → columns are neuron directions → transpose → (3072, 768)
            W_up = layer.mlp.c_fc.weight.T.float()   # (3072, 768)
            norms = W_up.norm(dim=1)                  # (3072,)
            W_up_n = F.normalize(W_up, dim=1)
            up_dirs.append(W_up_n)
            up_norms.append(norms)

            # c_proj.weight: (3072, 768) → rows are output directions
            W_down = layer.mlp.c_proj.weight.float()  # (3072, 768)
            W_down_n = F.normalize(W_down, dim=1)
            down_dirs.append(W_down_n)

            b = layer.mlp.c_fc.bias.float()
            biases.append(b)
    return up_dirs, up_norms, down_dirs, biases

up_dirs, up_norms, down_dirs, biases = extract_directions(model)

# Sanity check: first layer shapes
print(f"up_dirs[0]:   {up_dirs[0].shape}")
print(f"down_dirs[0]: {down_dirs[0].shape}")
print(f"biases[0]:    {biases[0].shape}")


# %% Step 2: Pairwise cosine similarity distributions
def cosine_sim_matrix(vecs: torch.Tensor) -> torch.Tensor:
    """(N, D) → (N, N) cosine similarity matrix."""
    return (vecs @ vecs.T).clamp(-1, 1)

# Random baseline: what does cosine similarity look like for random unit vectors in ℝ^768?
torch.manual_seed(42)
rand_vecs = F.normalize(torch.randn(d_ff, d_model), dim=1)
rand_sim = cosine_sim_matrix(rand_vecs)
rand_upper = rand_sim[torch.triu(torch.ones(d_ff, d_ff, dtype=bool), diagonal=1)]

print("Computing cosine similarity matrices for all layers...")
layer_sims = [cosine_sim_matrix(up_dirs[i]) for i in range(n_layers)]

# Summary stats per layer
print(f"\n{'Layer':>6} {'mean|cos|':>10} {'std':>8} {'frac>0.3':>10} {'frac>0.5':>10}")
print("-" * 50)
for i, sim in enumerate(layer_sims):
    upper = sim[torch.triu(torch.ones(d_ff, d_ff, dtype=bool), diagonal=1)]
    abs_u = upper.abs()
    print(f"{i:>6} {abs_u.mean():.4f}     {abs_u.std():.4f}   {(abs_u>0.3).float().mean():.4f}     {(abs_u>0.5).float().mean():.4f}")

abs_rand = rand_upper.abs()
print(f"{'RAND':>6} {abs_rand.mean():.4f}     {abs_rand.std():.4f}   {(abs_rand>0.3).float().mean():.4f}     {(abs_rand>0.5).float().mean():.4f}")

# %% Plot: cosine similarity distributions (all layers + random)
fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True)
axes = axes.flatten()

for i, (sim, ax) in enumerate(zip(layer_sims, axes)):
    upper = sim[torch.triu(torch.ones(d_ff, d_ff, dtype=bool), diagonal=1)].numpy()
    ax.hist(upper, bins=100, density=True, alpha=0.7, color=f"C{i}", label=f"Layer {i}")
    ax.hist(rand_upper.numpy(), bins=100, density=True, alpha=0.3, color="gray", label="Random")
    ax.set_title(f"Layer {i}", fontsize=10)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("|cos sim|")
    if i % 4 == 0:
        ax.set_ylabel("Density")

fig.suptitle("Pairwise Cosine Similarity Distributions (up-projection directions)\nvs. Random Baseline (gray)", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cosine_sim_distributions.png", dpi=150)
plt.show()
print("Saved cosine_sim_distributions.png")


# %% Step 3a: PCA effective dimensionality per layer
def pca_effective_dim(vecs: torch.Tensor, thresholds=(0.90, 0.95, 0.99)):
    """
    Returns number of PCs needed to capture each variance threshold.
    vecs: (N, D) torch tensor of direction vectors.
    """
    X = vecs.numpy()
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims = {}
    for t in thresholds:
        dims[t] = int(np.searchsorted(cumvar, t)) + 1
    return dims, pca.explained_variance_ratio_

print(f"\nPCA Effective Dimensionality of up-projection directions:")
print(f"{'Layer':>6} {'dim@90%':>9} {'dim@95%':>9} {'dim@99%':>9} {'compress@95%':>14}")
print("-" * 55)
pca_results = []
for i, vecs in enumerate(up_dirs):
    dims, var_ratio = pca_effective_dim(vecs)
    pca_results.append((dims, var_ratio))
    r = dims[0.95] / d_ff
    print(f"{i:>6} {dims[0.90]:>9} {dims[0.95]:>9} {dims[0.99]:>9} {r:>13.1%}")

# Random baseline PCA
dims_rand, _ = pca_effective_dim(rand_vecs)
print(f"{'RAND':>6} {dims_rand[0.90]:>9} {dims_rand[0.95]:>9} {dims_rand[0.99]:>9} {dims_rand[0.95]/d_ff:>13.1%}")


# %% Step 3b: Hierarchical clustering — effective direction counts at multiple thresholds
SIMILARITY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]  # cosine similarity thresholds

def count_clusters_at_thresholds(vecs: torch.Tensor, thresholds):
    """
    Returns number of clusters when merging neurons with cosine similarity > threshold.
    Uses average-linkage hierarchical clustering on cosine distance.
    """
    cos_dist = (1.0 - (vecs @ vecs.T).clamp(-1, 1)).numpy()
    np.fill_diagonal(cos_dist, 0)
    cos_dist = np.clip(cos_dist, 0, None)
    dist_vec = squareform(cos_dist, checks=False)
    Z = linkage(dist_vec, method="average")
    counts = {}
    for t in thresholds:
        labels = fcluster(Z, t=1.0 - t, criterion="distance")
        counts[t] = int(labels.max())
    return counts, Z

print("\nHierarchical clustering — neuron count after merging similar directions:")
print(f"{'Layer':>6}", end="")
for t in SIMILARITY_THRESHOLDS:
    print(f"  cos>{t}", end="")
print(f"  {'compress@0.7':>13}")
print("-" * 65)

cluster_results = []
for i, vecs in enumerate(up_dirs):
    counts, Z = count_clusters_at_thresholds(vecs, SIMILARITY_THRESHOLDS)
    cluster_results.append((counts, Z))
    compress = counts[0.7] / d_ff
    print(f"{i:>6}", end="")
    for t in SIMILARITY_THRESHOLDS:
        print(f"  {counts[t]:>6}", end="")
    print(f"  {compress:>12.1%}")


# %% Plot: cluster counts vs threshold for all layers
fig, ax = plt.subplots(figsize=(10, 5))
for i, (counts, _) in enumerate(cluster_results):
    vals = [counts[t] for t in SIMILARITY_THRESHOLDS]
    ax.plot(SIMILARITY_THRESHOLDS, vals, marker="o", label=f"L{i}", alpha=0.8)
ax.axhline(d_ff, color="black", linestyle="--", linewidth=1, label=f"Baseline ({d_ff} neurons)")
ax.set_xlabel("Cosine Similarity Threshold (merge if similarity > threshold)")
ax.set_ylabel("Number of Clusters (effective directions)")
ax.set_title("Effective Direction Counts vs. Merge Threshold\n(lower = more compressible)")
ax.legend(ncol=3, fontsize=8, loc="upper left")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cluster_counts_vs_threshold.png", dpi=150)
plt.show()


# %% Step 4: Joint input-output direction analysis
# Two neurons are functionally equivalent only if BOTH up AND down directions are similar.
def joint_cluster_counts(up_vecs: torch.Tensor, down_vecs: torch.Tensor,
                          thresholds, alpha=0.5):
    """
    Combined distance: alpha * cos_dist(up) + (1-alpha) * cos_dist(down).
    """
    up_cos = (up_vecs @ up_vecs.T).clamp(-1, 1).numpy()
    down_cos = (down_vecs @ down_vecs.T).clamp(-1, 1).numpy()
    combined_sim = alpha * up_cos + (1 - alpha) * down_cos
    combined_dist = 1.0 - combined_sim
    np.fill_diagonal(combined_dist, 0)
    combined_dist = np.clip(combined_dist, 0, None)
    dist_vec = squareform(combined_dist, checks=False)
    Z = linkage(dist_vec, method="average")
    counts = {}
    for t in thresholds:
        labels = fcluster(Z, t=1.0 - t, criterion="distance")
        counts[t] = int(labels.max())
    return counts

print("\nJoint (input+output) clustering vs. input-only clustering:")
print(f"{'Layer':>6}  {'input-only@0.7':>16}  {'joint@0.7':>11}  {'ratio':>6}")
print("-" * 48)
for i in range(n_layers):
    input_only = cluster_results[i][0][0.7]
    joint = joint_cluster_counts(up_dirs[i], down_dirs[i], [0.7])[0.7]
    print(f"{i:>6}  {input_only:>16}  {joint:>11}  {joint/input_only:>5.2f}x")


# %% Step 5: Within-cluster bias structure — visualize GeLU responses for a selected layer
# Pick the layer with strongest clustering (fewest clusters at 0.7 threshold)
best_layer = min(range(n_layers), key=lambda i: cluster_results[i][0][0.7])
print(f"\nMost compressible layer: {best_layer} ({cluster_results[best_layer][0][0.7]} clusters at cos>0.7)")

def analyze_cluster_bias_spread(layer_idx: int, n_clusters_to_plot: int = 4):
    """
    For each cluster, plot the combined GeLU response and fit a polynomial approximation.
    """
    vecs = up_dirs[layer_idx]
    dv = down_dirs[layer_idx]
    b = biases[layer_idx]

    # Get cluster assignments at threshold 0.7
    _, Z = cluster_results[layer_idx]
    labels = fcluster(Z, t=1.0 - 0.7, criterion="distance")
    n_clusters = labels.max()

    # Sort clusters by size
    cluster_sizes = np.bincount(labels)[1:]  # index from 1
    top_clusters = np.argsort(cluster_sizes)[::-1][:n_clusters_to_plot] + 1

    t_vals = torch.linspace(-4, 4, 500)

    fig, axes = plt.subplots(1, n_clusters_to_plot, figsize=(16, 4))
    for ax, cid in zip(axes, top_clusters):
        mask = labels == cid
        cluster_biases = b[mask]
        cluster_down_norms = dv[mask].norm(dim=1)  # output scale per neuron

        # Combined GeLU response: f(t) = Σ_j w_j * GeLU(t + b_j)
        # w_j = ||down_j|| (output magnitude)
        w = cluster_down_norms.float()
        w = w / w.sum()  # normalize
        responses = []
        for bj, wj in zip(cluster_biases, w):
            resp = wj * F.gelu(t_vals + bj)
            responses.append(resp)
        combined = torch.stack(responses).sum(0)

        # Bias spread
        b_std = cluster_biases.std().item()
        b_range = (cluster_biases.max() - cluster_biases.min()).item()

        ax.plot(t_vals.numpy(), combined.detach().numpy(), "b-", linewidth=2, label="Combined GeLU")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"Cluster {cid} (size={mask.sum()})\nbias range={b_range:.2f}, std={b_std:.2f}")
        ax.set_xlabel("Input projection value t")
        ax.set_ylabel("Collective response")
        ax.legend(fontsize=8)

    fig.suptitle(f"Layer {layer_idx}: Within-cluster GeLU collective responses\n(each cluster = neurons with similar input directions)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"within_cluster_gelu_layer{layer_idx}.png", dpi=150)
    plt.show()

analyze_cluster_bias_spread(best_layer)


# %% Step 6: Layer-wise summary plot
# Summarize all findings in a single figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

layers = list(range(n_layers))

# Panel 1: PCA effective dim at 95%
dims_95 = [pca_results[i][0][0.95] for i in layers]
axes[0, 0].bar(layers, dims_95, color="steelblue")
axes[0, 0].axhline(d_ff, color="red", linestyle="--", label=f"Full d_ff={d_ff}")
axes[0, 0].set_title("PCA Effective Dimensionality (95% variance)")
axes[0, 0].set_xlabel("Layer")
axes[0, 0].set_ylabel("Num. principal components")
axes[0, 0].legend()

# Panel 2: Cluster counts at 0.7 threshold
clusters_07 = [cluster_results[i][0][0.7] for i in layers]
axes[0, 1].bar(layers, clusters_07, color="coral")
axes[0, 1].axhline(d_ff, color="red", linestyle="--", label=f"Baseline ({d_ff})")
axes[0, 1].set_title("Effective Direction Count (cos sim > 0.7)")
axes[0, 1].set_xlabel("Layer")
axes[0, 1].set_ylabel("Number of clusters")
axes[0, 1].legend()

# Panel 3: Compression ratio at 0.7
compress_ratios = [c / d_ff for c in clusters_07]
axes[1, 0].bar(layers, compress_ratios, color="mediumseagreen")
axes[1, 0].axhline(1.0, color="red", linestyle="--")
axes[1, 0].axhline(0.5, color="orange", linestyle=":", label="50% (2× compression)")
axes[1, 0].axhline(1/6, color="green", linestyle=":", label="17% (6× compression)")
axes[1, 0].set_title("Compression Ratio (effective dirs / total neurons)")
axes[1, 0].set_xlabel("Layer")
axes[1, 0].set_ylabel("Ratio (lower = more compressible)")
axes[1, 0].legend()

# Panel 4: Mean |cos sim| per layer vs random baseline
mean_abs_cos = []
for i, sim in enumerate(layer_sims):
    upper = sim[torch.triu(torch.ones(d_ff, d_ff, dtype=bool), diagonal=1)]
    mean_abs_cos.append(upper.abs().mean().item())
rand_mean = rand_upper.abs().mean().item()

axes[1, 1].plot(layers, mean_abs_cos, "o-", color="purple", label="GPT-2 layers")
axes[1, 1].axhline(rand_mean, color="gray", linestyle="--", label=f"Random baseline ({rand_mean:.4f})")
axes[1, 1].set_title("Mean |Cosine Similarity| Between Direction Pairs")
axes[1, 1].set_xlabel("Layer")
axes[1, 1].set_ylabel("Mean |cos sim|")
axes[1, 1].legend()

fig.suptitle("GPT-2 MLP Direction Clustering — Layer-wise Summary", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_wise_summary.png", dpi=150)
plt.show()


# %% Final summary table
print("\n" + "="*80)
print("SUMMARY TABLE: Direction Clustering in GPT-2 MLP Blocks")
print("="*80)
print(f"{'Layer':>6} {'PCA@95%':>9} {'PCA%':>7} {'Clusters@0.7':>14} {'Compress':>10} {'Signal':>8}")
print("-"*60)
for i in range(n_layers):
    pca95 = pca_results[i][0][0.95]
    pca_pct = pca95 / d_ff
    c07 = cluster_results[i][0][0.7]
    compress = c07 / d_ff
    if compress < 1/6:
        signal = "STRONG"
    elif compress < 0.5:
        signal = "MODERATE"
    else:
        signal = "WEAK"
    print(f"{i:>6} {pca95:>9} {pca_pct:>6.1%} {c07:>14} {compress:>9.1%} {signal:>8}")

print()
best_compress = min(compress_ratios)
avg_compress = np.mean(compress_ratios)
print(f"Best compression (single layer): {best_compress:.1%} = {1/best_compress:.1f}× parameter reduction")
print(f"Average compression across layers: {avg_compress:.1%} = {1/avg_compress:.1f}× average")
print()

# Decision
if avg_compress < 1/6:
    verdict = "STRONG POSITIVE — proceed to Experiment 2 (KAN replacement)"
elif avg_compress < 0.5:
    verdict = "MODERATE SIGNAL — investigate further before Experiment 2"
else:
    verdict = "NEGATIVE — scaffolding hypothesis not supported for GPT-2 MLPs"
print(f"Decision: {verdict}")
print("="*80)
