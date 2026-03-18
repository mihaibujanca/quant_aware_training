"""Cycle 014: RTG spectral coarsening on GPT-2 FFN blocks.

For each of 4 selected GPT-2 FFN blocks, compute per-token Jacobians using the
analytical factored form J(x) = W_out @ diag(σ'(z(x))) @ W_in. Build a k-NN
graph (k=10) on flattened Jacobians, cluster via spectral embedding + k-means,
and compare reconstruction error to plain k-means baseline at matched K values.

Hypothesis (item 25): spectral coarsening reduces reconstruction error by ≥ 15%
relative to k-means at matched K for a majority of blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _activation_derivative,
    _extract_ffn_handles,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
)

RESULTS_DIR = Path("autoresearch/results")
SELECTED_BLOCKS = [0, 2, 5, 11]
K_VALUES = [2, 4, 8, 16]
KNN_K = 10
N_TOKENS = 128


def collect_preactivations_and_inputs(
    handle, teacher, corpus, device, n_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect (h_in, preact z, h_out) for n_tokens tokens from one block.

    Returns:
        h_in: [N, 768] input to the FFN block
        z:    [N, 3072] preactivations (before activation)
        h_out: [N, 768] output of the FFN block
    """
    h_in_chunks: list[torch.Tensor] = []
    z_chunks: list[torch.Tensor] = []
    h_out_chunks: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):
        x = inputs[0].detach()  # [B, T, 768]
        z_val = module.c_fc(x).detach()  # [B, T, 3072]
        y_val = output.detach()  # [B, T, 768]
        flat_x = x.reshape(-1, x.shape[-1])
        flat_z = z_val.reshape(-1, z_val.shape[-1])
        flat_y = y_val.reshape(-1, y_val.shape[-1])

        remaining = n_tokens - sum(c.shape[0] for c in h_in_chunks)
        if remaining > 0:
            take = min(remaining, flat_x.shape[0])
            h_in_chunks.append(flat_x[:take].cpu())
            z_chunks.append(flat_z[:take].cpu())
            h_out_chunks.append(flat_y[:take].cpu())

    hook = handle.mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for start in range(0, corpus.probe_holdout.shape[0], 4):
                batch = corpus.probe_holdout[start : start + 4].to(device)
                teacher.model(batch[:, :-1])
                if sum(c.shape[0] for c in h_in_chunks) >= n_tokens:
                    break
    finally:
        hook.remove()

    h_in = torch.cat(h_in_chunks, dim=0).float()[:n_tokens]
    z = torch.cat(z_chunks, dim=0).float()[:n_tokens]
    h_out = torch.cat(h_out_chunks, dim=0).float()[:n_tokens]
    return h_in, z, h_out


def compute_jacobians(
    z: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    activation_kind: str,
) -> torch.Tensor:
    """Compute per-token Jacobians using the analytical factored form.

    J(x) = W_out @ diag(σ'(z(x))) @ W_in

    Args:
        z:    [N, H] preactivations
        w_in: [H, D] weight matrix (from D → H)
        w_out: [D, H] weight matrix (from H → D)

    Returns:
        J: [N, D, D] Jacobian matrices
    """
    N = z.shape[0]
    # σ'(z) for each token: [N, H]
    deriv = _activation_derivative(z, activation_kind)  # [N, H]

    # J(x) = (W_out * d(x).unsqueeze(0)) @ W_in
    # w_out: [D, H], deriv[i]: [H] → w_out * deriv[i] is [D, H]
    # then @ w_in is [D, D]
    # Vectorise: [N, D, H] @ [H, D] → [N, D, D]
    scaled_wout = w_out.unsqueeze(0) * deriv.unsqueeze(1)  # [N, D, H]
    J = scaled_wout @ w_in  # [N, D, D]
    return J


def build_knn_graph(flat_J: torch.Tensor, k: int) -> torch.Tensor:
    """Build symmetric k-NN adjacency matrix from flattened Jacobians.

    Args:
        flat_J: [N, D*D] flattened Jacobians
        k: number of nearest neighbours

    Returns:
        A: [N, N] symmetric adjacency matrix (binary)
    """
    N = flat_J.shape[0]
    # Pairwise L2 distances
    dists = torch.cdist(flat_J, flat_J)  # [N, N]
    # k+1 because self is included at distance 0
    _, indices = torch.topk(dists, k=k + 1, largest=False, dim=1)  # [N, k+1]

    # Build adjacency (symmetric: edge if i in j's kNN OR j in i's kNN)
    A = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx].item()
            A[i, j] = 1.0
            A[j, i] = 1.0

    return A


def spectral_cluster(A: torch.Tensor, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Spectral clustering using graph Laplacian eigenvectors.

    Args:
        A: [N, N] adjacency matrix
        n_clusters: number of clusters K

    Returns:
        labels: [N] cluster assignments
    """
    N = A.shape[0]
    # Unnormalized Laplacian L = D - A
    deg = A.sum(dim=1)  # [N]
    D_mat = torch.diag(deg)
    L = D_mat - A  # [N, N]

    L_np = L.numpy().astype(np.float64)

    # Use numpy eigh (N=128 is small — dense is fine)
    eigenvalues, eigenvectors = np.linalg.eigh(L_np)

    # Take K smallest eigenvectors (indices 0..K-1)
    embedding = eigenvectors[:, :n_clusters]  # [N, K]

    # k-means on eigenvector embedding
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)
    return labels


def kmeans_cluster(flat_J: torch.Tensor, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Baseline: k-means directly on flattened Jacobians.

    Args:
        flat_J: [N, D*D]
        n_clusters: K

    Returns:
        labels: [N]
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(flat_J.numpy())
    return labels


def reconstruction_error(J: torch.Tensor, labels: np.ndarray) -> float:
    """Compute mean normalised Frobenius reconstruction error.

    For each cluster, the approximation is the cluster-mean Jacobian.
    Error = mean_x ||J(x) - J̄_{c(x)}||_F / ||J(x)||_F

    Args:
        J: [N, D, D]
        labels: [N]

    Returns:
        scalar error (normalised)
    """
    N = J.shape[0]
    n_clusters = int(labels.max()) + 1

    # Compute cluster means
    cluster_means = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            cluster_means.append(torch.zeros_like(J[0]))
        else:
            cluster_means.append(J[mask].mean(dim=0))

    errors = []
    for i in range(N):
        c = labels[i]
        diff = J[i] - cluster_means[c]
        numer = diff.norm().item()
        denom = J[i].norm().item()
        if denom < 1e-12:
            errors.append(0.0)
        else:
            errors.append(numer / denom)

    return float(np.mean(errors))


def check_graph_connectivity(A: torch.Tensor) -> int:
    """Return number of connected components via BFS."""
    N = A.shape[0]
    visited = [False] * N
    n_components = 0
    adj = A.numpy()
    for start in range(N):
        if visited[start]:
            continue
        n_components += 1
        queue = [start]
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            neighbors = np.where(adj[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(int(nb))
    return n_components


def main():
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load teacher and corpus
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device=str(device))
    )
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path="data/shakespeare.txt",
            sequence_length=128,
            probe_train_sequences=64,
            probe_holdout_sequences=32,
            eval_val_sequences=32,
            dataset_name="proxy_shakespeare",
        )
    )

    handles = _extract_ffn_handles(teacher.model, teacher.activation_kind)
    teacher.model.eval()

    # Filter to selected blocks
    selected_handles = [h for h in handles if h.layer_id in SELECTED_BLOCKS]
    print(f"Selected blocks: {[h.layer_id for h in selected_handles]}")

    results = {}

    for handle in selected_handles:
        layer_id = handle.layer_id
        print(f"\n=== Block {layer_id} ===")

        # Collect data
        h_in, z, h_out = collect_preactivations_and_inputs(
            handle, teacher, corpus, device, N_TOKENS
        )
        N = h_in.shape[0]
        print(f"  Collected {N} tokens")

        # Weights
        w_in = _standard_weight(handle.c_fc).float()   # [H, D]
        w_out = _standard_weight(handle.c_proj).float()  # [D, H]

        # Compute Jacobians: J(x) = W_out @ diag(σ'(z)) @ W_in
        J = compute_jacobians(z, w_in, w_out, handle.activation_kind)  # [N, D, D]
        print(f"  Jacobians shape: {J.shape}")

        # Flatten for k-NN
        D = J.shape[1]
        flat_J = J.reshape(N, -1)  # [N, D*D]
        print(f"  Flattened Jacobians: {flat_J.shape} ({flat_J.numel() * 4 / 1e6:.1f} MB)")

        # Build k-NN graph
        print(f"  Building k-NN graph (k={KNN_K})...")
        A = build_knn_graph(flat_J, KNN_K)
        n_components = check_graph_connectivity(A)
        print(f"  Graph connected components: {n_components}")

        layer_results = {
            "layer_id": layer_id,
            "n_tokens": N,
            "n_components": n_components,
            "K_values": K_VALUES,
            "spectral": {},
            "kmeans": {},
        }

        for K in K_VALUES:
            print(f"  K={K}...", end=" ", flush=True)

            # Spectral clustering
            spec_labels = spectral_cluster(A, K)
            spec_err = reconstruction_error(J, spec_labels)

            # K-means on raw Jacobians
            km_labels = kmeans_cluster(flat_J, K)
            km_err = reconstruction_error(J, km_labels)

            rel_improvement = (km_err - spec_err) / km_err if km_err > 1e-12 else 0.0

            layer_results["spectral"][K] = spec_err
            layer_results["kmeans"][K] = km_err
            layer_results[f"rel_improvement_K{K}"] = rel_improvement

            print(f"spectral={spec_err:.4f} | kmeans={km_err:.4f} | rel_imp={rel_improvement:.3f}")

        results[layer_id] = layer_results

    # ── Evaluate hypothesis ──
    print("\n=== Hypothesis Evaluation ===")
    blocks_confirmed = 0
    blocks_total = len(SELECTED_BLOCKS)

    for layer_id in SELECTED_BLOCKS:
        if layer_id not in results:
            continue
        r = results[layer_id]
        # Check majority of K values (>=3/4) have ≥15% relative improvement
        improvements_above_threshold = sum(
            1 for K in K_VALUES
            if r.get(f"rel_improvement_K{K}", 0.0) >= 0.15
        )
        block_confirmed = improvements_above_threshold > len(K_VALUES) // 2
        print(
            f"  Block {layer_id}: {improvements_above_threshold}/{len(K_VALUES)} K-values "
            f"with ≥15% improvement → {'CONFIRMED' if block_confirmed else 'NOT confirmed'}"
        )
        if block_confirmed:
            blocks_confirmed += 1

    majority = blocks_total // 2 + 1
    if blocks_confirmed >= majority:
        verdict = "CONFIRMED"
    else:
        # Check if benefit only at small K (inconclusive)
        small_k_only = all(
            results.get(lid, {}).get(f"rel_improvement_K{K}", 0.0) < 0.15
            for lid in SELECTED_BLOCKS if lid in results
            for K in [8, 16]
        ) and any(
            results.get(lid, {}).get(f"rel_improvement_K{K}", 0.0) >= 0.15
            for lid in SELECTED_BLOCKS if lid in results
            for K in [2, 4]
        )
        verdict = "INCONCLUSIVE" if small_k_only else "REFUTED"

    print(f"\nVerdict: {verdict} ({blocks_confirmed}/{blocks_total} blocks confirmed)")

    # Save results
    output = {
        "verdict": verdict,
        "blocks_confirmed": blocks_confirmed,
        "blocks_total": blocks_total,
        "selected_blocks": SELECTED_BLOCKS,
        "K_values": K_VALUES,
        "knn_k": KNN_K,
        "n_tokens": N_TOKENS,
        "results": {str(k): v for k, v in results.items()},
    }

    json_path = RESULTS_DIR / "cycle_014_rtg_spectral_gpt2.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # ── Plot ──
    fig, axes = plt.subplots(1, len(SELECTED_BLOCKS), figsize=(14, 4), sharey=False)
    if len(SELECTED_BLOCKS) == 1:
        axes = [axes]

    for ax, layer_id in zip(axes, SELECTED_BLOCKS):
        if layer_id not in results:
            ax.set_title(f"Block {layer_id}\n(missing)")
            continue
        r = results[layer_id]
        spec_errs = [r["spectral"][K] for K in K_VALUES]
        km_errs = [r["kmeans"][K] for K in K_VALUES]
        ax.plot(K_VALUES, km_errs, "o-", color="steelblue", label="k-means")
        ax.plot(K_VALUES, spec_errs, "s--", color="darkorange", label="spectral")
        ax.set_title(f"Block {layer_id}")
        ax.set_xlabel("K (clusters)")
        ax.set_ylabel("Reconstruction error")
        ax.legend(fontsize=8)
        ax.set_xticks(K_VALUES)

    fig.suptitle(f"RTG Spectral vs k-means — GPT-2 FFN blocks\nVerdict: {verdict}", fontsize=11)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_014_rtg_spectral_gpt2.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    return verdict, results


if __name__ == "__main__":
    main()
