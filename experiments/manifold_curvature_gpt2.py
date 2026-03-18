"""
Cycle 020: Manifold curvature trajectory — GPT-2

For each of 13 layer boundaries (embedding + 12 block outputs), estimate the
local manifold curvature of token representations using k-NN local PCA.

Curvature proxy = fraction of variance NOT explained by top-d PCs in a k=50
NN neighbourhood of each token, where d is the estimated local intrinsic
dimension (d such that top-d PCs explain >= 95% of variance).

Average over all tokens -> scalar per layer.
Correlate curvature trajectory with per-block compression damage from cycle 005.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aleph.lm_geometric_compression import (
    load_nanochat_teacher,
    TeacherLoadSpec,
    build_probe_corpus,
    ProbeCorpusSpec,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_NEIGHBORS = 50
VARIANCE_THRESHOLD = 0.95
SEQ_LEN = 512
N_SEQUENCES = 1  # one full sequence of 512 tokens is enough

# Per-block compression damage from cycle 005 (8x compression, geometric method)
CYCLE005_DAMAGE = {
    0:  0.17684,
    1:  0.00248,
    2:  0.02447,
    3:  0.01871,
    4:  0.01853,
    5:  0.02292,
    6:  0.02162,
    7:  0.02460,
    8:  0.02909,
    9:  0.02612,
    10: 0.03068,
    11: 0.06733,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "autoresearch", "results")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
t0 = time.time()
teacher = load_nanochat_teacher(TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device="cpu",
))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Load probe corpus
# ---------------------------------------------------------------------------
print("Building probe corpus...")
corpus = build_probe_corpus(ProbeCorpusSpec(
    text_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "shakespeare.txt"),
    sequence_length=SEQ_LEN,
    probe_train_sequences=N_SEQUENCES,
    probe_holdout_sequences=1,
    teacher=teacher,
))
probe_tokens = corpus.probe_train  # shape (N_sequences, seq_len)
print(f"  probe tokens shape: {probe_tokens.shape}")

# Use first sequence
input_ids = probe_tokens[0:1]  # (1, 512)

# ---------------------------------------------------------------------------
# Hook to collect hidden states at all 13 layer boundaries
# ---------------------------------------------------------------------------
hidden_states: list[torch.Tensor] = []

def make_hook(idx):
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h: (batch, seq_len, hidden_size)
        hidden_states.append(h.detach().cpu().squeeze(0))  # (seq_len, hidden)
    return hook

hooks = []
# Embedding output: register on wpe + wte combined via the transformer block
# GPT2Model has: transformer.wte, transformer.wpe, transformer.drop, transformer.h[*], transformer.ln_f
gpt2_transformer = model.transformer

# Hook on drop (after embedding sum = wte + wpe + drop) -> this IS embedding output
hooks.append(gpt2_transformer.drop.register_forward_hook(make_hook(0)))

# Hook on each block output
for i, block in enumerate(gpt2_transformer.h):
    hooks.append(block.register_forward_hook(make_hook(i + 1)))

print("Running forward pass...")
t1 = time.time()
with torch.no_grad():
    _ = model(input_ids=input_ids)
print(f"  forward pass: {time.time()-t1:.2f}s")

for h in hooks:
    h.remove()

print(f"Collected {len(hidden_states)} hidden state tensors")
for i, h in enumerate(hidden_states):
    print(f"  layer {i}: shape={h.shape}")

# ---------------------------------------------------------------------------
# Local PCA curvature estimation
# ---------------------------------------------------------------------------
def estimate_curvature(H: torch.Tensor, k: int = K_NEIGHBORS,
                       var_threshold: float = VARIANCE_THRESHOLD) -> float:
    """
    H: (N, D) float32 tensor of token representations.
    Returns mean curvature = mean unexplained variance fraction over tokens.
    """
    N, D = H.shape
    k = min(k, N - 1)

    # Pairwise distances
    dists = torch.cdist(H, H)  # (N, N)

    curvatures = []
    for i in range(N):
        # k nearest neighbors (excluding self)
        d_i = dists[i]
        d_i[i] = float("inf")
        nn_idx = torch.topk(d_i, k, largest=False).indices  # (k,)
        neighbors = H[nn_idx]  # (k, D)

        # Center
        center = neighbors.mean(0)
        X = neighbors - center  # (k, D)

        # Covariance: (D, D)
        C = (X.T @ X) / (k - 1)

        # Eigenvalues (all non-negative for PSD matrix)
        eigvals = torch.linalg.eigvalsh(C)  # ascending order, (D,)
        eigvals = eigvals.flip(0).clamp(min=0.0)  # descending

        total_var = eigvals.sum().item()
        if total_var < 1e-12:
            curvatures.append(0.0)
            continue

        # Intrinsic dimension: smallest d such that top-d explains >= var_threshold
        cumsum = torch.cumsum(eigvals, dim=0) / total_var
        d_choices = (cumsum >= var_threshold).nonzero(as_tuple=False)
        if len(d_choices) == 0:
            d = D
        else:
            d = int(d_choices[0].item()) + 1  # 1-indexed

        explained = eigvals[:d].sum().item() / total_var
        curvature = 1.0 - explained
        curvatures.append(curvature)

    return float(np.mean(curvatures))


print("\nEstimating curvature at each layer...")
curvature_per_layer = []
for i, H in enumerate(hidden_states):
    t_layer = time.time()
    c = estimate_curvature(H, k=K_NEIGHBORS)
    elapsed = time.time() - t_layer
    print(f"  layer {i:2d}: curvature={c:.6f}  ({elapsed:.1f}s)")
    curvature_per_layer.append(c)

# ---------------------------------------------------------------------------
# Correlation with cycle 005 damage
# ---------------------------------------------------------------------------
# Layer trajectory: layer 0 = embedding, layer 1..12 = after block 0..11
# Block i output = trajectory layer i+1
block_curvatures = [curvature_per_layer[i + 1] for i in range(12)]
damage_vals = [CYCLE005_DAMAGE[i] for i in range(12)]

rho, p_val = spearmanr(block_curvatures, damage_vals)
print(f"\nSpearman rho={rho:.4f}, p={p_val:.4f}")

# Monotone in final third: layers 9..12 in trajectory = indices [9,10,11,12]
final_third = curvature_per_layer[9:]  # 4 values
monotone_decreasing = all(final_third[j] >= final_third[j + 1]
                          for j in range(len(final_third) - 1))
print(f"Final-third monotone decreasing: {monotone_decreasing}")
print(f"Final-third values: {[round(v, 6) for v in final_third]}")

# Verdict
if monotone_decreasing and rho >= 0.4:
    verdict = "confirmed"
elif rho < 0.0 or (not monotone_decreasing and abs(rho) < 0.2):
    verdict = "refuted"
elif 0.2 <= rho < 0.4:
    verdict = "inconclusive"
else:
    verdict = "refuted"

print(f"Verdict: {verdict}")

# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
results = {
    "curvature_per_layer": [round(v, 8) for v in curvature_per_layer],
    "layer_labels": ["emb"] + [f"block_{i}" for i in range(12)],
    "damage_per_block": {str(k): v for k, v in CYCLE005_DAMAGE.items()},
    "block_curvatures": [round(v, 8) for v in block_curvatures],
    "spearman_rho": round(float(rho), 6),
    "spearman_p": round(float(p_val), 6),
    "monotone_final_third": bool(monotone_decreasing),
    "final_third_values": [round(v, 8) for v in final_third],
    "verdict": verdict,
    "k_neighbors": K_NEIGHBORS,
    "variance_threshold": VARIANCE_THRESHOLD,
}
json_path = os.path.join(OUTPUT_DIR, "cycle_020_curvature.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {json_path}")

# ---------------------------------------------------------------------------
# Plot 1: Curvature trajectory
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
layer_indices = list(range(len(curvature_per_layer)))
ax.plot(layer_indices, curvature_per_layer, "o-", color="steelblue", linewidth=2,
        markersize=6)
ax.axvspan(9, 12, alpha=0.12, color="orange", label="Final third (layers 9-12)")
ax.set_xlabel("Layer boundary (0=embedding, 1–12=after block 0–11)")
ax.set_ylabel("Mean curvature (unexplained variance fraction)")
ax.set_title("GPT-2 Manifold Curvature Trajectory")
ax.set_xticks(layer_indices)
ax.set_xticklabels(["emb"] + [str(i) for i in range(12)], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
traj_path = os.path.join(OUTPUT_DIR, "cycle_020_curvature_trajectory.png")
plt.savefig(traj_path, dpi=120)
plt.close()
print(f"Saved: {traj_path}")

# ---------------------------------------------------------------------------
# Plot 2: Curvature vs compression damage (scatter)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(block_curvatures, damage_vals, color="steelblue", s=60, zorder=3)
for i in range(12):
    ax.annotate(f"b{i}", (block_curvatures[i], damage_vals[i]),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set_xlabel("Local curvature at block output")
ax.set_ylabel("Relative perplexity increase (8x compression)")
ax.set_title(f"Curvature vs Compression Damage\nSpearman ρ={rho:.3f}, p={p_val:.3f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "cycle_020_curvature_vs_damage.png")
plt.savefig(scatter_path, dpi=120)
plt.close()
print(f"Saved: {scatter_path}")

print("\nDone.")
print(f"  Verdict: {verdict}")
print(f"  Spearman rho={rho:.4f}, p={p_val:.4f}")
print(f"  Monotone final third: {monotone_decreasing}")
