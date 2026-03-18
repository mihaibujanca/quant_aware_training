"""
Cycle 027: Centroid and Jacobian affinity as merge signals — GPT-2 FFN

For each of 12 FFN blocks, sample 200 random channel pairs from the 3072
hidden channels and compute three similarity signals:

  (a) Weight cosine: cos(c_proj[:,i], c_proj[:,j])  [data-free]
  (b) Centroid similarity: cos(act_timeseries_i, act_timeseries_j)  [data-conditioned]
  (c) Jacobian-channel similarity: cos(V_i, V_j) where V_i = mean_t(c_proj[:,i] * gelu'(z[:,i]))

Then measure the actual merge cost (MSE increase when pair is merged) and compute
Spearman(signal, merge_cost) per block.

GPT-2 Conv1D convention: weight shapes are transposed vs nn.Linear.
  c_fc.weight:  (768, 3072)   ->  z = x @ c_fc.weight + c_fc.bias   [768 -> 3072]
  c_proj.weight: (3072, 768)  ->  out = a @ c_proj.weight + c_proj.bias  [3072 -> 768]

So W_down_cols (contribution of channel i to output) = c_proj.weight[i, :]  shape (768,)
And W_up_rows (how input maps to channel i) = c_fc.weight[:, i]  shape (768,)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
N_SEQUENCES = 2       # 2 x 512 = 1024 probe tokens
SEQ_LEN = 512
N_PAIRS = 200         # channel pairs to sample per block
N_HIDDEN = 3072       # FFN hidden dim
SEED = 42
DEVICE = "cpu"

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
t0 = time.time()
teacher = load_nanochat_teacher(TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device=DEVICE,
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
    text_path=str(DATA_PATH),
    sequence_length=SEQ_LEN,
    probe_train_sequences=N_SEQUENCES,
    probe_holdout_sequences=1,
    teacher=teacher,
))
probe_tokens = corpus.probe_train  # (N_SEQUENCES, SEQ_LEN)
print(f"  probe tokens: {probe_tokens.shape}")

# Flatten to (T, ) tokens for a single long forward pass (batched by sequence)
# We'll run each sequence separately and collect FFN internals via hooks.

# ---------------------------------------------------------------------------
# GELU derivative (analytic)
# ---------------------------------------------------------------------------
def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Element-wise derivative of GELU (the tanh approximation used by GPT-2)."""
    # GPT-2 uses: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    import math
    k = math.sqrt(2.0 / math.pi)
    t = torch.tanh(k * (z + 0.044715 * z ** 3))
    dt = 1.0 - t ** 2  # sech^2
    inner_deriv = k * (1.0 + 3 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * dt * inner_deriv

# ---------------------------------------------------------------------------
# Collect FFN activations for all blocks
# ---------------------------------------------------------------------------
print("Collecting FFN activations via hooks...")

# Storage: block_idx -> dict
ffn_data: dict[int, dict] = {}
for b in range(12):
    ffn_data[b] = {"z": [], "h_in": []}  # pre-GELU and block input

hooks = []

def make_fc_hook(block_idx):
    """Hook on c_fc output to capture pre-GELU activations z."""
    def hook(module, input, output):
        # output is (batch, seq, 3072)
        # input[0] is (batch, seq, 768)
        ffn_data[block_idx]["z"].append(output.detach().cpu())    # pre-GELU: z
        ffn_data[block_idx]["h_in"].append(input[0].detach().cpu())  # block input h
    return hook

for b_idx in range(12):
    h = model.transformer.h[b_idx].mlp.c_fc.register_forward_hook(make_fc_hook(b_idx))
    hooks.append(h)

with torch.no_grad():
    for seq_idx in range(N_SEQUENCES):
        inp = probe_tokens[seq_idx:seq_idx+1].to(DEVICE)
        _ = model(inp, use_cache=False)

for h in hooks:
    h.remove()

# Concatenate sequences: z shape (N_SEQUENCES * SEQ_LEN, N_HIDDEN)
for b in range(12):
    ffn_data[b]["z"] = torch.cat(ffn_data[b]["z"], dim=0).squeeze(0)  # (T, 3072) or (N_SEQ, T, 3072)
    ffn_data[b]["h_in"] = torch.cat(ffn_data[b]["h_in"], dim=0).squeeze(0)

# Handle possible (N_SEQ, T, 3072) shape
for b in range(12):
    z = ffn_data[b]["z"]
    if z.dim() == 3:
        T_total = z.shape[0] * z.shape[1]
        ffn_data[b]["z"] = z.reshape(T_total, z.shape[2])
    h_in = ffn_data[b]["h_in"]
    if h_in.dim() == 3:
        T_total = h_in.shape[0] * h_in.shape[1]
        ffn_data[b]["h_in"] = h_in.reshape(T_total, h_in.shape[2])

print(f"  z shape (block 0): {ffn_data[0]['z'].shape}")

# ---------------------------------------------------------------------------
# Main analysis per block
# ---------------------------------------------------------------------------
results = {}

print("\nAnalyzing blocks...")
for b_idx in range(12):
    t_block = time.time()
    z = ffn_data[b_idx]["z"]  # (T, 3072)
    T = z.shape[0]

    # GPT-2 Conv1D weights (transposed convention)
    mlp = model.transformer.h[b_idx].mlp
    W_fc = mlp.c_fc.weight.detach()    # (768, 3072): z = h @ W_fc + bias_fc
    W_proj = mlp.c_proj.weight.detach()  # (3072, 768): out = a @ W_proj + bias_proj
    bias_proj = mlp.c_proj.bias.detach()  # (768,)

    # Activations
    a = F.gelu(z)  # (T, 3072)  -- standard GELU approximation
    dg = gelu_grad(z)  # (T, 3072) -- GELU derivative

    # --- Sample channel pairs ---
    all_pairs_i = rng.integers(0, N_HIDDEN, size=N_PAIRS)
    all_pairs_j = rng.integers(0, N_HIDDEN, size=N_PAIRS)
    # Avoid i==j
    mask = all_pairs_i == all_pairs_j
    all_pairs_j[mask] = (all_pairs_j[mask] + 1) % N_HIDDEN
    pairs = list(zip(all_pairs_i.tolist(), all_pairs_j.tolist()))

    # --- Signal (a): Weight cosine ---
    # c_proj.weight[i, :] is the contribution of channel i to output (shape 768)
    W_down = W_proj  # (3072, 768), W_down[i] = contribution of channel i
    W_down_norm = F.normalize(W_down, dim=1)  # (3072, 768)
    cos_w = (W_down_norm[all_pairs_i] * W_down_norm[all_pairs_j]).sum(dim=1).numpy()

    # --- Signal (b): Centroid similarity ---
    # Activation time-series for each channel: a[:, i]  shape (T,)
    a_i = a[:, all_pairs_i]  # (T, N_PAIRS)
    a_j = a[:, all_pairs_j]
    a_i_norm = F.normalize(a_i, dim=0)
    a_j_norm = F.normalize(a_j, dim=0)
    cos_c = (a_i_norm * a_j_norm).sum(dim=0).numpy()

    # --- Signal (c): Jacobian-channel similarity ---
    # Per-channel Jacobian contribution: v_i(t) = W_proj[i, :] * gelu'(z[t, i])
    # V_i = (1/T) * sum_t v_i(t) = W_proj[i, :] * mean_t(gelu'(z[:, i]))
    mean_dg = dg.mean(dim=0)  # (3072,)
    V = W_proj * mean_dg.unsqueeze(1)  # (3072, 768) -- V[i] = W_proj[i] * mean_dg[i]
    V_norm = F.normalize(V, dim=1)  # (3072, 768)
    cos_j = (V_norm[all_pairs_i] * V_norm[all_pairs_j]).sum(dim=1).numpy()

    # --- Merge cost ---
    # Baseline block output: out = a @ W_proj + bias_proj  (T, 768)
    out_orig = a @ W_proj + bias_proj  # (T, 768)

    merge_costs = []
    for pi, (i, j) in enumerate(pairs):
        # Merge: replace channels i and j with their average
        # New activations: channel i gets (a[:,i]+a[:,j])/2, channel j gets 0
        # (equivalent to one merged channel at position i, channel j zeroed)
        a_mod = a.clone()
        a_mod[:, i] = (a[:, i] + a[:, j]) / 2.0
        a_mod[:, j] = 0.0
        # Adjust W_proj: new channel i = (W_proj[i] + W_proj[j]), channel j = 0
        # Since we scaled activation by 1/2 and zero'd j, the output for those channels is:
        # (a[:,i]+a[:,j])/2 * (W_proj[i]+W_proj[j]) = effectively merged
        # To preserve the energy: merged output = a_mod[:,i]*(W_proj[i]+W_proj[j])
        # Let's compute it directly:
        out_mod = a_mod @ W_proj + bias_proj
        # But we also need to account for W_proj change at channel i:
        # Correct merged output: channel i contributes (a[:,i]+a[:,j])/2 * (W_proj[i]+W_proj[j])
        # Original: channel i: a[:,i]*W_proj[i] + channel j: a[:,j]*W_proj[j]
        # Merged:   channel i: (a[:,i]+a[:,j])/2 * (W_proj[i]+W_proj[j]), j: 0
        # Difference: merged - orig for these two channels:
        # = (a[:,i]+a[:,j])/2 * (W_proj[i]+W_proj[j]) - a[:,i]*W_proj[i] - a[:,j]*W_proj[j]
        # Compute this analytically:
        ai, aj = a[:, i], a[:, j]  # (T,)
        wi, wj = W_proj[i], W_proj[j]  # (768,)
        out_merged_contrib = torch.outer((ai + aj) / 2.0, wi + wj)  # (T, 768)
        out_orig_contrib = torch.outer(ai, wi) + torch.outer(aj, wj)  # (T, 768)
        delta = out_merged_contrib - out_orig_contrib  # (T, 768)
        cost = (delta ** 2).sum().item() / T
        merge_costs.append(cost)

    merge_costs = np.array(merge_costs)

    # --- Spearman correlations ---
    # Higher signal similarity = lower merge cost (negative correlation ideally)
    # We report ρ(signal, merge_cost) -- negative = signal is a good predictor
    rho_w, p_w = spearmanr(-cos_w, merge_costs)
    rho_c, p_c = spearmanr(-cos_c, merge_costs)
    rho_j, p_j = spearmanr(-cos_j, merge_costs)

    results[b_idx] = {
        "spearman_weight_cos": float(rho_w),
        "spearman_centroid": float(rho_c),
        "spearman_jacobian": float(rho_j),
        "p_weight_cos": float(p_w),
        "p_centroid": float(p_c),
        "p_jacobian": float(p_j),
        "mean_merge_cost": float(merge_costs.mean()),
        "n_pairs": N_PAIRS,
        "cos_w_range": [float(cos_w.min()), float(cos_w.max())],
        "cos_c_range": [float(cos_c.min()), float(cos_c.max())],
        "cos_j_range": [float(cos_j.min()), float(cos_j.max())],
    }

    print(f"  Block {b_idx:2d}: ρ_weight={rho_w:+.3f}  ρ_centroid={rho_c:+.3f}  ρ_jacobian={rho_j:+.3f} "
          f"  Δ(C-W)={rho_c-rho_w:+.3f}  Δ(J-W)={rho_j-rho_w:+.3f}  "
          f"[{time.time()-t_block:.1f}s]")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
rho_w_all = np.array([results[b]["spearman_weight_cos"] for b in range(12)])
rho_c_all = np.array([results[b]["spearman_centroid"] for b in range(12)])
rho_j_all = np.array([results[b]["spearman_jacobian"] for b in range(12)])

delta_c = rho_c_all - rho_w_all
delta_j = rho_j_all - rho_w_all

n_improved_c = int((delta_c >= 0.15).sum())
n_improved_j = int((delta_j >= 0.15).sum())

summary = {
    "n_blocks": 12,
    "n_pairs_per_block": N_PAIRS,
    "mean_rho_weight_cos": float(rho_w_all.mean()),
    "mean_rho_centroid": float(rho_c_all.mean()),
    "mean_rho_jacobian": float(rho_j_all.mean()),
    "mean_delta_centroid_vs_weight": float(delta_c.mean()),
    "mean_delta_jacobian_vs_weight": float(delta_j.mean()),
    "n_blocks_centroid_improves_by_0.15": n_improved_c,
    "n_blocks_jacobian_improves_by_0.15": n_improved_j,
    "confirmed_centroid": n_improved_c >= 7,
    "confirmed_jacobian": n_improved_j >= 7,
    "per_block": results,
}

print(f"\n=== SUMMARY ===")
print(f"Mean ρ: weight={rho_w_all.mean():+.3f}  centroid={rho_c_all.mean():+.3f}  jacobian={rho_j_all.mean():+.3f}")
print(f"Δ centroid vs weight: {delta_c.mean():+.3f}  (≥0.15 in {n_improved_c}/12 blocks)")
print(f"Δ jacobian vs weight: {delta_j.mean():+.3f}  (≥0.15 in {n_improved_j}/12 blocks)")
print(f"Criterion confirmed — centroid: {summary['confirmed_centroid']}, jacobian: {summary['confirmed_jacobian']}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
out_json = OUTPUT_DIR / "cycle_027_affinity_signals.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_json}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

blocks = list(range(12))
x = np.arange(12)
width = 0.25

ax = axes[0]
ax.bar(x - width, rho_w_all, width, label="Weight cosine", color="#4477AA", alpha=0.85)
ax.bar(x,         rho_c_all, width, label="Centroid",      color="#EE7733", alpha=0.85)
ax.bar(x + width, rho_j_all, width, label="Jacobian",      color="#009988", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"B{b}" for b in blocks])
ax.set_ylabel("Spearman ρ (−signal, cost)")
ax.set_title("Cycle 027: Merge signal quality per GPT-2 FFN block\n"
             "Higher ρ = signal better predicts safe merges")
ax.legend()
ax.set_ylim(-0.3, 0.8)

ax2 = axes[1]
ax2.bar(x - width/2, delta_c, width, label="Centroid − Weight", color="#EE7733", alpha=0.85)
ax2.bar(x + width/2, delta_j, width, label="Jacobian − Weight",  color="#009988", alpha=0.85)
ax2.axhline(0.15, color="red", linewidth=1.2, linestyle="--", label="Threshold +0.15")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([f"B{b}" for b in blocks])
ax2.set_ylabel("Δρ vs weight cosine")
ax2.set_title("Improvement over weight cosine baseline")
ax2.legend()

plt.tight_layout()
out_png = OUTPUT_DIR / "cycle_027_affinity_signals.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")
