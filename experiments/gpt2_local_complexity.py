"""
Cycle 048: Local complexity proxy as compression difficulty predictor — GPT-2.

For each of the 12 GPT-2 FFN blocks, compute three Jacobian-variation proxies
over a k-NN neighborhood in the FFN input space:
  P1: std of Jacobian Frobenius norms  (primary)
  P2: mean spectral norm of Jacobians
  P3: mean effective rank (nuclear norm / spectral norm)

Then correlate each proxy with cycle-005 single-block swap damage at 8x and
16x compression using one-tailed Spearman (alternative='greater').

Fallback: if OOM, subsample to 128 probes, k=5 neighbors.
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.neighbors import KDTree
from transformers import GPT2Model, GPT2Tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_PROBES = 128   # fallback: subsample to 128 probes (original 512 was OOM-slow)
K_NEIGHBORS = 5  # fallback: k=5 neighbors
DEVICE = "cpu"  # keep it safe on CPU to avoid MPS memory issues
DATA_PATH = "data/shakespeare.txt"
OUT_DIR = "autoresearch/results"
JSON_OUT = os.path.join(OUT_DIR, "cycle_048_local_complexity_gpt2.json")
PLOT_OUT = os.path.join(OUT_DIR, "cycle_048_local_complexity_gpt2.png")

# Cycle 005 damage vectors (relative perplexity increase as fraction, not %)
# hf=0.125 → 8x compression; hf=0.0625 → 16x compression
DAMAGE_8X = [
    0.176840,  # layer 0
    0.002476,  # layer 1
    0.024471,  # layer 2
    0.018710,  # layer 3
    0.018526,  # layer 4
    0.022925,  # layer 5
    0.021623,  # layer 6
    0.024599,  # layer 7
    0.029085,  # layer 8
    0.026125,  # layer 9
    0.030680,  # layer 10
    0.067325,  # layer 11
]
DAMAGE_16X = [
    0.331160,  # layer 0
    0.007408,  # layer 1
    0.011434,  # layer 2
    0.009973,  # layer 3
    0.006393,  # layer 4
    0.001226,  # layer 5
    0.010001,  # layer 6
   -0.006285,  # layer 7
    0.019531,  # layer 8
    0.016346,  # layer 9
    0.018099,  # layer 10
    0.065159,  # layer 11
]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)

# ---------------------------------------------------------------------------
# Tokenize Shakespeare, extract 512 probe tokens
# ---------------------------------------------------------------------------
print("Tokenizing...", flush=True)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
# Use a contiguous window from the middle of the text for diversity
start = len(tokens) // 4
token_ids = torch.tensor(tokens[start : start + N_PROBES], dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, 512]

print(f"Probe tokens: {token_ids.shape[1]}", flush=True)

# ---------------------------------------------------------------------------
# Collect FFN pre-activation inputs for all 12 blocks
# ---------------------------------------------------------------------------
# GPT-2 FFN: LayerNorm → c_fc (Linear 768→3072) → GELU → c_proj (Linear 3072→768)
# "pre-GELU hidden activations" = output of c_fc, before GELU
# We also need "FFN input" (input to c_fc) for the k-NN tree.

ffn_inputs = {}   # layer → [512, 768] tensor (input to FFN)
ffn_hidden = {}   # layer → [512, 3072] tensor (c_fc output before GELU)

hooks = []
for layer_idx in range(12):
    block = model.h[layer_idx]

    def make_hook(li):
        def hook_fn(module, input, output):
            # input[0] is the pre-GELU activation (c_fc output)
            ffn_hidden[li] = input[0].detach().squeeze(0)  # [512, 3072]
        return hook_fn

    def make_input_hook(li):
        def hook_fn(module, args):
            # args[0] is the input tensor to c_fc
            ffn_inputs[li] = args[0].detach().squeeze(0)  # [512, 768]
        return hook_fn

    hooks.append(block.mlp.act.register_forward_hook(make_hook(layer_idx)))
    hooks.append(block.mlp.c_fc.register_forward_pre_hook(make_input_hook(layer_idx)))

with torch.no_grad():
    model(token_ids)

for h in hooks:
    h.remove()

print(f"Collected ffn_inputs for {len(ffn_inputs)} layers, shape: {ffn_inputs[0].shape}", flush=True)
print(f"Collected ffn_hidden for {len(ffn_hidden)} layers, shape: {ffn_hidden[0].shape}", flush=True)


# ---------------------------------------------------------------------------
# Jacobian of the FFN block (c_fc → GELU → c_proj) at a single input vector
# ---------------------------------------------------------------------------
def ffn_forward(x_1d, block):
    """x_1d: [768] → [768] (output of FFN c_fc+GELU+c_proj, no residual)."""
    x = x_1d.unsqueeze(0)  # [1, 768]
    h = block.mlp.c_fc(x)  # [1, 3072]
    h = block.mlp.act(h)   # [1, 3072] GELU
    out = block.mlp.c_proj(h)  # [1, 768]
    return out.squeeze(0)  # [768]


def compute_jacobian(x_1d, block):
    """Returns Jacobian [768, 768] via torch.func.jacrev."""
    return torch.func.jacrev(ffn_forward)(x_1d, block)


# ---------------------------------------------------------------------------
# Main loop: compute proxies for each layer
# ---------------------------------------------------------------------------
def effective_rank(J):
    """nuclear_norm / spectral_norm for matrix J [768, 768]."""
    sv = torch.linalg.svdvals(J)
    spec = sv[0].item()
    nuc = sv.sum().item()
    if spec < 1e-12:
        return 0.0
    return nuc / spec


results_by_layer = {}
n_probes_actual = N_PROBES
k_actual = K_NEIGHBORS

for layer_idx in range(12):
    print(f"\nLayer {layer_idx}...", flush=True)

    X = ffn_inputs[layer_idx].float()  # [512, 768]
    block = model.h[layer_idx]

    # k-NN in FFN input space (768D, use PCA fallback if slow — but KDTree should be OK)
    X_np = X.numpy()
    tree = KDTree(X_np, leaf_size=40)
    dist, ind = tree.query(X_np, k=k_actual + 1)  # +1 because query returns self
    # ind: [512, k+1], first column is self → skip it
    ind = ind[:, 1:]  # [512, k]

    frob_norms_per_probe = []
    spec_norms_per_probe = []
    eff_ranks_per_probe = []

    for probe_idx in range(n_probes_actual):
        neighbor_idxs = ind[probe_idx]  # [k]
        frob_list = []
        spec_list = []
        erank_list = []

        for nb_idx in neighbor_idxs:
            x_nb = X[nb_idx]  # [768]
            J = compute_jacobian(x_nb, block)  # [768, 768]
            frob_list.append(torch.norm(J, 'fro').item())
            sv0 = torch.linalg.svdvals(J)[0].item()
            spec_list.append(sv0)
            erank_list.append(effective_rank(J))

        frob_norms_per_probe.append(frob_list)
        spec_norms_per_probe.append(spec_list)
        eff_ranks_per_probe.append(erank_list)

        if (probe_idx + 1) % 64 == 0:
            print(f"  probe {probe_idx+1}/{n_probes_actual}", flush=True)

    # Aggregate: std of Frobenius norms per neighborhood, then mean over probes
    p1_vals = [np.std(f) for f in frob_norms_per_probe]
    p2_vals = [np.mean(s) for s in spec_norms_per_probe]
    p3_vals = [np.mean(e) for e in eff_ranks_per_probe]

    p1 = float(np.mean(p1_vals))
    p2 = float(np.mean(p2_vals))
    p3 = float(np.mean(p3_vals))

    print(f"  P1 (frob_std)={p1:.4f}, P2 (spec_norm)={p2:.4f}, P3 (eff_rank)={p3:.4f}", flush=True)
    results_by_layer[layer_idx] = {"p1_frob_std": p1, "p2_spec_norm": p2, "p3_eff_rank": p3}


# ---------------------------------------------------------------------------
# Spearman correlations
# ---------------------------------------------------------------------------
layers = list(range(12))
p1 = [results_by_layer[l]["p1_frob_std"] for l in layers]
p2 = [results_by_layer[l]["p2_spec_norm"] for l in layers]
p3 = [results_by_layer[l]["p3_eff_rank"] for l in layers]

damage_8x = DAMAGE_8X
damage_16x = DAMAGE_16X

# Remove layer 0 for leave-one-out
layers_no0 = layers[1:]
p1_no0 = p1[1:]
p2_no0 = p2[1:]
p3_no0 = p3[1:]
d8_no0 = damage_8x[1:]
d16_no0 = damage_16x[1:]


def spearman_gt(x, y):
    """One-tailed Spearman (alternative='greater')."""
    rho, p_two = stats.spearmanr(x, y)
    # one-tailed p: if rho > 0, p_one = p_two/2; else p_one = 1 - p_two/2
    p_one = p_two / 2 if rho > 0 else 1.0 - p_two / 2
    return float(rho), float(p_one)


corr = {}
for proxy_name, proxy_full, proxy_no0 in [
    ("p1_frob_std", p1, p1_no0),
    ("p2_spec_norm", p2, p2_no0),
    ("p3_eff_rank", p3, p3_no0),
]:
    for dmg_name, dmg_full, dmg_no0 in [
        ("damage_8x", damage_8x, d8_no0),
        ("damage_16x", damage_16x, d16_no0),
    ]:
        rho_f, p_f = spearman_gt(proxy_full, dmg_full)
        rho_no0, p_no0 = spearman_gt(proxy_no0, dmg_no0)
        key = f"{proxy_name}_vs_{dmg_name}"
        corr[key] = {
            "rho_full": rho_f, "p_full": p_f,
            "rho_without_L0": rho_no0, "p_without_L0": p_no0,
        }
        print(f"{key}: rho_full={rho_f:.3f} (p={p_f:.3f}), rho_no_L0={rho_no0:.3f} (p={p_no0:.3f})")

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(corr_dict):
    confirmed = False
    refuted_all = True
    inconclusive_leverage = False

    for key, v in corr_dict.items():
        rf = v["rho_full"]
        rn = v["rho_without_L0"]
        if rf >= 0.5:
            refuted_all = False
            if rn >= 0.2:
                confirmed = True
            else:
                inconclusive_leverage = True
        elif rf >= 0.2:
            refuted_all = False

    if confirmed:
        return "confirmed"
    elif refuted_all:
        return "refuted"
    else:
        return "inconclusive"


outcome = classify(corr)
print(f"\nOutcome: {outcome}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
output = {
    "cycle": 48,
    "item": 30,
    "n_probes": n_probes_actual,
    "k_neighbors": k_actual,
    "proxies_by_layer": results_by_layer,
    "damage_8x": damage_8x,
    "damage_16x": damage_16x,
    "correlations": corr,
    "outcome": outcome,
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(JSON_OUT, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON: {JSON_OUT}")

# ---------------------------------------------------------------------------
# Plot: 3 proxies x 2 damage rates = 6 panels, layer 0 marked in red
# ---------------------------------------------------------------------------
proxy_labels = {
    "p1_frob_std": ("P1: Frob std", p1),
    "p2_spec_norm": ("P2: Spec norm", p2),
    "p3_eff_rank": ("P3: Eff rank", p3),
}
damage_labels = {
    "damage_8x": ("Damage 8x", damage_8x),
    "damage_16x": ("Damage 16x", damage_16x),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("GPT-2 FFN local complexity vs compression damage (cycle 048)", fontsize=12)

colors = ["red"] + ["steelblue"] * 11

for col_idx, (pkey, (plabel, pvals)) in enumerate(proxy_labels.items()):
    for row_idx, (dkey, (dlabel, dvals)) in enumerate(damage_labels.items()):
        ax = axes[row_idx][col_idx]
        ckey = f"{pkey}_vs_{dkey}"
        rho_f = corr[ckey]["rho_full"]
        rho_n = corr[ckey]["rho_without_L0"]

        for li in layers:
            c = "red" if li == 0 else "steelblue"
            ax.scatter(pvals[li], dvals[li], color=c, s=60, zorder=3)
            ax.annotate(str(li), (pvals[li], dvals[li]), textcoords="offset points",
                        xytext=(4, 2), fontsize=7)

        ax.set_xlabel(plabel, fontsize=9)
        ax.set_ylabel(dlabel, fontsize=9)
        ax.set_title(f"ρ_full={rho_f:.2f}, ρ_no_L0={rho_n:.2f}", fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=120)
print(f"Saved plot: {PLOT_OUT}")
print(f"\nFinal outcome: {outcome}")
