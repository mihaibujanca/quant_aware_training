"""
Cycle 055: Jacobian Frobenius std corpus-sensitivity — GPT-2 FFN.

Computes P1 (Jacobian Frobenius std) for all 12 GPT-2 FFN blocks using
WikiText-2 probes, then computes Spearman correlation between these values
and the Shakespeare P1 values from cycle 048.

Hypothesis: P1 block ranking is an intrinsic geometric property, not
corpus-dependent. Spearman(P1_shakespeare, P1_wikitext) >= 0.85.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.neighbors import KDTree
from transformers import GPT2Model, GPT2Tokenizer

# ---------------------------------------------------------------------------
# Config (identical to cycle 048 fallback params)
# ---------------------------------------------------------------------------
SEED = 42
N_PROBES = 128
K_NEIGHBORS = 5
DEVICE = "cpu"
OUT_DIR = "autoresearch/results"
CYCLE048_JSON = os.path.join(OUT_DIR, "cycle_048_local_complexity_gpt2.json")
JSON_OUT = os.path.join(OUT_DIR, "cycle_055_p1_corpus_sensitivity.json")
PLOT_OUT = os.path.join(OUT_DIR, "cycle_055_p1_corpus_sensitivity.png")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load Shakespeare P1 from cycle 048
# ---------------------------------------------------------------------------
print("Loading cycle 048 Shakespeare P1 values...", flush=True)
with open(CYCLE048_JSON) as f:
    c048 = json.load(f)

p1_shakespeare = [c048["proxies_by_layer"][str(i)]["p1_frob_std"] for i in range(12)]
print(f"Shakespeare P1: {[f'{v:.3f}' for v in p1_shakespeare]}", flush=True)

# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------
print("Loading GPT-2...", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)

# ---------------------------------------------------------------------------
# Load WikiText-2 validation set and extract 128 probe tokens
# ---------------------------------------------------------------------------
print("Loading WikiText-2...", flush=True)
import datasets as hf_datasets
wikitext = hf_datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
# Concatenate all text
full_text = " ".join([row["text"] for row in wikitext if row["text"].strip()])

tokens = tokenizer.encode(full_text)
print(f"WikiText-2 validation tokens: {len(tokens)}", flush=True)

# Use a window from the middle of the text for diversity, same approach as cycle 048
start = len(tokens) // 4
token_ids = torch.tensor(tokens[start: start + N_PROBES], dtype=torch.long).unsqueeze(0).to(DEVICE)
print(f"Probe tokens: {token_ids.shape[1]}", flush=True)

# ---------------------------------------------------------------------------
# Collect FFN pre-activation inputs for all 12 blocks
# ---------------------------------------------------------------------------
ffn_inputs = {}

hooks = []
for layer_idx in range(12):
    def make_input_hook(li):
        def hook_fn(module, args):
            ffn_inputs[li] = args[0].detach().squeeze(0)  # [N_PROBES, 768]
        return hook_fn

    hooks.append(model.h[layer_idx].mlp.c_fc.register_forward_pre_hook(make_input_hook(layer_idx)))

with torch.no_grad():
    model(token_ids)

for h in hooks:
    h.remove()

print(f"Collected ffn_inputs for {len(ffn_inputs)} layers, shape: {ffn_inputs[0].shape}", flush=True)

# ---------------------------------------------------------------------------
# Jacobian of the FFN block (c_fc → GELU → c_proj) at a single input vector
# ---------------------------------------------------------------------------
def ffn_forward(x_1d, block):
    """x_1d: [768] → [768] (output of FFN c_fc+GELU+c_proj, no residual)."""
    x = x_1d.unsqueeze(0)
    h = block.mlp.c_fc(x)
    h = block.mlp.act(h)
    out = block.mlp.c_proj(h)
    return out.squeeze(0)


def compute_jacobian(x_1d, block):
    """Returns Jacobian [768, 768] via torch.func.jacrev."""
    return torch.func.jacrev(ffn_forward)(x_1d, block)


# ---------------------------------------------------------------------------
# Main loop: compute P1 for each layer
# ---------------------------------------------------------------------------
results_by_layer = {}

for layer_idx in range(12):
    print(f"\nLayer {layer_idx}...", flush=True)

    X = ffn_inputs[layer_idx].float()  # [N_PROBES, 768]
    block = model.h[layer_idx]

    X_np = X.numpy()
    tree = KDTree(X_np, leaf_size=40)
    dist, ind = tree.query(X_np, k=K_NEIGHBORS + 1)
    ind = ind[:, 1:]  # [N_PROBES, k] — skip self

    frob_norms_per_probe = []

    for probe_idx in range(N_PROBES):
        neighbor_idxs = ind[probe_idx]
        frob_list = []
        for nb_idx in neighbor_idxs:
            x_nb = X[nb_idx]
            J = compute_jacobian(x_nb, block)
            frob_list.append(torch.norm(J, 'fro').item())
        frob_norms_per_probe.append(frob_list)

        if (probe_idx + 1) % 64 == 0:
            print(f"  probe {probe_idx+1}/{N_PROBES}", flush=True)

    p1_vals = [np.std(f) for f in frob_norms_per_probe]
    p1 = float(np.mean(p1_vals))

    print(f"  P1 (frob_std)={p1:.4f}", flush=True)
    results_by_layer[layer_idx] = {"p1_frob_std": p1}

# ---------------------------------------------------------------------------
# Spearman correlation: WikiText-2 P1 vs Shakespeare P1
# ---------------------------------------------------------------------------
p1_wikitext = [results_by_layer[i]["p1_frob_std"] for i in range(12)]

print(f"\nWikiText-2 P1:  {[f'{v:.3f}' for v in p1_wikitext]}", flush=True)
print(f"Shakespeare P1: {[f'{v:.3f}' for v in p1_shakespeare]}", flush=True)

rho_full, p_full = stats.spearmanr(p1_shakespeare, p1_wikitext)
rho_full = float(rho_full)
p_full = float(p_full)

# Remove extreme blocks (0 and 11) per design review note
p1_sha_inner = p1_shakespeare[1:11]
p1_wt_inner = p1_wikitext[1:11]
rho_inner, p_inner = stats.spearmanr(p1_sha_inner, p1_wt_inner)
rho_inner = float(rho_inner)
p_inner = float(p_inner)

print(f"\nSpearman (all 12 blocks):   rho={rho_full:.4f}, p={p_full:.4f}", flush=True)
print(f"Spearman (blocks 1-10):     rho={rho_inner:.4f}, p={p_inner:.4f}", flush=True)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
if rho_full >= 0.85:
    outcome = "confirmed"
elif rho_full < 0.5:
    outcome = "refuted"
else:
    outcome = "inconclusive"

print(f"\nOutcome: {outcome}", flush=True)

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
output = {
    "cycle": 55,
    "item": 71,
    "n_probes": N_PROBES,
    "k_neighbors": K_NEIGHBORS,
    "p1_wikitext": p1_wikitext,
    "p1_shakespeare": p1_shakespeare,
    "spearman_rho_full": rho_full,
    "spearman_p_full": p_full,
    "spearman_rho_inner": rho_inner,
    "spearman_p_inner": p_inner,
    "proxies_by_layer": results_by_layer,
    "outcome": outcome,
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(JSON_OUT, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON: {JSON_OUT}", flush=True)

# ---------------------------------------------------------------------------
# Bar chart: P1 per block, Shakespeare vs WikiText-2 side by side
# ---------------------------------------------------------------------------
layers = list(range(12))
x = np.arange(12)
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"GPT-2 FFN P1 (Jacobian Frob std): Shakespeare vs WikiText-2\n"
    f"Spearman rho (all 12)={rho_full:.3f}, rho (blocks 1-10)={rho_inner:.3f}",
    fontsize=12,
)

ax = axes[0]
ax.bar(x - width/2, p1_shakespeare, width, label="Shakespeare", color="steelblue", alpha=0.85)
ax.bar(x + width/2, p1_wikitext, width, label="WikiText-2", color="darkorange", alpha=0.85)
ax.set_xlabel("GPT-2 Block")
ax.set_ylabel("P1 (mean Frob std per neighborhood)")
ax.set_title("P1 by Block")
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in layers])
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

ax2 = axes[1]
ax2.scatter(p1_shakespeare, p1_wikitext, color="steelblue", s=80, zorder=3)
for i in layers:
    ax2.annotate(str(i), (p1_shakespeare[i], p1_wikitext[i]),
                 textcoords="offset points", xytext=(4, 2), fontsize=9)
mn = min(min(p1_shakespeare), min(p1_wikitext))
mx = max(max(p1_shakespeare), max(p1_wikitext))
ax2.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="y=x")
ax2.set_xlabel("P1 Shakespeare")
ax2.set_ylabel("P1 WikiText-2")
ax2.set_title(f"Rank correlation (rho={rho_full:.3f})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=120)
print(f"Saved plot: {PLOT_OUT}", flush=True)
print(f"\nFinal outcome: {outcome}", flush=True)
