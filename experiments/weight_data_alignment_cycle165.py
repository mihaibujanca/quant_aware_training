"""
Cycle 165 / Item 175: Weight-data alignment decomposition — GPT-2 FFN
Why is z_mean ≈ -1? Decompose W_up @ x contribution vs bias.
"""

import json
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# --- Config ---
N_TOKENS = 1024  # GPT-2 max context is 1024
N_RAND_DRAWS = 3
DEVICE = "cpu"
RESULTS_DIR = "autoresearch/results"
PLOT_PATH = "autoresearch/results/cycle_165_alignment.png"
JSON_PATH = "autoresearch/results/cycle_165_alignment.json"

# --- Load model and tokenizer ---
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()

# --- Load data ---
print("Loading wikitext-2...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join([t for t in ds["text"] if t.strip()])
tokens = tokenizer(text, return_tensors="pt")["input_ids"][0][:N_TOKENS]
input_ids = tokens.unsqueeze(0)  # (1, N_TOKENS)

# --- Hook storage ---
hooks_data = {}  # block_idx -> {"x": tensor, "z": tensor}

def make_hooks(block_idx):
    def hook_fn(module, inp, out):
        # inp[0]: (1, T, 768) — input to c_fc (W_up @ x + b)
        # out: (1, T, 3072) — output of c_fc
        hooks_data[block_idx] = {
            "x": inp[0].detach().squeeze(0),  # (T, 768)
            "z": out.detach().squeeze(0),     # (T, 3072)
        }
    return hook_fn

handles = []
for i, block in enumerate(model.transformer.h):
    h = block.mlp.c_fc.register_forward_hook(make_hooks(i))
    handles.append(h)

# --- Forward pass ---
print("Running forward pass...")
with torch.no_grad():
    _ = model(input_ids)

for h in handles:
    h.remove()

# --- Per-block analysis ---
results = []

for i, block in enumerate(model.transformer.h):
    W = block.mlp.c_fc.weight  # (3072, 768) — GPT-2 uses Conv1D so weight is (in, out)
    b = block.mlp.c_fc.bias    # (3072,)

    # GPT-2 Conv1D: weight shape is (in_features, out_features) = (768, 3072)
    # So W_up[j] = W[:, j] (column j), and z = x @ W + b
    W_np = W.float()  # (768, 3072)

    x = hooks_data[i]["x"].float()  # (T, 768)
    z = hooks_data[i]["z"].float()  # (T, 3072)

    # Verify: z = x @ W + b
    z_check = x @ W_np + b.float()
    assert torch.allclose(z, z_check, atol=1e-4), f"Block {i}: z mismatch"

    # (a) Mean decomposition
    z_mean = z.mean().item()
    b_mean = b.float().mean().item()
    Wx_mean = (x @ W_np).mean().item()  # should equal z_mean - b_mean

    # (b) Per-channel cosine alignment
    E_x = x.mean(0)  # (768,) — mean input across tokens
    E_x_norm = E_x / (E_x.norm() + 1e-8)
    # W columns = W_up rows
    W_rows = W_np.T  # (3072, 768) — each row is a W_up[j]
    W_rows_norm = W_rows / (W_rows.norm(dim=1, keepdim=True) + 1e-8)
    cos_per_channel = (W_rows_norm @ E_x_norm).detach().numpy()  # (3072,)
    cos_mean = float(cos_per_channel.mean())
    cos_median = float(np.median(cos_per_channel))
    frac_neg = float((cos_per_channel < 0).mean())

    # (c) PCA on inputs — top-10 PCs
    x_centered = x - x.mean(0)
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    # Vh: (min(T,768), 768), top-10 PCs = Vh[:10]
    top10_pcs = Vh[:10]  # (10, 768)
    # Project W_up rows onto top-10 PCs
    proj = W_rows @ top10_pcs.T  # (3072, 10) — projection of each W row onto each PC
    W_var_in_top10 = float((proj ** 2).sum() / (W_rows ** 2).sum())
    # Sign of mean projection onto PC1
    pc1_proj_mean = float((W_rows @ Vh[0]).mean())

    # (d) Random-init baseline — 3 draws
    init_std = (1.0 / math.sqrt(768)) ** 0.5  # GPT-2 init: std = 0.02
    # Actually GPT-2 c_fc: std=0.02 (from config)
    rand_Wx_means = []
    for _ in range(N_RAND_DRAWS):
        W_rand = torch.randn(768, 3072) * 0.02
        Wx_rand = (x @ W_rand).mean().item()
        rand_Wx_means.append(Wx_rand)
    Wx_rand_mean = float(np.mean(rand_Wx_means))
    Wx_rand_std = float(np.std(rand_Wx_means))

    shift = abs(Wx_mean - Wx_rand_mean)

    row = {
        "block": i,
        "z_mean": round(z_mean, 4),
        "b_mean": round(b_mean, 4),
        "Wx_mean": round(Wx_mean, 4),
        "cos_mean": round(cos_mean, 4),
        "cos_median": round(cos_median, 4),
        "frac_neg_cos": round(frac_neg, 4),
        "W_var_top10_pcs": round(W_var_in_top10, 4),
        "pc1_proj_mean": round(pc1_proj_mean, 4),
        "Wx_rand_mean": round(Wx_rand_mean, 4),
        "Wx_rand_std": round(Wx_rand_std, 6),
        "shift": round(shift, 4),
    }
    results.append(row)

    print(f"Block {i:2d}: z_mean={z_mean:+.3f} b_mean={b_mean:+.3f} "
          f"Wx_mean={Wx_mean:+.3f} Wx_rand={Wx_rand_mean:+.4f}±{Wx_rand_std:.4f} "
          f"shift={shift:.3f} cos_mean={cos_mean:+.3f} frac_neg={frac_neg:.2%}")

# --- Verdict ---
mean_shift = float(np.mean([r["shift"] for r in results]))
print(f"\nMean shift across blocks: {mean_shift:.4f}")
if mean_shift > 0.5:
    verdict = "CONFIRMED"
elif mean_shift < 0.1:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"Verdict: {verdict}")

# --- Save JSON ---
output = {
    "verdict": verdict,
    "mean_shift": round(mean_shift, 4),
    "n_tokens": N_TOKENS,
    "n_rand_draws": N_RAND_DRAWS,
    "per_block": results,
}
with open(JSON_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON: {JSON_PATH}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

blocks = [r["block"] for r in results]
Wx_trained = [r["Wx_mean"] for r in results]
Wx_rand = [r["Wx_rand_mean"] for r in results]
Wx_rand_std_vals = [r["Wx_rand_std"] for r in results]

ax1.plot(blocks, Wx_trained, "b-o", label="Trained W_up @ x mean", linewidth=2)
ax1.errorbar(blocks, Wx_rand, yerr=Wx_rand_std_vals, fmt="r--s",
             label="Random W_up @ x mean (±std)", linewidth=1.5, capsize=4)
ax1.axhline(0, color="k", linewidth=0.5, linestyle=":")
ax1.set_xlabel("Block")
ax1.set_ylabel("Mean activation")
ax1.set_title("Trained vs Random W_up @ x mean per block")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cosine alignment histogram — pooled across all blocks
# Re-compute pooled cosines (already computed per block, but not stored as arrays)
all_cos = []
for i, block in enumerate(model.transformer.h):
    W = block.mlp.c_fc.weight.float()  # (768, 3072)
    x = hooks_data[i]["x"].float()
    E_x = x.mean(0)
    E_x_norm = E_x / (E_x.norm() + 1e-8)
    W_rows = W.T  # (3072, 768)
    W_rows_norm = W_rows / (W_rows.norm(dim=1, keepdim=True) + 1e-8)
    cos_vals = (W_rows_norm @ E_x_norm).detach().numpy()
    all_cos.extend(cos_vals.tolist())

ax2.hist(all_cos, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
ax2.axvline(0, color="k", linewidth=1, linestyle="--")
ax2.axvline(float(np.mean(all_cos)), color="red", linewidth=1.5, linestyle="-",
            label=f"Mean={np.mean(all_cos):.3f}")
ax2.set_xlabel("cos(W_up[j], E[x])")
ax2.set_ylabel("Count")
ax2.set_title("Per-channel cosine alignment (all blocks pooled)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f"GPT-2 FFN Weight-Data Alignment — Cycle 165\n"
             f"Mean shift (trained vs random): {mean_shift:.3f} → {verdict}", fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved plot: {PLOT_PATH}")

print("\nPer-block summary table:")
print(f"{'Blk':>3} {'z_mean':>8} {'b_mean':>8} {'Wx_mean':>8} {'Wx_rand':>8} "
      f"{'shift':>7} {'cos_mean':>9} {'frac_neg':>9}")
for r in results:
    print(f"{r['block']:>3} {r['z_mean']:>8.3f} {r['b_mean']:>8.3f} {r['Wx_mean']:>8.3f} "
          f"{r['Wx_rand_mean']:>8.4f} {r['shift']:>7.3f} {r['cos_mean']:>9.3f} {r['frac_neg_cos']:>9.2%}")
