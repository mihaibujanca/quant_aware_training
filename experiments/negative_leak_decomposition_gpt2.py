"""
Cycle 109: Negative Leak Decomposition — GPT-2 FFN

Three-way decomposition of FFN output into:
  pos(t)    — positive-regime channels
  b_leak(t) — bias component from negative channels (token-dependent set, global per-channel mean)
  s_leak(t) — signal component (deviation from per-channel mean in negative regime)
"""

import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cpu"
N_TOKENS = 4096
RESULTS_DIR = "autoresearch/results"

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ── Load wikitext data ────────────────────────────────────────────────────────
print("Loading wikitext-2...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokens = []
for ex in ds:
    if ex["text"].strip():
        ids = tokenizer.encode(ex["text"])
        tokens.extend(ids)
    if len(tokens) >= N_TOKENS + 1024:
        break
tokens = torch.tensor(tokens[:N_TOKENS], dtype=torch.long).unsqueeze(0)  # (1, T)

# ── Hook infrastructure ──────────────────────────────────────────────────────
n_blocks = 12
pre_gelu_acts = {}   # block -> (T, 3072)
mlp_outputs   = {}   # block -> (T, 768)

hooks = []

def make_pre_gelu_hook(block_idx):
    def hook(module, inp, out):
        # inp[0] shape: (1, T, 3072)  — this is the c_fc output before GELU
        pre_gelu_acts[block_idx] = inp[0].squeeze(0).detach()  # (T, 3072)
    return hook

def make_mlp_out_hook(block_idx):
    def hook(module, inp, out):
        # out shape: (1, T, 768)
        mlp_outputs[block_idx] = out.squeeze(0).detach()  # (T, 768)
    return hook

for i in range(n_blocks):
    mlp = model.transformer.h[i].mlp
    # Hook the act (GELU) module to capture its input = pre-GELU activations
    hooks.append(mlp.act.register_forward_hook(make_pre_gelu_hook(i)))
    # Hook c_proj to capture full MLP output
    hooks.append(mlp.c_proj.register_forward_hook(make_mlp_out_hook(i)))

# GPT-2 context window is 1024; run in chunks and accumulate
CHUNK = 1024
print(f"Running forward pass in {N_TOKENS // CHUNK} chunks of {CHUNK}...")
all_pre_gelu = {i: [] for i in range(n_blocks)}
all_mlp_out  = {i: [] for i in range(n_blocks)}

with torch.no_grad():
    for c_start in range(0, N_TOKENS, CHUNK):
        chunk = tokens[:, c_start:c_start + CHUNK]
        _ = model(chunk)
        for i in range(n_blocks):
            all_pre_gelu[i].append(pre_gelu_acts[i].clone())
            all_mlp_out[i].append(mlp_outputs[i].clone())

for h in hooks:
    h.remove()

# Concatenate chunks
for i in range(n_blocks):
    pre_gelu_acts[i] = torch.cat(all_pre_gelu[i], dim=0)
    mlp_outputs[i]   = torch.cat(all_mlp_out[i],  dim=0)

T = pre_gelu_acts[0].shape[0]
print(f"Captured {T} tokens across {n_blocks} blocks")

# ── Main decomposition ────────────────────────────────────────────────────────
results = []

for i in range(n_blocks):
    z = pre_gelu_acts[i]          # (T, 3072)
    mlp_out = mlp_outputs[i]      # (T, 768) = c_proj output (includes bias)

    # GPT-2 Conv1D: weight stored (3072, 768), so W_out (768, 3072)
    W_out = model.transformer.h[i].mlp.c_proj.weight.T.detach()   # (768, 3072)
    b_out = model.transformer.h[i].mlp.c_proj.bias.detach()       # (768,)

    gelu_z = F.gelu(z, approximate='tanh')  # (T, 3072)

    # Per-channel mean/var GELU value in negative regime — vectorized
    neg_mask_chan = (z < 0)  # (T, 3072) bool
    neg_count = neg_mask_chan.float().sum(0)  # (3072,)

    # Masked sum for mu: zero-out positive positions
    gelu_neg = gelu_z * neg_mask_chan.float()   # (T, 3072), non-neg channels zeroed
    mu_leak = gelu_neg.sum(0) / neg_count.clamp(min=1)  # (3072,)

    # Variance: E[x^2] - mu^2
    gelu_neg_sq = (gelu_z ** 2) * neg_mask_chan.float()
    ex2 = gelu_neg_sq.sum(0) / neg_count.clamp(min=1)
    var_leak = (ex2 - mu_leak ** 2).clamp(min=0)

    valid_chan = neg_count >= 50
    mu_leak = mu_leak * valid_chan.float()   # zero out low-count channels
    var_leak = var_leak * valid_chan.float()

    n_valid = valid_chan.sum().item()
    print(f"Block {i:2d}: {n_valid}/3072 channels with ≥50 negative tokens")

    # Per-token decomposition
    # b_leak(t) = sum_{j in N(t)} W_out[:,j] * mu_leak_j
    # s_leak(t) = sum_{j in N(t)} W_out[:,j] * (gelu_z[t,j] - mu_leak_j)
    # pos(t)    = sum_{j not in N(t)} W_out[:,j] * gelu_z[t,j]

    neg_mask_tok = (z < 0)  # (T, 3072)

    # b_leak: for each token, W_out @ (neg_mask * mu_leak)
    # neg_mask_tok (T, 3072) * mu_leak (3072,) → (T, 3072), then @ W_out.T
    b_leak_weights = neg_mask_tok.float() * mu_leak.unsqueeze(0)  # (T, 3072)
    b_leak = b_leak_weights @ W_out.T  # (T, 768)

    # s_leak: deviation from mean for negative channels
    deviation = gelu_z - mu_leak.unsqueeze(0)  # (T, 3072)
    s_leak_weights = neg_mask_tok.float() * deviation  # (T, 3072)
    s_leak = s_leak_weights @ W_out.T  # (T, 768)

    # pos: positive regime contribution
    pos_weights = (~neg_mask_tok).float() * gelu_z  # (T, 3072)
    pos = pos_weights @ W_out.T  # (T, 768)

    # Reconstructed block output
    block_out_recon = pos + b_leak + s_leak + b_out.unsqueeze(0)  # (T, 768)

    # Verify reconstruction
    rel_err = (block_out_recon - mlp_out).norm(dim=-1) / (mlp_out.norm(dim=-1) + 1e-8)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()
    print(f"  Recon error: max={max_rel_err:.2e}, mean={mean_rel_err:.2e}")
    assert max_rel_err < 1e-3, f"Block {i}: reconstruction too inaccurate ({max_rel_err:.2e})"

    # Norms
    norm_out     = mlp_out.norm(dim=-1).mean().item()
    norm_b_leak  = b_leak.norm(dim=-1).mean().item()
    norm_s_leak  = s_leak.norm(dim=-1).mean().item()
    norm_pos     = pos.norm(dim=-1).mean().item()

    # neg_total = b_leak + s_leak
    neg_total = b_leak + s_leak
    norm_neg_total = neg_total.norm(dim=-1).mean().item()

    b_ratio = norm_b_leak / (norm_out + 1e-8)
    s_ratio = norm_s_leak / (norm_out + 1e-8)

    # b_energy: fraction of negative-contribution energy that is bias
    b_energy = (b_leak.norm(dim=-1)**2).mean().item() / ((neg_total.norm(dim=-1)**2).mean().item() + 1e-8)

    print(f"  b_ratio={b_ratio:.4f}  s_ratio={s_ratio:.4f}  b_energy={b_energy:.4f}")

    # Top-20 variance channels
    var_vals, var_idx = var_leak.topk(20)

    results.append({
        "block": i,
        "n_valid_channels": n_valid,
        "b_ratio": b_ratio,
        "s_ratio": s_ratio,
        "b_energy": b_energy,
        "norm_out": norm_out,
        "norm_b_leak": norm_b_leak,
        "norm_s_leak": norm_s_leak,
        "norm_pos": norm_pos,
        "norm_neg_total": norm_neg_total,
        "max_recon_err": max_rel_err,
        "mean_recon_err": mean_rel_err,
        "top20_var_channels": var_idx.tolist(),
        "top20_var_values": var_vals.tolist(),
    })

# ── Verdict ──────────────────────────────────────────────────────────────────
b_energies = [r["b_energy"] for r in results]
n_above_70 = sum(e > 0.70 for e in b_energies)
n_below_30 = sum(e < 0.30 for e in b_energies)

if n_above_70 >= 8:
    verdict = "CONFIRMED"
elif n_below_30 >= 8:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

b_ratios = [r["b_ratio"] for r in results]
s_ratios = [r["s_ratio"] for r in results]

# Also check original criteria from spec
n_b_lt5 = sum(r < 0.05 for r in b_ratios)
n_s_lt2 = sum(r < 0.02 for r in s_ratios)
n_s_gt10 = sum(r > 0.10 for r in s_ratios)

print(f"\n=== VERDICT: {verdict} ===")
print(f"b_energy > 0.70: {n_above_70}/12 blocks")
print(f"b_energy < 0.30: {n_below_30}/12 blocks")
print(f"b_ratio < 5%:    {n_b_lt5}/12 blocks")
print(f"s_ratio < 2%:    {n_s_lt2}/12 blocks")
print(f"s_ratio > 10%:   {n_s_gt10}/12 blocks")
print(f"b_energies: {[f'{e:.3f}' for e in b_energies]}")

# ── Save JSON ────────────────────────────────────────────────────────────────
output = {
    "verdict": verdict,
    "n_tokens": T,
    "n_above_70": n_above_70,
    "n_below_30": n_below_30,
    "n_b_ratio_lt5pct": n_b_lt5,
    "n_s_ratio_lt2pct": n_s_lt2,
    "n_s_ratio_gt10pct": n_s_gt10,
    "b_energies": b_energies,
    "b_ratios": b_ratios,
    "s_ratios": s_ratios,
    "blocks": results,
}

json_path = f"{RESULTS_DIR}/cycle_109_negative_leak.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved {json_path}")

# ── Plot ─────────────────────────────────────────────────────────────────────
blocks = list(range(n_blocks))
x = np.arange(n_blocks)
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.bar(x - width/2, b_ratios, width, label="b_ratio (bias component)")
ax1.bar(x + width/2, s_ratios, width, label="s_ratio (signal component)")
ax1.axhline(0.05, color="gray", linestyle="--", alpha=0.6, label="5% threshold")
ax1.axhline(0.02, color="orange", linestyle=":", alpha=0.6, label="2% threshold")
ax1.set_xlabel("Block")
ax1.set_ylabel("||component|| / ||block_out||")
ax1.set_title("Negative Leak: b_ratio and s_ratio per Block")
ax1.set_xticks(x)
ax1.legend(fontsize=8)
ax1.set_ylim(0, max(max(b_ratios), max(s_ratios)) * 1.2)

ax2 = axes[1]
ax2.bar(x, b_energies, color="steelblue", label="b_energy")
ax2.axhline(0.70, color="green", linestyle="--", label="0.70 (confirmed)")
ax2.axhline(0.30, color="red", linestyle="--", label="0.30 (refuted)")
ax2.set_xlabel("Block")
ax2.set_ylabel("||b_leak||² / ||neg_total||²")
ax2.set_title(f"b_energy per Block — Verdict: {verdict}")
ax2.set_xticks(x)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plot_path = f"{RESULTS_DIR}/cycle_109_negative_leak.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Saved {plot_path}")

print("\nDone.")
