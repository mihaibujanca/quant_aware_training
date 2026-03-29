"""
Cycle 083 — GELU near-linearity census v2 for GPT-2 FFN (item 94).
Corrects cycle 032's post-GELU magnitude thresholding by using the GELU derivative.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aleph.lm_geometric_compression import (
    TeacherLoadSpec,
    ProbeCorpusSpec,
    load_nanochat_teacher,
    build_probe_corpus,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Sanity checks ────────────────────────────────────────────────────────────
def check_gelu_derivative_sanity():
    # Note: tanh-approx GELU'(-3) ≈ -0.012 (slightly negative, not 0.004 from exact GELU)
    for x_val, expected_approx, tol in [(0.0, 0.5, 0.01), (3.0, 1.0, 0.02), (-3.0, -0.012, 0.01)]:
        z = torch.tensor([x_val], requires_grad=True)
        g = F.gelu(z, approximate='tanh')
        gp = torch.autograd.grad(g.sum(), z)[0]
        assert abs(gp.item() - expected_approx) < tol, (
            f"Sanity fail at x={x_val}: got {gp.item():.4f}, expected ~{expected_approx}"
        )
        print(f"  GELU'({x_val}) = {gp.item():.4f}  ✓ (expected ~{expected_approx})")

print("=== Sanity checks ===")
check_gelu_derivative_sanity()

# ── Load GPT-2 ───────────────────────────────────────────────────────────────
print("\n=== Loading GPT-2 ===")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
model = teacher.model
tokenizer = teacher.tokenizer

# ── Build 4096-token probe corpus ────────────────────────────────────────────
print("\n=== Building probe corpus ===")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
corpus_parts = []
for fname in ["shakespeare.txt", "wikitext2_valid.txt", "openwebtext_sample.txt", "c4_sample.txt"]:
    spec = ProbeCorpusSpec(
        text_path=os.path.join(DATA, fname),
        probe_train_sequences=8,
        sequence_length=128,
        teacher=teacher,
    )
    part = build_probe_corpus(spec)
    corpus_parts.append(part.probe_train)
    print(f"  {fname}: {part.probe_train.shape}")

# Concatenate all probe tokens: target shape (32, 128) = 4096 tokens
all_tokens = torch.cat(corpus_parts, dim=0)  # (32, 128)
print(f"  Combined: {all_tokens.shape}  → {all_tokens.numel()} tokens")

# ── Collect pre-GELU activations via hooks ───────────────────────────────────
print("\n=== Collecting pre-GELU activations per block ===")
DEAD_THRESH = 0.01
DEAD_FRAC_THRESH = 0.95
LINEAR_THRESH = 0.05
LINEAR_FRAC_THRESH = 0.80
N_BLOCKS = 12
BATCH_SIZE = 4

block_z = {i: [] for i in range(N_BLOCKS)}
hooks = []

def make_hook(block_idx):
    def hook_fn(module, input, output):
        # output of c_fc is the pre-GELU tensor
        block_z[block_idx].append(output.detach().cpu())
    return hook_fn

for i in range(N_BLOCKS):
    h = model.transformer.h[i].mlp.c_fc.register_forward_hook(make_hook(i))
    hooks.append(h)

model.eval()
with torch.no_grad():
    for batch_start in range(0, all_tokens.shape[0], BATCH_SIZE):
        batch = all_tokens[batch_start:batch_start + BATCH_SIZE]
        _ = model(batch)

for h in hooks:
    h.remove()

# Concatenate: each block gets shape (32*128, 3072) = (4096, 3072)
for i in range(N_BLOCKS):
    # Each hook capture has shape (batch_size, seq_len, 3072)
    z_list = block_z[i]
    z_cat = torch.cat(z_list, dim=0)  # (32, 128, 3072)
    block_z[i] = z_cat.view(-1, z_cat.shape[-1])  # (4096, 3072)
    print(f"  Block {i}: pre-GELU shape = {block_z[i].shape}")

# ── Gaussian null baseline ────────────────────────────────────────────────────
print("\n=== Gaussian null baseline ===")
z_null = torch.randn(100000, requires_grad=True)
g_null = F.gelu(z_null, approximate='tanh')
gp_null = torch.autograd.grad(g_null.sum(), z_null)[0].detach()
null_dead_frac = (gp_null.abs() < DEAD_THRESH).float().mean().item()
null_linear_frac = ((gp_null - 1).abs() < LINEAR_THRESH).float().mean().item()
print(f"  N(0,1): dead_frac={null_dead_frac:.4f} (~0.003 expected), linear_frac={null_linear_frac:.4f} (~0.28 expected)")

# ── Per-block GELU derivative classification ──────────────────────────────────
print("\n=== Computing GELU derivatives per block ===")
results = {}

for i in range(N_BLOCKS):
    z = block_z[i]  # (4096, 3072)
    n_channels = z.shape[1]

    # Compute GELU derivative via autograd
    z_grad = z.clone().requires_grad_(True)
    g = F.gelu(z_grad, approximate='tanh')
    gp = torch.autograd.grad(g.sum(), z_grad)[0].detach()  # (4096, 3072)

    # Per-channel fractions
    dead_fracs = (gp.abs() < DEAD_THRESH).float().mean(dim=0)    # (3072,)
    linear_fracs = ((gp - 1).abs() < LINEAR_THRESH).float().mean(dim=0)  # (3072,)

    is_dead = dead_fracs > DEAD_FRAC_THRESH
    is_linear = (~is_dead) & (linear_fracs > LINEAR_FRAC_THRESH)
    is_nonlinear = ~(is_dead | is_linear)

    n_dead = int(is_dead.sum())
    n_linear = int(is_linear.sum())
    n_nonlinear = int(is_nonlinear.sum())

    assert n_dead + n_linear + n_nonlinear == n_channels, "Category counts don't sum to n_channels"

    results[i] = {
        "n_channels": n_channels,
        "n_dead": n_dead,
        "n_linear": n_linear,
        "n_nonlinear": n_nonlinear,
        "pct_dead": n_dead / n_channels * 100,
        "pct_linear": n_linear / n_channels * 100,
        "pct_nonlinear": n_nonlinear / n_channels * 100,
        "channel_details": [
            {"j": int(j), "dead_frac": float(dead_fracs[j]), "linear_frac": float(linear_fracs[j])}
            for j in range(n_channels)
        ],
    }
    print(f"  Block {i:2d}: dead={n_dead:4d} ({n_dead/n_channels*100:.1f}%)  "
          f"near-linear={n_linear:4d} ({n_linear/n_channels*100:.1f}%)  "
          f"nonlinear={n_nonlinear:4d} ({n_nonlinear/n_channels*100:.1f}%)")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "experiment": "cycle_083_gelu_census_v2",
        "model": "gpt2",
        "n_probe_tokens": int(all_tokens.numel()),
        "dead_threshold": DEAD_THRESH,
        "dead_frac_threshold": DEAD_FRAC_THRESH,
        "linear_threshold": LINEAR_THRESH,
        "linear_frac_threshold": LINEAR_FRAC_THRESH,
        "gelu_variant": "tanh_approximate",
    },
    "null_baseline": {
        "distribution": "N(0,1)",
        "n_samples": 100000,
        "dead_frac": null_dead_frac,
        "linear_frac": null_linear_frac,
    },
    "blocks": {str(i): results[i] for i in range(N_BLOCKS)},
}

json_path = os.path.join(RESULTS_DIR, "cycle_083_gelu_census_v2.json")
# Remove channel_details for a slimmer JSON (keep summary only)
slim_output = {
    "metadata": output["metadata"],
    "null_baseline": output["null_baseline"],
    "blocks": {
        k: {kk: vv for kk, vv in v.items() if kk != "channel_details"}
        for k, v in output["blocks"].items()
    },
    "channel_details_per_block": {
        str(i): results[i]["channel_details"] for i in range(N_BLOCKS)
    },
}
with open(json_path, "w") as f:
    json.dump(slim_output, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n=== Plotting ===")
fig, ax = plt.subplots(figsize=(12, 6))

block_ids = list(range(N_BLOCKS))
pct_dead = [results[i]["pct_dead"] for i in block_ids]
pct_linear = [results[i]["pct_linear"] for i in block_ids]
pct_nonlinear = [results[i]["pct_nonlinear"] for i in block_ids]

x = np.arange(N_BLOCKS)
width = 0.65

bars_nonlinear = ax.bar(x, pct_nonlinear, width, label="Nonlinear", color="#4e79a7")
bars_linear = ax.bar(x, pct_linear, width, bottom=pct_nonlinear, label="Near-linear (|GELU'−1|<0.05 for >80%)", color="#f28e2b")
bottom_top = [a + b for a, b in zip(pct_nonlinear, pct_linear)]
bars_dead = ax.bar(x, pct_dead, width, bottom=bottom_top, label="Dead (|GELU'|<0.01 for >95%)", color="#e15759")

# Null baseline line
ax.axhline(y=null_linear_frac * 100, color="#f28e2b", linestyle="--", alpha=0.7, linewidth=1.5,
           label=f"Null N(0,1) near-linear frac: {null_linear_frac*100:.1f}%")
ax.axhline(y=null_dead_frac * 100, color="#e15759", linestyle="--", alpha=0.7, linewidth=1.5,
           label=f"Null N(0,1) dead frac: {null_dead_frac*100:.2f}%")

ax.set_xlabel("GPT-2 Block Index")
ax.set_ylabel("Fraction of Channels (%)")
ax.set_title("GPT-2 FFN Channel Classification by GELU Derivative\n(Cycle 083 — Corrected Methodology)")
ax.set_xticks(x)
ax.set_xticklabels([f"B{i}" for i in block_ids])
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "cycle_083_gelu_census_v2.png")
plt.savefig(png_path, dpi=150)
print(f"Saved plot: {png_path}")

# ── Evaluate hypothesis ───────────────────────────────────────────────────────
print("\n=== Hypothesis evaluation ===")
blocks_meeting_threshold = sum(
    1 for i in range(N_BLOCKS)
    if results[i]["pct_dead"] >= 10 or results[i]["pct_linear"] >= 20
)
any_dead_gt5 = any(results[i]["pct_dead"] >= 5 for i in range(N_BLOCKS))
any_linear_gt10 = any(results[i]["pct_linear"] >= 10 for i in range(N_BLOCKS))

print(f"  Blocks meeting threshold (>=10% dead OR >=20% linear): {blocks_meeting_threshold}/12")
print(f"  Any block with >=5% dead: {any_dead_gt5}")
print(f"  Any block with >=10% near-linear: {any_linear_gt10}")

if blocks_meeting_threshold >= 6:
    verdict = "CONFIRMED"
    reason = f"{blocks_meeting_threshold}/12 blocks meet ≥10% dead OR ≥20% near-linear threshold"
elif not any_dead_gt5 and not any_linear_gt10:
    verdict = "REFUTED"
    reason = "<5% dead AND <10% near-linear across all blocks"
else:
    verdict = "INCONCLUSIVE"
    reason = f"Only {blocks_meeting_threshold}/12 blocks meet threshold (need ≥6)"

print(f"\n  VERDICT: {verdict}")
print(f"  REASON: {reason}")

# Summary stats
total_dead = sum(results[i]["n_dead"] for i in range(N_BLOCKS))
total_linear = sum(results[i]["n_linear"] for i in range(N_BLOCKS))
total_channels = sum(results[i]["n_channels"] for i in range(N_BLOCKS))
print(f"\n  Summary across all 12 blocks:")
print(f"    Total dead channels: {total_dead}/{total_channels} ({total_dead/total_channels*100:.1f}%)")
print(f"    Total near-linear channels: {total_linear}/{total_channels} ({total_linear/total_channels*100:.1f}%)")

print(f"\nVERDICT={verdict}")
print(f"BLOCKS_MEETING_THRESHOLD={blocks_meeting_threshold}")
print(f"TOTAL_DEAD_PCT={total_dead/total_channels*100:.2f}")
print(f"TOTAL_LINEAR_PCT={total_linear/total_channels*100:.2f}")
