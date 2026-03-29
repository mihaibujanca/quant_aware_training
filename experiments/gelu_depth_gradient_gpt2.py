"""
Cycle 111: Depth Gradient of GELU Operating Regime — GPT-2 FFN
Item 122

Measures how the GELU pre-activation distribution shifts across the 12 transformer
blocks. Key metrics: negative fraction, E[GELU'(z)], CDF bins, attention effect.
"""

import json
import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformers import GPT2Model, GPT2Tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "data/openwebtext_sample.txt"
JSON_OUT = "autoresearch/results/cycle_111_gelu_depth_gradient.json"
PLOT_OUT = "autoresearch/results/cycle_111_gelu_depth_gradient.png"

N_TOKENS = 4096
BLOCK_SIZE = 1024  # GPT-2 context window

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(device).eval()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH} ...")
with open(DATA_PATH, "r") as f:
    text = f.read()

tokens = tokenizer.encode(text)[:N_TOKENS]
print(f"Tokens: {len(tokens)}")

# ---------------------------------------------------------------------------
# Storage for hooks
# ---------------------------------------------------------------------------
# We'll store:
#   c_fc_out[b]    : (n_tokens, 3072) pre-GELU activations (c_fc output)
#   ln2_out[b]     : (n_tokens, 768)  input to FFN (after LN2)
#   ln1_in[b]      : (n_tokens, 768)  residual stream before attention (before LN1)
c_fc_out = {}
ln2_out = {}
ln1_in = {}

hooks = []

for b in range(12):
    block = model.h[b]

    def make_c_fc_hook(blk_idx):
        def hook_fn(module, inp, out):
            c_fc_out[blk_idx] = out.detach().float().squeeze(0)  # (seq, 3072)
        return hook_fn

    def make_ln2_hook(blk_idx):
        def hook_fn(module, inp, out):
            # ln_2 output — this is the normed input to the MLP
            ln2_out[blk_idx] = out.detach().float().squeeze(0)  # (seq, 768)
        return hook_fn

    def make_ln1_in_hook(blk_idx):
        def hook_fn(module, inp, out):
            # ln_1 *input* is the residual stream before attention+LN
            # inp is a tuple; inp[0] is the tensor
            ln1_in[blk_idx] = inp[0].detach().float().squeeze(0)  # (seq, 768)
        return hook_fn

    hooks.append(block.mlp.c_fc.register_forward_hook(make_c_fc_hook(b)))
    hooks.append(block.ln_2.register_forward_hook(make_ln2_hook(b)))
    hooks.append(block.ln_1.register_forward_hook(make_ln1_in_hook(b)))

# ---------------------------------------------------------------------------
# Forward pass (chunked to fit context window)
# ---------------------------------------------------------------------------
print("Running forward pass...")
n_chunks = math.ceil(N_TOKENS / BLOCK_SIZE)

# We run in chunks; each chunk's hooks will overwrite the dicts.
# To aggregate, we'll collect lists and concatenate after.
c_fc_chunks = {b: [] for b in range(12)}
ln2_chunks = {b: [] for b in range(12)}
ln1_chunks = {b: [] for b in range(12)}

with torch.no_grad():
    for chunk_idx in range(n_chunks):
        start = chunk_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, N_TOKENS)
        chunk = tokens[start:end]
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        model(input_ids)
        for b in range(12):
            c_fc_chunks[b].append(c_fc_out[b].clone())
            ln2_chunks[b].append(ln2_out[b].clone())
            ln1_chunks[b].append(ln1_in[b].clone())

# Remove hooks
for h in hooks:
    h.remove()

# Concatenate chunks
for b in range(12):
    c_fc_chunks[b] = torch.cat(c_fc_chunks[b], dim=0)  # (N_tokens, 3072)
    ln2_chunks[b] = torch.cat(ln2_chunks[b], dim=0)    # (N_tokens, 768)
    ln1_chunks[b] = torch.cat(ln1_chunks[b], dim=0)    # (N_tokens, 768)

print(f"c_fc shape block 0: {c_fc_chunks[0].shape}")

# ---------------------------------------------------------------------------
# Abort checks
# ---------------------------------------------------------------------------
for b in range(12):
    if torch.isnan(c_fc_chunks[b]).any():
        print(f"NaN in block {b} c_fc output — ABORT")
        sys.exit(1)

# Quick PPL check via manual cross-entropy on one chunk
from transformers import GPT2LMHeadModel
lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
with torch.no_grad():
    chunk = tokens[:BLOCK_SIZE]
    input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
    loss = lm_model(input_ids, labels=input_ids).loss.item()
    ppl = math.exp(loss)
print(f"Baseline PPL: {ppl:.2f}")
if not (5 <= ppl <= 65):
    print(f"PPL {ppl:.2f} outside [5, 65] — ABORT")
    sys.exit(1)

del lm_model

# ---------------------------------------------------------------------------
# Helper: GELU derivative (exact)
# ---------------------------------------------------------------------------
def gelu_prime(z: torch.Tensor) -> torch.Tensor:
    """d/dz GELU(z) = 0.5*(1 + erf(z/sqrt(2))) + z*phi(z)"""
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / sqrt2))
    pdf = torch.exp(-0.5 * z * z) / sqrt2pi
    return cdf + z * pdf

# ---------------------------------------------------------------------------
# Per-block analysis
# ---------------------------------------------------------------------------
neg_frac = []
mean_gelu_prime_vals = []
cdf_bins = []  # 6 bins per block
mean_z = []
std_z = []
skewness = []

# Bins: z < -2, -2...-1, -1...0, 0...1, 1...2, z > 2
bin_edges = [-math.inf, -2, -1, 0, 1, 2, math.inf]
bin_labels = ["z<-2", "-2<z<-1", "-1<z<0", "0<z<1", "1<z<2", "z>2"]

for b in range(12):
    z = c_fc_chunks[b]  # (N_tokens, 3072)
    total = z.numel()

    # Negative fraction
    nf = (z < 0).float().mean().item()
    neg_frac.append(nf)

    # GELU derivative
    gp = gelu_prime(z).mean().item()
    mean_gelu_prime_vals.append(gp)

    # CDF bins
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        frac = ((z > lo) & (z < hi)).float().mean().item()
        bins.append(round(frac, 5))
    cdf_bins.append(bins)

    # Distribution stats
    mz = z.mean().item()
    sz = z.std().item()
    # Skewness via scipy on a subsample for speed
    z_flat = z.flatten().cpu().numpy()
    if len(z_flat) > 100000:
        rng = np.random.default_rng(0)
        z_flat = rng.choice(z_flat, 100000, replace=False)
    sk = float(stats.skew(z_flat))
    mean_z.append(mz)
    std_z.append(sz)
    skewness.append(sk)

print("\nNeg fraction per block:")
for b, nf in enumerate(neg_frac):
    print(f"  Block {b:2d}: {nf:.4f}  E[GELU']: {mean_gelu_prime_vals[b]:.4f}  mean_z: {mean_z[b]:.3f}  std_z: {std_z[b]:.3f}  skew: {skewness[b]:.3f}")

# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------
block_indices = np.arange(12, dtype=float)
neg_frac_arr = np.array(neg_frac)

slope, intercept, r_value, p_value, se = stats.linregress(block_indices, neg_frac_arr)
r2 = r_value ** 2
print(f"\nOLS: slope={slope:.5f}/block ({slope*100:.3f}%/block), R²={r2:.4f}, p={p_value:.4e}")

# ---------------------------------------------------------------------------
# Monotonicity violations
# ---------------------------------------------------------------------------
violations = sum(1 for i in range(11) if neg_frac[i+1] < neg_frac[i])
print(f"Monotonicity violations: {violations}/11")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
if violations <= 1 and slope > 0.005 and r2 > 0.7:
    verdict = "CONFIRMED"
elif violations >= 4 or slope < 0.002:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"Verdict: {verdict}")

# ---------------------------------------------------------------------------
# Attention effect
# ---------------------------------------------------------------------------
# For each block: actual neg_frac uses ln2_out (after attention+LN2).
# Hypothetical: use ln1_in (residual before attention) projected through W_in.
# W_in for GPT-2 Conv1D: c_fc.weight shape is (768, 3072), bias (3072,)
# Linear: z = x @ W_in + b

attention_effect = []
for b in range(12):
    W = model.h[b].mlp.c_fc.weight.detach().float()   # (768, 3072)
    bias = model.h[b].mlp.c_fc.bias.detach().float()  # (3072,)

    # Actual: ln2_out @ W + bias
    z_actual = ln2_chunks[b].to(device) @ W.to(device) + bias.to(device)
    nf_actual = (z_actual < 0).float().mean().item()

    # Hypothetical: apply LN manually to ln1_in, then project
    # Use the block's ln_2 parameters on the raw residual
    # Actually ln1_in is the residual BEFORE ln_1; we want pre-FFN residual.
    # The pre-FFN residual = ln1_in + attention output.
    # We don't have attention output separately, but we have ln2_chunks[b]
    # which IS the LN2-normed pre-FFN residual.
    # For the hypothetical "no attention" we use ln1_in directly normalized by LN2 params.
    ln2_weight = model.h[b].ln_2.weight.detach().float()  # (768,)
    ln2_bias_ln = model.h[b].ln_2.bias.detach().float()    # (768,)

    x_pre_attn = ln1_in[b] if b in ln1_in else ln1_chunks[b]
    # Re-apply LN2 normalization to pre-attention residual
    mean_pre = x_pre_attn.mean(dim=-1, keepdim=True)
    var_pre = x_pre_attn.var(dim=-1, keepdim=True, unbiased=False)
    x_normed = (x_pre_attn - mean_pre) / (var_pre + 1e-5).sqrt()
    x_normed = x_normed * ln2_weight + ln2_bias_ln  # apply LN2 affine

    z_hypo = x_normed.to(device) @ W.to(device) + bias.to(device)
    nf_hypo = (z_hypo < 0).float().mean().item()

    delta = nf_actual - nf_hypo
    attention_effect.append({
        "block": b,
        "nf_actual": round(nf_actual, 5),
        "nf_no_attn": round(nf_hypo, 5),
        "delta": round(delta, 5),
    })

print("\nAttention effect (actual - no_attn neg_frac):")
for ae in attention_effect:
    print(f"  Block {ae['block']:2d}: actual={ae['nf_actual']:.4f}  no_attn={ae['nf_no_attn']:.4f}  delta={ae['delta']:+.4f}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 111,
    "item": 122,
    "n_tokens": N_TOKENS,
    "baseline_ppl": round(ppl, 3),
    "per_block": [
        {
            "block": b,
            "neg_frac": round(neg_frac[b], 5),
            "mean_gelu_prime": round(mean_gelu_prime_vals[b], 5),
            "mean_z": round(mean_z[b], 4),
            "std_z": round(std_z[b], 4),
            "skewness": round(skewness[b], 4),
            "cdf_bins": dict(zip(bin_labels, cdf_bins[b])),
            "attention_effect": attention_effect[b],
        }
        for b in range(12)
    ],
    "regression": {
        "slope": round(slope, 7),
        "slope_pct_per_block": round(slope * 100, 5),
        "intercept": round(intercept, 5),
        "r2": round(r2, 5),
        "p_value": float(p_value),
    },
    "monotonicity_violations": violations,
    "verdict": verdict,
}

with open(JSON_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved: {JSON_OUT}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 7))

blocks = list(range(12))
ols_line = intercept + slope * np.array(blocks)

ax0 = axes[0]
ax0.plot(blocks, [nf * 100 for nf in neg_frac], "o-", color="steelblue", linewidth=2, markersize=6, label="neg_frac (%)")
ax0.plot(blocks, ols_line * 100, "--", color="tomato", linewidth=1.5, label=f"OLS: slope={slope*100:.3f}%/blk, R²={r2:.3f}")
ax0.set_xlabel("Block index")
ax0.set_ylabel("Negative fraction (%)")
ax0.set_title(f"GELU pre-activation negative fraction by depth  [violations={violations}, verdict={verdict}]")
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.3)
ax0.set_xticks(blocks)

ax1 = axes[1]
ax1.plot(blocks, mean_gelu_prime_vals, "s-", color="darkorange", linewidth=2, markersize=6)
ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="fully linear (1.0)")
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="half-gated (0.5)")
ax1.set_xlabel("Block index")
ax1.set_ylabel("E[GELU'(z)]")
ax1.set_title("Mean GELU derivative by depth (effective linearity)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(blocks)
ax1.set_ylim(0.4, 1.1)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {PLOT_OUT}")

print("\nDone.")
