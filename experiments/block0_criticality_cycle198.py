"""
Cycle 198 — Item 205: Block 0 criticality analysis
Compare block 0 vs blocks 1-11 on data-active geometry metrics.
Block 0 dropping gives PPL 1923; all others <41 (cycle 187).
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_SEQS = 50
SEQ_LEN = 64
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_BLOCKS = 12
ENERGY_THRESH = 0.90
OUTPUT_PATH = "autoresearch/results/cycle_198_block0_criticality.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data ──────────────────────────────────────────────────────────
print("Loading data...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
calib_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        calib_ids.append(enc["input_ids"][:, :SEQ_LEN].to(DEVICE))
    if len(calib_ids) >= N_SEQS:
        break
N_tokens = len(calib_ids) * SEQ_LEN
print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} = {N_tokens} tokens")

# ── Collect per-block data via hooks ──────────────────────────────────────────
print("Collecting per-block data via hooks...")

preact_samples = {b: [] for b in range(N_BLOCKS)}   # pre-GELU activations (z)
delta_samples = {b: [] for b in range(N_BLOCKS)}    # block output change per token

# Hook 1: pre-GELU activations from c_fc
preact_hooks = []
for b in range(N_BLOCKS):
    def mk_preact(bb):
        def fn(m, inp, out):
            # out = c_fc(ln_out) = z (pre-GELU), shape (1, SEQ_LEN, 3072)
            preact_samples[bb].append(out.detach().float().cpu().reshape(-1, 3072))
        return fn
    preact_hooks.append(model.transformer.h[b].mlp.c_fc.register_forward_hook(mk_preact(b)))

# Hook 2: block input/output to compute delta
block_in = {b: [] for b in range(N_BLOCKS)}
block_out = {b: [] for b in range(N_BLOCKS)}
block_hooks = []
for b in range(N_BLOCKS):
    def mk_block(bb):
        def fn(m, inp, out):
            # inp[0]: (1, SEQ_LEN, 768) block input
            # out[0]: (1, SEQ_LEN, 768) block output (tuple)
            block_in[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            out_tensor = out[0] if isinstance(out, tuple) else out
            block_out[bb].append(out_tensor.detach().float().cpu().reshape(-1, 768))
        return fn
    block_hooks.append(model.transformer.h[b].register_forward_hook(mk_block(b)))

with torch.no_grad():
    for ids in calib_ids:
        model(ids)

for h in preact_hooks + block_hooks:
    h.remove()

# Concatenate
for b in range(N_BLOCKS):
    preact_samples[b] = torch.cat(preact_samples[b], dim=0)   # (N_tokens, 3072)
    block_in[b] = torch.cat(block_in[b], dim=0)               # (N_tokens, 768)
    block_out[b] = torch.cat(block_out[b], dim=0)             # (N_tokens, 768)

print(f"Collected. Pre-act shape: {preact_samples[0].shape}")

# ── Helper functions ──────────────────────────────────────────────────────────

def effective_rank(vals, threshold=ENERGY_THRESH):
    """Effective rank at given energy threshold of squared singular values."""
    energy = (vals ** 2).cumsum(0) / ((vals ** 2).sum() + 1e-12)
    return int((energy < threshold).sum()) + 1


def gini(vals):
    """Gini coefficient of sorted values (inequality measure, 0=equal, 1=concentrated)."""
    v = vals.sort().values
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float((2 * (idx * v).sum() / (n * v.sum()) - (n + 1) / n).item())

# ── Compute per-block metrics ─────────────────────────────────────────────────
print("Computing per-block metrics...")

per_block = []
for b in range(N_BLOCKS):
    z = preact_samples[b]   # (N_tokens, 3072) pre-GELU
    act = F.gelu(z)         # (N_tokens, 3072)
    delta = block_out[b] - block_in[b]  # (N_tokens, 768) residual change

    # 1. Rank of pre-activation distribution
    z_centered = z - z.mean(0)
    # SVD of centered data: singular values of z_centered (N_tokens, 3072)
    # Use svdvals; expensive for full matrix — use randomized SVD estimate
    # For (N_tokens=3200, 3072): manageable
    S_z = torch.linalg.svdvals(z_centered)  # (min(N, 3072),)
    rank_90_preact = effective_rank(S_z)

    # 2. Channel utilization
    util_01 = float((act.abs() > 0.1).float().mean().item())
    util_10 = float((act.abs() > 1.0).float().mean().item())

    # 3. Block delta (output - input) norms per token
    delta_norms = delta.norm(dim=1)  # (N_tokens,)
    mean_delta = float(delta_norms.mean().item())
    std_delta = float(delta_norms.std().item())

    # 4. c_fc weight singular value concentration
    W = model.transformer.h[b].mlp.c_fc.weight.data.float().cpu()  # (768, 3072)
    S_w = torch.linalg.svdvals(W)  # (768,)
    w_gini = gini(S_w)
    w_top1_frac = float((S_w[0] / (S_w.sum() + 1e-12)).item())

    # 5. Input activation rank (rank of block input covariance)
    x_in = block_in[b]  # (N_tokens, 768)
    x_centered = x_in - x_in.mean(0)
    S_x = torch.linalg.svdvals(x_centered)  # (min(N, 768),)
    rank_90_input = effective_rank(S_x)

    rec = {
        "block": b,
        "preact_rank_90": rank_90_preact,
        "input_rank_90": rank_90_input,
        "channel_util_01": util_01,
        "channel_util_10": util_10,
        "mean_delta_norm": mean_delta,
        "std_delta_norm": std_delta,
        "weight_sv_gini": w_gini,
        "weight_sv_top1_frac": w_top1_frac,
    }
    per_block.append(rec)
    print(f"  Block {b:2d}: preact_rank={rank_90_preact}, input_rank={rank_90_input}, "
          f"util_01={util_01:.3f}, delta={mean_delta:.3f}, gini={w_gini:.3f}", flush=True)

# ── Verdict ───────────────────────────────────────────────────────────────────
def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

others = per_block[1:]
other_preact_ranks = [r["preact_rank_90"] for r in others]
other_deltas = [r["mean_delta_norm"] for r in others]
other_utils = [r["channel_util_01"] for r in others]
other_input_ranks = [r["input_rank_90"] for r in others]

block0 = per_block[0]
med_preact = median(other_preact_ranks)
med_delta = median(other_deltas)
med_util = median(other_utils)
med_input = median(other_input_ranks)

outlier_count = 0
outlier_flags = {}

# Block 0 unique if its preact rank is <50% of median
if block0["preact_rank_90"] < 0.5 * med_preact:
    outlier_count += 1
    outlier_flags["low_preact_rank"] = True
else:
    outlier_flags["low_preact_rank"] = False

# Block 0 unique if its delta norm is >2x median
if block0["mean_delta_norm"] > 2.0 * med_delta:
    outlier_count += 1
    outlier_flags["high_delta_norm"] = True
else:
    outlier_flags["high_delta_norm"] = False

# Block 0 unique if its channel utilization is <50% of median
if block0["channel_util_01"] < 0.5 * med_util:
    outlier_count += 1
    outlier_flags["low_channel_util"] = True
else:
    outlier_flags["low_channel_util"] = False

# Block 0 unique if input rank is <50% of median (raw embeddings less distributed)
if block0["input_rank_90"] < 0.5 * med_input:
    outlier_count += 1
    outlier_flags["low_input_rank"] = True
else:
    outlier_flags["low_input_rank"] = False

if outlier_count >= 2:
    verdict = "CONFIRMED"
else:
    verdict = "REFUTED"

print(f"\nBlock 0 vs others (median):")
print(f"  preact_rank: {block0['preact_rank_90']} vs {med_preact:.0f}")
print(f"  input_rank:  {block0['input_rank_90']} vs {med_input:.0f}")
print(f"  delta_norm:  {block0['mean_delta_norm']:.3f} vs {med_delta:.3f}")
print(f"  util_01:     {block0['channel_util_01']:.3f} vs {med_util:.3f}")
print(f"Outlier flags: {outlier_flags} (count={outlier_count})")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 198,
    "item": 205,
    "verdict": verdict,
    "outlier_count": outlier_count,
    "outlier_flags": outlier_flags,
    "block0_vs_median_others": {
        "preact_rank_90": {"block0": block0["preact_rank_90"], "median_others": med_preact},
        "input_rank_90": {"block0": block0["input_rank_90"], "median_others": med_input},
        "mean_delta_norm": {"block0": block0["mean_delta_norm"], "median_others": med_delta},
        "channel_util_01": {"block0": block0["channel_util_01"], "median_others": med_util},
    },
    "per_block": per_block,
    "config": {
        "N_SEQS": N_SEQS,
        "SEQ_LEN": SEQ_LEN,
        "N_tokens": N_tokens,
        "ENERGY_THRESH": ENERGY_THRESH,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
