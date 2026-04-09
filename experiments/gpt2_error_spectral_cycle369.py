#!/usr/bin/env python3
"""Cycle 358 / Item 369 — Spectral analysis of topk vs LASER error matrices.

Why does topk compose and LASER doesn't?

Hypothesis: LASER error is low-rank (concentrated), topk error is higher-rank
(diffuse). Low-rank errors compound coherently; diffuse errors partially cancel.

Approach:
  For each method applied to block b2 (most concentrated, used in all stages):
  1. Compute block output error E = f_compressed(X) - f_original(X) on calib data
  2. SVD of E: analyze singular value distribution
  3. Compare effective rank, top-1 energy fraction, coherence with residual stream

CONFIRMED if LASER error eff_rank <= 10 AND topk error eff_rank >= 50
REFUTED if both have similar eff_rank (within 2x)
INCONCLUSIVE otherwise
"""
import json
import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_BLOCK = 2  # b2: most concentrated block (eff_rank 1.05)
LASER_RANK = 2    # rank used in cycle 343/352

# topk: same as cycle 348 (1/6 of 3072 = 512 channels)
HIDDEN_FRACTION = 1 / 6  # k = 512

N_CALIB = 200
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOAD_SEED = 42

OUTPUT_PATH = "autoresearch/results/cycle_358_spectral.json"

CONFIRMED_LASER_LE = 10
CONFIRMED_TOPK_GE = 50

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def effective_rank(S):
    """Effective rank from singular values via entropy of normalized squared spectrum."""
    s2 = S ** 2
    p = s2 / s2.sum()
    # Clip for numerical safety
    p = p.clamp(min=1e-12)
    ent = -(p * p.log()).sum()
    return math.exp(float(ent))


def spectral_stats(E, label):
    """Compute and print spectral stats for error matrix E (N, D)."""
    # SVD on CPU float32
    E_cpu = E.float().cpu()
    # Use economy SVD
    U, S, Vh = torch.linalg.svd(E_cpu, full_matrices=False)
    total_energy = float((S ** 2).sum())
    top1_frac = float(S[0] ** 2) / total_energy
    top5_frac = float((S[:5] ** 2).sum()) / total_energy
    top10_frac = float((S[:10] ** 2).sum()) / total_energy
    eff_r = effective_rank(S)
    print(f"\n[{label}] Error matrix E shape: {E_cpu.shape}", flush=True)
    print(f"  Effective rank:    {eff_r:.2f}", flush=True)
    print(f"  Top-1 energy frac: {top1_frac:.4f}", flush=True)
    print(f"  Top-5 energy frac: {top5_frac:.4f}", flush=True)
    print(f"  Top-10 energy frac:{top10_frac:.4f}", flush=True)
    print(f"  Top-10 S values:   {S[:10].tolist()}", flush=True)
    print(f"  Frobenius norm:    {float(E_cpu.norm()):.4f}", flush=True)
    return {
        "label": label,
        "shape": list(E_cpu.shape),
        "eff_rank": eff_r,
        "top1_frac": top1_frac,
        "top5_frac": top5_frac,
        "top10_frac": top10_frac,
        "S_top10": S[:10].tolist(),
        "frobenius": float(E_cpu.norm()),
    }


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB:
        break

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
print(f"Calibration: {len(calib_ids)} seqs", flush=True)


# ── Load model ───────────────────────────────────────────────────────────────
print("Loading GPT-2...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# Extract block b2 original weights
blk = TARGET_BLOCK
mlp = model.transformer.h[blk].mlp
W_fc = mlp.c_fc.weight.data.detach().clone()    # (3072, 768) in HF Conv1D convention
b_fc = mlp.c_fc.bias.data.detach().clone()      # (3072,)
W_proj = mlp.c_proj.weight.data.detach().clone() # (768, 3072) in HF Conv1D (actually 3072,768)
b_proj = mlp.c_proj.bias.data.detach().clone()   # (768,)

# HF GPT-2 uses Conv1D where weight is (in, out), i.e. x @ weight + bias
# c_fc: (768, 3072), c_proj: (3072, 768)
print(f"Block {blk} c_fc weight shape:   {W_fc.shape}", flush=True)
print(f"Block {blk} c_proj weight shape: {W_proj.shape}", flush=True)


# ── Collect MLP inputs and original outputs ──────────────────────────────────
print("\nCollecting block activations...", flush=True)
mlp_inputs = []
mlp_outputs = []

def hook_in_out(module, inputs, output):
    mlp_inputs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu().float())
    mlp_outputs.append(output.detach().reshape(-1, output.shape[-1]).cpu().float())

hook = model.transformer.h[blk].mlp.register_forward_hook(hook_in_out)
with torch.no_grad():
    for ids in calib_ids:
        model(ids)
hook.remove()

X = torch.cat(mlp_inputs, dim=0)   # (N, 768)
Y_orig = torch.cat(mlp_outputs, dim=0)  # (N, 768)
print(f"Collected X: {X.shape}, Y_orig: {Y_orig.shape}", flush=True)

# Also compute hidden activations for topk scoring
with torch.no_grad():
    X_dev = X.to(DEVICE)
    # HF Conv1D: weight stored as (in_features, out_features)
    # c_fc: (768, 3072) → x @ W_fc + b_fc gives (N, 3072)
    if W_fc.shape == (768, 3072):
        preact = X_dev @ W_fc + b_fc
    else:
        # (3072, 768) case — transpose
        preact = X_dev @ W_fc.T + b_fc
    hidden = F.gelu(preact)  # (N, 3072)
    print(f"Hidden activations shape: {hidden.shape}", flush=True)

del model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: LASER (SVD rank-r truncation of c_proj)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"LASER: SVD rank-{LASER_RANK} truncation of c_proj", flush=True)
print(f"{'='*60}", flush=True)

# c_proj weight: HF Conv1D (3072, 768) stored as weight (in=3072, out=768)
# Forward: hidden @ W_proj + b_proj → (N, 768)
if W_proj.shape == (3072, 768):
    W_proj_mat = W_proj.float().cpu()   # (3072, 768)
elif W_proj.shape == (768, 3072):
    W_proj_mat = W_proj.T.float().cpu() # (3072, 768)
else:
    raise ValueError(f"Unexpected c_proj shape: {W_proj.shape}")

U, S_svd, Vh = torch.linalg.svd(W_proj_mat, full_matrices=False)
# W_proj_mat ≈ U @ diag(S_svd) @ Vh  shape: (3072, 768), rank min(3072,768)=768
print(f"W_proj SVD: U={U.shape}, S={S_svd.shape}, Vh={Vh.shape}", flush=True)
print(f"Top-10 singular values: {S_svd[:10].tolist()}", flush=True)

# Truncate to rank r
W_laser = (U[:, :LASER_RANK] * S_svd[:LASER_RANK]) @ Vh[:LASER_RANK, :]  # (3072, 768)

# Compute LASER output on calib data (on CPU)
hidden_cpu = hidden.cpu().float()
Y_laser = hidden_cpu @ W_laser + b_proj.cpu().float()  # (N, 768)

E_laser = Y_laser - Y_orig  # (N, 768)
print(f"LASER error range: [{float(E_laser.min()):.4f}, {float(E_laser.max()):.4f}]", flush=True)
laser_stats = spectral_stats(E_laser, f"LASER rank-{LASER_RANK}")


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: topk+lstsq (k = 512 channels)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
k = int(3072 * HIDDEN_FRACTION)
print(f"topk+lstsq: k={k} channels (fraction={HIDDEN_FRACTION:.3f})", flush=True)
print(f"{'='*60}", flush=True)

hidden_cpu = hidden.cpu().float()

# Score channels: std * (|mean| + |mean_deriv| + eps)
preact_cpu = preact.cpu().float()
inv_sqrt_2 = 1.0 / math.sqrt(2.0)
inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
deriv = (0.5 * (1.0 + torch.erf(preact_cpu * inv_sqrt_2))
         + preact_cpu * torch.exp(-0.5 * preact_cpu.square()) * inv_sqrt_2pi)

std_h = hidden_cpu.std(dim=0)        # (3072,)
mean_h = hidden_cpu.mean(dim=0)      # (3072,)
mean_d = deriv.mean(dim=0)           # (3072,)

# Include c_proj column norms in scoring
if W_proj_mat.shape == (3072, 768):
    col_norms = W_proj_mat.norm(dim=1)  # (3072,) — one norm per hidden channel
else:
    col_norms = torch.ones(3072)

scores = std_h * col_norms * (mean_h.abs() + mean_d.abs() + 1e-6)
top_indices = scores.topk(k).indices.sort().values  # (k,)
print(f"Selected {k} channels. Score range: [{float(scores.min()):.4f}, {float(scores.max()):.4f}]", flush=True)

# Lstsq: fit W_out s.t. h_sel @ W_out ≈ Y_orig
h_sel = hidden_cpu[:, top_indices]  # (N, k)
# Add bias column
h_aug = torch.cat([h_sel, torch.ones(h_sel.shape[0], 1)], dim=1)  # (N, k+1)
# Lstsq: min ||h_aug @ P - Y_orig||_F
result = torch.linalg.lstsq(h_aug, Y_orig, driver="gelsd")
P = result.solution  # (k+1, 768)
W_out = P[:k]        # (k, 768)
b_out = P[k]         # (768,)

Y_topk = h_aug @ P  # (N, 768)
E_topk = Y_topk - Y_orig

fit_res = float((Y_topk - Y_orig).norm() / Y_orig.norm())
print(f"lstsq relative residual: {fit_res:.4f}", flush=True)
print(f"topk error range: [{float(E_topk.min()):.4f}, {float(E_topk.max()):.4f}]", flush=True)
topk_stats = spectral_stats(E_topk, f"topk k={k}+lstsq")


# ══════════════════════════════════════════════════════════════════════════════
# Extra: check coherence with residual stream direction
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("Coherence with residual stream (X):", flush=True)

# Normalize X and error matrices for cosine
X_cpu = X.float()
X_norm = X_cpu / (X_cpu.norm(dim=1, keepdim=True) + 1e-8)  # (N, 768)

# For each error vector, compute cosine with corresponding residual
cos_laser = (E_laser * X_norm).sum(dim=1)  # (N,)
cos_topk  = (E_topk  * X_norm).sum(dim=1)  # (N,)
print(f"LASER error-residual cosine: mean={float(cos_laser.mean()):.4f}, "
      f"std={float(cos_laser.std()):.4f}, |mean|={float(cos_laser.abs().mean()):.4f}", flush=True)
print(f"topk  error-residual cosine: mean={float(cos_topk.mean()):.4f}, "
      f"std={float(cos_topk.std()):.4f}, |mean|={float(cos_topk.abs().mean()):.4f}", flush=True)

laser_coherence = {
    "mean": float(cos_laser.mean()), "std": float(cos_laser.std()),
    "abs_mean": float(cos_laser.abs().mean()),
}
topk_coherence = {
    "mean": float(cos_topk.mean()), "std": float(cos_topk.std()),
    "abs_mean": float(cos_topk.abs().mean()),
}


# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════
laser_eff = laser_stats["eff_rank"]
topk_eff  = topk_stats["eff_rank"]
ratio = topk_eff / laser_eff

print(f"\n{'='*60}", flush=True)
print("VERDICT", flush=True)
print(f"  LASER eff_rank: {laser_eff:.2f}  (threshold: <= {CONFIRMED_LASER_LE})", flush=True)
print(f"  topk  eff_rank: {topk_eff:.2f}   (threshold: >= {CONFIRMED_TOPK_GE})", flush=True)
print(f"  Ratio topk/LASER: {ratio:.1f}x  (need 10x+ for CONFIRMED)", flush=True)

if laser_eff <= CONFIRMED_LASER_LE and topk_eff >= CONFIRMED_TOPK_GE:
    verdict = "CONFIRMED"
elif topk_eff / max(laser_eff, 1e-6) < 2.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n  Verdict: {verdict}", flush=True)


# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "cycle": 358,
    "item": 369,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "config": {
        "target_block": TARGET_BLOCK,
        "laser_rank": LASER_RANK,
        "topk_k": k,
        "n_calib": N_CALIB,
    },
    "laser": laser_stats,
    "topk": topk_stats,
    "laser_coherence": laser_coherence,
    "topk_coherence": topk_coherence,
    "ratio_topk_over_laser": ratio,
    "verdict": verdict,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
