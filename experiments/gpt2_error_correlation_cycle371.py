#!/usr/bin/env python3
"""Cycle 362 / Item 371 — Cross-block error decorrelation for channel selection.

Current topk selects channels per-block by importance (output norm contribution).
Since topk error is diffuse (eff_rank ~463 from cycle 358), errors may partially
cancel across blocks. But even topk fails beyond 3 blocks (c357: 12-block PPL 186,
c361: 6-block PPL 65.54).

This is an ANALYSIS item: measure whether per-block errors are correlated.

Approach:
  For each of blocks [b2, b1, b10, b4, b3, b11]:
    1. Apply topk k=512 channel selection (same as c348) independently
    2. Compute error E_b = f_compressed(X) - f_original(X), shape (N*128, 768)
    3. Flatten to a vector: vec(E_b)
  Then compute the 6x6 pairwise cosine correlation matrix.

Verdict:
  CONFIRMED if mean off-diagonal |correlation| >= 0.3 (errors correlated → decorrelation could help)
  REFUTED if mean |correlation| < 0.1 (already decorrelated → problem is elsewhere)
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
BLOCKS = [2, 1, 10, 4, 3, 11]   # same combo as c357 6-block test
HIDDEN_FRACTION = 1 / 6          # k = 512 of 3072

N_CALIB = 200
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOAD_SEED = 42

CONFIRMED_CORR_GE = 0.3
REFUTED_CORR_LT   = 0.1

OUTPUT_PATH = "autoresearch/results/cycle_362_error_correlation.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)
print(f"Blocks: {BLOCKS}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def effective_rank(S):
    s2 = S ** 2
    p = (s2 / s2.sum()).clamp(min=1e-12)
    return math.exp(float(-(p * p.log()).sum()))


def topk_lstsq_error(X, Y_orig, W_fc, b_fc, W_proj, b_proj, k, device):
    """Compute topk+lstsq block error for one block, returns E (N, 768) on CPU float32."""
    X_dev = X.to(device)

    # HF GPT-2 Conv1D: weight is (in_features, out_features)
    # c_fc: (768, 3072)  →  preact = x @ W_fc + b_fc  →  (N, 3072)
    if W_fc.shape == (768, 3072):
        preact = X_dev @ W_fc.to(device) + b_fc.to(device)
    else:
        preact = X_dev @ W_fc.T.to(device) + b_fc.to(device)

    hidden = F.gelu(preact)  # (N, 3072)

    # c_proj: (3072, 768)  →  output = hidden @ W_proj + b_proj
    if W_proj.shape == (3072, 768):
        W_proj_mat = W_proj.float()
    else:
        W_proj_mat = W_proj.T.float()

    # Score channels for topk selection
    preact_cpu = preact.cpu().float()
    hidden_cpu = hidden.cpu().float()

    inv_sqrt_2    = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi  = 1.0 / math.sqrt(2.0 * math.pi)
    deriv = (0.5 * (1.0 + torch.erf(preact_cpu * inv_sqrt_2))
             + preact_cpu * torch.exp(-0.5 * preact_cpu.square()) * inv_sqrt_2pi)

    std_h    = hidden_cpu.std(dim=0)    # (3072,)
    mean_h   = hidden_cpu.mean(dim=0)   # (3072,)
    mean_d   = deriv.mean(dim=0)        # (3072,)

    W_proj_cpu = W_proj_mat.cpu()
    col_norms  = W_proj_cpu.norm(dim=1)  # (3072,) — one norm per hidden channel

    scores      = std_h * col_norms * (mean_h.abs() + mean_d.abs() + 1e-6)
    top_indices = scores.topk(k).indices.sort().values  # (k,)

    # lstsq: fit W_out s.t. h_sel @ W_out ≈ Y_orig
    h_sel = hidden_cpu[:, top_indices]                         # (N, k)
    h_aug = torch.cat([h_sel, torch.ones(h_sel.shape[0], 1)], dim=1)  # (N, k+1)
    result = torch.linalg.lstsq(h_aug, Y_orig.float().cpu(), driver="gelsd")
    P      = result.solution  # (k+1, 768)

    Y_topk = h_aug @ P         # (N, 768)
    E      = Y_topk - Y_orig.float().cpu()  # (N, 768)

    rel_res = float((Y_topk - Y_orig.float().cpu()).norm() / Y_orig.float().cpu().norm())
    print(f"  lstsq rel residual: {rel_res:.4f}", flush=True)

    return E, top_indices


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]

all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB:
        break

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} tokens = "
      f"{len(calib_ids)*SEQ_LEN} token positions", flush=True)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

k = int(3072 * HIDDEN_FRACTION)
print(f"k = {k} channels ({HIDDEN_FRACTION:.3f} of 3072)", flush=True)

errors = {}  # block_idx → E (N, 768) on CPU

for blk in BLOCKS:
    print(f"\n{'='*60}", flush=True)
    print(f"Block {blk}", flush=True)

    mlp    = model.transformer.h[blk].mlp
    W_fc   = mlp.c_fc.weight.data.detach().clone()
    b_fc   = mlp.c_fc.bias.data.detach().clone()
    W_proj = mlp.c_proj.weight.data.detach().clone()
    b_proj = mlp.c_proj.bias.data.detach().clone()

    print(f"  c_fc shape: {W_fc.shape}, c_proj shape: {W_proj.shape}", flush=True)

    # Collect MLP inputs and original outputs for this block
    mlp_inputs  = []
    mlp_outputs = []

    def _hook(module, inputs, output,
              _in=mlp_inputs, _out=mlp_outputs):
        _in.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu().float())
        _out.append(output.detach().reshape(-1, output.shape[-1]).cpu().float())

    hook = model.transformer.h[blk].mlp.register_forward_hook(_hook)
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    hook.remove()

    X      = torch.cat(mlp_inputs,  dim=0)  # (N, 768)
    Y_orig = torch.cat(mlp_outputs, dim=0)  # (N, 768)
    print(f"  X: {X.shape}, Y_orig: {Y_orig.shape}", flush=True)

    t0 = time.time()
    E, top_idx = topk_lstsq_error(X, Y_orig, W_fc, b_fc, W_proj, b_proj, k, DEVICE)
    print(f"  Error shape: {E.shape}, time: {time.time()-t0:.1f}s", flush=True)

    # Quick spectral summary
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    er = effective_rank(S)
    fro = float(E.norm())
    print(f"  Error eff_rank: {er:.1f}, Frobenius: {fro:.4f}", flush=True)

    errors[blk] = E
    del X, Y_orig, mlp_inputs, mlp_outputs

if DEVICE == "mps":
    torch.mps.empty_cache()

del model


# ── Compute pairwise cosine correlation of flattened errors ───────────────────
print(f"\n{'='*60}", flush=True)
print("Computing 6×6 pairwise cosine correlation matrix...", flush=True)

# Stack errors: for each block, flatten to (N*768,) vector then normalize
vecs = []
for blk in BLOCKS:
    E = errors[blk]  # (N, 768)
    v = E.reshape(-1).float()  # (N*768,)
    v = v / (v.norm() + 1e-12)
    vecs.append(v)

# Pairwise cosine
n_blk = len(BLOCKS)
corr_matrix = torch.zeros(n_blk, n_blk)
for i in range(n_blk):
    for j in range(n_blk):
        corr_matrix[i, j] = float(torch.dot(vecs[i], vecs[j]))

print("\nCorrelation matrix (rows/cols = blocks in order):", flush=True)
header = "       " + "  ".join(f"b{b:2d}" for b in BLOCKS)
print(header, flush=True)
for i, blk_i in enumerate(BLOCKS):
    row = f"b{blk_i:2d}  " + "  ".join(f"{corr_matrix[i,j]:+.3f}" for j in range(n_blk))
    print(row, flush=True)

# Compute mean off-diagonal |correlation|
off_diag = []
for i in range(n_blk):
    for j in range(n_blk):
        if i != j:
            off_diag.append(abs(float(corr_matrix[i, j])))

mean_off_diag = sum(off_diag) / len(off_diag)
max_off_diag  = max(off_diag)
min_off_diag  = min(off_diag)

print(f"\nOff-diagonal |correlation|: mean={mean_off_diag:.4f}, "
      f"max={max_off_diag:.4f}, min={min_off_diag:.4f}", flush=True)


# ── Per-block error spectral stats ────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("Per-block error spectral summary:", flush=True)
block_stats = {}
for blk in BLOCKS:
    E = errors[blk]
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    s2 = S ** 2
    total = float(s2.sum())
    er = effective_rank(S)
    top1 = float(s2[0]) / total
    fro  = float(E.norm())
    print(f"  b{blk}: eff_rank={er:.1f}, top1_frac={top1:.3f}, frob={fro:.4f}", flush=True)
    block_stats[blk] = {"eff_rank": er, "top1_frac": top1, "frobenius": fro}


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("VERDICT", flush=True)
print(f"  Mean off-diagonal |cosine|: {mean_off_diag:.4f}", flush=True)
print(f"  CONFIRMED threshold: >= {CONFIRMED_CORR_GE}", flush=True)
print(f"  REFUTED   threshold: <  {REFUTED_CORR_LT}", flush=True)

if mean_off_diag >= CONFIRMED_CORR_GE:
    verdict = "CONFIRMED"
    print("  → CONFIRMED: errors are correlated across blocks; "
          "deliberate decorrelation could help", flush=True)
elif mean_off_diag < REFUTED_CORR_LT:
    verdict = "REFUTED"
    print("  → REFUTED: errors already decorrelated; "
          "composition failure is elsewhere", flush=True)
else:
    verdict = "INCONCLUSIVE"
    print(f"  → INCONCLUSIVE: moderate correlation ({mean_off_diag:.4f}), "
          "neither clearly correlated nor decorrelated", flush=True)

print(f"\n  Verdict: {verdict}", flush=True)


# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "cycle": 362,
    "item": 371,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "config": {
        "blocks": BLOCKS,
        "k": k,
        "hidden_fraction": HIDDEN_FRACTION,
        "n_calib": N_CALIB,
        "seq_len": SEQ_LEN,
    },
    "correlation_matrix": corr_matrix.tolist(),
    "blocks_order": BLOCKS,
    "off_diagonal": {
        "mean_abs": mean_off_diag,
        "max_abs": max_off_diag,
        "min_abs": min_off_diag,
    },
    "block_stats": {str(b): v for b, v in block_stats.items()},
    "verdict": verdict,
    "confirmed_threshold": CONFIRMED_CORR_GE,
    "refuted_threshold": REFUTED_CORR_LT,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {OUTPUT_PATH}", flush=True)
