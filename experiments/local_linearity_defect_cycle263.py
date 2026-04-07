"""
Cycle 263 — Item 273: Local linearity defect ε_l(x, h).

For each (block l, activation σ ∈ {GELU, ReLU+b_leak}, calib token x):
    f(x) = c_proj(σ(c_fc(x))) + c_proj.bias       — block MLP function
    J(x) = W_fc.T @ diag(σ'(z)) @ W_proj          — Jacobian (768 × 768)
    ε(x, h) = ‖f(x+h) − f(x) − J(x) h‖_G / ‖h‖²_G under G = U^T U.

64 calib tokens × 32 isotropic Gaussian directions × 4 ε scales × 12 blocks × 2 σ types.

Verdict (item 273):
  - CONFIRMED iff median ε_G/ε_R ≥ 5 at smallest ‖h‖ AND p99(ε_R) / p99(ε_G) ≥ 1
    in ≥3 of blocks 0..4.
  - REFUTED   iff median AND p99 ratios both ∈ [0.5, 2] at all 12 blocks.
  - INCONCLUSIVE otherwise.
"""
import json
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from aleph.pullback_geometry import pullback_norm, get_unembedding

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB_SEQS_FOR_LOADER = 200
SEQ_LEN = 128
N_TOKENS = 64                  # 64 token positions of the first sequence
N_DIRECTIONS = 32              # 32 isotropic Gaussian directions per token
EPS_FRACS = [0.01, 0.03, 0.1, 0.3]  # × ‖x‖_G
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LINALG = "cpu"          # do all per-token math on CPU float32 for reliability

RESULT_PATH = "autoresearch/results/cycle_263_local_linearity_defect.json"
SUMMARY_PATH = "autoresearch/results/cycle_263_summary.md"

torch.manual_seed(SEED)
print(f"Model device: {DEVICE}", flush=True)
print(f"Linalg device: {DEVICE_LINALG}", flush=True)

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data (matches cycle 218/258/260 loader) ──────────────────────
print("Loading wikitext-2 validation split...", flush=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB_SEQS_FOR_LOADER:
        break
calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB_SEQS_FOR_LOADER]]
print(f"Calib loader: {len(calib_ids)} seqs × {SEQ_LEN} tokens "
      f"= {len(calib_ids) * SEQ_LEN} tokens (b_leak source)", flush=True)
print(f"Token sample for ε_defect: first {N_TOKENS} positions of seq 0", flush=True)

# ── Hook ln_2 outputs (== c_fc inputs) on every block ────────────────────────
N_BLOCKS = 12
cfc_in_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_hook(l):
    def fn(module, input, output):
        cfc_in_stash[l].append(input[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_cfc_hook(l))
    hooks.append(h)

print("Forward pass to capture ln_2 outputs across all calib tokens...", flush=True)
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

# Concatenate (n_seqs * seq_len, 768) per block
ln2_out_full = {}
for l in range(N_BLOCKS):
    x = torch.cat(cfc_in_stash[l], dim=0).reshape(-1, 768)
    ln2_out_full[l] = x.contiguous()
cfc_in_stash = None
n_total_tokens = ln2_out_full[0].shape[0]
print(f"Captured ln_2 outputs: {ln2_out_full[0].shape} per block "
      f"({n_total_tokens} tokens total)", flush=True)

# Token sample for ε_defect = first N_TOKENS positions of the first sequence
ln2_token_sample = {l: ln2_out_full[l][:N_TOKENS].clone() for l in range(N_BLOCKS)}

# ── Cache weights on linalg device ───────────────────────────────────────────
W_fc_per_block = {}
b_fc_per_block = {}
W_proj_per_block = {}
b_proj_per_block = {}
for l in range(N_BLOCKS):
    block = model.transformer.h[l]
    W_fc_per_block[l] = block.mlp.c_fc.weight.detach().float().to(DEVICE_LINALG)    # (768, 3072)
    b_fc_per_block[l] = block.mlp.c_fc.bias.detach().float().to(DEVICE_LINALG)      # (3072,)
    W_proj_per_block[l] = block.mlp.c_proj.weight.detach().float().to(DEVICE_LINALG)  # (3072, 768)
    b_proj_per_block[l] = block.mlp.c_proj.bias.detach().float().to(DEVICE_LINALG)    # (768,)
    assert W_fc_per_block[l].shape == (768, 3072)
    assert W_proj_per_block[l].shape == (3072, 768)

# ── Precompute b_leak per block (cycle 258 recipe) ───────────────────────────
print("\nPrecomputing b_leak per block (E[gelu - relu] over all calib tokens)...", flush=True)
b_leak_per_block = {}
for l in range(N_BLOCKS):
    x_all = ln2_out_full[l]                          # (N, 768)
    z_all = x_all @ W_fc_per_block[l] + b_fc_per_block[l]  # (N, 3072)
    gelu_all = F.gelu(z_all, approximate='tanh')
    relu_all = F.relu(z_all)
    b_leak = (gelu_all - relu_all).mean(dim=0)       # (3072,)
    b_leak_per_block[l] = b_leak.contiguous()
    print(f"  block {l:2d}: b_leak mean={float(b_leak.mean()):+.5f} "
          f"min={float(b_leak.min()):+.5f} max={float(b_leak.max()):+.5f} "
          f"|b|={float(b_leak.abs().mean()):.5f}", flush=True)
ln2_out_full = None  # free

# ── Unembedding U for pullback metric ────────────────────────────────────────
U = get_unembedding(model).detach().float().to(DEVICE_LINALG)  # (50257, 768)
print(f"\nU shape = {tuple(U.shape)}", flush=True)

# ── Activation function table ────────────────────────────────────────────────
def gelu_apply(z):
    return F.gelu(z, approximate='tanh')

def gelu_prime(z):
    # fresh autograd
    z_leaf = z.detach().clone().requires_grad_(True)
    y = F.gelu(z_leaf, approximate='tanh').sum()
    return torch.autograd.grad(y, z_leaf)[0].detach()

def relu_bleak_apply(z, b_leak):
    return F.relu(z) + b_leak

def relu_prime(z):
    return (z > 0).float()

# ── MLP function and Jacobian ────────────────────────────────────────────────
def mlp_forward(x_batch, W_fc, b_fc, W_proj, b_proj, sigma_type, b_leak):
    """f(x) = c_proj(σ(x @ W_fc + b_fc)) + b_proj. x_batch: (B, 768) → (B, 768)."""
    z = x_batch @ W_fc + b_fc                   # (B, 3072)
    if sigma_type == "GELU":
        a = gelu_apply(z)
    elif sigma_type == "ReLU_bleak":
        a = relu_bleak_apply(z, b_leak)
    else:
        raise ValueError(sigma_type)
    return a @ W_proj + b_proj                  # (B, 768)

def jacobian_at(x, W_fc, b_fc, W_proj, sigma_type):
    """J(x) = W_fc @ diag(σ'(z)) @ W_proj  →  shape (768, 768).  x: (768,)."""
    z = x @ W_fc + b_fc                         # (3072,)
    if sigma_type == "GELU":
        sp = gelu_prime(z)                      # (3072,)
    elif sigma_type == "ReLU_bleak":
        sp = relu_prime(z)
    else:
        raise ValueError(sigma_type)
    # J = W_fc * diag(sp) @ W_proj  →  (768, 3072) * (3072,) @ (3072, 768)
    return (W_fc * sp.unsqueeze(0)) @ W_proj    # (768, 768)

# ── Main loop ────────────────────────────────────────────────────────────────
print("\n──────── Computing ε_defect per (l, σ, token, scale) ────────", flush=True)

# Pre-sample directions: shared across all (l, σ, token) for reproducibility.
# (32, 768) unit vectors.
gen = torch.Generator().manual_seed(SEED)
H_unit = torch.randn(N_DIRECTIONS, 768, generator=gen)
H_unit = H_unit / H_unit.norm(dim=1, keepdim=True)
H_unit = H_unit.to(DEVICE_LINALG).float()       # (32, 768)
print(f"Pre-sampled directions H_unit: {tuple(H_unit.shape)} "
      f"unit-norm check: ‖H[0]‖={float(H_unit[0].norm()):.6f}", flush=True)

n_scales = len(EPS_FRACS)

# results[(l, σ, scale_idx)] = list of ε_defect values, one per (token, dir) — 64*32 = 2048
results_raw = {}

for sigma_type in ["GELU", "ReLU_bleak"]:
    print(f"\n=== σ = {sigma_type} ===", flush=True)
    for l in range(N_BLOCKS):
        W_fc = W_fc_per_block[l]
        b_fc = b_fc_per_block[l]
        W_proj = W_proj_per_block[l]
        b_proj = b_proj_per_block[l]
        b_leak = b_leak_per_block[l] if sigma_type == "ReLU_bleak" else None
        x_tokens = ln2_token_sample[l]  # (64, 768) on cpu

        # buckets
        for si in range(n_scales):
            results_raw[(l, sigma_type, si)] = []

        for t in range(N_TOKENS):
            x = x_tokens[t]                          # (768,)
            # ‖x‖_G via pullback metric
            x_norm_g = float(pullback_norm(x.unsqueeze(0), U).item())
            if x_norm_g < 1e-12:
                # degenerate (shouldn't happen for ln_2 outputs)
                continue

            # f(x) once
            f_x = mlp_forward(x.unsqueeze(0), W_fc, b_fc, W_proj, b_proj, sigma_type, b_leak)  # (1, 768)

            # J(x) once
            J = jacobian_at(x, W_fc, b_fc, W_proj, sigma_type)  # (768, 768)

            # For each scale, build the 32-direction batch δ = ε * H_unit, ε = frac * x_norm_g
            for si, frac in enumerate(EPS_FRACS):
                eps_mag = frac * x_norm_g
                delta = H_unit * eps_mag                 # (32, 768)

                # f(x + δ) batched (32, 768)
                x_perturbed = x.unsqueeze(0) + delta     # (32, 768)
                f_xpd = mlp_forward(x_perturbed, W_fc, b_fc, W_proj, b_proj, sigma_type, b_leak)

                # J @ δ.T → (768, 32) → .T → (32, 768)
                J_delta = (J @ delta.T).T                # (32, 768)

                # residual
                residual = f_xpd - f_x - J_delta         # (32, 768)

                # G-norms
                num = pullback_norm(residual, U)         # (32,)
                den = pullback_norm(delta, U)            # (32,)  (NB: ‖δ‖_G is the same for all 32 dirs only if H_unit dirs share same G-norm — they do NOT, so denominator varies per dir)
                # ε_defect = ‖residual‖_G / ‖δ‖_G^2
                eps_defect = num / den.square().clamp_min(1e-30)

                results_raw[(l, sigma_type, si)].extend(eps_defect.tolist())

        # Per-block summary print
        for si, frac in enumerate(EPS_FRACS):
            vals = torch.tensor(results_raw[(l, sigma_type, si)], dtype=torch.float64)
            med = float(torch.quantile(vals, 0.5).item())
            p90 = float(torch.quantile(vals, 0.9).item())
            p99 = float(torch.quantile(vals, 0.99).item())
            print(f"  block {l:2d} σ={sigma_type:10s} ε={frac:0.2f}·‖x‖: "
                  f"n={len(vals)} med={med:.3e} p90={p90:.3e} p99={p99:.3e}", flush=True)

# ── Aggregate ────────────────────────────────────────────────────────────────
print("\n──────── Aggregate per (l, σ, scale) ────────", flush=True)
agg = {}
for l in range(N_BLOCKS):
    agg[l] = {}
    for sigma_type in ["GELU", "ReLU_bleak"]:
        agg[l][sigma_type] = {}
        for si, frac in enumerate(EPS_FRACS):
            vals = torch.tensor(results_raw[(l, sigma_type, si)], dtype=torch.float64)
            agg[l][sigma_type][f"frac_{frac}"] = {
                "median": float(torch.quantile(vals, 0.5).item()),
                "p90": float(torch.quantile(vals, 0.9).item()),
                "p99": float(torch.quantile(vals, 0.99).item()),
                "n": int(vals.numel()),
            }

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n──────── Verdict logic ────────", flush=True)

# Per-block ratios at smallest scale
smallest_frac = EPS_FRACS[0]
key = f"frac_{smallest_frac}"

per_block_summary = []
n_pass_clause_a = 0  # blocks 0..4 with median ε_G/ε_R ≥ 5
n_pass_clause_b = 0  # blocks 0..4 with p99 ε_R / p99 ε_G ≥ 1
all_med_in_band = True
all_p99_in_band = True

for l in range(N_BLOCKS):
    g_med = agg[l]["GELU"][key]["median"]
    r_med = agg[l]["ReLU_bleak"][key]["median"]
    g_p99 = agg[l]["GELU"][key]["p99"]
    r_p99 = agg[l]["ReLU_bleak"][key]["p99"]
    med_ratio = g_med / max(r_med, 1e-30)
    p99_ratio_RG = r_p99 / max(g_p99, 1e-30)
    p99_ratio_GR = g_p99 / max(r_p99, 1e-30)
    in_band_med = (0.5 <= med_ratio <= 2.0)
    in_band_p99 = (0.5 <= p99_ratio_GR <= 2.0)
    if not in_band_med:
        all_med_in_band = False
    if not in_band_p99:
        all_p99_in_band = False
    per_block_summary.append({
        "block": l,
        "g_median": g_med,
        "r_median": r_med,
        "g_p99": g_p99,
        "r_p99": r_p99,
        "median_ratio_GoverR": med_ratio,
        "p99_ratio_RoverG": p99_ratio_RG,
        "in_band_median": bool(in_band_med),
        "in_band_p99": bool(in_band_p99),
    })
    if l <= 4:
        if med_ratio >= 5.0:
            n_pass_clause_a += 1
        if p99_ratio_RG >= 1.0:
            n_pass_clause_b += 1
    print(f"  block {l:2d}: med G={g_med:.3e} R={r_med:.3e} ratio G/R={med_ratio:.3e}  "
          f"p99 G={g_p99:.3e} R={r_p99:.3e} ratio R/G={p99_ratio_RG:.3e}", flush=True)

confirmed_a = n_pass_clause_a >= 3
confirmed_b = n_pass_clause_b >= 3
confirmed = confirmed_a and confirmed_b
refuted = all_med_in_band and all_p99_in_band

if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n  Clause A (med G/R ≥ 5 in ≥3 of blocks 0-4): {n_pass_clause_a}/5 -> "
      f"{'PASS' if confirmed_a else 'FAIL'}", flush=True)
print(f"  Clause B (p99 R/G ≥ 1 in ≥3 of blocks 0-4):  {n_pass_clause_b}/5 -> "
      f"{'PASS' if confirmed_b else 'FAIL'}", flush=True)
print(f"  REFUTED if all 12 blocks: med ratio ∈[0.5,2] AND p99 ratio ∈[0.5,2]: "
      f"med_in_band={all_med_in_band} p99_in_band={all_p99_in_band}", flush=True)
print(f"\n  VERDICT: {verdict}", flush=True)

# ── Quadratic-scaling check (informational) ─────────────────────────────────
print("\n──────── Scale dependence: median ε(scale) ────────", flush=True)
scale_check = {}
for sigma_type in ["GELU", "ReLU_bleak"]:
    scale_check[sigma_type] = {}
    for l in range(N_BLOCKS):
        meds = []
        for si, frac in enumerate(EPS_FRACS):
            meds.append(agg[l][sigma_type][f"frac_{frac}"]["median"])
        scale_check[sigma_type][l] = meds
        print(f"  σ={sigma_type:10s} block {l:2d} medians @ {EPS_FRACS}: "
              f"[{meds[0]:.2e} {meds[1]:.2e} {meds[2]:.2e} {meds[3]:.2e}]", flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 263,
    "item": 273,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_calib_seqs_for_loader": N_CALIB_SEQS_FOR_LOADER,
        "n_tokens": N_TOKENS,
        "n_directions": N_DIRECTIONS,
        "eps_fracs_of_pullback_norm": EPS_FRACS,
        "seq_len": SEQ_LEN,
        "n_blocks": N_BLOCKS,
        "seed": SEED,
        "device_model": DEVICE,
        "device_linalg": DEVICE_LINALG,
    },
    "trainable_params": 0,
    "per_block_summary_at_smallest_scale": per_block_summary,
    "agg_per_block_per_sigma_per_scale": {
        str(l): {
            sigma: agg[l][sigma]
            for sigma in ["GELU", "ReLU_bleak"]
        }
        for l in range(N_BLOCKS)
    },
    "scale_dependence_medians": {
        sigma: {str(l): scale_check[sigma][l] for l in range(N_BLOCKS)}
        for sigma in ["GELU", "ReLU_bleak"]
    },
    "verdict_logic": {
        "smallest_scale_frac": smallest_frac,
        "n_pass_clause_a_med_GR_ge_5_in_blocks_0_4": n_pass_clause_a,
        "n_pass_clause_b_p99_RG_ge_1_in_blocks_0_4": n_pass_clause_b,
        "confirmed_a_passes": bool(confirmed_a),
        "confirmed_b_passes": bool(confirmed_b),
        "all_blocks_med_ratio_in_0p5_2": bool(all_med_in_band),
        "all_blocks_p99_ratio_in_0p5_2": bool(all_p99_in_band),
        "confirmed_iff": "n_clause_a >=3 AND n_clause_b >=3",
        "refuted_iff": "all 12 blocks have BOTH med-ratio AND p99-ratio in [0.5, 2]",
    },
}

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {RESULT_PATH}", flush=True)

# ── Summary md ───────────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 263 — Item 273: Local linearity defect ε_l(x, h)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- N_TOKENS={N_TOKENS}, N_DIRECTIONS={N_DIRECTIONS}, EPS_FRACS={EPS_FRACS}, "
             f"trainable_params=0.")
lines.append(f"- ε(x, h) = ‖f(x+h) − f(x) − J(x) h‖_G / ‖h‖²_G,  G = U^T U.")
lines.append(f"- f(x) = c_proj(σ(c_fc(x) + b_fc) [+ b_leak if ReLU]) + c_proj.bias.")
lines.append("")
lines.append(f"## Per-block summary at smallest scale (ε = {smallest_frac} · ‖x‖_G)")
lines.append("")
lines.append("| block | med G | med R | med G/R | p99 G | p99 R | p99 R/G |")
lines.append("|------:|------:|------:|--------:|------:|------:|--------:|")
for r in per_block_summary:
    lines.append(f"| {r['block']} | {r['g_median']:.2e} | {r['r_median']:.2e} | "
                 f"{r['median_ratio_GoverR']:.2e} | {r['g_p99']:.2e} | "
                 f"{r['r_p99']:.2e} | {r['p99_ratio_RoverG']:.2e} |")
lines.append("")
lines.append("## Verdict logic")
lines.append("")
lines.append(f"- CONFIRMED clause A (med G/R ≥ 5 in ≥3 of blocks 0-4): "
             f"**{n_pass_clause_a}/5** -> {'PASS' if confirmed_a else 'FAIL'}")
lines.append(f"- CONFIRMED clause B (p99 R/G ≥ 1 in ≥3 of blocks 0-4): "
             f"**{n_pass_clause_b}/5** -> {'PASS' if confirmed_b else 'FAIL'}")
lines.append(f"- REFUTED (all 12 blocks med AND p99 ratios ∈ [0.5, 2]): "
             f"med_in_band={all_med_in_band}, p99_in_band={all_p99_in_band}")
lines.append(f"- **Verdict: {verdict}**")
lines.append("")
lines.append("## Scale dependence (median ε per scale)")
lines.append("")
lines.append(f"Scales: {EPS_FRACS} × ‖x‖_G")
lines.append("")
for sigma_type in ["GELU", "ReLU_bleak"]:
    lines.append(f"### σ = {sigma_type}")
    lines.append("")
    lines.append("| block | ε=0.01 | ε=0.03 | ε=0.10 | ε=0.30 |")
    lines.append("|------:|-------:|-------:|-------:|-------:|")
    for l in range(N_BLOCKS):
        m = scale_check[sigma_type][l]
        lines.append(f"| {l} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} |")
    lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
