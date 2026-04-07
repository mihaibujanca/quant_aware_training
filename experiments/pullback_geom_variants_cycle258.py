"""
Cycle 258 — Item 269: Pullback curvature & salience for GELU vs ReLU+bias vs gelu_noneg+μ.

For each of three GPT-2 small model variants on the standard 200×128 wikitext-2 calib slice:
  (A) baseline GELU (untouched HF gpt2)
  (B) ReLU + per-block per-channel b_leak (cycle 181 first-moment recipe, expressed as
      a post-activation offset on the intermediate (3072-dim) space)
  (C) gelu_noneg + per-channel μ_l (cycle 228 GeluNonegCorrected, verbatim)

Collect residual stream trajectory at all 13 layer boundaries, compute pullback κ_l and
S_l per token, aggregate (mean / p10 / p90) per layer, and report Pearson ρ(S_l, κ_l)
per variant.

Verdict (item 269):
  - CONFIRMED if max_l |κ_GELU - κ_ReLU| / κ_GELU >= 0.30 in some layer l in 0..6
    AND |ρ_ReLU(S, κ)| <= 0.5
  - REFUTED if all per-layer relative κ differences < 0.10
    AND ρ_ReLU in [-0.85, -0.65]
  - INCONCLUSIVE otherwise
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from aleph.pullback_geometry import (
    collect_residual_trajectory,
    compute_curvature,
    compute_salience,
    get_unembedding,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

RESULT_PATH = "autoresearch/results/cycle_258_pullback_variants.json"
SUMMARY_PATH = "autoresearch/results/cycle_258_summary.md"
PLOT_PATH = "autoresearch/results/cycle_258_kappa_salience.png"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data (cycle 220/228 loader) ─────────────────────────────────
print("Loading GPT-2 and data...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_EVAL:
        break
calib_ids = [t.to(DEVICE) for t in all_ids[:N_EVAL]]
print(f"Calib: {len(calib_ids)} seqs × {SEQ_LEN} tokens", flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
originals = [model.transformer.h[i].mlp.act for i in range(12)]
print(f"Original activation type: {type(originals[0]).__name__}", flush=True)


# ── PPL eval (sanity) ─────────────────────────────────────────────────────────
def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


baseline_ppl = eval_ppl(model, calib_ids)
print(f"Baseline GELU PPL on calib slice = {baseline_ppl:.4f}", flush=True)


# ── Phase 1: Precompute b_leak per block (variant B) and μ per block (variant C) ────
print("\n──────── Phase 1: Precompute b_leak (B) and μ (C) ────────", flush=True)
preact_stash = {l: [] for l in range(12)}
hooks = []

def make_hook(l):
    def hook(module, inputs, output):
        # inputs[0] is the LN2 output going into c_fc
        preact_stash[l].append(inputs[0].detach())
    return hook

for l in range(12):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_hook(l))
    hooks.append(h)

print(f"Hooks registered on {len(hooks)} c_fc modules. Calibration forward pass...", flush=True)
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()
print("Hooks removed.", flush=True)

b_leak_dict = {}  # variant B: relu offset (gelu - relu) per channel
mu_dict = {}      # variant C: cycle 228 (gelu - relu(gelu)) per channel
b_leak_stats = []
mu_stats = []
for l in range(12):
    x_l = torch.cat(preact_stash[l], dim=0).reshape(-1, 768)  # (n_tokens, 768)
    c_fc = model.transformer.h[l].mlp.c_fc
    # GPT-2 Conv1D forward: x @ W + b with weight shape (768, 3072)
    preact = x_l @ c_fc.weight + c_fc.bias  # (n_tokens, 3072)
    gelu_l = F.gelu(preact, approximate='tanh')
    relu_l = F.relu(preact)
    noneg_l = torch.relu(gelu_l)

    b_leak_l = (gelu_l - relu_l).mean(dim=0)  # (3072,) — should be <=0 elementwise
    mu_l = (gelu_l - noneg_l).mean(dim=0)     # (3072,) — should be <=0 elementwise

    b_leak_dict[l] = b_leak_l.to(DEVICE)
    mu_dict[l] = mu_l.to(DEVICE)

    b_leak_stats.append({
        "block": l,
        "mean": float(b_leak_l.mean().item()),
        "min": float(b_leak_l.min().item()),
        "max": float(b_leak_l.max().item()),
        "abs_mean": float(b_leak_l.abs().mean().item()),
        "n_positive_entries": int((b_leak_l > 1e-6).sum().item()),
    })
    mu_stats.append({
        "block": l,
        "mean": float(mu_l.mean().item()),
        "min": float(mu_l.min().item()),
        "max": float(mu_l.max().item()),
        "abs_mean": float(mu_l.abs().mean().item()),
        "n_positive_entries": int((mu_l > 1e-6).sum().item()),
    })
    print(f"  block {l:2d}: b_leak mean={b_leak_stats[l]['mean']:+.5f} "
          f"min={b_leak_stats[l]['min']:+.5f} max={b_leak_stats[l]['max']:+.5f} "
          f"|b|={b_leak_stats[l]['abs_mean']:.5f} | "
          f"μ mean={mu_stats[l]['mean']:+.5f} |μ|={mu_stats[l]['abs_mean']:.5f}", flush=True)
    preact_stash[l] = None

# Sign sanity (informational)
b_leak_max_pos = max(s["max"] for s in b_leak_stats)
mu_max_pos = max(s["max"] for s in mu_stats)
print(f"\nb_leak sign sanity: max positive entry across blocks = {b_leak_max_pos:+.5f} "
      f"(expected near 0 — gelu(z) - relu(z) is dominated by negative regime)", flush=True)
print(f"μ sign sanity: max positive entry = {mu_max_pos:+.5f} "
      f"(expected ≤ 0 elementwise; cycle 228 invariant)", flush=True)


# ── Variant patch modules ─────────────────────────────────────────────────────
class ReluPlusBleak(nn.Module):
    """ReLU + per-channel offset b_leak: returns relu(z) + b_leak.
    The offset b_leak is precomputed as E_x[gelu(z) - relu(z)] in the intermediate
    (3072-dim) space — the per-channel first-moment correction matching GELU's mean.
    """
    def __init__(self, b_leak):
        super().__init__()
        self.register_buffer('b_leak', b_leak)

    def forward(self, z):
        return F.relu(z) + self.b_leak


class GeluNonegCorrected(nn.Module):
    """Cycle 228 patch verbatim: relu(gelu(z)) + μ, μ = E[gelu - relu(gelu)] per channel."""
    def __init__(self, mu):
        super().__init__()
        self.register_buffer('mu', mu)

    def forward(self, z):
        return torch.relu(F.gelu(z, approximate='tanh')) + self.mu


# ── Phase 2: Trajectory + geometry per variant ────────────────────────────────
def collect_concat_trajectory(model, calib_ids):
    """Forward all calib seqs, concatenate per-layer states along the batch dim."""
    per_seq_trajectories = []
    for ids in calib_ids:
        traj = collect_residual_trajectory(model, ids)
        per_seq_trajectories.append(traj)
    n_layers_plus = len(per_seq_trajectories[0])
    concat = []
    for li in range(n_layers_plus):
        concat.append(torch.cat([t[li] for t in per_seq_trajectories], dim=0))
    return concat


def aggregate(t, axes):
    """Per-layer mean / p10 / p50 / p90 over given axes (returns dict with lists)."""
    mean_v = t.mean(dim=axes)
    p10 = torch.quantile(t.reshape(t.shape[0], -1), 0.10, dim=1)
    p50 = torch.quantile(t.reshape(t.shape[0], -1), 0.50, dim=1)
    p90 = torch.quantile(t.reshape(t.shape[0], -1), 0.90, dim=1)
    return {
        "mean": mean_v.tolist(),
        "p10": p10.tolist(),
        "p50": p50.tolist(),
        "p90": p90.tolist(),
    }


def pearson_rho(x, y):
    """Pearson correlation between two 1-D tensors of equal length."""
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.norm() * ym.norm()).clamp_min(1e-12)
    return float((xm * ym).sum() / denom)


def restore_originals():
    for l in range(12):
        model.transformer.h[l].mlp.act = originals[l]


def patch_variant_b():
    for l in range(12):
        model.transformer.h[l].mlp.act = ReluPlusBleak(b_leak_dict[l]).to(DEVICE)
    ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
    assert len(ids_check) == 12, f"Variant B patches not distinct: {len(ids_check)}/12"


def patch_variant_c():
    for l in range(12):
        model.transformer.h[l].mlp.act = GeluNonegCorrected(mu_dict[l]).to(DEVICE)
    ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
    assert len(ids_check) == 12, f"Variant C patches not distinct: {len(ids_check)}/12"


print("\n──────── Phase 2: Trajectory + geometry per variant ────────", flush=True)
U = get_unembedding(model)
print(f"U shape = {tuple(U.shape)}", flush=True)

variants = {}

for variant_name, patch_fn in [
    ("A_gelu", None),
    ("B_relu_bleak", patch_variant_b),
    ("C_gelu_noneg_mu", patch_variant_c),
]:
    print(f"\n=== Variant {variant_name} ===", flush=True)
    restore_originals()
    if patch_fn is not None:
        patch_fn()
        ppl = eval_ppl(model, calib_ids)
        print(f"  Patched PPL on calib slice = {ppl:.4f}", flush=True)
    else:
        ppl = baseline_ppl

    print(f"  Collecting trajectory ({N_EVAL} seqs)...", flush=True)
    traj = collect_concat_trajectory(model, calib_ids)
    print(f"  trajectory: {len(traj)} layers, x^(0) shape = {tuple(traj[0].shape)}", flush=True)

    print(f"  Computing salience and curvature (reduce='none')...", flush=True)
    S = compute_salience(traj, U, reduce='none')   # [12, B, T]
    K = compute_curvature(traj, U, reduce='none')  # [11, B, T]
    print(f"  S shape={tuple(S.shape)}  K shape={tuple(K.shape)}", flush=True)

    # Move to CPU float32 for aggregation
    S_cpu = S.to('cpu').float()
    K_cpu = K.to('cpu').float()

    S_agg = aggregate(S_cpu, axes=(1, 2))
    K_agg = aggregate(K_cpu, axes=(1, 2))

    # Pearson ρ over the 11 layers where κ is defined (l = 1..11):
    # κ index 0 corresponds to layer 1, κ index 10 to layer 11.
    # Salience index l corresponds to layer l. Use S[1:12] vs κ[0:11].
    S_for_rho = S_agg["mean"][1:12]   # length 11
    K_for_rho = K_agg["mean"]         # length 11
    rho = pearson_rho(S_for_rho, K_for_rho)
    print(f"  ρ(S_l, κ_l) over layers 1..11 = {rho:+.4f}", flush=True)

    # Free big tensors
    del traj, S, K, S_cpu, K_cpu

    variants[variant_name] = {
        "ppl_calib_slice": float(ppl),
        "salience": S_agg,
        "curvature": K_agg,
        "rho_S_kappa": rho,
    }

restore_originals()
post_restore_ppl = eval_ppl(model, calib_ids)
print(f"\nPost-restore baseline PPL = {post_restore_ppl:.4f} "
      f"(should match {baseline_ppl:.4f})", flush=True)
assert abs(post_restore_ppl - baseline_ppl) < 1e-4, "Restore round-trip failed"


# ── Verdict logic ─────────────────────────────────────────────────────────────
print("\n──────── Verdict logic ────────", flush=True)
kappa_gelu = variants["A_gelu"]["curvature"]["mean"]   # length 11, layers 1..11
kappa_relu = variants["B_relu_bleak"]["curvature"]["mean"]
kappa_noneg = variants["C_gelu_noneg_mu"]["curvature"]["mean"]

# Relative diffs over κ layers — κ index 0 → layer 1, κ index 6 → layer 7.
# Item 269 says "in at least one of layers 0–6". Layer 0 has no κ; the early-layer band
# we can probe is κ index 0..5 (which corresponds to layers 1..6).
rel_diffs_relu = []
for ki in range(11):
    g = kappa_gelu[ki]
    r = kappa_relu[ki]
    rd = abs(g - r) / abs(g) if abs(g) > 1e-12 else 0.0
    rel_diffs_relu.append(rd)

rel_diffs_noneg = []
for ki in range(11):
    g = kappa_gelu[ki]
    r = kappa_noneg[ki]
    rd = abs(g - r) / abs(g) if abs(g) > 1e-12 else 0.0
    rel_diffs_noneg.append(rd)

rho_relu = variants["B_relu_bleak"]["rho_S_kappa"]
rho_gelu = variants["A_gelu"]["rho_S_kappa"]
rho_noneg = variants["C_gelu_noneg_mu"]["rho_S_kappa"]

# Early-band (κ indices 0..5 → layers 1..6, the available subset of 0..6)
early_max_rel_relu = max(rel_diffs_relu[:6])
all_max_rel_relu = max(rel_diffs_relu)
all_below_10pct = all(rd < 0.10 for rd in rel_diffs_relu)

confirmed = (early_max_rel_relu >= 0.30) and (abs(rho_relu) <= 0.5)
refuted = all_below_10pct and (-0.85 <= rho_relu <= -0.65)
if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  ρ(S, κ) — A_gelu={rho_gelu:+.4f}  B_relu={rho_relu:+.4f}  C_noneg={rho_noneg:+.4f}", flush=True)
print(f"  early-band (κ idx 0..5) max relative κ diff (ReLU vs GELU) = {early_max_rel_relu:.4f}", flush=True)
print(f"  all-layer max relative κ diff (ReLU vs GELU) = {all_max_rel_relu:.4f}", flush=True)
print(f"  all-layer max relative κ diff (gelu_noneg vs GELU) = {max(rel_diffs_noneg):.4f}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_k, ax_s = axes

kappa_layers = list(range(1, 12))   # κ defined on layers 1..11
sal_layers = list(range(0, 12))      # S defined on layers 0..11

color_map = {
    "A_gelu": ("tab:blue", "GELU baseline"),
    "B_relu_bleak": ("tab:orange", "ReLU + b_leak (cycle 181 first-moment)"),
    "C_gelu_noneg_mu": ("tab:green", "gelu_noneg + μ (cycle 228)"),
}

for v_name, (color, label) in color_map.items():
    K_agg = variants[v_name]["curvature"]
    S_agg = variants[v_name]["salience"]
    rho = variants[v_name]["rho_S_kappa"]
    ax_k.plot(kappa_layers, K_agg["mean"], color=color, marker='o', label=f"{label}  ρ={rho:+.3f}")
    ax_k.fill_between(kappa_layers, K_agg["p10"], K_agg["p90"], color=color, alpha=0.15)
    ax_s.plot(sal_layers, S_agg["mean"], color=color, marker='o', label=label)
    ax_s.fill_between(sal_layers, S_agg["p10"], S_agg["p90"], color=color, alpha=0.15)

ax_k.set_xlabel("layer l")
ax_k.set_ylabel(r"pullback curvature $\kappa_l$")
ax_k.set_title(f"Per-layer $\\kappa_l$ — verdict: {verdict}")
ax_k.legend(fontsize=8)
ax_k.grid(True, alpha=0.3)

ax_s.set_xlabel("layer l")
ax_s.set_ylabel(r"pullback salience $S_l = \|x^{l+1}-x^l\|_G$")
ax_s.set_title("Per-layer $S_l$")
ax_s.legend(fontsize=8)
ax_s.grid(True, alpha=0.3)

fig.suptitle(f"Cycle 258 — Pullback geometry: GELU vs ReLU+b_leak vs gelu_noneg+μ "
             f"(N={N_EVAL}×{SEQ_LEN})", fontsize=11)
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
print(f"Plot saved to {PLOT_PATH}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 258,
    "item": 269,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
    },
    "baseline_ppl_calib_slice": float(baseline_ppl),
    "post_restore_baseline_ppl": float(post_restore_ppl),
    "variants": {
        v_name: {
            "ppl_calib_slice": variants[v_name]["ppl_calib_slice"],
            "salience": variants[v_name]["salience"],
            "curvature": variants[v_name]["curvature"],
            "rho_S_kappa": variants[v_name]["rho_S_kappa"],
        }
        for v_name in ["A_gelu", "B_relu_bleak", "C_gelu_noneg_mu"]
    },
    "rel_diffs_kappa": {
        "ReLU_vs_GELU_per_layer_index_0_to_10": rel_diffs_relu,
        "gelu_noneg_vs_GELU_per_layer_index_0_to_10": rel_diffs_noneg,
        "early_band_idx_0_to_5_max_rel_diff_ReLU": early_max_rel_relu,
        "all_layer_max_rel_diff_ReLU": all_max_rel_relu,
        "all_layer_max_rel_diff_noneg": max(rel_diffs_noneg),
    },
    "kappa_layer_indices": kappa_layers,
    "salience_layer_indices": sal_layers,
    "b_leak_stats_per_block": b_leak_stats,
    "mu_stats_per_block": mu_stats,
    "verdict_logic": {
        "confirmed_iff": "early_band_max_rel_diff_ReLU>=0.30 AND |rho_ReLU|<=0.5",
        "refuted_iff": "all_layers_rel_diff<0.10 AND rho_ReLU in [-0.85,-0.65]",
        "rho_ReLU": rho_relu,
        "rho_GELU": rho_gelu,
        "rho_noneg": rho_noneg,
    },
    "plot_path": PLOT_PATH,
}
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {RESULT_PATH}", flush=True)


# ── Save summary md ───────────────────────────────────────────────────────────
summary = f"""# Cycle 258 — Pullback κ/S: GELU vs ReLU+b_leak vs gelu_noneg+μ (item 269)

**Verdict: {verdict}**

For three variants of GPT-2 small on the standard 200×128 wikitext-2 calibration slice
(seed {SEED}), collect the residual stream trajectory at all 13 layer boundaries via
`aleph.pullback_geometry.collect_residual_trajectory`, and compute pullback metric
salience S_l and curvature κ_l per token under G = U^T U.

| Variant | calib PPL | ρ(S, κ) over layers 1..11 |
|---|---|---|
| A — GELU baseline | {variants['A_gelu']['ppl_calib_slice']:.3f} | {variants['A_gelu']['rho_S_kappa']:+.4f} |
| B — ReLU + per-block per-channel b_leak | {variants['B_relu_bleak']['ppl_calib_slice']:.3f} | {variants['B_relu_bleak']['rho_S_kappa']:+.4f} |
| C — gelu_noneg + per-channel μ (cycle 228) | {variants['C_gelu_noneg_mu']['ppl_calib_slice']:.3f} | {variants['C_gelu_noneg_mu']['rho_S_kappa']:+.4f} |

**κ profile (mean per layer 1..11)**

| layer | A_gelu | B_relu+b_leak | C_gelu_noneg+μ | rel ΔReLU | rel Δnoneg |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {kappa_layers[i]} | {kappa_gelu[i]:.4f} | {kappa_relu[i]:.4f} | {kappa_noneg[i]:.4f} | "
    f"{rel_diffs_relu[i]:.3f} | {rel_diffs_noneg[i]:.3f} |"
    for i in range(11)
) + f"""

**Verdict logic.**
- early-band (layer 1..6) max |Δκ|/κ_GELU for ReLU = **{early_max_rel_relu:.4f}** (need ≥0.30 for CONFIRMED clause 1)
- |ρ_ReLU(S, κ)| = **{abs(rho_relu):.4f}** (need ≤0.50 for CONFIRMED clause 2)
- all-layer max |Δκ|/κ_GELU for ReLU = **{all_max_rel_relu:.4f}** (need <0.10 for REFUTED clause 1)
- ρ_ReLU = **{rho_relu:+.4f}** (need ∈ [−0.85, −0.65] for REFUTED clause 2)
- Result: **{verdict}**

Plot: `{PLOT_PATH}`. Per-token κ/S aggregates (mean / p10 / p50 / p90) saved in JSON.
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
