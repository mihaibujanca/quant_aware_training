"""
Cycle 259 — Item 270: Per-token trajectory divergence under depth.

For 200x128 wikitext-2 val tokens, capture residual stream trajectories under
two GPT-2 small variants:
  (A) baseline GELU
  (B) ReLU + per-block per-channel b_leak (cycle 181 first-moment recipe,
      per-channel intermediate-space form; cycle 258 script's ReluPlusBleak)

At each layer l in {0..12}, compute per-token d_l(x) = ||x_G^l - x_R^l||_G
under pullback metric G = U^T U. Report:
  (i)   median, p10, p90 of d_l over the (B*T)=25600 token axis for l in 0..12
  (ii)  fit log-linear, linear, saturating curves to median d_l over l in 1..12
        (l=0 is trivially zero since both variants share the embedding output)
        and report best-fit type + growth constant
  (iii) per-layer median cos(d_l, d_{l+1}) under G over the (B*T) token axis for
        l in 0..11 (cos[0] is numerically undefined where d_0 ~ 0 and is skipped
        from the verdict count)

Verdict (item 270):
  CONFIRMED  if log-fit growth constant >= 0.20/layer estimated on l in [1, 6]
             AND median per-token cos >= 0.6 in >= 4 of cos indices 1..6.
  REFUTED    if best-fit is linear or saturating with sub-linear growth through
             l in [1, 6] OR median per-token cos <= 0.2 in >= 3 early layers.
  INCONCLUSIVE otherwise.
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
    get_unembedding,
    pullback_norm,
    pullback_inner,
)

# --- Config ------------------------------------------------------------------
N_EVAL = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

RESULT_PATH = "autoresearch/results/cycle_259_per_token_divergence.json"
SUMMARY_PATH = "autoresearch/results/cycle_259_summary.md"
PLOT_PATH = "autoresearch/results/cycle_259_divergence.png"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)


# --- Load model + data (cycle 258 loader) -----------------------------------
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
print(f"Calib: {len(calib_ids)} seqs x {SEQ_LEN} tokens", flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
originals = [model.transformer.h[i].mlp.act for i in range(12)]
print(f"Original activation type: {type(originals[0]).__name__}", flush=True)


# --- PPL eval (sanity) ------------------------------------------------------
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


# --- Precompute b_leak per block (variant B) --------------------------------
print("\n-------- Precompute b_leak (variant B) --------", flush=True)
preact_stash = {l: [] for l in range(12)}
hooks = []


def make_hook(l):
    def hook(module, inputs, output):
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

b_leak_dict = {}
b_leak_stats = []
for l in range(12):
    x_l = torch.cat(preact_stash[l], dim=0).reshape(-1, 768)
    c_fc = model.transformer.h[l].mlp.c_fc
    preact = x_l @ c_fc.weight + c_fc.bias  # (n_tokens, 3072)
    gelu_l = F.gelu(preact, approximate='tanh')
    relu_l = F.relu(preact)
    b_leak_l = (gelu_l - relu_l).mean(dim=0)  # (3072,) — expected elementwise <= 0
    b_leak_dict[l] = b_leak_l.to(DEVICE)
    b_leak_stats.append({
        "block": l,
        "mean": float(b_leak_l.mean().item()),
        "min": float(b_leak_l.min().item()),
        "max": float(b_leak_l.max().item()),
        "abs_mean": float(b_leak_l.abs().mean().item()),
        "n_positive_entries": int((b_leak_l > 1e-6).sum().item()),
    })
    print(f"  block {l:2d}: b_leak mean={b_leak_stats[l]['mean']:+.5f} "
          f"min={b_leak_stats[l]['min']:+.5f} max={b_leak_stats[l]['max']:+.5f} "
          f"|b|={b_leak_stats[l]['abs_mean']:.5f}", flush=True)
    preact_stash[l] = None

b_leak_max_pos = max(s["max"] for s in b_leak_stats)
print(f"\nb_leak sign sanity: max positive entry across blocks = {b_leak_max_pos:+.5f} "
      f"(expected <= 0)", flush=True)


class ReluPlusBleak(nn.Module):
    """ReLU + per-channel offset b_leak: returns relu(z) + b_leak. Cycle 258 verbatim."""
    def __init__(self, b_leak):
        super().__init__()
        self.register_buffer('b_leak', b_leak)

    def forward(self, z):
        return F.relu(z) + self.b_leak


def restore_originals():
    for l in range(12):
        model.transformer.h[l].mlp.act = originals[l]


def patch_variant_b():
    for l in range(12):
        model.transformer.h[l].mlp.act = ReluPlusBleak(b_leak_dict[l]).to(DEVICE)
    ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
    assert len(ids_check) == 12, f"Variant B patches not distinct: {len(ids_check)}/12"


# --- Collect trajectories per variant ---------------------------------------
def collect_concat_trajectory(model, calib_ids):
    """Forward all calib seqs, concatenate per-layer states along batch dim."""
    per_seq = []
    for ids in calib_ids:
        traj = collect_residual_trajectory(model, ids)
        per_seq.append([t.detach() for t in traj])
    n_layers_plus = len(per_seq[0])
    concat = []
    for li in range(n_layers_plus):
        concat.append(torch.cat([t[li] for t in per_seq], dim=0))  # [B, T, 768]
    return concat


U = get_unembedding(model).to(DEVICE)
print(f"\nU shape = {tuple(U.shape)}", flush=True)

# Variant A
print("\n=== Variant A: GELU baseline ===", flush=True)
restore_originals()
print("Collecting A trajectory...", flush=True)
traj_A = collect_concat_trajectory(model, calib_ids)
print(f"  trajectory: {len(traj_A)} layers, x^(0) shape = {tuple(traj_A[0].shape)}", flush=True)

# Variant B
print("\n=== Variant B: ReLU + b_leak ===", flush=True)
patch_variant_b()
ppl_B = eval_ppl(model, calib_ids)
print(f"  Patched PPL on calib slice = {ppl_B:.4f}", flush=True)
print("Collecting B trajectory...", flush=True)
traj_B = collect_concat_trajectory(model, calib_ids)
print(f"  trajectory: {len(traj_B)} layers, x^(0) shape = {tuple(traj_B[0].shape)}", flush=True)

restore_originals()
post_restore_ppl = eval_ppl(model, calib_ids)
print(f"\nPost-restore baseline PPL = {post_restore_ppl:.4f} "
      f"(should match {baseline_ppl:.4f})", flush=True)
assert abs(post_restore_ppl - baseline_ppl) < 1e-4, "Restore round-trip failed"


# --- Per-layer per-token d_l ------------------------------------------------
print("\n-------- Per-token d_l under G = U^T U --------", flush=True)

# To avoid the [B, T, 50257] blow-up in pullback_norm, process each layer's
# d_l per-sequence (B=1, T=128 -> [1, 128, 50257] ~ 26 MB peak).

L_plus = len(traj_A)  # 13
n_total_tokens = N_EVAL * SEQ_LEN

# Store per-layer flattened per-token d (length 25600, float32 on CPU)
per_token_d = torch.zeros((L_plus, n_total_tokens), dtype=torch.float32)
# And the pre-aggregated residual d_l per seq (to compute cos without re-reading traj)
# To keep memory bounded, we will compute cos on the fly: load d_l and d_{l+1} one
# sequence at a time and accumulate cos values. Simpler approach: store the full
# (n_total, 768) delta CPU tensor per layer -> 13 * 25600 * 768 * 4 = ~1 GB which
# is too much. Instead compute (a) per-token d per layer from delta sequences
# streamed to CPU, and (b) per-token cos with a second pass that pairs (l, l+1).

print("  Pass 1: compute per-token d_l per layer (per-seq streaming)...", flush=True)
for l in range(L_plus):
    xg = traj_A[l]  # [B, T, 768]
    xr = traj_B[l]  # [B, T, 768]
    # Per-sequence loop to cap peak memory at ~26 MB per call.
    out_rows = []
    for b in range(xg.shape[0]):
        d_bt = xg[b:b+1] - xr[b:b+1]  # [1, T, 768]
        nrm = pullback_norm(d_bt, U)   # [1, T]
        out_rows.append(nrm.to('cpu').float().reshape(-1))
    d_flat = torch.cat(out_rows, dim=0)  # [n_total_tokens]
    assert d_flat.numel() == n_total_tokens, f"{d_flat.numel()} != {n_total_tokens}"
    per_token_d[l] = d_flat
    m = float(d_flat.median().item())
    p10 = float(torch.quantile(d_flat, 0.10).item())
    p90 = float(torch.quantile(d_flat, 0.90).item())
    print(f"    layer {l:2d}: median={m:.6e}  p10={p10:.6e}  p90={p90:.6e}", flush=True)

# Aggregates
median_d = [float(per_token_d[l].median().item()) for l in range(L_plus)]
p10_d = [float(torch.quantile(per_token_d[l], 0.10).item()) for l in range(L_plus)]
p50_d = median_d
p90_d = [float(torch.quantile(per_token_d[l], 0.90).item()) for l in range(L_plus)]

print(f"\n  d_0 = {median_d[0]:.4e}  (expected ~0: both variants share embedding output)", flush=True)
print(f"  d_12 = {median_d[12]:.4e}", flush=True)
print(f"  d_12 / d_1 (if d_1>0) = "
      f"{median_d[12] / max(median_d[1], 1e-12):.2f}x", flush=True)


# --- Per-token cos(d_l, d_{l+1}) under G ------------------------------------
print("\n-------- Per-token cos(d_l, d_{l+1}) under G --------", flush=True)
# There are L_plus - 1 = 12 cos entries for l in 0..11.
n_cos = L_plus - 1
per_token_cos = torch.zeros((n_cos, n_total_tokens), dtype=torch.float32)

for l in range(n_cos):
    xg_a = traj_A[l]
    xr_a = traj_B[l]
    xg_b = traj_A[l + 1]
    xr_b = traj_B[l + 1]
    out_rows = []
    for b in range(xg_a.shape[0]):
        d_l = xg_a[b:b+1] - xr_a[b:b+1]       # [1, T, 768]
        d_lp1 = xg_b[b:b+1] - xr_b[b:b+1]     # [1, T, 768]
        inner = pullback_inner(d_l, d_lp1, U)  # [1, T]
        nrm_l = pullback_norm(d_l, U)          # [1, T]
        nrm_lp1 = pullback_norm(d_lp1, U)      # [1, T]
        denom = (nrm_l * nrm_lp1).clamp_min(1e-20)
        cos_bt = (inner / denom).to('cpu').float().reshape(-1)
        out_rows.append(cos_bt)
    cos_flat = torch.cat(out_rows, dim=0)
    per_token_cos[l] = cos_flat
    m = float(cos_flat.median().item())
    p10 = float(torch.quantile(cos_flat, 0.10).item())
    p90 = float(torch.quantile(cos_flat, 0.90).item())
    print(f"    cos({l:2d},{l+1:2d}): median={m:+.4f}  p10={p10:+.4f}  p90={p90:+.4f}", flush=True)

median_cos = [float(per_token_cos[l].median().item()) for l in range(n_cos)]
p10_cos = [float(torch.quantile(per_token_cos[l], 0.10).item()) for l in range(n_cos)]
p90_cos = [float(torch.quantile(per_token_cos[l], 0.90).item()) for l in range(n_cos)]


# --- Curve fits to median d_l over l in [1, 12] -----------------------------
print("\n-------- Curve fits to median d_l (l in 1..12) --------", flush=True)
# Work in float64 to avoid cancellation. l starts at 1 because d_0 ~ 0.
layers = torch.arange(1, L_plus, dtype=torch.float64)   # [12] values 1..12
y = torch.tensor(median_d[1:], dtype=torch.float64)
log_y = torch.log(y.clamp_min(1e-30))

# (i) log-linear: log(y) = a + b*l
ones = torch.ones_like(layers)
X_lin = torch.stack([ones, layers], dim=1)  # [12, 2]
sol_loglin = torch.linalg.lstsq(X_lin, log_y.unsqueeze(1)).solution.flatten()
a_loglin, b_loglin = float(sol_loglin[0]), float(sol_loglin[1])
pred_loglin = torch.exp(X_lin @ sol_loglin)
ssr_loglin = float(((y - pred_loglin) ** 2).sum().item())

# (i') log-linear fit restricted to l in [1, 6] for the verdict constant
mask_early = (layers >= 1) & (layers <= 6)
X_early = X_lin[mask_early]
log_y_early = log_y[mask_early]
sol_early = torch.linalg.lstsq(X_early, log_y_early.unsqueeze(1)).solution.flatten()
a_early, b_early = float(sol_early[0]), float(sol_early[1])

# (ii) linear: y = a + b*l
sol_lin = torch.linalg.lstsq(X_lin, y.unsqueeze(1)).solution.flatten()
a_lin, b_lin = float(sol_lin[0]), float(sol_lin[1])
pred_lin = X_lin @ sol_lin
ssr_lin = float(((y - pred_lin) ** 2).sum().item())

# (iii) saturating: y = c*(1 - exp(-b*l)) + a with a = y[0], asymptote = c + a
# Grid on b, closed-form c = (y[-1] - a) / (1 - exp(-b * l_max))
a_sat = float(y[0].item())
l_max = float(layers[-1].item())
b_grid = torch.linspace(0.02, 2.0, steps=200, dtype=torch.float64)
best_ssr_sat = float('inf')
best_b_sat = None
best_c_sat = None
for b_try in b_grid.tolist():
    denom = 1.0 - math.exp(-b_try * l_max)
    if denom <= 1e-9:
        continue
    c_try = (float(y[-1].item()) - a_sat) / denom
    pred_try = c_try * (1.0 - torch.exp(-b_try * layers)) + a_sat
    ssr_try = float(((y - pred_try) ** 2).sum().item())
    if ssr_try < best_ssr_sat:
        best_ssr_sat = ssr_try
        best_b_sat = b_try
        best_c_sat = c_try
ssr_sat = best_ssr_sat

fit_ssrs = {"log_linear": ssr_loglin, "linear": ssr_lin, "saturating": ssr_sat}
best_fit = min(fit_ssrs, key=fit_ssrs.get)
print(f"  SSR  log_linear={ssr_loglin:.4e}  linear={ssr_lin:.4e}  saturating={ssr_sat:.4e}", flush=True)
print(f"  best-fit: {best_fit}", flush=True)
print(f"  log_linear full     : a={a_loglin:+.4f}  b={b_loglin:+.4f} (per-layer log-growth)", flush=True)
print(f"  log_linear early[1-6]: a={a_early:+.4f}  b={b_early:+.4f} (per-layer log-growth)", flush=True)
print(f"  linear              : a={a_lin:+.4e}  b={b_lin:+.4e}", flush=True)
print(f"  saturating          : a={a_sat:+.4e}  c={best_c_sat:+.4e}  b={best_b_sat:+.4f}", flush=True)


# --- Verdict logic ----------------------------------------------------------
print("\n-------- Verdict logic --------", flush=True)
# Clause 1: b_loglin (over l in [1,6]) >= 0.20 per layer
clause1 = b_early >= 0.20

# Clause 2: median per-token cos >= 0.6 in >= 4 of cos indices 1..6 (cos index 0
# corresponds to (d_0, d_1) and is numerically undefined because d_0 ~ 0).
cos_early_indices = list(range(1, 7))  # 1..6 (6 values)
cos_early_values = [median_cos[i] for i in cos_early_indices]
n_cos_ge_06 = sum(1 for v in cos_early_values if v >= 0.6)
clause2 = n_cos_ge_06 >= 4

confirmed = clause1 and clause2

# REFUTED clause: best-fit is linear or saturating with sub-linear growth through
# l in [1, 6] OR median per-token cos <= 0.2 in >= 3 early layers (cos idx 1..6).
best_is_sublinear = best_fit in {"linear", "saturating"}
# Evaluate "sub-linear growth" in the best-fit regime. For CONFIRMED we need
# log-linear b_early >= 0.20. The REFUTED clause is "best-fit is linear or
# saturating AND growth is sub-linear". Define "sub-linear" as: linear fit has
# b_lin/median_d[6] normalised growth such that the ratio median_d[6]/median_d[1] < e^{1.0}
# (i.e. the early-band multiplicative growth is less than a factor of ~2.7
# across 5 layer steps). This formalises "sub-linear" as "doesn't double+ over
# the early band". Keep it conservative — only flag REFUTED if log_linear is a
# worse fit than linear OR saturating AND the multiplicative ratio is small.
early_ratio = median_d[6] / max(median_d[1], 1e-20)
refuted_growth = best_is_sublinear and (early_ratio < math.exp(1.0))

n_cos_le_02 = sum(1 for v in cos_early_values if v <= 0.2)
refuted_cos = n_cos_le_02 >= 3

refuted = refuted_growth or refuted_cos

if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  b_loglin (l in [1,6]) = {b_early:+.4f}  (need >= 0.20 for CONFIRMED clause 1)", flush=True)
print(f"  cos idx 1..6 median values = "
      f"[{', '.join(f'{v:+.3f}' for v in cos_early_values)}]", flush=True)
print(f"  n(cos>=0.6) in idx 1..6 = {n_cos_ge_06}  (need >= 4 for CONFIRMED clause 2)", flush=True)
print(f"  best-fit curve = {best_fit}", flush=True)
print(f"  early-band multiplicative ratio median_d[6]/median_d[1] = {early_ratio:.2f}", flush=True)
print(f"  n(cos<=0.2) in idx 1..6 = {n_cos_le_02}  (REFUTED if >= 3)", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# --- Plot --------------------------------------------------------------------
print("\nPlotting...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_d, ax_c = axes

ls = list(range(L_plus))
ax_d.plot(ls, median_d, 'o-', color='tab:red', label='median')
ax_d.fill_between(ls, p10_d, p90_d, color='tab:red', alpha=0.2, label='p10-p90')
# Overlay best-fit curves on l in [1, 12]
ll_plot = torch.linspace(1, L_plus - 1, steps=60, dtype=torch.float64)
y_loglin = torch.exp(a_loglin + b_loglin * ll_plot)
y_lin = a_lin + b_lin * ll_plot
y_sat = best_c_sat * (1.0 - torch.exp(-best_b_sat * ll_plot)) + a_sat
ax_d.plot(ll_plot.tolist(), y_loglin.tolist(), '--', color='tab:blue',
          label=f'log-lin (b={b_loglin:+.3f})')
ax_d.plot(ll_plot.tolist(), y_lin.tolist(), '--', color='tab:green', label='linear')
ax_d.plot(ll_plot.tolist(), y_sat.tolist(), '--', color='tab:orange',
          label=f'saturating (b={best_b_sat:+.3f})')
ax_d.set_xlabel('layer l')
ax_d.set_ylabel(r'per-token $\|x_G^l - x_R^l\|_G$')
ax_d.set_title(f'Per-token divergence — best-fit {best_fit}')
ax_d.set_yscale('log')
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

cls = list(range(n_cos))
ax_c.plot(cls, median_cos, 'o-', color='tab:purple', label='median cos')
ax_c.fill_between(cls, p10_cos, p90_cos, color='tab:purple', alpha=0.2, label='p10-p90')
ax_c.axhline(0.6, color='tab:gray', ls=':', label='0.6 (CONFIRMED threshold)')
ax_c.axhline(0.2, color='tab:gray', ls='--', label='0.2 (REFUTED threshold)')
ax_c.set_xlabel('layer l')
ax_c.set_ylabel(r'per-token $\cos(d_l, d_{l+1})_G$')
ax_c.set_title(f'Per-token directional coherence — verdict: {verdict}')
ax_c.set_ylim(-1.05, 1.05)
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3)

fig.suptitle(f'Cycle 259 — Per-token trajectory divergence GELU vs ReLU+b_leak '
             f'(N={N_EVAL}x{SEQ_LEN})', fontsize=11)
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.savefig(PLOT_PATH, dpi=120, bbox_inches='tight')
print(f"Plot saved to {PLOT_PATH}", flush=True)


# --- Save JSON ---------------------------------------------------------------
result = {
    "cycle": 259,
    "item": 270,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_eval": N_EVAL,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
        "n_total_tokens": n_total_tokens,
    },
    "baseline_ppl_calib_slice": float(baseline_ppl),
    "variant_B_ppl_calib_slice": float(ppl_B),
    "post_restore_baseline_ppl": float(post_restore_ppl),
    "layer_indices": ls,
    "median_d": median_d,
    "p10_d": p10_d,
    "p50_d": p50_d,
    "p90_d": p90_d,
    "cos_layer_indices": cls,
    "median_cos": median_cos,
    "p10_cos": p10_cos,
    "p90_cos": p90_cos,
    "curve_fits": {
        "log_linear_full_l1_to_12": {"a": a_loglin, "b": b_loglin, "ssr": ssr_loglin},
        "log_linear_early_l1_to_6":  {"a": a_early, "b": b_early},
        "linear":       {"a": a_lin, "b": b_lin, "ssr": ssr_lin},
        "saturating":   {"a": a_sat, "c": best_c_sat, "b": best_b_sat, "ssr": ssr_sat},
        "best_fit": best_fit,
    },
    "verdict_logic": {
        "clause1_b_loglin_early_ge_0.20": {
            "value": b_early,
            "threshold": 0.20,
            "passes": clause1,
        },
        "clause2_n_cos_ge_0.6_in_idx_1_to_6_ge_4": {
            "cos_early_values_idx_1_to_6": cos_early_values,
            "n_ge_0.6": n_cos_ge_06,
            "threshold": 4,
            "passes": clause2,
        },
        "refuted_growth_sublinear": refuted_growth,
        "refuted_n_cos_le_0.2_ge_3": refuted_cos,
        "early_ratio_median_d6_over_d1": early_ratio,
    },
    "b_leak_stats_per_block": b_leak_stats,
    "plot_path": PLOT_PATH,
    "notes": [
        "d_0 ~ 0 by construction (both variants share transformer.drop output); "
        "log-linear fit uses l in [1, 12], verdict clause 1 uses restricted fit on l in [1, 6]",
        "cos index 0 corresponds to (d_0, d_1) and is numerically undefined for d_0 ~ 0; "
        "verdict clause 2 counts cos indices 1..6 (6 values, need >= 4 at >= 0.6)",
        "Per-sequence streaming of pullback_norm avoids the [B, T, 50257] blow-up on MPS",
    ],
}
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {RESULT_PATH}", flush=True)


# --- Save summary md ---------------------------------------------------------
cos_table = "\n".join(
    f"| {l} | {median_cos[l]:+.4f} | {p10_cos[l]:+.4f} | {p90_cos[l]:+.4f} |"
    for l in range(n_cos)
)
d_table = "\n".join(
    f"| {l} | {median_d[l]:.4e} | {p10_d[l]:.4e} | {p90_d[l]:.4e} |"
    for l in range(L_plus)
)

summary = f"""# Cycle 259 — Per-token trajectory divergence GELU vs ReLU+b_leak (item 270)

**Verdict: {verdict}**

Single forward pass each for GPT-2 small with (A) GELU baseline and (B) ReLU+per-block
per-channel b_leak (cycle 258 recipe) on the standard 200x128 wikitext-2 calib slice
(seed {SEED}). For each layer l in {{0..12}}, compute per-token
d_l(x) = ||x_G^l - x_R^l||_G under the pullback metric G = U^T U where U is the
unembedding matrix. Report median/p10/p90 of d_l and median per-token cos(d_l, d_{{l+1}})
under G.

- calib PPL A (GELU) = {baseline_ppl:.4f}
- calib PPL B (ReLU+b_leak) = {ppl_B:.4f}
- post-restore baseline PPL = {post_restore_ppl:.4f} (round-trip clean)

## Per-layer d_l (median / p10 / p90)

| layer l | median ||d_l||_G | p10 | p90 |
|---|---|---|---|
{d_table}

## Per-layer per-token cos(d_l, d_{{l+1}}) under G

| l (pair l,l+1) | median | p10 | p90 |
|---|---|---|---|
{cos_table}

## Curve fits (median d_l over l in [1, 12])

| fit | parameters | SSR |
|---|---|---|
| log-linear | a={a_loglin:+.4f}, b={b_loglin:+.4f} | {ssr_loglin:.4e} |
| linear | a={a_lin:+.4e}, b={b_lin:+.4e} | {ssr_lin:.4e} |
| saturating | a={a_sat:+.4e}, c={best_c_sat:+.4e}, b={best_b_sat:+.4f} | {ssr_sat:.4e} |

Best-fit: **{best_fit}**. Log-linear fit restricted to l in [1, 6] for the verdict
threshold: b_early = **{b_early:+.4f}**.

## Verdict logic

- Clause 1: b_loglin on l in [1, 6] >= 0.20 -> **{clause1}** (value {b_early:+.4f}).
- Clause 2: median per-token cos >= 0.6 in >= 4 of cos indices 1..6
  -> **{clause2}** ({n_cos_ge_06} / 6 values >= 0.6).
- REFUTED growth clause: best-fit is linear or saturating AND early-band
  multiplicative ratio median_d[6]/median_d[1] = {early_ratio:.2f} < e^1.0 -> **{refuted_growth}**.
- REFUTED cos clause: n(cos<=0.2) in idx 1..6 = {n_cos_le_02} -> **{refuted_cos}**.
- Verdict: **{verdict}**.

Notes:
- d_0 ~ {median_d[0]:.2e} by construction (both variants share transformer.drop output);
  log-linear fit uses l in [1, 12] and clause 1 uses l in [1, 6] restricted fit.
- cos index 0 = cos(d_0, d_1) is numerically undefined for d_0 ~ 0; the verdict
  counts cos indices 1..6 (6 values), not 0..5.

Plot: `{PLOT_PATH}`. Per-layer aggregates and per-token arrays in JSON.
"""
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"Saved {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
