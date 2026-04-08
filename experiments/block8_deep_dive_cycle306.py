"""
Cycle 306 — Item 316: Block 8 deep dive — why is E_kept_top_2(b8) = 0.47, the
lowest across all 12 blocks?

Geometric object: for activation pair (sigma, sigma_lin) = (GELU tanh, ReLU)
and each GPT-2 small block l in PRIMARY = {7, 8, 9}, with extra reference set
CONCENTRATORS = {1, 2, 10, 11},

    z(x)   = (LN2(x) @ W_fc + b_fc)            in R^3072  (per token)
    delta  = GELU(z) - ReLU(z)                 in R^3072
    Dy(x)  = delta @ W_proj                    in R^768

We measure four things at each block (per item 316 verbatim Setup):
  1. Full SV spectrum of Dy (768 singular values).
  2. Per-token norms ||Dy_t||_2.
  3. Top-K concentration of Dy (E_top_1, eff_rank, k_95).
  4. Cosine between Dy top-5 right SVs (Vh_y[:5]) and W_proj top-5 right SVs
     (Vh_w[:5]) — both in R^768.

Verdict per item 316 verbatim:
  CONFIRMED iff (a) block 8 W_proj top-2/top-5 SV ratio <= 0.5 * median
              concentrator block's ratio
              AND (b) block 8 Dy eff_rank >= 1.5 * median concentrator block's
              Dy eff_rank.
  REFUTED   iff neither (a) nor (b) holds.
  INCONCLUSIVE otherwise.

No training. Pure SVD/cosine analysis. Reuses cycle 302 framework.
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from experiments.curvature_residual_geometric import UnifiedCurvatureFramework

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
PRIMARY = [7, 8, 9]
CONCENTRATORS = [1, 2, 10, 11]
ALL_BLOCKS = sorted(set(PRIMARY + CONCENTRATORS))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FLOAT_BASELINE_REF = 45.0637  # cycle 302/304 calib slice reference

# Cycle 302 panel 4 E_kept_top_2 reference per block (sanity check)
C302_E_TOP2_REF = {
    0: 0.5902, 1: 0.7463, 2: 0.9190, 3: 0.8649, 4: 0.8426, 5: 0.8109,
    6: 0.7197, 7: 0.6622, 8: 0.4722, 9: 0.5356, 10: 0.8374, 11: 0.9803,
}

# Cycle 304 k_95 dense reference per block (cross-check)
C304_K95_REF = {
    0: 445, 1: 348, 2: 63, 3: 131, 4: 152, 5: 228,
    6: 325, 7: 411, 8: 511, 9: 542, 10: 397, 11: 2,
}

OUT_JSON = "autoresearch/results/cycle_306_block8_deep_dive.json"
OUT_SUMMARY = "autoresearch/results/cycle_306_summary.md"
OUT_PLOT = "autoresearch/results/cycle_306_panels.png"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data ────────────────────────────────────────────────────────
print("Loading GPT-2 + wikitext-2 val...", flush=True)
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
N_TOK = N_CALIB * SEQ_LEN
print(f"Calib: {len(calib_ids)} seqs x {SEQ_LEN} tokens = {N_TOK} tokens", flush=True)

t0 = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)


def eval_ppl(m, eval_ids):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = m(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


t0 = time.time()
float_ppl = eval_ppl(model, calib_ids)
print(f"Float PPL on calib slice = {float_ppl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} off reference {FLOAT_BASELINE_REF}"
)

# ── Framework ────────────────────────────────────────────────────────────────
fw = UnifiedCurvatureFramework(
    model=model,
    sigma=lambda z: F.gelu(z, approximate="tanh"),
    sigma_lin=F.relu,
    blocks=ALL_BLOCKS,
    calib_ids=calib_ids,
    device=DEVICE,
    tokenizer=tokenizer,
    A_factors=None,
    sigma_name="GELU",
    sigma_lin_name="ReLU",
)

print("Capturing c_fc inputs (single forward pass)...", flush=True)
t0 = time.time()
fw.capture_inputs()
print(f"  captured in {time.time() - t0:.1f}s", flush=True)


def eff_rank_from_singvals(s: torch.Tensor) -> float:
    """Stable rank: (sum s^2)^2 / sum s^4."""
    e = (s * s).double()
    num = float(e.sum().item()) ** 2
    den = float((e * e).sum().item())
    if den <= 0:
        return 0.0
    return num / den


def k_at_threshold(s: torch.Tensor, frac: float) -> int:
    """Smallest k such that cumulative s^2 >= frac * total."""
    e = (s * s).double()
    cum = e.cumsum(0)
    total = float(cum[-1].item())
    if total <= 0:
        return 0
    return int((cum < frac * total).sum().item()) + 1


per_block: dict = {}

print("\n── Per-block deep-dive ──", flush=True)
for l in ALL_BLOCKS:
    t0 = time.time()
    c = fw.compute_Dy(l)
    Dy = c["Dy"]              # (N_TOK, 768)  CPU float32
    W_proj = c["W_proj"]      # (3072, 768)
    assert Dy.shape == (N_TOK, 768), f"block {l}: Dy {tuple(Dy.shape)}"
    assert W_proj.shape == (3072, 768), f"block {l}: W_proj {tuple(W_proj.shape)}"

    # ── (3) W_proj SVD ───────────────────────────────────────────────────────
    Uw, Sw, Vhw = torch.linalg.svd(W_proj, full_matrices=False)
    assert Sw.shape == (768,) and Vhw.shape == (768, 768), f"block {l}: Vhw {tuple(Vhw.shape)}"

    s_w_top10 = [float(s) for s in Sw[:10].tolist()]
    s_w_full = [float(s) for s in Sw.tolist()]
    # Item 316 verbatim metric: top-2/top-5 ratio = s_w[1] / s_w[4]
    wproj_top2_top5_ratio = float(Sw[1].item() / Sw[4].item())
    wproj_top1_top5_ratio = float(Sw[0].item() / Sw[4].item())
    wproj_eff_rank = eff_rank_from_singvals(Sw)

    # ── (1) Δy SVD ────────────────────────────────────────────────────────────
    Uy, Sy, Vhy = torch.linalg.svd(Dy, full_matrices=False)
    # Dy is (N=25600, 768), full_matrices=False -> Vhy is (768, 768).
    assert Sy.shape == (768,) and Vhy.shape == (768, 768), f"block {l}: Vhy {tuple(Vhy.shape)}"
    Dy_frob_sq = float((Sy * Sy).sum().item())
    Dy_top1_energy = float(Sy[0].item() ** 2 / Dy_frob_sq)
    Dy_top2_energy = float((Sy[:2] ** 2).sum().item() / Dy_frob_sq)
    Dy_top5_energy = float((Sy[:5] ** 2).sum().item() / Dy_frob_sq)
    Dy_top10_energy = float((Sy[:10] ** 2).sum().item() / Dy_frob_sq)
    Dy_eff_rank = eff_rank_from_singvals(Sy)
    Dy_k95 = k_at_threshold(Sy, 0.95)
    Dy_k99 = k_at_threshold(Sy, 0.99)
    s_y_top10 = [float(s) for s in Sy[:10].tolist()]
    s_y_full = [float(s) for s in Sy.tolist()]

    # Cross-check k_95 against cycle 304 reference for ALL_BLOCKS in C304_K95_REF
    if l in C304_K95_REF:
        ref = C304_K95_REF[l]
        rel_err = abs(Dy_k95 - ref) / max(ref, 1)
        # NOTE: cycle 304's k_95 was measured on Dy@Vh_w (W_proj basis projection),
        # not on Dy SVD directly. They will differ; do not assert tight match here.

    # Sanity: E_kept_top_2 derived from Dy @ Vh_w[:2].T must match cycle 302 ref
    Dy_kept_top2 = Dy @ Vhw[:2, :].T
    e_kept_top2 = float((Dy_kept_top2 * Dy_kept_top2).sum().item() / Dy_frob_sq)
    ref_e2 = C302_E_TOP2_REF[l]
    assert abs(e_kept_top2 - ref_e2) < 1e-4, (
        f"block {l}: E_kept_top_2 {e_kept_top2:.6f} vs cycle 302 ref {ref_e2:.6f} "
        f"(delta {e_kept_top2 - ref_e2:+.6f})"
    )
    del Dy_kept_top2

    # ── (2) Per-token Δy norms ────────────────────────────────────────────────
    Dy_norms = torch.linalg.norm(Dy, dim=1)
    qs = torch.quantile(
        Dy_norms,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=Dy_norms.dtype),
    )
    norm_stats = {
        "mean": float(Dy_norms.mean().item()),
        "std": float(Dy_norms.std().item()),
        "min": float(Dy_norms.min().item()),
        "max": float(Dy_norms.max().item()),
        "q05": float(qs[0].item()),
        "q25": float(qs[1].item()),
        "median": float(qs[2].item()),
        "q75": float(qs[3].item()),
        "q95": float(qs[4].item()),
    }
    del Dy_norms

    # ── (4) Cosine between top-5 right SVs of Δy and W_proj ──────────────────
    Vy5 = Vhy[:5, :]   # (5, 768)
    Vw5 = Vhw[:5, :]   # (5, 768)
    M = Vy5 @ Vw5.T    # (5, 5)
    M_abs = M.abs()
    cos_principal = torch.linalg.svdvals(M)  # (5,)  principal-angle cosines
    cos_principal_list = [float(s) for s in cos_principal.tolist()]
    cos_top5_sum = float(cos_principal.sum().item())   # max 5.0
    cos_top1 = float(cos_principal[0].item())          # max 1.0
    M_diag = [float(M_abs[i, i].item()) for i in range(5)]
    M_max_off = float((M_abs - torch.diag(torch.diag(M_abs))).max().item())

    per_block[str(l)] = {
        "block": l,
        # W_proj
        "wproj_singvals_top10": s_w_top10,
        "wproj_singvals_full": s_w_full,
        "wproj_top2_top5_ratio": wproj_top2_top5_ratio,
        "wproj_top1_top5_ratio": wproj_top1_top5_ratio,
        "wproj_eff_rank": wproj_eff_rank,
        # Dy spectrum
        "Dy_singvals_top10": s_y_top10,
        "Dy_singvals_full": s_y_full,
        "Dy_frob_sq": Dy_frob_sq,
        "Dy_top1_energy": Dy_top1_energy,
        "Dy_top2_energy": Dy_top2_energy,
        "Dy_top5_energy": Dy_top5_energy,
        "Dy_top10_energy": Dy_top10_energy,
        "Dy_eff_rank": Dy_eff_rank,
        "Dy_k95": Dy_k95,
        "Dy_k99": Dy_k99,
        "Dy_norms": norm_stats,
        # Cycle 302 cross-check (W_proj-basis projection energy)
        "E_kept_top_2_via_Wproj": e_kept_top2,
        "E_kept_top_2_cycle302_ref": ref_e2,
        # Cosine top-5(Δy) vs top-5(W_proj)
        "cos_principal_top5": cos_principal_list,
        "cos_top5_sum": cos_top5_sum,
        "cos_top1": cos_top1,
        "cos_M_diag": M_diag,
        "cos_M_max_off_diag": M_max_off,
    }

    print(
        f"  block {l:2d}: "
        f"S_w[0:5]={[round(x, 3) for x in s_w_top10[:5]]} "
        f"top2/top5={wproj_top2_top5_ratio:.3f}  "
        f"Dy eff_rank={Dy_eff_rank:.2f}  k95={Dy_k95}  "
        f"top1_E={Dy_top1_energy:.3f}  cos_top5_sum={cos_top5_sum:.3f}  "
        f"E_kept_top_2_via_Wproj={e_kept_top2:.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    fw.free_block(l)

# ── Concentrator references for verdict ─────────────────────────────────────
conc_ratios = [per_block[str(l)]["wproj_top2_top5_ratio"] for l in CONCENTRATORS]
conc_dy_eff = [per_block[str(l)]["Dy_eff_rank"] for l in CONCENTRATORS]


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


conc_ratio_med = median(conc_ratios)
conc_dy_eff_med = median(conc_dy_eff)

b8 = per_block["8"]
b8_ratio = b8["wproj_top2_top5_ratio"]
b8_dy_eff = b8["Dy_eff_rank"]

# Item 316 verbatim CONFIRM:
#   (a) b8 top2/top5 ratio <= 0.5 * median concentrator ratio  (b8 FLATTER)
#   (b) b8 Dy eff_rank   >= 1.5 * median concentrator Dy eff_rank  (b8 BROADER)
ratio_threshold = 0.5 * conc_ratio_med
eff_rank_threshold = 1.5 * conc_dy_eff_med

cond_a = b8_ratio <= ratio_threshold
cond_b = b8_dy_eff >= eff_rank_threshold

if cond_a and cond_b:
    verdict = "CONFIRMED"
elif (not cond_a) and (not cond_b):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

verdict_reason = (
    f"(a) b8 W_proj top2/top5 ratio = {b8_ratio:.4f}; "
    f"concentrator median = {conc_ratio_med:.4f}; "
    f"threshold (≤ 0.5×median) = {ratio_threshold:.4f}; "
    f"cond_a (b8 flatter) = {cond_a}. "
    f"(b) b8 Dy eff_rank = {b8_dy_eff:.2f}; "
    f"concentrator median = {conc_dy_eff_med:.2f}; "
    f"threshold (≥ 1.5×median) = {eff_rank_threshold:.2f}; "
    f"cond_b (b8 broader) = {cond_b}. "
    f"CONFIRMED iff (a) AND (b); REFUTED iff NOT (a) AND NOT (b); else INCONCLUSIVE."
)

print(f"\n== VERDICT: {verdict} ==", flush=True)
print(verdict_reason, flush=True)

# ── Save JSON ───────────────────────────────────────────────────────────────
result = {
    "cycle": 306,
    "item": 316,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "config": {
        "N_CALIB": N_CALIB,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "PRIMARY": PRIMARY,
        "CONCENTRATORS": CONCENTRATORS,
        "ALL_BLOCKS": ALL_BLOCKS,
        "DEVICE": DEVICE,
        "sigma": "GELU (tanh approx)",
        "sigma_lin": "ReLU",
        "model": "gpt2",
    },
    "float_ppl_calib": float_ppl,
    "concentrator_ratio_median": conc_ratio_med,
    "concentrator_Dy_eff_rank_median": conc_dy_eff_med,
    "ratio_threshold_b8_le": ratio_threshold,
    "eff_rank_threshold_b8_ge": eff_rank_threshold,
    "cond_a_b8_flatter_wproj": bool(cond_a),
    "cond_b_b8_broader_dy": bool(cond_b),
    "per_block": per_block,
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUT_JSON}", flush=True)

# ── Plot panels ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel 1: W_proj spectrum (log) for all reported blocks
ax = axes[0, 0]
cmap = matplotlib.colormaps.get_cmap("tab10")
for i, l in enumerate(ALL_BLOCKS):
    s = per_block[str(l)]["wproj_singvals_full"]
    color = cmap(i % 10)
    style = "-" if l in PRIMARY else "--"
    ax.plot(range(1, len(s) + 1), s, style, color=color, label=f"b{l}", linewidth=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("rank index (1..768)")
ax.set_ylabel("W_proj singular value")
ax.set_title("W_proj spectrum (log–log)")
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.grid(alpha=0.3)

# Panel 2: Dy spectrum (log) for all reported blocks
ax = axes[0, 1]
for i, l in enumerate(ALL_BLOCKS):
    s = per_block[str(l)]["Dy_singvals_full"]
    color = cmap(i % 10)
    style = "-" if l in PRIMARY else "--"
    ax.plot(range(1, len(s) + 1), s, style, color=color, label=f"b{l}", linewidth=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("rank index (1..768)")
ax.set_ylabel("Δy singular value")
ax.set_title("Δy spectrum (log–log)")
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.grid(alpha=0.3)

# Panel 3: Bar — W_proj top2/top5 ratio per block (lower = flatter)
ax = axes[1, 0]
ratios = [per_block[str(l)]["wproj_top2_top5_ratio"] for l in ALL_BLOCKS]
bar_colors = ["tab:red" if l == 8 else ("tab:orange" if l in CONCENTRATORS else "tab:blue")
              for l in ALL_BLOCKS]
ax.bar([str(l) for l in ALL_BLOCKS], ratios, color=bar_colors)
ax.axhline(conc_ratio_med, color="black", linestyle="--",
           label=f"conc median = {conc_ratio_med:.3f}")
ax.axhline(ratio_threshold, color="green", linestyle=":",
           label=f"0.5×conc med = {ratio_threshold:.3f}")
ax.set_ylabel("s_w[1] / s_w[4]")
ax.set_title("W_proj top-2/top-5 ratio (lower = flatter; red=b8, orange=conc)")
ax.legend(fontsize=8)
for x, v in zip([str(l) for l in ALL_BLOCKS], ratios):
    ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)

# Panel 4: Bar — Dy eff_rank per block (higher = broader)
ax = axes[1, 1]
effs = [per_block[str(l)]["Dy_eff_rank"] for l in ALL_BLOCKS]
ax.bar([str(l) for l in ALL_BLOCKS], effs, color=bar_colors)
ax.axhline(conc_dy_eff_med, color="black", linestyle="--",
           label=f"conc median = {conc_dy_eff_med:.2f}")
ax.axhline(eff_rank_threshold, color="green", linestyle=":",
           label=f"1.5×conc med = {eff_rank_threshold:.2f}")
ax.set_ylabel("Δy eff_rank (stable rank)")
ax.set_title("Δy stable rank per block (higher = broader; red=b8, orange=conc)")
ax.legend(fontsize=8)
for x, v in zip([str(l) for l in ALL_BLOCKS], effs):
    ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

fig.suptitle(f"Cycle 306 / Item 316 — Block 8 deep dive — {verdict}")
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=120)
print(f"Wrote {OUT_PLOT}", flush=True)

# ── Summary markdown ────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 306 — Item 316: Block 8 deep dive (curvature residual)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(verdict_reason)
lines.append("")
lines.append("## Geometric object")
lines.append(
    "For GPT-2 small block l, z = LN2(x) @ W_fc + b_fc in R^3072, "
    "delta = GELU(z) - ReLU(z) in R^3072, "
    "Dy = delta @ W_proj in R^768 (W_proj HF Conv1D (3072, 768)). "
    "Two SVDs: W_proj = U_w diag(S_w) Vh_w; Dy = U_y diag(S_y) Vh_y."
)
lines.append("")
lines.append("## Setup")
lines.append(
    f"- Calib: {N_CALIB} wikitext-2 val seqs × {SEQ_LEN} tokens (seed {SEED}); "
    f"N_TOKENS = {N_TOK}."
)
lines.append(
    f"- Float PPL on calib slice = {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})."
)
lines.append(f"- Primary blocks: {PRIMARY}.  Concentrator reference: {CONCENTRATORS}.")
lines.append("- Trainable params: 0.")
lines.append("")

lines.append("## W_proj spectrum & Δy spectrum per block")
lines.append("")
lines.append(
    "| block | s_w[0] | s_w[1] | s_w[4] | top2/top5 | s_y[0] | "
    "Δy eff_rank | Δy k_95 | top1 E | top5 E | E_kept_top_2 (W_proj basis) |"
)
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for l in ALL_BLOCKS:
    pb = per_block[str(l)]
    sw = pb["wproj_singvals_top10"]
    sy = pb["Dy_singvals_top10"]
    lines.append(
        f"| {l} | {sw[0]:.3f} | {sw[1]:.3f} | {sw[4]:.3f} | "
        f"{pb['wproj_top2_top5_ratio']:.3f} | {sy[0]:.3e} | "
        f"{pb['Dy_eff_rank']:.2f} | {pb['Dy_k95']} | "
        f"{pb['Dy_top1_energy']:.4f} | {pb['Dy_top5_energy']:.4f} | "
        f"{pb['E_kept_top_2_via_Wproj']:.4f} |"
    )
lines.append("")

lines.append("## Per-token Δy norms")
lines.append("")
lines.append("| block | mean | std | q05 | median | q95 | max |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
for l in ALL_BLOCKS:
    n = per_block[str(l)]["Dy_norms"]
    lines.append(
        f"| {l} | {n['mean']:.3f} | {n['std']:.3f} | {n['q05']:.3f} | "
        f"{n['median']:.3f} | {n['q95']:.3f} | {n['max']:.3f} |"
    )
lines.append("")

lines.append("## Top-5 right-SV cosine alignment Δy ↔ W_proj")
lines.append("")
lines.append(
    "Principal-angle cosines of (Vh_y[:5], Vh_w[:5]); both are orthonormal "
    "5-frames in R^768. cos_top5_sum max = 5.0 (perfect overlap), 0 = orthogonal."
)
lines.append("")
lines.append("| block | cos_1 | cos_2 | cos_3 | cos_4 | cos_5 | sum top-5 |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
for l in ALL_BLOCKS:
    cs = per_block[str(l)]["cos_principal_top5"]
    s = per_block[str(l)]["cos_top5_sum"]
    lines.append(
        f"| {l} | {cs[0]:.4f} | {cs[1]:.4f} | {cs[2]:.4f} | "
        f"{cs[3]:.4f} | {cs[4]:.4f} | {s:.4f} |"
    )
lines.append("")

lines.append("## Item 316 verdict logic (verbatim)")
lines.append("- **CONFIRMED** iff (a) b8 W_proj top-2/top-5 SV ratio ≤ 0.5 × median concentrator ratio AND (b) b8 Δy eff_rank ≥ 1.5 × median concentrator Δy eff_rank.")
lines.append("- **REFUTED** iff neither (a) nor (b) holds.")
lines.append("- **INCONCLUSIVE** otherwise.")
lines.append("")
lines.append(
    f"- Concentrator W_proj top2/top5 ratios = {[round(r, 4) for r in conc_ratios]}, "
    f"median = {conc_ratio_med:.4f}, 0.5× = {ratio_threshold:.4f}."
)
lines.append(
    f"- Concentrator Δy eff_rank = {[round(e, 2) for e in conc_dy_eff]}, "
    f"median = {conc_dy_eff_med:.2f}, 1.5× = {eff_rank_threshold:.2f}."
)
lines.append(
    f"- b8: top2/top5 = {b8_ratio:.4f} (≤ {ratio_threshold:.4f}? {cond_a}); "
    f"Δy eff_rank = {b8_dy_eff:.2f} (≥ {eff_rank_threshold:.2f}? {cond_b})."
)
lines.append("")
lines.append(f"**Result: {verdict}**.")
lines.append("")

lines.append("## Artifacts")
for p in [
    "experiments/block8_deep_dive_cycle306.py",
    "experiments/curvature_residual_geometric.py",
    OUT_JSON,
    OUT_PLOT,
    OUT_SUMMARY,
]:
    lines.append(f"- {p}")

with open(OUT_SUMMARY, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT_SUMMARY}", flush=True)
print("\nDone.", flush=True)
