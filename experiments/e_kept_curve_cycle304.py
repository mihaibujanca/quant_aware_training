"""
Cycle 304 — Item 314: E_kept_top_k curve per block.

For each GPT-2 small block l in {0..11} and each k in a sweep grid, compute
    E_kept_top_k(l) = ||Dy @ Vh[:k, :]^T||_F^2 / ||Dy||_F^2
where
    z       = LN2(x) @ W_fc + b_fc        in R^3072 (per token)
    delta   = GELU(z) - ReLU(z)           in R^3072
    Dy      = delta @ W_proj              in R^768
    W_proj  = U diag(S) Vh                (3072, 768)  [HF Conv1D]
    Vh[:k]  = top-k right singular vectors of W_proj, orthonormal rows in R^768.

Find k_95(l) = smallest k in the sweep grid with E_kept_top_k(l) >= 0.95.

Cycle 302's panel 4 measured only k=2; cycle 301 the same 4 blocks at k=2.
This script extends to the full E_kept_top_k curve.

Item 314 verdict thresholds:
  CONFIRMED  iff k_95 <= 10 at >= 8 of 12 blocks AND k_95 <= 5 at ALL
             concentrator blocks {1, 2, 10, 11}.
  REFUTED    iff k_95 >= 50 at >= 6 of 12 blocks.
  INCONCLUSIVE otherwise.

Config: 200 wikitext-2 val x 128 tok, seed 42, DEVICE=mps; no training.
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
BLOCKS = list(range(12))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FLOAT_BASELINE_REF = 45.0637  # cycle 302 calib slice reference

# Sweep grid: covers 1..500 (< 768). Includes brief-constraint ks {2, 5, 10, 20, 50, 100}.
K_GRID = [1, 2, 3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 233, 377, 500]

# Concentrator blocks per item 314 verbatim
CONCENTRATORS = [1, 2, 10, 11]

# Cycle 302 panel 4 E_kept_top_2 reference per block (for sanity check at k=2)
C302_E_TOP2_REF = {
    0: 0.5902, 1: 0.7463, 2: 0.9190, 3: 0.8649, 4: 0.8426, 5: 0.8109,
    6: 0.7197, 7: 0.6622, 8: 0.4722, 9: 0.5356, 10: 0.8374, 11: 0.9803,
}

RESULT_PATH = "autoresearch/results/cycle_304_curve.json"
SUMMARY_PATH = "autoresearch/results/cycle_304_summary.md"
PLOT_PATH = "autoresearch/results/cycle_304_curve.png"

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
print(f"Calib: {len(calib_ids)} seqs x {SEQ_LEN} tokens = {N_CALIB * SEQ_LEN} tokens",
      flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()


def eval_ppl(m, eval_ids):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = m(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


float_ppl = eval_ppl(model, calib_ids)
print(f"Float PPL on calib slice = {float_ppl:.4f}", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} off reference {FLOAT_BASELINE_REF}"
)

# ── Framework ────────────────────────────────────────────────────────────────
fw = UnifiedCurvatureFramework(
    model=model,
    sigma=lambda z: F.gelu(z, approximate="tanh"),
    sigma_lin=F.relu,
    blocks=BLOCKS,
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

# ── Sweep E_kept_top_k per block ────────────────────────────────────────────
per_block_curves: dict = {}
per_block_k95: dict = {}
per_block_Dy_frob_sq: dict = {}
per_block_top10_S_wproj: dict = {}

print("\n── Sweeping E_kept_top_k per block ──", flush=True)
t_sweep = time.time()
for l in BLOCKS:
    t0 = time.time()
    c = fw.compute_Dy(l)
    W_proj = c["W_proj"]          # (3072, 768)
    Dy = c["Dy"]                  # (N, 768) CPU float32
    assert W_proj.shape == (3072, 768), f"block {l} W_proj {W_proj.shape}"

    # Full SVD of W_proj (768 right-singular vectors in Vh, rows).
    U_, S_, Vh = torch.linalg.svd(W_proj, full_matrices=False)
    assert Vh.shape == (768, 768), f"block {l} Vh {Vh.shape}"

    Dy_norm_sq = float((Dy * Dy).sum().item())
    per_block_Dy_frob_sq[l] = Dy_norm_sq
    per_block_top10_S_wproj[l] = [float(s) for s in S_[:10].tolist()]

    # Compute all k in grid. Projection coefficients are Dy @ Vh[:k,:].T = (Dy @ Vh.T)[:, :k];
    # compute (Dy @ Vh.T) once then slice.
    proj_all = Dy @ Vh.T  # (N, 768)
    per_col_energy = (proj_all * proj_all).sum(dim=0)  # (768,)
    cum_energy = per_col_energy.cumsum(0)              # monotone non-decreasing

    curve = {}
    for k in K_GRID:
        # top-k rows of Vh => top-k columns of proj_all
        ek = float(cum_energy[k - 1].item()) / Dy_norm_sq
        curve[k] = ek
    per_block_curves[l] = curve

    # k_95 over a dense 1..768 sweep to avoid grid-coarseness artifacts
    frac_all = cum_energy / Dy_norm_sq
    k95_dense = int((frac_all < 0.95).sum().item()) + 1
    # Also record k_95 within K_GRID (for the stated grid) for reporting
    k95_grid = None
    for k in K_GRID:
        if curve[k] >= 0.95:
            k95_grid = k
            break
    per_block_k95[l] = {"k95_dense": k95_dense, "k95_grid": k95_grid}

    # Cycle 302 sanity check at k=2
    ref = C302_E_TOP2_REF.get(l)
    assert abs(curve[2] - ref) < 1e-4, (
        f"block {l}: E_kept_top_2 {curve[2]:.6f} vs cycle 302 ref {ref:.6f} "
        f"(delta {curve[2] - ref:+.6f})"
    )

    # Monotone sanity on the grid
    prev = -1.0
    for k in K_GRID:
        assert curve[k] >= prev - 1e-6, (
            f"block {l} k={k}: E_kept {curve[k]} < prev {prev}"
        )
        prev = curve[k]

    print(
        f"  block {l:2d}: k95_dense={k95_dense:3d}  E_k2={curve[2]:.4f}  "
        f"E_k5={curve[5]:.4f}  E_k10={curve[10]:.4f}  E_k20={curve[20]:.4f}  "
        f"E_k50={curve[50]:.4f}  E_k100={curve[100]:.4f}  "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    fw.free_block(l)

print(f"\n  sweep done in {time.time() - t_sweep:.1f}s", flush=True)

# ── Verdict per item 314 ─────────────────────────────────────────────────────
# Use dense k_95 as the primary metric (smallest k in 1..768 s.t. E_kept_top_k >= 0.95)
k95_dense_list = [per_block_k95[l]["k95_dense"] for l in BLOCKS]
n_k95_le_10 = sum(1 for k in k95_dense_list if k <= 10)
concentrator_k95 = [per_block_k95[l]["k95_dense"] for l in CONCENTRATORS]
concentrator_k95_le_5 = all(k <= 5 for k in concentrator_k95)
n_k95_ge_50 = sum(1 for k in k95_dense_list if k >= 50)

confirmed = (n_k95_le_10 >= 8) and concentrator_k95_le_5
refuted = n_k95_ge_50 >= 6
if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

verdict_reason = (
    f"n_blocks k95_dense <= 10 = {n_k95_le_10}/12 (need >= 8); "
    f"concentrator k95 {CONCENTRATORS} = {concentrator_k95} "
    f"(all <= 5: {concentrator_k95_le_5}); "
    f"n_blocks k95_dense >= 50 = {n_k95_ge_50}/12 (refute if >= 6); "
    f"k95_dense per block = {k95_dense_list}."
)

print(f"\n== VERDICT: {verdict} ==", flush=True)
print(verdict_reason, flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 304,
    "item": 314,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "config": {
        "N_CALIB": N_CALIB,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "BLOCKS": BLOCKS,
        "DEVICE": DEVICE,
        "sigma": "GELU (tanh approx)",
        "sigma_lin": "ReLU",
        "model": "gpt2",
        "K_GRID": K_GRID,
        "CONCENTRATORS": CONCENTRATORS,
    },
    "float_ppl_calib": float_ppl,
    "per_block_curve": {
        str(l): {str(k): per_block_curves[l][k] for k in K_GRID} for l in BLOCKS
    },
    "per_block_k95": {
        str(l): per_block_k95[l] for l in BLOCKS
    },
    "per_block_Dy_frob_sq": {str(l): per_block_Dy_frob_sq[l] for l in BLOCKS},
    "per_block_W_proj_top10_S": {str(l): per_block_top10_S_wproj[l] for l in BLOCKS},
    "cycle_302_E_top2_ref": C302_E_TOP2_REF,
    "validation": {
        "k95_dense_per_block": k95_dense_list,
        "n_k95_le_10": n_k95_le_10,
        "concentrator_k95": concentrator_k95,
        "concentrator_k95_le_5": bool(concentrator_k95_le_5),
        "n_k95_ge_50": n_k95_ge_50,
        "confirmed": bool(confirmed),
        "refuted": bool(refuted),
    },
}

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {RESULT_PATH}", flush=True)

# ── Plot per-block curve ─────────────────────────────────────────────────────
fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5))

cmap = matplotlib.colormaps.get_cmap("viridis")
for i, l in enumerate(BLOCKS):
    ks = K_GRID
    es = [per_block_curves[l][k] for k in ks]
    color = cmap(i / 11.0)
    label = f"b{l} (k95={per_block_k95[l]['k95_dense']})"
    ax_curve.plot(ks, es, marker="o", markersize=3, color=color, label=label)
ax_curve.axhline(0.95, color="red", linestyle="--", linewidth=1, label="0.95")
ax_curve.set_xscale("log")
ax_curve.set_xlabel("k (top-k right SVs of W_proj)")
ax_curve.set_ylabel("E_kept_top_k")
ax_curve.set_title("Cycle 304 — E_kept_top_k curve per block")
ax_curve.legend(fontsize=7, loc="lower right", ncol=2)
ax_curve.grid(alpha=0.3)

bar_colors = ["tab:orange" if l in CONCENTRATORS else "tab:blue" for l in BLOCKS]
ax_bar.bar(BLOCKS, k95_dense_list, color=bar_colors)
ax_bar.axhline(10, color="red", linestyle="--", label="k95<=10 (confirm)")
ax_bar.axhline(5, color="green", linestyle=":", label="k95<=5 (concentrator)")
ax_bar.axhline(50, color="black", linestyle="-.", label="k95>=50 (refute)")
ax_bar.set_xlabel("block")
ax_bar.set_ylabel("k_95 (dense 1..768)")
ax_bar.set_title("k_95 per block (orange = concentrator blocks {1,2,10,11})")
ax_bar.legend(fontsize=8)
for x, v in zip(BLOCKS, k95_dense_list):
    ax_bar.text(x, v + 0.5, str(v), ha="center", fontsize=8)

fig.suptitle(f"Cycle 304 / Item 314 — {verdict}")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=120)
print(f"Wrote {PLOT_PATH}", flush=True)

# ── Summary markdown ────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 304 — Item 314: E_kept_top_k curve per block")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(verdict_reason)
lines.append("")
lines.append("## Geometric object")
lines.append(
    "For GPT-2 small block l, z = LN2(x)@W_fc+b_fc in R^3072, delta = GELU(z) - ReLU(z), "
    "Dy = delta @ W_proj in R^768 (W_proj HF Conv1D (3072, 768)). For SVD "
    "W_proj = U diag(S) Vh, Vh[:k,:] are k orthonormal rows in R^768 (top-k right "
    "singular vectors). E_kept_top_k = ||Dy @ Vh[:k,:]^T||_F^2 / ||Dy||_F^2. "
    "k_95(l) = smallest k with E_kept_top_k >= 0.95, computed densely over 1..768."
)
lines.append("")
lines.append("## Per-block E_kept_top_k curve")
lines.append("")
header = "| block | " + " | ".join(f"k={k}" for k in K_GRID) + " | k95_dense |"
sep = "|---:|" + "---:|" * (len(K_GRID) + 1)
lines.append(header)
lines.append(sep)
for l in BLOCKS:
    row = f"| {l} | " + " | ".join(
        f"{per_block_curves[l][k]:.4f}" for k in K_GRID
    ) + f" | {per_block_k95[l]['k95_dense']} |"
    lines.append(row)
lines.append("")
lines.append("## k_95 summary")
lines.append("")
lines.append("| block | k_95 (dense) | concentrator? | flag |")
lines.append("|---:|---:|:---:|---|")
for l in BLOCKS:
    k95 = per_block_k95[l]["k95_dense"]
    conc = "YES" if l in CONCENTRATORS else "-"
    if l in CONCENTRATORS:
        flag = "conc<=5 ok" if k95 <= 5 else "conc>5 FAIL"
    elif k95 <= 10:
        flag = "<=10"
    elif k95 >= 50:
        flag = ">=50 (refute bucket)"
    else:
        flag = "11-49"
    lines.append(f"| {l} | {k95} | {conc} | {flag} |")
lines.append("")
lines.append(
    f"n_blocks k95 <= 10: **{n_k95_le_10}/12** (confirm threshold: >= 8). "
    f"concentrator k_95 = {concentrator_k95} (confirm: all <= 5 => "
    f"{concentrator_k95_le_5}). n_blocks k95 >= 50: **{n_k95_ge_50}/12** "
    f"(refute threshold: >= 6)."
)
lines.append("")
lines.append("## Item 314 verdict logic (verbatim)")
lines.append(
    "- **CONFIRMED** iff k_95 <= 10 at >= 8 of 12 blocks AND k_95 <= 5 at ALL 4 "
    "concentrator blocks (1, 2, 10, 11)."
)
lines.append(
    "- **REFUTED** iff k_95 >= 50 at >= 6 of 12 blocks."
)
lines.append("- **INCONCLUSIVE** otherwise.")
lines.append("")
lines.append(f"**Result: {verdict}**.")
lines.append("")
lines.append("## Artifacts")
for p in [
    "experiments/e_kept_curve_cycle304.py",
    "experiments/curvature_residual_geometric.py",
    RESULT_PATH,
    PLOT_PATH,
    SUMMARY_PATH,
]:
    lines.append(f"- {p}")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
