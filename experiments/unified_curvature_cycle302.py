"""
Cycle 302 — Item 312: Unified curvature residual framework validation.

Validates `experiments/curvature_residual_geometric.py::UnifiedCurvatureFramework`
by reproducing three prior-cycle findings on (sigma=GELU, sigma_lin=ReLU) for
GPT-2 small across all 12 blocks:

  (a) cycle 217 raw rank_95 ≤ 64 in ≥ 9 blocks, per-block within ±5% of
      reference `[10,132,1,1,1,1,1,8,73,110,1,1]`.
  (b) cycle 218 centred Δ̃ c_fc top-5 cos sum ≥ 2.0 in ≥ 6 of 11 cross pairs.
      (Reference: all 11 ≥ 2.49.)
  (c) cycle 258 pullback κ shift under sigma_lin + b_leak swap ≥ 35% in at
      least one of layers 3–5 (κ indices 2–4). Reference: 0.477 at layer 4.

Verdict:
  - CONFIRMED if all three land within prior margins.
  - REFUTED if any reproduction is off by > 2× (rank95 count < 5, or
    n_pairs ≥ 2.0 < 3, or max_kappa_shift < 0.18).
  - INCONCLUSIVE otherwise.

Run cap ~20 min. No training.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from aleph.pullback_geometry import (
    collect_residual_trajectory,
    compute_curvature,
    get_unembedding,
)
from experiments.curvature_residual_geometric import (
    UnifiedCurvatureFramework,
    LinearisedAct,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
BLOCKS = list(range(12))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FLOAT_BASELINE_REF = 45.0637  # cycle 258 baseline on the 200-seq calib slice

RESULT_PATH = "autoresearch/results/cycle_302_unified.json"
SUMMARY_PATH = "autoresearch/results/cycle_302_summary.md"
PLOT_PATH = "autoresearch/results/cycle_302_panels.png"

# Cycle 217 reference rank_95 per block (from autoresearch/results/cycle_217_delta_per_token.json)
C217_RANK95_REF = [10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1]
# Cycle 258 reference kappa per κ-index (layer l+1), for variants A_gelu and B_relu_bleak
C258_KAPPA_A_GELU = [0.0001, 0.0022, 0.0022, 0.0019, 0.0011, 0.0007, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000]
C258_KAPPA_B_RELU = [0.0001, 0.0026, 0.0030, 0.0027, 0.0015, 0.0008, 0.0004, 0.0002, 0.0001, 0.0000, 0.0000]

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data ─────────────────────────────────────────────────────────
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
print(f"Calib: {len(calib_ids)} seqs × {SEQ_LEN} tokens = {N_CALIB * SEQ_LEN} tokens", flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
originals = [model.transformer.h[i].mlp.act for i in range(12)]


def eval_ppl(model, eval_ids):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


float_ppl = eval_ppl(model, calib_ids)
print(f"Float PPL on calib slice = {float_ppl:.4f}", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} off reference {FLOAT_BASELINE_REF}"
)

# ── Instantiate framework (sigma=GELU, sigma_lin=ReLU) ───────────────────────
print("\n── Building UnifiedCurvatureFramework(sigma=GELU, sigma_lin=ReLU) ──", flush=True)
def sigma_gelu(z):
    return F.gelu(z, approximate="tanh")
def sigma_relu(z):
    return F.relu(z)

fw = UnifiedCurvatureFramework(
    model=model,
    sigma=sigma_gelu,
    sigma_lin=sigma_relu,
    blocks=BLOCKS,
    calib_ids=calib_ids,
    device=DEVICE,
    tokenizer=tokenizer,
    A_factors=None,
    sigma_name="GELU",
    sigma_lin_name="ReLU",
)

# Shared forward pass to cache c_fc inputs at all 12 blocks.
print("Capturing c_fc inputs (single forward pass)...", flush=True)
t0 = time.time()
fw.capture_inputs()
print(f"  captured in {time.time() - t0:.1f}s", flush=True)

# ── Panels 1, 2, 3, 4 per block ─────────────────────────────────────────────
wte = model.transformer.wte.weight.detach().float().cpu()  # (50257, 768)

per_block_out = {}
rank95_raw_list = []
cross_alignment_rows = []

print("\n── Running panels 1–4 per block ──", flush=True)
t0 = time.time()
for l in BLOCKS:
    print(f"  block {l}...", end=" ", flush=True)
    p1 = fw.panel_raw_rank(l)
    rank95_raw_list.append(p1["rank_95_raw"])

    p2 = fw.panel_centred_alignment(l, l_next=l + 1 if l + 1 < 12 else None)
    if l + 1 < 12:
        cross_alignment_rows.append({
            "l": l,
            "lp1": l + 1,
            "c_fc_top5_cos_sum": p2["c_fc_top5_cos_sum"],
            "Q_top5_cos_sum": p2["Q_top5_cos_sum"],
            "K_top5_cos_sum": p2["K_top5_cos_sum"],
            "V_top5_cos_sum": p2["V_top5_cos_sum"],
            "c_fc_top20_cos_sum": p2["c_fc_top20_cos_sum"],
        })

    p3 = fw.panel_vocab_projection(l, wte, top_k_dirs=5, top_n_tokens=20)

    # Panel 4 runs only if A_factors provided; we compute Vh[:2] on the fly
    # for all blocks as a general A-projection measurement (cycle 301-style).
    p4 = fw.panel_A_projection(l, rank=2)

    per_block_out[l] = {
        "block": l,
        "panel1_raw_rank": {
            "rank_95_raw": p1["rank_95_raw"],
            "rank_99_raw": p1["rank_99_raw"],
            "top5_singular_values_raw": p1["top5_singular_values_raw"],
            "delta_norm_mean": p1["delta_norm_mean"],
        },
        "panel2_centred_alignment": {
            "rank_95_centred": p2["rank_95_centred"],
            "rank_99_centred": p2["rank_99_centred"],
            "sigma0": p2["sigma0"],
            "sigma1": p2["sigma1"],
            "sigma2": p2["sigma2"],
            **{k: v for k, v in p2.items() if k.endswith("_cos_sum")},
        },
        "panel3_vocab_projection": p3,
        "panel4_A_projection": p4,
    }
    print(f"rank_95_raw={p1['rank_95_raw']}  E_kept_top2={p4['E_kept']:.4f}", flush=True)

    # Drop heavy Dy caches; Panel 5 does a separate forward pass.
    fw.free_block(l)

print(f"  panels 1–4 done in {time.time() - t0:.1f}s", flush=True)

# ── Panel 5: κ shift under (sigma_lin + b_leak) swap (cycle 258) ────────────
print("\n── Panel 5: κ shift under LinearisedAct(sigma_lin + b_leak) ──", flush=True)
# Need b_leak per block — but we cleared the per-block caches. Re-capture.
fw._per_block_cache = {}
print("  re-capturing Dy to compute b_leak...", flush=True)
t0 = time.time()
for l in BLOCKS:
    fw.compute_Dy(l)
b_leak = fw.compute_b_leak()
print(f"  b_leak ready in {time.time() - t0:.1f}s", flush=True)
# Now free the per-block caches again.
for l in BLOCKS:
    fw.free_block(l)

print("  collecting baseline trajectory + kappa (unpatched)...", flush=True)
t0 = time.time()
U_un = get_unembedding(model)
per_seq_trajs_A = []
for ids in calib_ids:
    per_seq_trajs_A.append(collect_residual_trajectory(model, ids))
n_layers_plus = len(per_seq_trajs_A[0])
traj_A = [torch.cat([t[li] for t in per_seq_trajs_A], dim=0) for li in range(n_layers_plus)]
K_A = compute_curvature(traj_A, U_un, reduce="none").to("cpu").float()  # (11, B, T)
kappa_A_mean = K_A.mean(dim=(1, 2)).tolist()
del per_seq_trajs_A, traj_A, K_A
print(f"    κ_A mean per layer = {[round(x, 4) for x in kappa_A_mean]}", flush=True)
print(f"    done in {time.time() - t0:.1f}s", flush=True)

print("  installing LinearisedAct (sigma_lin + b_leak) on all 12 blocks...", flush=True)
for l in range(12):
    model.transformer.h[l].mlp.act = LinearisedAct(sigma_relu, b_leak[l]).to(DEVICE)
ids_check = set(id(model.transformer.h[i].mlp.act) for i in range(12))
assert len(ids_check) == 12, f"Patches not distinct: {len(ids_check)}/12"

patched_ppl = eval_ppl(model, calib_ids)
print(f"  patched PPL = {patched_ppl:.4f}", flush=True)

print("  collecting patched trajectory + kappa...", flush=True)
t0 = time.time()
per_seq_trajs_B = []
for ids in calib_ids:
    per_seq_trajs_B.append(collect_residual_trajectory(model, ids))
traj_B = [torch.cat([t[li] for t in per_seq_trajs_B], dim=0) for li in range(n_layers_plus)]
K_B = compute_curvature(traj_B, U_un, reduce="none").to("cpu").float()
kappa_B_mean = K_B.mean(dim=(1, 2)).tolist()
del per_seq_trajs_B, traj_B, K_B
print(f"    κ_B mean per layer = {[round(x, 4) for x in kappa_B_mean]}", flush=True)
print(f"    done in {time.time() - t0:.1f}s", flush=True)

# Restore
for l in range(12):
    model.transformer.h[l].mlp.act = originals[l]
post_restore_ppl = eval_ppl(model, calib_ids)
assert abs(post_restore_ppl - float_ppl) < 1e-4, (
    f"Restore round-trip failed: {post_restore_ppl} vs {float_ppl}"
)

# Per-layer relative kappa diffs. Index 0 -> layer 1, index 10 -> layer 11.
rel_kappa_shift = []
for ki in range(11):
    g = kappa_A_mean[ki]
    r = kappa_B_mean[ki]
    if abs(g) > 1e-10:
        rel_kappa_shift.append(abs(g - r) / abs(g))
    else:
        rel_kappa_shift.append(0.0)
print(f"  rel_kappa_shift per layer (κ idx 0..10 -> layers 1..11): "
      f"{[round(x, 3) for x in rel_kappa_shift]}", flush=True)

# Layers 3-5 correspond to κ indices 2..4 (κ index k => layer k+1).
layers_3_5_shift = rel_kappa_shift[2:5]
max_layer_3_5_shift = max(layers_3_5_shift)
argmax_layer_3_5 = 3 + int(torch.tensor(layers_3_5_shift).argmax().item())
print(f"  max rel κ shift in layers 3–5 = {max_layer_3_5_shift:.4f} at layer {argmax_layer_3_5}",
      flush=True)

# Cycle 258 comparison
c258_rel_shift = [
    abs(g - r) / abs(g) if abs(g) > 1e-10 else 0.0
    for g, r in zip(C258_KAPPA_A_GELU, C258_KAPPA_B_RELU)
]
print(f"  cycle 258 reference rel κ shift: {[round(x, 3) for x in c258_rel_shift]}", flush=True)

# ── Validation metrics ──────────────────────────────────────────────────────
print("\n── Validation metrics ──", flush=True)

# (a) cycle 217 rank_95 ≤ 64 count + per-block ±5%
n_rank95_le_64 = sum(1 for r in rank95_raw_list if r <= 64)
rank95_ref_diff_pct = []
for l, (r_new, r_ref) in enumerate(zip(rank95_raw_list, C217_RANK95_REF)):
    if r_ref > 0:
        d = abs(r_new - r_ref) / max(r_ref, 1)
    else:
        d = 0.0 if r_new == 0 else float("inf")
    rank95_ref_diff_pct.append(d)
n_within_5pct_of_c217 = sum(1 for d in rank95_ref_diff_pct if d <= 0.05)
print(f"  (a) n_blocks rank_95 ≤ 64: {n_rank95_le_64}/12 (ref 9/12)")
print(f"      per-block rank_95 (this cycle): {rank95_raw_list}")
print(f"      per-block rank_95 (c217 ref):   {C217_RANK95_REF}")
print(f"      per-block |diff|/ref: {[round(d, 3) for d in rank95_ref_diff_pct]}")
print(f"      n within ±5%: {n_within_5pct_of_c217}/12")
a_pass = (n_rank95_le_64 >= 9) and (n_within_5pct_of_c217 >= 9)

# (b) cycle 218 c_fc top-5 cos sum ≥ 2.0 in ≥ 6 of 11 pairs
c_fc_sums = [row["c_fc_top5_cos_sum"] for row in cross_alignment_rows]
n_cfc_ge_2 = sum(1 for x in c_fc_sums if x >= 2.0)
print(f"  (b) n_pairs c_fc_top5_cos_sum ≥ 2.0: {n_cfc_ge_2}/11 (ref 11/11)")
print(f"      per-pair: {[round(x, 3) for x in c_fc_sums]}")
b_pass = n_cfc_ge_2 >= 6

# (c) cycle 258 κ shift in layers 3–5
print(f"  (c) max κ shift in layers 3–5 = {max_layer_3_5_shift:.4f} (ref 0.477 at layer 4)")
c_pass = max_layer_3_5_shift >= 0.35

# Refutation thresholds (2x off)
a_refuted = n_rank95_le_64 < 5 or n_within_5pct_of_c217 < 5
b_refuted = n_cfc_ge_2 < 3
c_refuted = max_layer_3_5_shift < 0.18

if a_pass and b_pass and c_pass:
    verdict = "CONFIRMED"
elif a_refuted or b_refuted or c_refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

verdict_reason = (
    f"(a) n_blocks rank_95 ≤ 64 = {n_rank95_le_64}/12 (pass ≥9: {a_pass}), "
    f"n within ±5% = {n_within_5pct_of_c217}/12; "
    f"(b) n_pairs c_fc_top5 ≥ 2.0 = {n_cfc_ge_2}/11 (pass ≥6: {b_pass}); "
    f"(c) max κ shift (layers 3–5) = {max_layer_3_5_shift:.4f} "
    f"(pass ≥0.35: {c_pass})."
)
print(f"\n== VERDICT: {verdict} ==")
print(verdict_reason, flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
# Scrub torch tensors from per_block_out for JSON. panel3 has boost_ids etc.
def scrub(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


result = {
    "cycle": 302,
    "item": 312,
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
    },
    "float_ppl_calib": float_ppl,
    "patched_ppl_calib": patched_ppl,
    "per_block": scrub(per_block_out),
    "cross_block_alignment": cross_alignment_rows,
    "panel5_kappa_shift": {
        "kappa_A_mean_per_layer": kappa_A_mean,
        "kappa_B_mean_per_layer": kappa_B_mean,
        "rel_kappa_shift_per_layer": rel_kappa_shift,
        "max_layer_3_5_shift": max_layer_3_5_shift,
        "argmax_layer_in_3_5": argmax_layer_3_5,
        "cycle_258_reference_rel_shift": c258_rel_shift,
    },
    "validation": {
        "cycle_217_ref_rank_95": C217_RANK95_REF,
        "rank_95_raw_per_block": rank95_raw_list,
        "n_rank95_le_64": n_rank95_le_64,
        "n_within_5pct_of_c217": n_within_5pct_of_c217,
        "c_fc_top5_cos_sum_per_pair": c_fc_sums,
        "n_cfc_ge_2": n_cfc_ge_2,
        "max_kappa_shift_3_5": max_layer_3_5_shift,
        "a_pass": bool(a_pass),
        "b_pass": bool(b_pass),
        "c_pass": bool(c_pass),
        "a_refuted": bool(a_refuted),
        "b_refuted": bool(b_refuted),
        "c_refuted": bool(c_refuted),
    },
}

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {RESULT_PATH}", flush=True)

# ── Plot: 4-panel figure — rank_95, c_fc_top5, E_kept_top2, rel_kappa ──────
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
# P1: rank_95 per block
axes[0].plot(BLOCKS, rank95_raw_list, marker="o", label="cycle 302")
axes[0].plot(BLOCKS, C217_RANK95_REF, marker="x", linestyle="--", label="cycle 217 ref")
axes[0].axhline(64, color="gray", linestyle=":", label="64")
axes[0].set_title("P1 rank_95 raw Δy per block")
axes[0].set_xlabel("block")
axes[0].set_ylabel("rank_95")
axes[0].legend()
# P2: c_fc top-5 cos sum per pair
pair_lbls = [f"{r['l']}→{r['lp1']}" for r in cross_alignment_rows]
axes[1].plot(range(len(c_fc_sums)), c_fc_sums, marker="o")
axes[1].axhline(2.0, color="red", linestyle="--", label="threshold 2.0")
axes[1].set_xticks(range(len(c_fc_sums)))
axes[1].set_xticklabels(pair_lbls, rotation=45, fontsize=7)
axes[1].set_title("P2 c_fc top-5 cos sum (centred Δ̃)")
axes[1].set_ylabel("cos sum")
axes[1].legend()
# P3 (P4 really): E_kept_top2 per block
e_kept = [per_block_out[l]["panel4_A_projection"]["E_kept"] for l in BLOCKS]
axes[2].bar(BLOCKS, e_kept)
axes[2].set_title("P4 E_kept_top2 (Vh[:2] of W_proj) per block")
axes[2].set_xlabel("block")
axes[2].set_ylabel("E_kept")
axes[2].axhline(0.05, color="red", linestyle=":", label="0.05")
axes[2].axhline(0.30, color="orange", linestyle=":", label="0.30")
axes[2].legend()
# P5: rel kappa shift per layer
layer_ids = list(range(1, 12))
axes[3].plot(layer_ids, rel_kappa_shift, marker="o", label="cycle 302")
axes[3].plot(layer_ids, c258_rel_shift, marker="x", linestyle="--", label="cycle 258 ref")
axes[3].axhline(0.35, color="red", linestyle=":", label="0.35")
axes[3].set_title("P5 rel κ shift (GELU vs ReLU+b_leak)")
axes[3].set_xlabel("layer")
axes[3].set_ylabel("|Δκ|/κ")
axes[3].legend()

fig.suptitle(f"Cycle 302 Unified Curvature Framework — {verdict}")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=120)
print(f"Wrote {PLOT_PATH}", flush=True)

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 302 — Item 312: Unified curvature residual geometric framework")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(verdict_reason)
lines.append("")
lines.append("## Framework")
lines.append(
    "Module `experiments/curvature_residual_geometric.py::UnifiedCurvatureFramework` "
    "parameterises the curvature residual Δy = W_proj · (sigma − sigma_lin)(z) for any "
    "activation pair. This run: sigma=GELU (tanh approx), sigma_lin=ReLU, GPT-2 small, "
    f"calib = {N_CALIB} wikitext-2 val seqs × {SEQ_LEN} tokens = {N_CALIB * SEQ_LEN} "
    "tokens, seed 42."
)
lines.append("")
lines.append("## Panel 1 — raw Δy rank_95 (cycle 217 reproduction)")
lines.append("")
lines.append("| block | this cycle | cycle 217 ref | |diff|/ref |")
lines.append("|---:|---:|---:|---:|")
for l in BLOCKS:
    lines.append(
        f"| {l} | {rank95_raw_list[l]} | {C217_RANK95_REF[l]} | "
        f"{rank95_ref_diff_pct[l]:.3f} |"
    )
lines.append("")
lines.append(
    f"n blocks rank_95 ≤ 64 = **{n_rank95_le_64}/12** (ref 9/12). "
    f"n within ±5% of c217 reference = **{n_within_5pct_of_c217}/12**."
)
lines.append("")
lines.append("## Panel 2 — centred Δ̃ c_fc top-5 cos sum (cycle 218 reproduction)")
lines.append("")
lines.append("| l→l+1 | c_fc top5 | Q top5 | K top5 | V top5 |")
lines.append("|---|---:|---:|---:|---:|")
for r in cross_alignment_rows:
    lines.append(
        f"| {r['l']}→{r['lp1']} | {r['c_fc_top5_cos_sum']:.3f} | "
        f"{r['Q_top5_cos_sum']:.3f} | {r['K_top5_cos_sum']:.3f} | "
        f"{r['V_top5_cos_sum']:.3f} |"
    )
lines.append("")
lines.append(f"n pairs c_fc_top5 ≥ 2.0 = **{n_cfc_ge_2}/11** (ref 11/11).")
lines.append("")
lines.append("## Panel 4 — E_kept_top2 vs Vh[:2] of W_proj (cycle 301 reproduction)")
lines.append("")
lines.append("| block | E_kept_top2 | cycle 301 ref |")
lines.append("|---:|---:|---:|")
c301_ref = {0: 0.5902, 1: 0.7463, 2: 0.9190, 11: 0.9803}
for l in BLOCKS:
    ref = c301_ref.get(l, "-")
    ref_str = f"{ref:.4f}" if isinstance(ref, float) else "-"
    lines.append(
        f"| {l} | {per_block_out[l]['panel4_A_projection']['E_kept']:.4f} | {ref_str} |"
    )
lines.append("")
lines.append("## Panel 5 — pullback κ shift under sigma_lin + b_leak swap (cycle 258 reproduction)")
lines.append("")
lines.append("| layer | κ_A (GELU) | κ_B (ReLU+b_leak) | rel shift | c258 ref shift |")
lines.append("|---:|---:|---:|---:|---:|")
for ki in range(11):
    lines.append(
        f"| {ki + 1} | {kappa_A_mean[ki]:.5f} | {kappa_B_mean[ki]:.5f} | "
        f"{rel_kappa_shift[ki]:.3f} | {c258_rel_shift[ki]:.3f} |"
    )
lines.append("")
lines.append(
    f"max rel κ shift in layers 3–5 = **{max_layer_3_5_shift:.4f}** at layer "
    f"{argmax_layer_3_5} (ref 0.477 at layer 4)."
)
lines.append("")
lines.append("## Verdict logic (item 312 verbatim)")
lines.append(
    "- CONFIRMED iff (a) n_blocks rank_95 ≤ 64 ≥ 9 AND per-block within ±5% of "
    "cycle 217 reference, AND (b) n_pairs c_fc_top5_cos_sum ≥ 2.0 ≥ 6, AND "
    "(c) max κ shift in layers 3–5 ≥ 0.35."
)
lines.append(
    "- REFUTED iff any reproduction off by > 2× (n_rank95_le_64 < 5, "
    "n_cfc_ge_2 < 3, or max_kappa_shift < 0.18)."
)
lines.append("- INCONCLUSIVE otherwise.")
lines.append("")
lines.append(f"**Result: {verdict}**.")
lines.append("")
lines.append("## Artifacts")
for p in [
    "experiments/curvature_residual_geometric.py",
    "experiments/unified_curvature_cycle302.py",
    RESULT_PATH,
    PLOT_PATH,
    SUMMARY_PATH,
]:
    lines.append(f"- {p}")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
