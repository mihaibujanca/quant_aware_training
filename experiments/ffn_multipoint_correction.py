"""Cycle 043: Multi-point residual correction at every compressed block boundary.

Fits N sequential affine corrections (one per compressed block, in layer order)
and compares against cycle 039's single-point correction.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ── Config ────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K4_BLOCKS = [7, 5, 4, 1]
K5_BLOCKS = [7, 5, 1, 3, 6]
HIDDEN_FRACTION = 1 / 6
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
SEED = 42


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


def collect_hidden_states(model, input_ids, hook_layer_idx, batch_size=4):
    collected = []
    def hook_fn(module, args):
        collected.append(args[0].detach().clone())
    handle = model.transformer.h[hook_layer_idx].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            model(input_ids[i:i+batch_size])
    handle.remove()
    return torch.cat(collected, dim=0)


def fit_correction(h_float, h_comp):
    h_f = h_float.reshape(-1, h_float.shape[-1]).float()
    h_c = h_comp.reshape(-1, h_comp.shape[-1]).float()
    mse_before = F.mse_loss(h_c, h_f).item()
    H = torch.cat([h_c, torch.ones(h_c.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(H, h_f).solution
    W = sol[:h_f.shape[1]].T
    b = sol[h_f.shape[1]]
    h_fixed = h_c @ W.T + b
    mse_after = F.mse_loss(h_fixed, h_f).item()
    frob = torch.norm(W - torch.eye(W.shape[0]), p='fro').item()
    b_norm = torch.norm(b).item()
    return W, b, mse_before, mse_after, frob, b_norm


def make_correction_hook(W, b):
    def hook_fn(module, args):
        h = args[0]
        return (h @ W.T + b,) + args[1:]
    return hook_fn


def run_multipoint_correction(teacher, comp_model, probe_ids, compressed_blocks):
    """Fit sequential corrections at each compressed block boundary."""
    sorted_blocks = sorted(compressed_blocks)
    corrections = []
    hooks = []

    for block_id in sorted_blocks:
        correction_layer = block_id + 1
        if correction_layer >= 12:  # skip if last block
            continue

        # Collect hidden states (comp_model already has prior corrections applied)
        h_float = collect_hidden_states(teacher.model, probe_ids, correction_layer, BATCH_SIZE)
        h_comp = collect_hidden_states(comp_model, probe_ids, correction_layer, BATCH_SIZE)

        W, b, mse_before, mse_after, frob, b_norm = fit_correction(h_float, h_comp)

        # Install correction on comp_model
        hook = comp_model.transformer.h[correction_layer].register_forward_pre_hook(
            make_correction_hook(W, b)
        )
        hooks.append(hook)

        reduction_pct = (1 - mse_after / mse_before) * 100 if mse_before > 0 else 0
        corrections.append({
            "block": block_id,
            "correction_layer": correction_layer,
            "mse_before": mse_before,
            "mse_after": mse_after,
            "mse_reduction_pct": reduction_pct,
            "frob_W_delta": frob,
            "b_norm": b_norm,
        })
        print(f"    Block {block_id} → layer {correction_layer}: "
              f"MSE {mse_before:.4f} → {mse_after:.4f} ({reduction_pct:.1f}% red), "
              f"||W-I||={frob:.3f}")

    return corrections, hooks


# ── Main ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

print("Building probe corpus...")
corpus_spec = ProbeCorpusSpec(
    teacher=teacher, text_path=SHAKESPEARE_PATH, sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
    eval_val_sequences=PROBE_HOLDOUT_SEQ, dataset_name="proxy_shakespeare",
)
corpus = build_probe_corpus(corpus_spec)

baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
print(f"  Baseline PPL: {baseline_ppl:.2f}")

all_blocks = sorted(set(K4_BLOCKS + K5_BLOCKS))
print(f"Probing geometry for blocks {all_blocks}...")
atlas = probe_ffn_geometry(
    teacher, corpus, layer_ids=tuple(all_blocks),
    max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
)

budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
full_pack = derive_ffn_compression(atlas, budget)
compressed_by_id = {l.layer_id: l for l in full_pack.layers}
probe_ids = corpus.probe_train[:, :-1]

results = {"baseline_ppl": float(baseline_ppl), "configs": {}}

# ── k=4 ──────────────────────────────────────────────────────────────────────
for config_name, blocks in [("k4", K4_BLOCKS), ("k5", K5_BLOCKS)]:
    print(f"\n{'='*60}")
    print(f"{config_name.upper()} — blocks {blocks}")
    print(f"{'='*60}")

    sub_layers = tuple(compressed_by_id[b] for b in blocks)
    sub_pack = CompressedFfnPack(
        method=full_pack.method, model_family=full_pack.model_family,
        dataset_name=full_pack.dataset_name, layers=sub_layers, metadata=full_pack.metadata,
    )

    # Uncorrected
    comp_model = swap_compressed_ffns(teacher, sub_pack, inplace=False)
    uncorr_ppl = eval_ppl(comp_model, teacher, corpus, BATCH_SIZE)
    uncorr_rel = (uncorr_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"  Uncorrected: {uncorr_ppl:.2f} ({uncorr_rel:+.2f}%)")
    del comp_model

    # Single-point correction (at max(blocks)+1)
    comp_model_sp = swap_compressed_ffns(teacher, sub_pack, inplace=False)
    correction_layer_sp = max(blocks) + 1
    h_f_sp = collect_hidden_states(teacher.model, probe_ids, correction_layer_sp, BATCH_SIZE)
    h_c_sp = collect_hidden_states(comp_model_sp, probe_ids, correction_layer_sp, BATCH_SIZE)
    W_sp, b_sp, _, _, _, _ = fit_correction(h_f_sp, h_c_sp)
    hook_sp = comp_model_sp.transformer.h[correction_layer_sp].register_forward_pre_hook(
        make_correction_hook(W_sp, b_sp)
    )
    sp_ppl = eval_ppl(comp_model_sp, teacher, corpus, BATCH_SIZE)
    sp_rel = (sp_ppl - baseline_ppl) / baseline_ppl * 100
    hook_sp.remove()
    print(f"  Single-point corrected: {sp_ppl:.2f} ({sp_rel:+.2f}%)")
    del comp_model_sp

    # Multi-point correction
    print(f"  Multi-point corrections (layer order):")
    comp_model_mp = swap_compressed_ffns(teacher, sub_pack, inplace=False)
    corrections, hooks = run_multipoint_correction(teacher, comp_model_mp, probe_ids, blocks)
    mp_ppl = eval_ppl(comp_model_mp, teacher, corpus, BATCH_SIZE)
    mp_rel = (mp_ppl - baseline_ppl) / baseline_ppl * 100
    for h in hooks:
        h.remove()
    print(f"  Multi-point corrected: {mp_ppl:.2f} ({mp_rel:+.2f}%)")
    del comp_model_mp

    sp_recovery = uncorr_rel - sp_rel
    mp_recovery = uncorr_rel - mp_rel
    mp_vs_sp = sp_rel - mp_rel  # positive = multi-point is better

    print(f"\n  Single-point recovery: {sp_recovery:+.2f}pp")
    print(f"  Multi-point recovery:  {mp_recovery:+.2f}pp")
    print(f"  Multi-point gain over single-point: {mp_vs_sp:+.2f}pp")

    results["configs"][config_name] = {
        "blocks": blocks,
        "uncorrected": {"ppl": float(uncorr_ppl), "rel_pct": float(uncorr_rel)},
        "single_point": {"ppl": float(sp_ppl), "rel_pct": float(sp_rel), "recovery_pp": float(sp_recovery)},
        "multi_point": {
            "ppl": float(mp_ppl), "rel_pct": float(mp_rel),
            "recovery_pp": float(mp_recovery),
            "gain_over_sp_pp": float(mp_vs_sp),
            "per_correction": corrections,
        },
    }

# ── Verdict ───────────────────────────────────────────────────────────────────
k4_mp_rel = results["configs"]["k4"]["multi_point"]["rel_pct"]
k5_mp_rel = results["configs"]["k5"]["multi_point"]["rel_pct"]
k4_sp_rel = results["configs"]["k4"]["single_point"]["rel_pct"]
k4_mp_gain = results["configs"]["k4"]["multi_point"]["gain_over_sp_pp"]

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")
print(f"  k=4 single-point: {k4_sp_rel:+.2f}%")
print(f"  k=4 multi-point:  {k4_mp_rel:+.2f}% (gain: {k4_mp_gain:+.2f}pp)")
print(f"  k=5 multi-point:  {k5_mp_rel:+.2f}%")

confirmed = k5_mp_rel <= 12.0 or k4_mp_gain >= 1.5
refuted = k4_mp_gain < 0.0 and (k5_mp_rel >= results["configs"]["k5"]["single_point"]["rel_pct"])

if confirmed:
    verdict = "confirmed"
    if k5_mp_rel <= 12.0:
        print(f"Verdict: CONFIRMED — k=5 multi-point {k5_mp_rel:+.2f}% ≤ 12%")
    else:
        print(f"Verdict: CONFIRMED — k=4 multi-point gains {k4_mp_gain:+.2f}pp ≥ 1.5pp over single-point")
elif refuted:
    verdict = "refuted"
    print(f"Verdict: REFUTED — multi-point does not improve over single-point")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — k=4 gain {k4_mp_gain:+.2f}pp, k=5 at {k5_mp_rel:+.2f}%")

results["verdict"] = verdict

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: PPL comparison bars
configs_labels = ["k4\nuncorr", "k4\nSP", "k4\nMP", "k5\nuncorr", "k5\nSP", "k5\nMP"]
ppls = [
    results["configs"]["k4"]["uncorrected"]["rel_pct"],
    results["configs"]["k4"]["single_point"]["rel_pct"],
    results["configs"]["k4"]["multi_point"]["rel_pct"],
    results["configs"]["k5"]["uncorrected"]["rel_pct"],
    results["configs"]["k5"]["single_point"]["rel_pct"],
    results["configs"]["k5"]["multi_point"]["rel_pct"],
]
colors = ["coral", "lightblue", "steelblue", "coral", "lightblue", "steelblue"]

x = np.arange(len(configs_labels))
ax1.bar(x, ppls, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, p in enumerate(ppls):
    ax1.annotate(f"{p:+.1f}%", (i, p), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=9)
ax1.axhline(12.0, color="red", ls="--", alpha=0.7, label="12% threshold")
ax1.set_xticks(x)
ax1.set_xticklabels(configs_labels, fontsize=9)
ax1.set_ylabel("Relative PPL (%)")
ax1.set_title("PPL: Uncorrected vs Single-point vs Multi-point")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# Right: per-correction MSE breakdown for k=4
k4_corrs = results["configs"]["k4"]["multi_point"]["per_correction"]
if k4_corrs:
    blocks_c = [c["block"] for c in k4_corrs]
    mse_before = [c["mse_before"] for c in k4_corrs]
    mse_after = [c["mse_after"] for c in k4_corrs]
    xc = np.arange(len(blocks_c))
    w = 0.35
    ax2.bar(xc - w/2, mse_before, w, label="MSE before", color="coral", alpha=0.8)
    ax2.bar(xc + w/2, mse_after, w, label="MSE after", color="steelblue", alpha=0.8)
    ax2.set_xticks(xc)
    ax2.set_xticklabels([f"Block {b}" for b in blocks_c])
    ax2.set_ylabel("MSE at correction point")
    ax2.set_title("k=4: Per-Correction MSE Breakdown")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Cycle 043: Multi-point Correction — Verdict: {verdict.upper()}", fontsize=13)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_043_multipoint_correction.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_043_multipoint_correction.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")
