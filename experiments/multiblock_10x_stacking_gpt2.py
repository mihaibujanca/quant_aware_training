"""Cycle 042: Multi-block stacking at 10x per-block — GPT-2 FFN.

Tests whether 10x single-block headroom (cycle 040) composes at k=4,
with and without residual correction (cycle 039 method).
3 seeds, 200 eval sequences.
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

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 0.1  # 10× compression
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
CORRECTION_LAYER = max(TARGET_BLOCKS) + 1  # block 8


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
    H = torch.cat([h_c, torch.ones(h_c.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(H, h_f).solution
    W = sol[:h_f.shape[1]].T
    b = sol[h_f.shape[1]]
    return W, b


def make_correction_hook(W, b):
    def hook_fn(module, args):
        h = args[0]
        return (h @ W.T + b,) + args[1:]
    return hook_fn


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_results = {"accumulation": [], "corrected": [], "baselines": {}}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    corpus_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=SHAKESPEARE_PATH, sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=PROBE_EVAL_SEQ,
        eval_val_sequences=PROBE_EVAL_SEQ, dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(corpus_spec)

    baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
    all_results["baselines"][str(seed)] = float(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print("Probing geometry...")
    atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )

    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
    full_pack = derive_ffn_compression(atlas, budget)
    compressed_by_id = {l.layer_id: l for l in full_pack.layers}

    # ── Accumulation ──────────────────────────────────────────────────────
    print(f"\n{'k':>3} {'blocks':>15} {'PPL':>8} {'rel%':>8}")
    print("-" * 40)

    for k in range(1, len(TARGET_BLOCKS) + 1):
        active = TARGET_BLOCKS[:k]
        sub_layers = tuple(compressed_by_id[b] for b in active)
        sub_pack = CompressedFfnPack(
            method=full_pack.method, model_family=full_pack.model_family,
            dataset_name=full_pack.dataset_name, layers=sub_layers, metadata=full_pack.metadata,
        )
        student = swap_compressed_ffns(teacher, sub_pack, inplace=False)
        ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"  {k:>2}  {str(active):>13}  {ppl:8.2f}  {rel:+7.2f}%")

        all_results["accumulation"].append({
            "seed": seed, "k": k, "blocks": active,
            "ppl": float(ppl), "rel_pct": float(rel),
        })

        if k == len(TARGET_BLOCKS):
            # Keep model for correction
            k4_model = student
        else:
            del student

    # ── Correction at k=4 ────────────────────────────────────────────────
    print(f"\nFitting correction at block {CORRECTION_LAYER}...")
    probe_ids = corpus.probe_train[:, :-1]
    h_float = collect_hidden_states(teacher.model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    h_comp = collect_hidden_states(k4_model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    W_corr, b_corr = fit_correction(h_float, h_comp)

    mse_before = F.mse_loss(h_comp.reshape(-1, 768).float(), h_float.reshape(-1, 768).float()).item()
    h_fixed = h_comp.reshape(-1, 768).float() @ W_corr.T + b_corr
    mse_after = F.mse_loss(h_fixed, h_float.reshape(-1, 768).float()).item()
    print(f"  MSE: {mse_before:.4f} → {mse_after:.4f} ({(1-mse_after/mse_before)*100:.1f}% reduction)")

    hook = k4_model.transformer.h[CORRECTION_LAYER].register_forward_pre_hook(
        make_correction_hook(W_corr, b_corr)
    )
    k4_corr_ppl = eval_ppl(k4_model, teacher, corpus, BATCH_SIZE)
    k4_corr_rel = (k4_corr_ppl - baseline_ppl) / baseline_ppl * 100
    hook.remove()

    k4_uncorr = next(r for r in all_results["accumulation"]
                     if r["seed"] == seed and r["k"] == 4)
    recovery = k4_uncorr["rel_pct"] - k4_corr_rel

    print(f"  k=4@10x corrected: {k4_corr_ppl:.2f} ({k4_corr_rel:+.2f}%), recovery={recovery:+.2f}pp")

    all_results["corrected"].append({
        "seed": seed, "ppl": float(k4_corr_ppl), "rel_pct": float(k4_corr_rel),
        "recovery_pp": float(recovery), "mse_before": mse_before, "mse_after": mse_after,
    })

    del k4_model

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE")
print(f"{'='*60}")

# Per-k uncorrected
for k in range(1, 5):
    entries = [r for r in all_results["accumulation"] if r["k"] == k]
    rels = [r["rel_pct"] for r in entries]
    m, s = float(np.mean(rels)), float(np.std(rels, ddof=1))
    print(f"  k={k} uncorrected: {m:+.2f}% ±{s:.2f}")

# k=4 corrected
corr_rels = [r["rel_pct"] for r in all_results["corrected"]]
corr_mean = float(np.mean(corr_rels))
corr_std = float(np.std(corr_rels, ddof=1))
recoveries = [r["recovery_pp"] for r in all_results["corrected"]]
print(f"  k=4 corrected: {corr_mean:+.2f}% ±{corr_std:.2f}, recovery={np.mean(recoveries):+.2f}pp")

# ── Verdict ───────────────────────────────────────────────────────────────────
all_under_15 = all(r["rel_pct"] < 15.0 for r in all_results["corrected"])

if corr_mean <= 12.0 and all_under_15:
    verdict = "confirmed"
    print(f"\nVerdict: CONFIRMED — k=4@10x corrected mean={corr_mean:+.2f}% (≤12%)")
elif corr_mean > 20.0:
    verdict = "refuted"
    print(f"\nVerdict: REFUTED — k=4@10x corrected mean={corr_mean:+.2f}% (>20%)")
else:
    verdict = "inconclusive"
    print(f"\nVerdict: INCONCLUSIVE — k=4@10x corrected mean={corr_mean:+.2f}% (12-20%)")

all_results["summary"] = {
    "k4_corrected_mean": corr_mean, "k4_corrected_std": corr_std,
    "mean_recovery_pp": float(np.mean(recoveries)),
}
all_results["verdict"] = verdict

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# 10× uncorrected accumulation
ks = list(range(1, 5))
means_10x = []
stds_10x = []
for k in ks:
    entries = [r for r in all_results["accumulation"] if r["k"] == k]
    rels = [r["rel_pct"] for r in entries]
    means_10x.append(float(np.mean(rels)))
    stds_10x.append(float(np.std(rels, ddof=1)))

ax.errorbar(ks, means_10x, yerr=stds_10x, marker="o", color="darkred", lw=2, capsize=4,
            label="10× uncorrected")
ax.errorbar([4], [corr_mean], yerr=[corr_std], marker="s", color="green", ms=10, capsize=4,
            label="10× corrected")

# 6× reference from cycle 036 (single seed)
ref_6x = [0.56, 2.60, 6.40, 9.59]  # from cycle 036
ax.plot(ks, ref_6x, marker="^", color="steelblue", lw=2, ls="--", label="6× uncorrected (c036)")
ax.plot([4], [8.35], marker="D", color="lightblue", ms=10, label="6× corrected (c039)")

ax.axhline(12.0, color="orange", ls="--", alpha=0.7, label="12% threshold")
ax.axhline(20.0, color="red", ls="--", alpha=0.7, label="20% threshold")

for i, (m, s) in enumerate(zip(means_10x, stds_10x)):
    ax.annotate(f"{m:+.1f}%", (ks[i], m), textcoords="offset points", xytext=(8, -8), fontsize=9)
ax.annotate(f"{corr_mean:+.1f}%", (4, corr_mean), textcoords="offset points",
            xytext=(8, 8), fontsize=9, color="green")

ax.set_xlabel("Number of blocks compressed (k)")
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 042: 10× Multi-Block Stacking (3 seeds)\n"
             f"k=4@10x corrected: {corr_mean:+.1f}%  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xticks(ks)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_042_multiblock_10x.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_042_multiblock_10x.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved: {out_json}")
