"""Cycle 045: Compression recipe validation — end-to-end replication.

Runs the documented recipe on Shakespeare (replication) and WikiText-2 (reality check).
"""
from __future__ import annotations

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

# ── Config (from recipe) ─────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
WIKITEXT_PATH = str(Path(__file__).parent.parent / "data" / "wikitext2_valid.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
CORRECTION_LAYER = 8


def eval_ppl(model, teacher, corpus, batch_size):
    return float(_evaluate_model(teacher, model, corpus, batch_size=batch_size)["perplexity"])


def collect_hidden(model, ids, layer, bs=4):
    out = []
    def hook(mod, args): out.append(args[0].detach().clone())
    h = model.transformer.h[layer].register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(0, len(ids), bs): model(ids[i:i+bs])
    h.remove()
    return torch.cat(out, dim=0)


def fit_correction(h_f, h_c):
    hf, hc = h_f.reshape(-1, h_f.shape[-1]).float(), h_c.reshape(-1, h_c.shape[-1]).float()
    H = torch.cat([hc, torch.ones(hc.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(H, hf).solution
    return sol[:hf.shape[1]].T, sol[hf.shape[1]]


def run_recipe(teacher, text_path, corpus_name, seed):
    """Run the full recipe on one corpus with one seed."""
    torch.manual_seed(seed)
    spec = ProbeCorpusSpec(
        teacher=teacher, text_path=text_path, sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=50,
        eval_val_sequences=PROBE_EVAL_SEQ, dataset_name=corpus_name, shuffle_seed=seed,
    )
    corpus = build_probe_corpus(spec)
    baseline = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)

    atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
    pack = derive_ffn_compression(atlas, budget)

    # Uncorrected
    model_u = swap_compressed_ffns(teacher, pack, inplace=False)
    ppl_u = eval_ppl(model_u, teacher, corpus, BATCH_SIZE)

    # Correction
    probe_ids = corpus.probe_train[:, :-1]
    h_f = collect_hidden(teacher.model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    h_c = collect_hidden(model_u, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    W, b = fit_correction(h_f, h_c)

    def hook_fn(mod, args): return (args[0] @ W.T + b,) + args[1:]
    handle = model_u.transformer.h[CORRECTION_LAYER].register_forward_pre_hook(hook_fn)
    ppl_c = eval_ppl(model_u, teacher, corpus, BATCH_SIZE)
    handle.remove()
    del model_u

    rel_u = (ppl_u - baseline) / baseline * 100
    rel_c = (ppl_c - baseline) / baseline * 100

    return {
        "baseline": float(baseline), "uncorrected": float(ppl_u), "corrected": float(ppl_c),
        "rel_uncorrected": float(rel_u), "rel_corrected": float(rel_c),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

results = {"shakespeare": [], "wikitext2": []}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"SEED {seed}")
    print(f"{'='*50}")

    for name, path in [("shakespeare", SHAKESPEARE_PATH), ("wikitext2", WIKITEXT_PATH)]:
        print(f"\n  {name}:")
        r = run_recipe(teacher, path, name, seed)
        print(f"    baseline={r['baseline']:.2f}, uncorr={r['uncorrected']:.2f} ({r['rel_uncorrected']:+.2f}%), "
              f"corr={r['corrected']:.2f} ({r['rel_corrected']:+.2f}%)")
        results[name].append({"seed": seed, **r})

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("AGGREGATE")
print(f"{'='*50}")

summaries = {}
for name in ["shakespeare", "wikitext2"]:
    rels = [r["rel_corrected"] for r in results[name]]
    m, s = float(np.mean(rels)), float(np.std(rels, ddof=1))
    summaries[name] = {"mean": m, "std": s}
    print(f"  {name}: corrected {m:+.2f}% ±{s:.2f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
shk_mean = summaries["shakespeare"]["mean"]
wiki_mean = summaries["wikitext2"]["mean"]
target_shk = 8.35  # cycle 039 reference

shk_match = abs(shk_mean - target_shk) <= 1.0
shk_close = abs(shk_mean - target_shk) <= 3.0
corpus_specific = wiki_mean > 20.0  # clearly corpus-specific

if shk_match and not corpus_specific:
    verdict = "confirmed"
elif not shk_close:
    verdict = "refuted"
else:
    verdict = "inconclusive"

print(f"\n  Shakespeare match (target {target_shk:+.2f}%): {shk_mean:+.2f}% ({'MATCH' if shk_match else 'CLOSE' if shk_close else 'MISS'})")
print(f"  WikiText-2 corpus-specific: {wiki_mean:+.2f}% ({'corpus-specific' if corpus_specific else 'OK'})")
print(f"  Verdict: {verdict.upper()}")

if verdict == "inconclusive":
    print(f"  Reason: recipe works but requires corpus-specific tuning (cycle 044 finding)")

results["summaries"] = summaries
results["verdict"] = verdict

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Shakespeare\n(narrow)", "WikiText-2\n(broad)"]
means = [summaries["shakespeare"]["mean"], summaries["wikitext2"]["mean"]]
stds = [summaries["shakespeare"]["std"], summaries["wikitext2"]["std"]]
colors = ["steelblue", "coral"]
ax.bar([0, 1], means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:+.1f}%\n±{s:.1f}", (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)
ax.axhline(10, color="green", ls="--", alpha=0.5, label="10% practical threshold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Corrected Relative PPL (%)")
ax.set_title(f"Cycle 045: Recipe Validation — k=4@6× + Correction\nVerdict: {verdict.upper()}")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out_png = RESULTS_DIR / "cycle_045_validation.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_045_validation.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")
