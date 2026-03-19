"""Cycle 054: Training-distribution probing with OpenWebText — GPT-2 FFN.

Tests whether using GPT-2's training distribution (OpenWebText) as BOTH probe
and eval corpus yields ≤+15% corrected PPL degradation at k=4@6×.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
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
OWT_SAMPLE_PATH = str(Path(__file__).parent.parent / "data" / "openwebtext_sample.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
CORRECTION_LAYER = 8
NUM_OWT_DOCS = 500


# ── Download OpenWebText sample ───────────────────────────────────────────────
def download_owt_sample(path: str, n_docs: int = NUM_OWT_DOCS) -> None:
    """Stream first n_docs documents from OpenWebText and save to file."""
    print(f"Downloading {n_docs} OpenWebText documents...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
    docs = []
    for i, example in enumerate(ds):
        if i >= n_docs:
            break
        docs.append(example["text"].strip())
    text = "\n\n".join(docs)
    Path(path).write_text(text, encoding="utf-8")
    print(f"Saved {n_docs} docs ({len(text):,} chars) to {path}")


# ── Helpers (same as cycle 045) ───────────────────────────────────────────────
def eval_ppl(model, teacher, corpus, batch_size):
    return float(_evaluate_model(teacher, model, corpus, batch_size=batch_size)["perplexity"])


def collect_hidden(model, ids, layer, bs=4):
    out = []
    def hook(mod, args):
        out.append(args[0].detach().clone())
    h = model.transformer.h[layer].register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            model(ids[i:i+bs])
    h.remove()
    return torch.cat(out, dim=0)


def fit_correction(h_f, h_c):
    hf = h_f.reshape(-1, h_f.shape[-1]).float()
    hc = h_c.reshape(-1, h_c.shape[-1]).float()
    H = torch.cat([hc, torch.ones(hc.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(H, hf).solution
    return sol[:hf.shape[1]].T, sol[hf.shape[1]]


def run_recipe(teacher, text_path, corpus_name, seed):
    """Run the full recipe on OpenWebText with one seed."""
    torch.manual_seed(seed)
    spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=text_path,
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=50,
        eval_val_sequences=PROBE_EVAL_SEQ,
        dataset_name=corpus_name,
        shuffle_seed=seed,
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

    # Affine correction at layer 8
    probe_ids = corpus.probe_train[:, :-1]
    h_f = collect_hidden(teacher.model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    h_c = collect_hidden(model_u, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    W, b = fit_correction(h_f, h_c)

    def hook_fn(mod, args):
        return (args[0] @ W.T + b,) + args[1:]

    handle = model_u.transformer.h[CORRECTION_LAYER].register_forward_pre_hook(hook_fn)
    ppl_c = eval_ppl(model_u, teacher, corpus, BATCH_SIZE)
    handle.remove()
    del model_u

    rel_u = (ppl_u - baseline) / baseline * 100
    rel_c = (ppl_c - baseline) / baseline * 100

    return {
        "baseline": float(baseline),
        "uncorrected": float(ppl_u),
        "corrected": float(ppl_c),
        "rel_uncorrected": float(rel_u),
        "rel_corrected": float(rel_c),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
# Download sample if not already present
if not Path(OWT_SAMPLE_PATH).exists():
    download_owt_sample(OWT_SAMPLE_PATH)
else:
    print(f"Using cached OpenWebText sample: {OWT_SAMPLE_PATH}")

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

results = {"openwebtext": []}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"SEED {seed}")
    print(f"{'='*50}")
    r = run_recipe(teacher, OWT_SAMPLE_PATH, "openwebtext", seed)
    print(f"  baseline={r['baseline']:.2f}, uncorr={r['uncorrected']:.2f} ({r['rel_uncorrected']:+.2f}%), "
          f"corr={r['corrected']:.2f} ({r['rel_corrected']:+.2f}%)")
    results["openwebtext"].append({"seed": seed, **r})

# ── Aggregate ─────────────────────────────────────────────────────────────────
import numpy as np

print(f"\n{'='*50}")
print("AGGREGATE")
print(f"{'='*50}")

rels = [r["rel_corrected"] for r in results["openwebtext"]]
mean_rel = float(np.mean(rels))
std_rel = float(np.std(rels, ddof=1))
print(f"  OpenWebText: corrected {mean_rel:+.2f}% ±{std_rel:.2f}")

# Reference values from cycle 045
shk_ref = 8.5
wiki_ref = 44.0
print(f"\n  Reference — Shakespeare-probed: +{shk_ref:.1f}%")
print(f"  Reference — WikiText-2-probed:  +{wiki_ref:.1f}%")
print(f"  This run  — OpenWebText-probed: {mean_rel:+.2f}%")

# ── Verdict ───────────────────────────────────────────────────────────────────
if mean_rel <= 15.0:
    verdict = "confirmed"
elif mean_rel >= 30.0:
    verdict = "refuted"
else:
    verdict = "inconclusive"

print(f"\n  Verdict: {verdict.upper()}")
if verdict == "confirmed":
    print("  Training-distribution probing resolves corpus sensitivity.")
elif verdict == "refuted":
    print("  Even training-distribution data cannot save the method at k=4@6×.")
else:
    print("  Better than WikiText-2 but still too high for practical use.")

results["summary"] = {
    "openwebtext": {"mean": mean_rel, "std": std_rel},
    "references": {"shakespeare": shk_ref, "wikitext2": wiki_ref},
    "verdict": verdict,
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["Shakespeare\n(narrow OOD)", "WikiText-2\n(broad OOD)", "OpenWebText\n(training dist)"]
means = [shk_ref, wiki_ref, mean_rel]
stds = [1.5, 5.0, std_rel]  # Shakespeare/WT2 stds from cycle 045 approximate
colors = ["steelblue", "coral", "forestgreen"]
ax.bar([0, 1, 2], means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
# Error bar only for OpenWebText (which has 3-seed std)
ax.errorbar([2], [mean_rel], yerr=[std_rel], fmt="none", capsize=5,
            capthick=1.5, color="black", linewidth=1.5)
# Annotate bars
for i, m in enumerate(means):
    suf = f"\n(ref)" if i < 2 else f"\n±{std_rel:.1f}"
    ax.annotate(f"{m:+.1f}%{suf}", (i, m), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10)
ax.axhline(15, color="green", ls="--", alpha=0.6, label="+15% confirm threshold")
ax.axhline(30, color="red", ls="--", alpha=0.6, label="+30% refute threshold")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Corrected Relative PPL Degradation (%)")
ax.set_title(f"Cycle 054: OpenWebText Training-Distribution Probe\n"
             f"k=4@6× + correction | Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out_png = RESULTS_DIR / "cycle_054_openwebtext_probe.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved plot: {out_png}")

out_json = RESULTS_DIR / "cycle_054_openwebtext_probe.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {out_json}")
