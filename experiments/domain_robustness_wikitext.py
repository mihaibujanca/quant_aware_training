"""Cycle 044: Domain robustness — compression quality on WikiText-2 vs Shakespeare.

Tests whether k=4@6x channel selection + lstsq correction transfers across domains.
Conditions: A (Shakespeare→Shakespeare), B (WikiText→WikiText), C (Shakespeare→WikiText),
D (Shakespeare channels + WikiText correction refit → WikiText eval).
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
WIKITEXT_PATH = str(Path(__file__).parent.parent / "data" / "wikitext2_valid.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6  # 6×
SEEDS = [42, 137, 2024]
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


def compress_and_correct(teacher, atlas, corpus, probe_ids):
    """Compress k=4 blocks and fit correction. Returns compressed pack and correction."""
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
    pack = derive_ffn_compression(atlas, budget)
    comp_model = swap_compressed_ffns(teacher, pack, inplace=False)

    h_float = collect_hidden_states(teacher.model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    h_comp = collect_hidden_states(comp_model, probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    W, b = fit_correction(h_float, h_comp)

    del comp_model
    return pack, W, b


def eval_with_correction(teacher, pack, corpus, W_corr, b_corr):
    """Evaluate compressed model with correction."""
    model = swap_compressed_ffns(teacher, pack, inplace=False)
    # Uncorrected
    uncorr_ppl = eval_ppl(model, teacher, corpus, BATCH_SIZE)
    # Corrected
    hook = model.transformer.h[CORRECTION_LAYER].register_forward_pre_hook(
        make_correction_hook(W_corr, b_corr)
    )
    corr_ppl = eval_ppl(model, teacher, corpus, BATCH_SIZE)
    hook.remove()
    del model
    return uncorr_ppl, corr_ppl


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_results = {"conditions": {c: [] for c in ["A", "B", "C", "D"]}, "baselines": {}}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    # Build both corpora
    shk_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=SHAKESPEARE_PATH, sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=50,
        eval_val_sequences=PROBE_EVAL_SEQ, dataset_name="shakespeare",
        shuffle_seed=seed,
    )
    shk_corpus = build_probe_corpus(shk_spec)

    wiki_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=WIKITEXT_PATH, sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=50,
        eval_val_sequences=PROBE_EVAL_SEQ, dataset_name="wikitext2",
        shuffle_seed=seed,
    )
    wiki_corpus = build_probe_corpus(wiki_spec)

    # Baselines
    shk_baseline = eval_ppl(teacher.model, teacher, shk_corpus, BATCH_SIZE)
    wiki_baseline = eval_ppl(teacher.model, teacher, wiki_corpus, BATCH_SIZE)
    all_results["baselines"][str(seed)] = {
        "shakespeare": float(shk_baseline), "wikitext2": float(wiki_baseline),
    }
    print(f"  Shakespeare baseline: {shk_baseline:.2f}")
    print(f"  WikiText-2 baseline:  {wiki_baseline:.2f}")

    # Probe geometry on both corpora
    shk_atlas = probe_ffn_geometry(
        teacher, shk_corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )
    wiki_atlas = probe_ffn_geometry(
        teacher, wiki_corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )

    # ── Condition A: Shakespeare → Shakespeare ────────────────────────────
    shk_probe_ids = shk_corpus.probe_train[:, :-1]
    shk_pack, shk_W, shk_b = compress_and_correct(teacher, shk_atlas, shk_corpus, shk_probe_ids)
    a_uncorr, a_corr = eval_with_correction(teacher, shk_pack, shk_corpus, shk_W, shk_b)
    a_rel_uncorr = (a_uncorr - shk_baseline) / shk_baseline * 100
    a_rel_corr = (a_corr - shk_baseline) / shk_baseline * 100
    print(f"  A (shk→shk): uncorr={a_uncorr:.2f} ({a_rel_uncorr:+.2f}%), corr={a_corr:.2f} ({a_rel_corr:+.2f}%)")
    all_results["conditions"]["A"].append({
        "seed": seed, "uncorr_ppl": float(a_uncorr), "corr_ppl": float(a_corr),
        "uncorr_rel": float(a_rel_uncorr), "corr_rel": float(a_rel_corr),
    })

    # ── Condition B: WikiText → WikiText ──────────────────────────────────
    wiki_probe_ids = wiki_corpus.probe_train[:, :-1]
    wiki_pack, wiki_W, wiki_b = compress_and_correct(teacher, wiki_atlas, wiki_corpus, wiki_probe_ids)
    b_uncorr, b_corr = eval_with_correction(teacher, wiki_pack, wiki_corpus, wiki_W, wiki_b)
    b_rel_uncorr = (b_uncorr - wiki_baseline) / wiki_baseline * 100
    b_rel_corr = (b_corr - wiki_baseline) / wiki_baseline * 100
    print(f"  B (wiki→wiki): uncorr={b_uncorr:.2f} ({b_rel_uncorr:+.2f}%), corr={b_corr:.2f} ({b_rel_corr:+.2f}%)")
    all_results["conditions"]["B"].append({
        "seed": seed, "uncorr_ppl": float(b_uncorr), "corr_ppl": float(b_corr),
        "uncorr_rel": float(b_rel_uncorr), "corr_rel": float(b_rel_corr),
    })

    # ── Condition C: Shakespeare probe → WikiText eval (zero-shot) ────────
    c_uncorr, c_corr = eval_with_correction(teacher, shk_pack, wiki_corpus, shk_W, shk_b)
    c_rel_uncorr = (c_uncorr - wiki_baseline) / wiki_baseline * 100
    c_rel_corr = (c_corr - wiki_baseline) / wiki_baseline * 100
    print(f"  C (shk→wiki): uncorr={c_uncorr:.2f} ({c_rel_uncorr:+.2f}%), corr={c_corr:.2f} ({c_rel_corr:+.2f}%)")
    all_results["conditions"]["C"].append({
        "seed": seed, "uncorr_ppl": float(c_uncorr), "corr_ppl": float(c_corr),
        "uncorr_rel": float(c_rel_uncorr), "corr_rel": float(c_rel_corr),
    })

    # ── Condition D: Shakespeare channels + WikiText correction refit ─────
    d_comp_model = swap_compressed_ffns(teacher, shk_pack, inplace=False)
    h_f_d = collect_hidden_states(teacher.model, wiki_probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    h_c_d = collect_hidden_states(d_comp_model, wiki_probe_ids, CORRECTION_LAYER, BATCH_SIZE)
    W_d, b_d = fit_correction(h_f_d, h_c_d)
    del d_comp_model

    d_uncorr, d_corr = eval_with_correction(teacher, shk_pack, wiki_corpus, W_d, b_d)
    d_rel_uncorr = (d_uncorr - wiki_baseline) / wiki_baseline * 100
    d_rel_corr = (d_corr - wiki_baseline) / wiki_baseline * 100
    print(f"  D (shk channels+wiki corr→wiki): uncorr={d_uncorr:.2f} ({d_rel_uncorr:+.2f}%), corr={d_corr:.2f} ({d_rel_corr:+.2f}%)")
    all_results["conditions"]["D"].append({
        "seed": seed, "uncorr_ppl": float(d_uncorr), "corr_ppl": float(d_corr),
        "uncorr_rel": float(d_rel_uncorr), "corr_rel": float(d_rel_corr),
    })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE (corrected PPL %)")
print(f"{'='*60}")

summaries = {}
for cond in ["A", "B", "C", "D"]:
    rels = [r["corr_rel"] for r in all_results["conditions"][cond]]
    m, s = float(np.mean(rels)), float(np.std(rels, ddof=1))
    summaries[cond] = {"mean": m, "std": s}
    label = {"A": "shk→shk", "B": "wiki→wiki", "C": "shk→wiki(zero-shot)", "D": "shk+wiki_corr→wiki"}[cond]
    print(f"  {cond} ({label}): {m:+.2f}% ±{s:.2f}")

# Verdict: compare C (zero-shot) vs B (corpus-specific)
gap = abs(summaries["C"]["mean"] - summaries["B"]["mean"])
print(f"\n  Zero-shot gap (C-B): {gap:.2f}pp")

b_mean = summaries["B"]["mean"]
c_mean = summaries["C"]["mean"]

if gap <= 3.0:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — zero-shot transfer within {gap:.1f}pp of corpus-specific (≤3pp)")
elif c_mean > 20.0 and b_mean < 10.0:
    verdict = "refuted"
    print(f"Verdict: REFUTED — zero-shot >20% while corpus-specific <10%")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — gap {gap:.1f}pp > 3pp")

all_results["summaries"] = summaries
all_results["zero_shot_gap_pp"] = float(gap)
all_results["verdict"] = verdict

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

labels = ["A: shk→shk", "B: wiki→wiki", "C: shk→wiki\n(zero-shot)", "D: shk+wiki_corr\n→wiki"]
means = [summaries[c]["mean"] for c in ["A", "B", "C", "D"]]
stds = [summaries[c]["std"] for c in ["A", "B", "C", "D"]]
colors = ["steelblue", "coral", "gold", "mediumseagreen"]

x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:+.1f}%\n±{s:.1f}", (i, m), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)

ax.axhline(0, color="black", ls="-", alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean Corrected Relative PPL (%)")
ax.set_title(f"Cycle 044: Domain Robustness — k=4@6× + Correction\n"
             f"Zero-shot gap (C-B): {gap:.1f}pp  |  Verdict: {verdict.upper()}")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_044_domain_robustness.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_044_domain_robustness.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved: {out_json}")
