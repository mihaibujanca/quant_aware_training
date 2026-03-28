"""Cycle 070: Corpus-diverse probing — multi-domain channel selection stability.

For blocks [7, 5, 4, 1] at 6x compression, runs topk channel selection independently
on 4 probe corpora: Shakespeare, WikiText-2, OpenWebText, C4 (fallback to 3 if C4 fails).
Measures pairwise Jaccard similarity and 4×4 cross-corpus PPL matrix per block.
"""
from __future__ import annotations

import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1.0 / 6  # 6× → k≈512 of 3072
SEEDS = [42, 123, 7]
SEQ_LEN = 128
PROBE_TRAIN_SEQ = 16
PROBE_HOLDOUT_SEQ = 8
EVAL_VAL_SEQ = 200
BATCH_SIZE = 4
JACOBIAN_SAMPLES = 8

JSON_PATH = RESULTS_DIR / "cycle_070_corpus_diverse_channels.json"
PLOT_PATH = RESULTS_DIR / "cycle_070_corpus_diverse_channels.png"

# ── Corpus setup ────────────────────────────────────────────────────────────────
CORPUS_PATHS = {
    "shakespeare": str(DATA_DIR / "shakespeare.txt"),
    "wikitext2": str(DATA_DIR / "wikitext2_valid.txt"),
    "openwebtext": str(DATA_DIR / "openwebtext_sample.txt"),
}

# Try to download C4 if not already present
C4_PATH = DATA_DIR / "c4_sample.txt"
if not C4_PATH.exists():
    print("Attempting to download C4 sample...")
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
        docs = []
        for i, ex in enumerate(ds):
            docs.append(ex["text"])
            if i >= 199:
                break
        C4_PATH.write_text("\n\n".join(docs), encoding="utf-8")
        CORPUS_PATHS["c4"] = str(C4_PATH)
        print(f"  Saved {len(docs)} C4 docs to {C4_PATH}")
    except Exception as e:
        print(f"  C4 download failed ({e}), using 3 corpora only.")
else:
    CORPUS_PATHS["c4"] = str(C4_PATH)
    print(f"C4 sample found at {C4_PATH}")

CORPUS_NAMES = list(CORPUS_PATHS.keys())
print(f"Using corpora: {CORPUS_NAMES}")

# ── Load model ──────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
print(f"  device={teacher.device}, activation={teacher.activation_kind}")


def eval_ppl(model, corpus) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
    return float(metrics["perplexity"])


def jaccard(s1: set, s2: set) -> float:
    u = s1 | s2
    return len(s1 & s2) / len(u) if u else 1.0


# ── Build eval corpora (one per corpus name, fixed seed=42) ────────────────────
print("\nBuilding eval corpora...")
eval_corpora: dict[str, object] = {}
for cname, cpath in CORPUS_PATHS.items():
    spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=cpath,
        sequence_length=SEQ_LEN,
        probe_train_sequences=1,
        probe_holdout_sequences=1,
        eval_val_sequences=EVAL_VAL_SEQ,
        dataset_name=cname,
        shuffle_seed=42,
    )
    eval_corpora[cname] = build_probe_corpus(spec)
    print(f"  {cname}: eval_val shape={eval_corpora[cname].eval_val.shape}")

# ── Baselines (uncompressed) ────────────────────────────────────────────────────
print("\nComputing uncompressed baselines...")
baselines: dict[str, float] = {}
for cname, ec in eval_corpora.items():
    baselines[cname] = eval_ppl(teacher.model, ec)
    print(f"  baseline PPL on {cname}: {baselines[cname]:.4f}")

# ── Main loop ──────────────────────────────────────────────────────────────────
# channels_store[block_id][corpus_name][seed] = sorted list of channel indices
channels_store: dict[int, dict[str, dict[int, list]]] = {b: {c: {} for c in CORPUS_NAMES} for b in TARGET_BLOCKS}

# ppl_store[block_id][probe_corpus][seed][eval_corpus] = ppl
ppl_store: dict[int, dict[str, dict[int, dict[str, float]]]] = {
    b: {c: {s: {} for s in SEEDS} for c in CORPUS_NAMES} for b in TARGET_BLOCKS
}

budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")

for block_id in TARGET_BLOCKS:
    print(f"\n{'='*60}")
    print(f"Block {block_id}")
    for probe_corpus_name in CORPUS_NAMES:
        probe_path = CORPUS_PATHS[probe_corpus_name]
        print(f"\n  Probe corpus: {probe_corpus_name}")
        for seed in SEEDS:
            torch.manual_seed(seed)
            # Build probe corpus for channel selection
            probe_spec = ProbeCorpusSpec(
                teacher=teacher,
                text_path=probe_path,
                sequence_length=SEQ_LEN,
                probe_train_sequences=PROBE_TRAIN_SEQ,
                probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
                eval_val_sequences=EVAL_VAL_SEQ,
                dataset_name=probe_corpus_name,
                shuffle_seed=seed,
            )
            probe_corpus = build_probe_corpus(probe_spec)

            # Build atlas for this block only
            atlas = probe_ffn_geometry(
                teacher,
                probe_corpus,
                layer_ids=(block_id,),
                max_tokens_per_layer=2048,
                batch_size=BATCH_SIZE,
                jacobian_samples=JACOBIAN_SAMPLES,
            )

            # Derive compression
            pack = derive_ffn_compression(atlas, budget)
            layer_pack = pack.layers[0]
            selected = layer_pack.selected_indices.sort().values.tolist()
            channels_store[block_id][probe_corpus_name][seed] = selected
            k = len(selected)
            print(f"    seed={seed}: k={k} channels")

            # Swap and evaluate on all eval corpora
            student = swap_compressed_ffns(teacher, pack, inplace=False)
            for eval_corpus_name, ec in eval_corpora.items():
                ppl = eval_ppl(student, ec)
                ppl_store[block_id][probe_corpus_name][seed][eval_corpus_name] = ppl
                rel = (ppl - baselines[eval_corpus_name]) / baselines[eval_corpus_name] * 100
                print(f"      eval={eval_corpus_name}: PPL={ppl:.3f} ({rel:+.1f}%)")

# ── Compute Jaccard matrices ─────────────────────────────────────────────────────
# Per block: cross-corpus Jaccard (mean over seeds), within-corpus cross-seed Jaccard
print("\n== Jaccard Analysis ==")
jaccard_results: dict[int, dict] = {}

for block_id in TARGET_BLOCKS:
    block_j: dict = {"cross_corpus": {}, "within_corpus": {}}

    # Cross-corpus Jaccard (mean across seeds, for each pair)
    corpus_pairs = list(combinations(CORPUS_NAMES, 2))
    for c1, c2 in corpus_pairs:
        pair_key = f"{c1}_vs_{c2}"
        seed_jaccards = []
        for seed in SEEDS:
            s1 = set(channels_store[block_id][c1][seed])
            s2 = set(channels_store[block_id][c2][seed])
            seed_jaccards.append(jaccard(s1, s2))
        j_mean = sum(seed_jaccards) / len(seed_jaccards)
        j_std = (sum((x - j_mean) ** 2 for x in seed_jaccards) / len(seed_jaccards)) ** 0.5
        block_j["cross_corpus"][pair_key] = {"mean": j_mean, "std": j_std, "per_seed": seed_jaccards}
        print(f"  Block {block_id} {pair_key}: J={j_mean:.4f} ± {j_std:.4f}")

    # Within-corpus cross-seed Jaccard (reference for stability)
    seed_pairs = list(combinations(SEEDS, 2))
    for cname in CORPUS_NAMES:
        sp_jaccards = []
        for s1, s2 in seed_pairs:
            c1s = set(channels_store[block_id][cname][s1])
            c2s = set(channels_store[block_id][cname][s2])
            sp_jaccards.append(jaccard(c1s, c2s))
        sp_mean = sum(sp_jaccards) / len(sp_jaccards)
        sp_std = (sum((x - sp_mean) ** 2 for x in sp_jaccards) / len(sp_jaccards)) ** 0.5
        block_j["within_corpus"][cname] = {"mean": sp_mean, "std": sp_std, "per_pair": sp_jaccards}
        print(f"  Block {block_id} within-{cname}: J={sp_mean:.4f} ± {sp_std:.4f}")

    jaccard_results[block_id] = block_j

# ── Compute PPL matrices ─────────────────────────────────────────────────────────
print("\n== PPL Cross-Eval Matrices ==")
ppl_matrix_results: dict[int, dict] = {}

for block_id in TARGET_BLOCKS:
    block_ppl: dict = {}
    for probe_cname in CORPUS_NAMES:
        block_ppl[probe_cname] = {}
        for eval_cname in CORPUS_NAMES:
            ppls = [ppl_store[block_id][probe_cname][seed][eval_cname] for seed in SEEDS]
            mean_ppl = sum(ppls) / len(ppls)
            std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5
            rel = (mean_ppl - baselines[eval_cname]) / baselines[eval_cname] * 100
            block_ppl[probe_cname][eval_cname] = {
                "mean": mean_ppl, "std": std_ppl, "rel_pct": rel,
                "per_seed": ppls,
            }
            print(f"  Block {block_id} probe={probe_cname} eval={eval_cname}: "
                  f"PPL={mean_ppl:.3f}±{std_ppl:.3f} ({rel:+.1f}%)")
    ppl_matrix_results[block_id] = block_ppl

# ── Adjudication ──────────────────────────────────────────────────────────────────
# Confirmed: cross-corpus Jaccard ≥ 0.7 for ≥ 3/4 blocks AND PPL gap ≤ 10% for ≥ 2/4 blocks
# Refuted: Jaccard < 0.4 for majority of blocks

jaccard_threshold = 0.7
ppl_gap_threshold = 10.0  # %

verdict_blocks: dict[int, dict] = {}
for block_id in TARGET_BLOCKS:
    # Mean cross-corpus Jaccard for this block
    cc_means = [v["mean"] for v in jaccard_results[block_id]["cross_corpus"].values()]
    avg_j = sum(cc_means) / len(cc_means)
    passes_jaccard = avg_j >= jaccard_threshold

    # PPL gap: max(probe row diagonal vs off-diagonal) per eval corpus
    # For each eval corpus: range of PPL across different probe corpora
    ppl_gaps = []
    for eval_cname in CORPUS_NAMES:
        vals = [ppl_matrix_results[block_id][p][eval_cname]["mean"] for p in CORPUS_NAMES]
        ppl_gaps.append(max(vals) - min(vals))
    max_ppl_gap_pct = max(
        (ppl_matrix_results[block_id][p][e]["rel_pct"] for p in CORPUS_NAMES for e in CORPUS_NAMES),
        default=0.0,
    )
    # Relative gap: max PPL - min PPL over baseline per eval corpus
    rel_ppl_gaps = []
    for eval_cname in CORPUS_NAMES:
        ppls_for_eval = [ppl_matrix_results[block_id][p][eval_cname]["mean"] for p in CORPUS_NAMES]
        gap_rel = (max(ppls_for_eval) - min(ppls_for_eval)) / baselines[eval_cname] * 100
        rel_ppl_gaps.append(gap_rel)
    max_rel_ppl_gap = max(rel_ppl_gaps)
    passes_ppl = max_rel_ppl_gap <= ppl_gap_threshold

    verdict_blocks[block_id] = {
        "avg_cross_corpus_jaccard": avg_j,
        "passes_jaccard_threshold": passes_jaccard,
        "max_rel_ppl_gap_pct": max_rel_ppl_gap,
        "passes_ppl_threshold": passes_ppl,
    }
    print(f"\nBlock {block_id}: J={avg_j:.4f} ({'PASS' if passes_jaccard else 'FAIL'}), "
          f"max_rel_PPL_gap={max_rel_ppl_gap:.1f}% ({'PASS' if passes_ppl else 'FAIL'})")

n_jaccard_pass = sum(v["passes_jaccard_threshold"] for v in verdict_blocks.values())
n_ppl_pass = sum(v["passes_ppl_threshold"] for v in verdict_blocks.values())

if n_jaccard_pass >= 3 and n_ppl_pass >= 2:
    verdict = "CONFIRMED"
elif all(v["avg_cross_corpus_jaccard"] < 0.4 for v in verdict_blocks.values()):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nOverall verdict: {verdict}")
print(f"  Jaccard ≥ 0.7 in {n_jaccard_pass}/4 blocks, PPL gap ≤ 10% in {n_ppl_pass}/4 blocks")

# ── Save JSON ──────────────────────────────────────────────────────────────────────
results = {
    "config": {
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "seq_len": SEQ_LEN,
        "probe_train_sequences": PROBE_TRAIN_SEQ,
        "probe_holdout_sequences": PROBE_HOLDOUT_SEQ,
        "eval_val_sequences": EVAL_VAL_SEQ,
        "corpus_names": CORPUS_NAMES,
        "random_baseline_jaccard": 512 / (2 * 3072 - 512),
    },
    "baselines": baselines,
    "jaccard_results": {str(k): v for k, v in jaccard_results.items()},
    "ppl_matrix_results": {str(k): v for k, v in ppl_matrix_results.items()},
    "channels_store": {
        str(bid): {
            cn: {str(s): ch for s, ch in sd.items()}
            for cn, sd in cd.items()
        }
        for bid, cd in channels_store.items()
    },
    "verdict_blocks": {str(k): v for k, v in verdict_blocks.items()},
    "overall_verdict": verdict,
    "n_jaccard_pass": n_jaccard_pass,
    "n_ppl_pass": n_ppl_pass,
}

JSON_PATH.write_text(json.dumps(results, indent=2))
print(f"\nJSON saved to {JSON_PATH}")

# ── Plots ─────────────────────────────────────────────────────────────────────────
n_blocks = len(TARGET_BLOCKS)
n_corpora = len(CORPUS_NAMES)
fig, axes = plt.subplots(n_blocks, 2, figsize=(14, 4 * n_blocks))

for row_idx, block_id in enumerate(TARGET_BLOCKS):
    # Left: Jaccard heatmap (corpus × corpus)
    ax_j = axes[row_idx, 0]
    jmat = torch.zeros(n_corpora, n_corpora)
    for i in range(n_corpora):
        jmat[i, i] = jaccard_results[block_id]["within_corpus"][CORPUS_NAMES[i]]["mean"]
        for j in range(i + 1, n_corpora):
            pair_key = f"{CORPUS_NAMES[i]}_vs_{CORPUS_NAMES[j]}"
            if pair_key not in jaccard_results[block_id]["cross_corpus"]:
                pair_key = f"{CORPUS_NAMES[j]}_vs_{CORPUS_NAMES[i]}"
            j_val = jaccard_results[block_id]["cross_corpus"].get(pair_key, {}).get("mean", 0.0)
            jmat[i, j] = j_val
            jmat[j, i] = j_val

    im = ax_j.imshow(jmat.numpy(), vmin=0, vmax=1, cmap="YlOrRd")
    ax_j.set_xticks(range(n_corpora))
    ax_j.set_yticks(range(n_corpora))
    ax_j.set_xticklabels(CORPUS_NAMES, rotation=30, ha="right", fontsize=8)
    ax_j.set_yticklabels(CORPUS_NAMES, fontsize=8)
    ax_j.set_title(f"Block {block_id} — Jaccard", fontsize=10)
    for i in range(n_corpora):
        for j in range(n_corpora):
            ax_j.text(j, i, f"{jmat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax_j)

    # Right: relative PPL heatmap (probe × eval)
    ax_p = axes[row_idx, 1]
    pmat = torch.zeros(n_corpora, n_corpora)
    for i, probe_c in enumerate(CORPUS_NAMES):
        for j, eval_c in enumerate(CORPUS_NAMES):
            pmat[i, j] = ppl_matrix_results[block_id][probe_c][eval_c]["rel_pct"]

    im2 = ax_p.imshow(pmat.numpy(), cmap="RdYlGn_r")
    ax_p.set_xticks(range(n_corpora))
    ax_p.set_yticks(range(n_corpora))
    ax_p.set_xticklabels([f"eval: {c}" for c in CORPUS_NAMES], rotation=30, ha="right", fontsize=7)
    ax_p.set_yticklabels([f"probe: {c}" for c in CORPUS_NAMES], fontsize=7)
    ax_p.set_title(f"Block {block_id} — Relative PPL % vs baseline", fontsize=10)
    for i in range(n_corpora):
        for j in range(n_corpora):
            ax_p.text(j, i, f"{pmat[i,j]:+.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im2, ax=ax_p)

fig.suptitle(
    f"Corpus-diverse channel selection — GPT-2 6x ({', '.join(CORPUS_NAMES)})\n"
    f"Verdict: {verdict} (J≥0.7: {n_jaccard_pass}/4, PPL-gap≤10%: {n_ppl_pass}/4)",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved to {PLOT_PATH}")
