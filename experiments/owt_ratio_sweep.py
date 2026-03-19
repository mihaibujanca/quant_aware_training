"""Cycle 056: Compression ratio sweep on OpenWebText — GPT-2 FFN.

Tests ratios {2x, 3x, 4x, 6x} on blocks [7, 5, 4] with no correction,
to find the viable compression frontier on GPT-2's training distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

TARGET_BLOCKS = [7, 5, 4]
RATIOS = {2: 1 / 2, 3: 1 / 3, 4: 1 / 4, 6: 1 / 6}
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
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


# ── Helper ────────────────────────────────────────────────────────────────────
def eval_ppl(model, teacher, corpus, batch_size):
    return float(_evaluate_model(teacher, model, corpus, batch_size=batch_size)["perplexity"])


# ── Main ──────────────────────────────────────────────────────────────────────
if not Path(OWT_SAMPLE_PATH).exists():
    download_owt_sample(OWT_SAMPLE_PATH)
else:
    print(f"Using cached OpenWebText sample: {OWT_SAMPLE_PATH}")

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

# results[block][ratio][seed] = {baseline, uncorrected, rel_uncorrected}
all_results: dict = {}

total_configs = len(SEEDS) * len(TARGET_BLOCKS) * len(RATIOS)
done = 0

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed)

    # Build probe corpus once per seed
    spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=OWT_SAMPLE_PATH,
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=50,
        eval_val_sequences=PROBE_EVAL_SEQ,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(spec)
    baseline = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
    print(f"  baseline PPL: {baseline:.3f}")

    for block in TARGET_BLOCKS:
        block_key = f"block{block}"
        if block_key not in all_results:
            all_results[block_key] = {}

        # Probe geometry once per block per seed
        atlas = probe_ffn_geometry(
            teacher, corpus, layer_ids=(block,),
            max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
        )

        for ratio_int, hidden_frac in RATIOS.items():
            ratio_key = f"{ratio_int}x"
            if ratio_key not in all_results[block_key]:
                all_results[block_key][ratio_key] = []

            budget = CompressionBudgetSpec(hidden_fraction=hidden_frac, selection_mode="topk")
            pack = derive_ffn_compression(atlas, budget)

            model_u = swap_compressed_ffns(teacher, pack, inplace=False)
            ppl_u = eval_ppl(model_u, teacher, corpus, BATCH_SIZE)
            del model_u

            rel_u = (ppl_u - baseline) / baseline * 100
            done += 1
            print(f"  block={block} ratio={ratio_int}x  PPL={ppl_u:.3f} ({rel_u:+.2f}%)  [{done}/{total_configs}]")

            all_results[block_key][ratio_key].append({
                "seed": seed,
                "baseline": float(baseline),
                "uncorrected": float(ppl_u),
                "rel_uncorrected": float(rel_u),
            })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE (mean ± std rel PPL%)")
print(f"{'='*60}")

summary: dict = {}
for block_key in all_results:
    summary[block_key] = {}
    for ratio_key in sorted(all_results[block_key].keys()):
        rels = [r["rel_uncorrected"] for r in all_results[block_key][ratio_key]]
        mean_rel = float(np.mean(rels))
        std_rel = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
        summary[block_key][ratio_key] = {"mean": mean_rel, "std": std_rel}
        print(f"  {block_key} {ratio_key:4s}: {mean_rel:+7.2f}% ± {std_rel:.2f}%")

# ── Criterion evaluation ───────────────────────────────────────────────────────
THRESHOLD = 10.0  # % PPL degradation
QUORUM = 2        # blocks out of 3

print(f"\n{'='*60}")
print(f"CRITERION: ratio <=4x achieves <{THRESHOLD}% PPL for >={QUORUM}/3 blocks")
print(f"{'='*60}")

qualifying_ratios = []
for ratio_int in [2, 3, 4]:
    ratio_key = f"{ratio_int}x"
    blocks_passing = 0
    for block_key in all_results:
        mean_rel = summary[block_key][ratio_key]["mean"]
        if mean_rel < THRESHOLD:
            blocks_passing += 1
    qualifies = blocks_passing >= QUORUM
    status = "PASS" if qualifies else "FAIL"
    print(f"  {ratio_key}: {blocks_passing}/3 blocks below {THRESHOLD}%  [{status}]")
    if qualifies:
        qualifying_ratios.append(ratio_int)

# Also check 2x specifically for inconclusive
ratio2_blocks_ok = sum(
    1 for bk in all_results if summary[bk]["2x"]["mean"] < THRESHOLD
)
ratio3_blocks_ok = sum(
    1 for bk in all_results if summary[bk]["3x"]["mean"] < THRESHOLD
)

if qualifying_ratios:
    verdict = "confirmed"
elif ratio2_blocks_ok >= QUORUM and ratio3_blocks_ok < QUORUM:
    verdict = "inconclusive"
else:
    verdict = "refuted"

print(f"\n  Verdict: {verdict.upper()}")
if verdict == "confirmed":
    best = min(qualifying_ratios)
    print(f"  Best ratio achieving <{THRESHOLD}% PPL: {best}x")
elif verdict == "inconclusive":
    print(f"  2x works but 3x already fails (narrow window)")
else:
    print(f"  Even 2x exceeds {THRESHOLD}% for majority of blocks")

# ── Save results ──────────────────────────────────────────────────────────────
out_data = {
    "config": {
        "target_blocks": TARGET_BLOCKS,
        "ratios": list(RATIOS.keys()),
        "seeds": SEEDS,
        "probe_train_seq": PROBE_TRAIN_SEQ,
        "eval_val_seq": PROBE_EVAL_SEQ,
        "correction": False,
        "threshold_pct": THRESHOLD,
        "quorum": QUORUM,
    },
    "per_seed": all_results,
    "summary": summary,
    "verdict": verdict,
}

out_json = RESULTS_DIR / "cycle_056_owt_ratio_sweep.json"
with open(out_json, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"\nSaved results: {out_json}")

# ── Plot: heatmap block x ratio ────────────────────────────────────────────────
ratio_labels = ["2x", "3x", "4x", "6x"]
block_labels = [f"Block {b}" for b in TARGET_BLOCKS]

heat = np.zeros((len(TARGET_BLOCKS), len(ratio_labels)))
for i, block in enumerate(TARGET_BLOCKS):
    block_key = f"block{block}"
    for j, rl in enumerate(ratio_labels):
        heat[i, j] = summary[block_key][rl]["mean"]

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(heat, cmap="RdYlGn_r", aspect="auto",
               vmin=0, vmax=max(50, heat.max() * 1.05))
plt.colorbar(im, ax=ax, label="Mean Rel PPL Degradation (%)")

ax.set_xticks(range(len(ratio_labels)))
ax.set_xticklabels(ratio_labels, fontsize=11)
ax.set_yticks(range(len(block_labels)))
ax.set_yticklabels(block_labels, fontsize=11)
ax.set_xlabel("Compression Ratio", fontsize=12)
ax.set_ylabel("Transformer Block", fontsize=12)
ax.set_title(f"Cycle 056: OWT Ratio Sweep — GPT-2 FFN (no correction)\n"
             f"Verdict: {verdict.upper()}", fontsize=11)

# Annotate cells
for i in range(len(TARGET_BLOCKS)):
    for j, rl in enumerate(ratio_labels):
        block_key = f"block{TARGET_BLOCKS[i]}"
        m = summary[block_key][rl]["mean"]
        s = summary[block_key][rl]["std"]
        color = "white" if abs(m) > 30 else "black"
        ax.text(j, i, f"{m:+.1f}%\n±{s:.1f}", ha="center", va="center",
                fontsize=9, color=color)

# Threshold line annotation
ax.axvline(2.5, color="blue", lw=2, ls="--", alpha=0.7, label="≤4x threshold")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
out_png = RESULTS_DIR / "cycle_056_owt_ratio_sweep.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"Saved plot: {out_png}")
