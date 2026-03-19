"""Cycle 046: Lstsq refit quality degradation analysis — GPT-2 FFN.

Measures condition number and held-out MSE of the lstsq W_down refit
at compression ratios {2,4,6,8,10} on blocks [7,5,4] for both
Shakespeare and WikiText-2 probes.
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

import gc
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _select_hidden_units,
    _layer_target_hidden,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
)

# ── Config ────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
WIKITEXT_PATH = str(Path(__file__).parent.parent / "data" / "wikitext2_valid.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

TARGET_BLOCKS = [7, 5, 4]
RATIOS = [2, 4, 6, 8, 10]
SEEDS = [42, 137, 2024]
CORPORA = {"shakespeare": SHAKESPEARE_PATH, "wikitext2": WIKITEXT_PATH}
PROBE_TRAIN_SEQ = 100  # reduced from 200 to avoid OOM
BATCH_SIZE = 4


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    for corpus_name, text_path in CORPORA.items():
        print(f"\n  Corpus: {corpus_name}")
        spec = ProbeCorpusSpec(
            teacher=teacher, text_path=text_path, sequence_length=128,
            probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=50,
            eval_val_sequences=50, dataset_name=corpus_name, shuffle_seed=seed,
        )
        corpus = build_probe_corpus(spec)

        for block_id in TARGET_BLOCKS:
            atlas = probe_ffn_geometry(
                teacher, corpus, layer_ids=(block_id,),
                max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
            )
            layer = atlas.layer_geometries[0]

            # Get full hidden activations and output targets
            hidden_full = layer.hidden_sample.float()  # [N, 3072]
            output_full = layer.output_sample.float()   # [N, 768]
            N = hidden_full.shape[0]

            # 80/20 train/holdout split
            n_train = int(0.8 * N)
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
            train_idx = perm[:n_train]
            hold_idx = perm[n_train:]

            for ratio in RATIOS:
                budget = CompressionBudgetSpec(
                    hidden_fraction=1.0/ratio, selection_mode="topk",
                )
                k = _layer_target_hidden(layer, budget)
                selected = _select_hidden_units(layer, budget, k)

                # Design matrix: selected hidden activations
                A = hidden_full[:, selected]  # [N, k]
                A_train = A[train_idx]
                A_hold = A[hold_idx]
                B_train = output_full[train_idx]
                B_hold = output_full[hold_idx]

                # Condition number
                cond = float(torch.linalg.cond(A_train).item())

                # Lstsq solve
                sol = torch.linalg.lstsq(A_train, B_train).solution  # [k, 768]

                # MSE
                pred_train = A_train @ sol
                pred_hold = A_hold @ sol
                mse_train = float(F.mse_loss(pred_train, B_train).item())
                mse_hold = float(F.mse_loss(pred_hold, B_hold).item())
                overfit_ratio = mse_hold / mse_train if mse_train > 0 else float('inf')

                print(f"    Block {block_id} {ratio:>2}x: cond={cond:.1f}, "
                      f"mse_train={mse_train:.6f}, mse_hold={mse_hold:.6f}, "
                      f"overfit={overfit_ratio:.3f}")

                all_results.append({
                    "seed": seed, "corpus": corpus_name, "block": block_id,
                    "ratio": ratio, "k_channels": int(k), "n_train": n_train,
                    "condition_number": cond,
                    "mse_train": mse_train, "mse_hold": mse_hold,
                    "overfit_ratio": overfit_ratio,
                })

            del atlas, layer, hidden_full, output_full, A, A_train, A_hold, B_train, B_hold
            gc.collect()

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE (mean across seeds)")
print(f"{'='*60}")

summaries = {}
for corpus_name in CORPORA:
    for block_id in TARGET_BLOCKS:
        for ratio in RATIOS:
            entries = [r for r in all_results
                      if r["corpus"] == corpus_name and r["block"] == block_id and r["ratio"] == ratio]
            conds = [r["condition_number"] for r in entries]
            overfits = [r["overfit_ratio"] for r in entries]
            key = f"{corpus_name}_b{block_id}_{ratio}x"
            summaries[key] = {
                "mean_cond": float(np.mean(conds)),
                "mean_overfit": float(np.mean(overfits)),
                "std_overfit": float(np.std(overfits, ddof=1)) if len(overfits) > 1 else 0,
            }

print(f"\n{'corpus':>12} {'block':>6} {'ratio':>6} {'cond':>12} {'overfit':>8}")
print("-" * 50)
for corpus_name in CORPORA:
    for block_id in TARGET_BLOCKS:
        for ratio in RATIOS:
            key = f"{corpus_name}_b{block_id}_{ratio}x"
            s = summaries[key]
            print(f"  {corpus_name:>10} {block_id:>4} {ratio:>4}x {s['mean_cond']:>11.1f} {s['mean_overfit']:>7.3f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
# Check: cond increases >10x from 4x to 8x AND overfit >2.0 at 8x
cond_increase_blocks = 0
overfit_blocks = 0
for block_id in TARGET_BLOCKS:
    # Use Shakespeare for the original criterion
    cond_4x = summaries[f"shakespeare_b{block_id}_4x"]["mean_cond"]
    cond_8x = summaries[f"shakespeare_b{block_id}_8x"]["mean_cond"]
    overfit_8x = summaries[f"shakespeare_b{block_id}_8x"]["mean_overfit"]
    if cond_8x / cond_4x > 10:
        cond_increase_blocks += 1
    if overfit_8x > 2.0:
        overfit_blocks += 1

# Cross-corpus comparison
wiki_cond_higher = 0
for block_id in TARGET_BLOCKS:
    for ratio in RATIOS:
        s_cond = summaries[f"shakespeare_b{block_id}_{ratio}x"]["mean_cond"]
        w_cond = summaries[f"wikitext2_b{block_id}_{ratio}x"]["mean_cond"]
        if w_cond > s_cond * 1.5:
            wiki_cond_higher += 1

print(f"\nCond >10x increase (4x→8x): {cond_increase_blocks}/3 blocks")
print(f"Overfit ratio >2.0 at 8x: {overfit_blocks}/3 blocks")
print(f"WikiText cond >1.5x Shakespeare: {wiki_cond_higher}/{len(TARGET_BLOCKS)*len(RATIOS)} pairs")

if cond_increase_blocks >= 2 and overfit_blocks >= 2:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — cond explodes and refit overfits at high compression")
elif cond_increase_blocks == 0 and overfit_blocks == 0:
    verdict = "refuted"
    print(f"Verdict: REFUTED — refit is well-conditioned at all ratios")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — cond rises but overfit mixed")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = {"shakespeare": "steelblue", "wikitext2": "coral"}
markers = {7: "o", 5: "s", 4: "^"}

for corpus_name in CORPORA:
    for block_id in TARGET_BLOCKS:
        conds = [summaries[f"{corpus_name}_b{block_id}_{r}x"]["mean_cond"] for r in RATIOS]
        overfits = [summaries[f"{corpus_name}_b{block_id}_{r}x"]["mean_overfit"] for r in RATIOS]
        label = f"{corpus_name} block {block_id}"
        ax1.semilogy(RATIOS, conds, marker=markers[block_id], color=colors[corpus_name],
                     label=label, alpha=0.8)
        ax2.plot(RATIOS, overfits, marker=markers[block_id], color=colors[corpus_name],
                label=label, alpha=0.8)

ax1.set_xlabel("Compression Ratio")
ax1.set_ylabel("Condition Number (log scale)")
ax1.set_title("Lstsq Condition Number vs Ratio")
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

ax2.axhline(2.0, color="red", ls="--", alpha=0.7, label="overfit=2.0 threshold")
ax2.set_xlabel("Compression Ratio")
ax2.set_ylabel("Overfit Ratio (held-out MSE / train MSE)")
ax2.set_title("Refit Overfit Ratio vs Ratio")
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

fig.suptitle(f"Cycle 046: Lstsq Refit Quality — Verdict: {verdict.upper()}", fontsize=13)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_046_refit_quality.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved: {out_png}")

out_json = RESULTS_DIR / "cycle_046_refit_quality.json"
with open(out_json, "w") as f:
    json.dump({"results": all_results, "summaries": summaries, "verdict": verdict}, f, indent=2)
print(f"Saved: {out_json}")
