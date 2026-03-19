"""Cycle 034: Block-specific compression ratio sweep — GPT-2 FFN.

Sweep compression ratios {2x, 3x, 4x, 5x, 6x} independently on each of the
6 easiest FFN blocks [7, 5, 4, 1, 3, 6] from cycle 005/033. Single-block swaps
only. For each block, find max ratio within 5% PPL threshold.

Uses topk channel selection + lstsq output refit (cycles 026-027).
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _load_compressed_layer,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]
RATIOS = [2, 3, 4, 5, 6]  # compression ratios
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


def compute_recon_mse(atlas: FfnGeometryAtlas, layer_id: int, compressed_layer) -> float:
    """Compute reconstruction MSE of compressed FFN on probe data."""
    source = next(lg for lg in atlas.layer_geometries if lg.layer_id == layer_id)
    mlp = CompressedMLP(
        input_dim=compressed_layer.c_fc_weight.shape[1],
        hidden_dim=compressed_layer.compressed_hidden_dim,
        output_dim=compressed_layer.c_proj_weight.shape[0],
        activation_kind=compressed_layer.activation_kind,
        c_fc_bias=compressed_layer.c_fc_bias is not None,
        c_proj_bias=compressed_layer.c_proj_bias is not None,
    )
    _load_compressed_layer(mlp, compressed_layer)
    with torch.no_grad():
        predicted = mlp(source.input_sample)
        mse = F.mse_loss(predicted, source.output_sample).item()
    return mse


# ── Main ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
print(f"  device: {teacher.device}")

print("Building probe corpus...")
corpus_spec = ProbeCorpusSpec(
    teacher=teacher,
    text_path=SHAKESPEARE_PATH,
    sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ,
    probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
    eval_val_sequences=PROBE_HOLDOUT_SEQ,
    dataset_name="proxy_shakespeare",
)
corpus = build_probe_corpus(corpus_spec)
print(f"  probe: {corpus.probe_train.shape}, holdout: {corpus.probe_holdout.shape}")

print("Computing baseline perplexity...")
baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
threshold_ppl = baseline_ppl * 1.05  # dynamic 5% threshold
print(f"  Baseline PPL: {baseline_ppl:.2f}")
print(f"  5% threshold: {threshold_ppl:.2f}")

print(f"Probing geometry for all target blocks: {TARGET_BLOCKS}...")
atlas = probe_ffn_geometry(
    teacher,
    corpus,
    layer_ids=tuple(TARGET_BLOCKS),
    max_tokens_per_layer=2048,
    batch_size=BATCH_SIZE,
    jacobian_samples=8,
)
print(f"  Probed {len(atlas.layer_geometries)} blocks")

# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"\nBlock-specific ratio sweep:")
print(f"{'block':>6} {'ratio':>6} {'frac':>6} {'PPL':>8} {'rel%':>8} {'MSE':>10} {'safe':>5}")
print("-" * 60)

sweep_results = []
per_block_max_ratio = {}

for block_id in TARGET_BLOCKS:
    # Build single-block atlas
    sub_layers = tuple(lg for lg in atlas.layer_geometries if lg.layer_id == block_id)
    sub_atlas = FfnGeometryAtlas(
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        sequence_length=atlas.sequence_length,
        layer_geometries=sub_layers,
        metadata=atlas.metadata,
    )

    max_safe_ratio = 1  # no compression baseline

    for ratio in RATIOS:
        hidden_fraction = 1.0 / ratio
        budget = CompressionBudgetSpec(
            hidden_fraction=hidden_fraction,
            selection_mode="topk",
        )
        pack = derive_ffn_compression(sub_atlas, budget)
        compressed_layer = pack.layers[0]
        actual_ratio = compressed_layer.source_param_count / max(compressed_layer.compressed_param_count, 1)

        # Reconstruction MSE
        recon_mse = compute_recon_mse(atlas, block_id, compressed_layer)

        # Full-model PPL with single-block swap
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        safe = ppl <= threshold_ppl

        if safe:
            max_safe_ratio = ratio

        print(f"  {block_id:>4}  {ratio:>4}x  {hidden_fraction:.3f}  {ppl:8.2f}  {rel:+7.2f}%  {recon_mse:10.4f}  {'YES' if safe else 'NO':>5}")

        sweep_results.append({
            "block": block_id,
            "ratio": ratio,
            "hidden_fraction": hidden_fraction,
            "actual_ratio": float(actual_ratio),
            "ppl": float(ppl),
            "rel_pct": float(rel),
            "recon_mse": float(recon_mse),
            "safe": safe,
        })

        del student

    per_block_max_ratio[block_id] = max_safe_ratio
    print(f"  → Block {block_id}: max safe ratio = {max_safe_ratio}x")
    print()

# ── Verdict ───────────────────────────────────────────────────────────────────
blocks_at_5x = sum(1 for r in per_block_max_ratio.values() if r >= 5)
blocks_above_4x = sum(1 for r in per_block_max_ratio.values() if r > 4)

print(f"\nPer-block max safe ratios: {per_block_max_ratio}")
print(f"Blocks with max_ratio >= 5x: {blocks_at_5x}/6")

if blocks_at_5x >= 4:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — {blocks_at_5x} blocks tolerate >=5x compression within 5% PPL threshold")
elif blocks_above_4x == 0:
    verdict = "refuted"
    print(f"Verdict: REFUTED — no block tolerates >4x compression within 5% threshold")
elif blocks_at_5x >= 1:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — {blocks_at_5x} blocks tolerate >=5x (partial headroom, insufficient for broad stacking)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — bimodal results without clear predictor")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Build heatmap data: blocks x ratios
block_labels = [str(b) for b in TARGET_BLOCKS]
ratio_labels = [f"{r}x" for r in RATIOS]
heatmap = np.full((len(TARGET_BLOCKS), len(RATIOS)), np.nan)

for entry in sweep_results:
    bi = TARGET_BLOCKS.index(entry["block"])
    ri = RATIOS.index(entry["ratio"])
    heatmap[bi, ri] = entry["rel_pct"]

im = ax.imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=15, aspect="auto")
ax.set_xticks(range(len(RATIOS)))
ax.set_xticklabels(ratio_labels)
ax.set_yticks(range(len(TARGET_BLOCKS)))
ax.set_yticklabels([f"Block {b}" for b in TARGET_BLOCKS])
ax.set_xlabel("Compression Ratio")
ax.set_ylabel("FFN Block")

# Annotate cells
for entry in sweep_results:
    bi = TARGET_BLOCKS.index(entry["block"])
    ri = RATIOS.index(entry["ratio"])
    color = "white" if entry["rel_pct"] > 7 else "black"
    marker = " *" if entry["safe"] else ""
    ax.text(ri, bi, f"{entry['rel_pct']:+.1f}%{marker}", ha="center", va="center",
            fontsize=8, color=color, fontweight="bold" if entry["safe"] else "normal")

# Threshold line annotation
ax.set_title(f"Cycle 034: Block-Specific Compression Ratio Sweep (GPT-2 FFN)\n"
             f"Baseline PPL={baseline_ppl:.1f}, 5% threshold={threshold_ppl:.1f}  |  "
             f"Blocks ≥5x: {blocks_at_5x}/6  |  Verdict: {verdict.upper()}")
fig.colorbar(im, ax=ax, label="Relative PPL change (%)")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_034_ratio_sweep.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {
        "target_blocks": TARGET_BLOCKS,
        "ratios": RATIOS,
        "probe_train_seq": PROBE_TRAIN_SEQ,
        "probe_holdout_seq": PROBE_HOLDOUT_SEQ,
        "seed": SEED,
    },
    "baseline_ppl": float(baseline_ppl),
    "threshold_ppl": float(threshold_ppl),
    "sweep": sweep_results,
    "per_block_max_ratio": {str(k): v for k, v in per_block_max_ratio.items()},
    "blocks_at_5x": blocks_at_5x,
    "verdict": verdict,
}

out_json = RESULTS_DIR / "cycle_034_ratio_sweep.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
