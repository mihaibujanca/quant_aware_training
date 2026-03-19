"""Cycle 035: Multi-seed confirmation of 6x single-block compression — GPT-2 FFN.

Replicates cycle 034's 6x result with 3 seeds (varying probe corpus via shuffle_seed)
and 200 eval sequences. Also includes random-channel null baseline per block per seed.
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
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _load_compressed_layer,
    _standard_weight,
    _standard_bias,
    _layer_target_hidden,
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

TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]
RATIO = 6
HIDDEN_FRACTION = 1.0 / RATIO
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 200  # 200 eval sequences per plan
BATCH_SIZE = 4


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


def random_channel_compress(
    atlas: FfnGeometryAtlas, layer_id: int, k: int, seed: int
) -> CompressedFfnLayer:
    """Compress a single block using random channel selection + lstsq refit."""
    layer = next(lg for lg in atlas.layer_geometries if lg.layer_id == layer_id)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(layer.hidden_dim, generator=rng)
    selected = perm[:k].sort().values

    source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
    source_b_in = _standard_bias(layer.handle.c_fc)
    use_output_bias = _standard_bias(layer.handle.c_proj) is not None

    student_w_in = source_w_in[selected].contiguous()
    student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
    hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
    student_w_out, student_b_out = _fit_output_projection(
        hidden, layer.output_sample, use_bias=use_output_bias
    )

    source_param_count = (
        source_w_in.numel() + (source_b_in.numel() if source_b_in is not None else 0)
        + _standard_weight(layer.handle.c_proj).numel()
        + (_standard_bias(layer.handle.c_proj).numel() if _standard_bias(layer.handle.c_proj) is not None else 0)
    )
    compressed_param_count = (
        student_w_in.numel() + (student_b_in.numel() if student_b_in is not None else 0)
        + student_w_out.numel() + (student_b_out.numel() if student_b_out is not None else 0)
    )

    return CompressedFfnLayer(
        layer_id=layer.layer_id,
        activation_kind=layer.activation_kind,
        source_hidden_dim=layer.hidden_dim,
        compressed_hidden_dim=int(k),
        c_fc_weight=student_w_in,
        c_fc_bias=student_b_in,
        c_proj_weight=student_w_out,
        c_proj_bias=student_b_out,
        selected_indices=selected,
        source_param_count=source_param_count,
        compressed_param_count=compressed_param_count,
        notes={"selection_mode": "random", "seed": seed},
    )


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
print(f"  device: {teacher.device}")

all_results = []
baselines = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    print("Building probe corpus (shuffle_seed={})...".format(seed))
    corpus_spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=SHAKESPEARE_PATH,
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
        eval_val_sequences=PROBE_HOLDOUT_SEQ,
        dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(corpus_spec)
    print(f"  probe: {corpus.probe_train.shape}, holdout: {corpus.probe_holdout.shape}")

    print("Computing baseline perplexity...")
    baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
    baselines[seed] = baseline_ppl
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print(f"Probing geometry for target blocks...")
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048,
        batch_size=BATCH_SIZE,
        jacobian_samples=8,
    )

    # Topk compression at 6x
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )
    topk_pack = derive_ffn_compression(atlas, budget)
    topk_by_id = {layer.layer_id: layer for layer in topk_pack.layers}

    # Target hidden dim for random baseline
    k_target = topk_pack.layers[0].compressed_hidden_dim

    print(f"\n{'block':>6} {'mode':>8} {'PPL':>8} {'rel%':>8}")
    print("-" * 40)

    for block_id in TARGET_BLOCKS:
        # --- Topk ---
        sub_pack = CompressedFfnPack(
            method=topk_pack.method,
            model_family=topk_pack.model_family,
            dataset_name=topk_pack.dataset_name,
            layers=(topk_by_id[block_id],),
            metadata=topk_pack.metadata,
        )
        student = swap_compressed_ffns(teacher, sub_pack, inplace=False)
        topk_ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
        topk_rel = (topk_ppl - baseline_ppl) / baseline_ppl * 100
        del student

        # --- Random ---
        rand_layer = random_channel_compress(atlas, block_id, k_target, seed)
        rand_pack = CompressedFfnPack(
            method="random_lstsq",
            model_family=topk_pack.model_family,
            dataset_name=topk_pack.dataset_name,
            layers=(rand_layer,),
            metadata={},
        )
        student = swap_compressed_ffns(teacher, rand_pack, inplace=False)
        rand_ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
        rand_rel = (rand_ppl - baseline_ppl) / baseline_ppl * 100
        del student

        print(f"  {block_id:>4}    topk  {topk_ppl:8.2f}  {topk_rel:+7.2f}%")
        print(f"  {block_id:>4}  random  {rand_ppl:8.2f}  {rand_rel:+7.2f}%")

        all_results.append({
            "seed": seed,
            "block": block_id,
            "baseline_ppl": float(baseline_ppl),
            "topk_ppl": float(topk_ppl),
            "topk_rel_pct": float(topk_rel),
            "random_ppl": float(rand_ppl),
            "random_rel_pct": float(rand_rel),
        })

# ── Validation: baselines differ across seeds ────────────────────────────────
baseline_vals = list(baselines.values())
print(f"\nBaseline PPLs across seeds: {baselines}")
if len(set(f'{v:.4f}' for v in baseline_vals)) == 1:
    print("WARNING: All baselines identical — shuffle_seed may not be propagating!")

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")
print(f"\n{'block':>6} {'mean_rel%':>10} {'std_rel%':>10} {'mean_rand%':>11} {'topk>rand':>10}")
print("-" * 55)

block_summaries = {}
for block_id in TARGET_BLOCKS:
    block_data = [r for r in all_results if r["block"] == block_id]
    topk_rels = [r["topk_rel_pct"] for r in block_data]
    rand_rels = [r["random_rel_pct"] for r in block_data]

    mean_topk = float(np.mean(topk_rels))
    std_topk = float(np.std(topk_rels, ddof=1)) if len(topk_rels) > 1 else 0.0
    mean_rand = float(np.mean(rand_rels))
    topk_beats_rand = mean_topk < mean_rand

    block_summaries[block_id] = {
        "mean_topk_rel": mean_topk,
        "std_topk_rel": std_topk,
        "mean_random_rel": mean_rand,
        "topk_beats_random": topk_beats_rand,
        "topk_rels": topk_rels,
        "random_rels": rand_rels,
    }
    print(f"  {block_id:>4}  {mean_topk:+9.2f}%  {std_topk:9.2f}%  {mean_rand:+10.2f}%  {'YES' if topk_beats_rand else 'NO':>10}")

# ── Verdict ───────────────────────────────────────────────────────────────────
blocks_within_5 = sum(1 for s in block_summaries.values() if s["mean_topk_rel"] <= 5.0)
blocks_sub_baseline = sum(1 for s in block_summaries.values() if s["mean_topk_rel"] < 0.0)
blocks_high_std = sum(1 for s in block_summaries.values() if s["std_topk_rel"] > 2.0)

print(f"\nBlocks with mean rel PPL <= 5%: {blocks_within_5}/6")
print(f"Blocks with sub-baseline mean PPL: {blocks_sub_baseline}/6")
print(f"Blocks with std > 2%: {blocks_high_std}/6")

if blocks_within_5 >= 5 and blocks_sub_baseline == 0:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — {blocks_within_5}/6 blocks within 5%, no sub-baseline means")
elif sum(1 for s in block_summaries.values() if s["mean_topk_rel"] > 5.0) >= 3:
    verdict = "refuted"
    print(f"Verdict: REFUTED — >=3 blocks exceed 5% mean relative PPL")
elif blocks_high_std > 3:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — high cross-seed std (>2%) for majority of blocks")
else:
    verdict = "inconclusive"
    reason = []
    if blocks_sub_baseline > 0:
        reason.append(f"{blocks_sub_baseline} blocks sub-baseline")
    if blocks_high_std > 0:
        reason.append(f"{blocks_high_std} blocks high std")
    print(f"Verdict: INCONCLUSIVE — {'; '.join(reason) if reason else 'mixed results'}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

blocks = TARGET_BLOCKS
x = np.arange(len(blocks))
means = [block_summaries[b]["mean_topk_rel"] for b in blocks]
stds = [block_summaries[b]["std_topk_rel"] for b in blocks]
rand_means = [block_summaries[b]["mean_random_rel"] for b in blocks]

ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8, label="Topk 6x (mean ± std)")
ax.scatter(x, rand_means, marker="D", color="grey", s=60, zorder=5, label="Random 6x (mean)")
ax.axhline(5.0, color="red", ls="--", alpha=0.7, label="5% threshold")
ax.axhline(0.0, color="black", ls="-", alpha=0.3)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:+.2f}%\n±{s:.2f}", (i, m), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"Block {b}" for b in blocks])
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 035: Multi-Seed 6x Compression Confirmation (3 seeds, 200 eval seqs)\n"
             f"Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_035_multiseed_6x.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {
        "target_blocks": TARGET_BLOCKS,
        "ratio": RATIO,
        "seeds": SEEDS,
        "probe_train_seq": PROBE_TRAIN_SEQ,
        "probe_holdout_seq": PROBE_HOLDOUT_SEQ,
    },
    "baselines": {str(k): v for k, v in baselines.items()},
    "per_seed_results": all_results,
    "block_summaries": {str(k): v for k, v in block_summaries.items()},
    "verdict": verdict,
    "blocks_within_5pct": blocks_within_5,
    "blocks_sub_baseline": blocks_sub_baseline,
    "blocks_high_std": blocks_high_std,
}

out_json = RESULTS_DIR / "cycle_035_multiseed_6x.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
