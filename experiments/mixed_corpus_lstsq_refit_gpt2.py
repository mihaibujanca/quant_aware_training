"""Cycle 076: Lstsq refit with mixed-corpus probe data — GPT-2 FFN.

Fix channel selection from Shakespeare. Run lstsq W_down refit with:
  A) Shakespeare-only 512 tokens
  B) Mixed 512 tokens (128 each from 4 corpora)
  C) Mixed 2048 tokens (512 each from 4 corpora)

Evaluate all on WikiText-2 validation PPL.
Blocks [7, 5, 4, 1] at 6x compression. Seeds [42, 123, 7].
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    CompressionBudgetSpec,
    ProbeCorpus,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_bias,
    _standard_weight,
    _apply_activation,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
SHAKESPEARE_PATH = str(DATA_DIR / "shakespeare.txt")
WIKITEXT_PATH    = str(DATA_DIR / "wikitext2_valid.txt")
OPENWEBTEXT_PATH = str(DATA_DIR / "openwebtext_sample.txt")
C4_PATH          = str(DATA_DIR / "c4_sample.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

TARGET_BLOCKS  = [7, 5, 4, 1]
COMPRESSION    = 6          # 6x → hidden_fraction = 1/6
SEEDS          = [42, 123, 7]
BATCH_SIZE     = 4
SEQ_LEN        = 128
EVAL_SEQUENCES = 200        # WikiText-2 validation sequences

CORPUS_PATHS = {
    "shakespeare":  SHAKESPEARE_PATH,
    "wikitext2":    WIKITEXT_PATH,
    "openwebtext":  OPENWEBTEXT_PATH,
    "c4":           C4_PATH,
}

budget = CompressionBudgetSpec(hidden_fraction=1.0 / COMPRESSION, selection_mode="topk")


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_hidden_activations(
    teacher: TeacherBundle,
    token_tensor: torch.Tensor,
    block_id: int,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect hidden-layer activations (post-activation) and MLP outputs for one block."""
    from aleph.lm_geometric_compression import _extract_ffn_handles, _sample_tensor

    all_handles = _extract_ffn_handles(teacher.model, teacher.activation_kind)
    handle = next(h for h in all_handles if h.layer_id == block_id)

    hidden_samples: list[torch.Tensor] = []
    output_samples: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0].detach()
        z = module.c_fc(x).detach()
        h = _apply_activation(z, handle.activation_kind).detach()
        remaining = n_tokens - sum(item.shape[0] for item in hidden_samples)
        taken = _sample_tensor(hidden_samples, h, remaining)
        if taken > 0:
            _sample_tensor(output_samples, output, taken)

    hook = handle.mlp.register_forward_hook(hook_fn)
    try:
        teacher.model.eval()
        with torch.no_grad():
            for input_ids, _targets in _iter_lm_batches(token_tensor, BATCH_SIZE, teacher.device):
                teacher.model(input_ids)
                if sum(item.shape[0] for item in hidden_samples) >= n_tokens:
                    break
    finally:
        hook.remove()

    h_cat = torch.cat(hidden_samples, dim=0).float()[:n_tokens]
    y_cat = torch.cat(output_samples, dim=0).float()[:n_tokens]
    return h_cat, y_cat


def build_compressed_pack_from_refits(
    teacher: TeacherBundle,
    atlas_shakes: object,   # FfnGeometryAtlas from Shakespeare
    refit_data: dict[int, tuple[torch.Tensor, torch.Tensor]],  # block_id -> (hidden, output)
    dataset_label: str,
) -> CompressedFfnPack:
    """Build a CompressedFfnPack using Shakespeare channel selection + custom refit data."""
    compressed_layers = []
    for layer in atlas_shakes.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        # Use provided hidden/output for refit
        hidden_full, output = refit_data[layer.layer_id]
        hidden_sel = hidden_full[:, selected].float()
        student_w_out, student_b_out = _fit_output_projection(
            hidden_sel, output, use_bias=use_output_bias
        )

        source_param_count = (
            source_w_in.numel() + (source_b_in.numel() if source_b_in is not None else 0)
            + _standard_weight(layer.handle.c_proj).numel()
            + (use_output_bias and _standard_bias(layer.handle.c_proj).numel() or 0)
        )
        compressed_param_count = (
            student_w_in.numel() + (student_b_in.numel() if student_b_in is not None else 0)
            + student_w_out.numel() + (student_b_out.numel() if student_b_out is not None else 0)
        )

        compressed_layers.append(
            CompressedFfnLayer(
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
                notes={"selection_corpus": "shakespeare", "refit_corpus": dataset_label},
            )
        )

    return CompressedFfnPack(
        method="geometric_mixed_refit",
        model_family=atlas_shakes.model_family,
        dataset_name=dataset_label,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": SEQ_LEN},
    )


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

# Build WikiText-2 eval corpus (200 sequences, held-out)
print("Building WikiText-2 eval corpus...")
wiki_eval_spec = ProbeCorpusSpec(
    teacher=teacher, text_path=WIKITEXT_PATH,
    sequence_length=SEQ_LEN,
    probe_train_sequences=32, probe_holdout_sequences=32,
    eval_val_sequences=EVAL_SEQUENCES,
    dataset_name="wikitext2", shuffle_seed=999,
)
wiki_eval_corpus = build_probe_corpus(wiki_eval_spec)

# Baseline: uncompressed teacher PPL on WikiText-2
print("Computing uncompressed baseline PPL...")
baseline_metrics = _evaluate_model(teacher, teacher.model, wiki_eval_corpus, batch_size=BATCH_SIZE)
baseline_ppl = baseline_metrics["perplexity"]
print(f"Baseline PPL: {baseline_ppl:.2f}")

all_results = []

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    torch.manual_seed(seed)

    # ── Step 1: Channel selection from Shakespeare (fixed) ─────────────────
    print(f"  [seed={seed}] Building Shakespeare corpus (512 tokens = 4 holdout seqs × 128)...")
    # probe_ffn_geometry uses probe_holdout; 4 seqs × 128 = 512 tokens
    shakes_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=SHAKESPEARE_PATH,
        sequence_length=SEQ_LEN,
        probe_train_sequences=32, probe_holdout_sequences=4,
        eval_val_sequences=32,
        dataset_name="shakespeare", shuffle_seed=seed,
    )
    shakes_corpus = build_probe_corpus(shakes_spec)

    print(f"  [seed={seed}] Probing FFN geometry on Shakespeare (channel selection)...")
    atlas_shakes = probe_ffn_geometry(
        teacher, shakes_corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=512, batch_size=BATCH_SIZE, jacobian_samples=4,
    )

    # Channel selection is embedded in atlas_shakes; log k values
    for layer in atlas_shakes.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        print(f"    Block {layer.layer_id}: k={k}, hidden={layer.hidden_dim}")

    # ── Step 2: Build per-corpus token tensors for refit ──────────────────
    print(f"  [seed={seed}] Building per-corpus token tensors...")
    # Build 512-token tensors for each corpus (for condition B/C)
    corpus_tokens: dict[str, torch.Tensor] = {}
    for cname, cpath in CORPUS_PATHS.items():
        c_spec = ProbeCorpusSpec(
            teacher=teacher, text_path=cpath,
            sequence_length=SEQ_LEN,
            probe_train_sequences=4, probe_holdout_sequences=4,
            eval_val_sequences=8,
            dataset_name=cname, shuffle_seed=seed,
        )
        c_corpus = build_probe_corpus(c_spec)
        # probe_holdout: [4, 129] → flatten to token stream for activation collection
        corpus_tokens[cname] = c_corpus.probe_holdout  # [4, 129]

    # ── Step 3: Collect hidden activations for each condition ─────────────
    # Condition A: Shakespeare 512 tokens (from atlas_shakes.hidden_sample)
    # Condition B: Mixed 512 tokens (128 per corpus = 1 seq × 128)
    # Condition C: Mixed 2048 tokens (512 per corpus = 4 seqs × 128)

    print(f"  [seed={seed}] Collecting activations for conditions B and C...")
    # For each block, collect 128 and 512 tokens from each corpus
    block_hidden_B: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {b: {} for b in TARGET_BLOCKS}
    block_hidden_C: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {b: {} for b in TARGET_BLOCKS}

    for cname, tok_tensor in corpus_tokens.items():
        # Build larger token tensor for condition C (512 tokens per corpus)
        c_spec_large = ProbeCorpusSpec(
            teacher=teacher, text_path=CORPUS_PATHS[cname],
            sequence_length=SEQ_LEN,
            probe_train_sequences=16, probe_holdout_sequences=4,
            eval_val_sequences=8,
            dataset_name=cname, shuffle_seed=seed,
        )
        c_corpus_large = build_probe_corpus(c_spec_large)
        tok_tensor_large = c_corpus_large.probe_holdout  # [4, 129] = 512 tokens

        for block_id in TARGET_BLOCKS:
            # 128 tokens from this corpus (1 seq)
            h128, y128 = collect_hidden_activations(teacher, tok_tensor[:1], block_id, n_tokens=128)
            block_hidden_B[block_id][cname] = (h128, y128)

            # 512 tokens from this corpus (4 seqs)
            h512, y512 = collect_hidden_activations(teacher, tok_tensor_large, block_id, n_tokens=512)
            block_hidden_C[block_id][cname] = (h512, y512)

    # ── Step 4: Build refit data dicts ─────────────────────────────────────
    # Condition A: use atlas_shakes hidden_sample (already 512 tokens max)
    refit_A: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in atlas_shakes.layer_geometries:
        h = layer.hidden_sample.float()
        y = layer.output_sample.float()
        refit_A[layer.layer_id] = (h[:512], y[:512])

    # Condition B: concatenate 128 from each of 4 corpora = 512 total
    refit_B: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_id in TARGET_BLOCKS:
        hs = [block_hidden_B[block_id][c][0] for c in CORPUS_PATHS]
        ys = [block_hidden_B[block_id][c][1] for c in CORPUS_PATHS]
        refit_B[block_id] = (torch.cat(hs, dim=0), torch.cat(ys, dim=0))

    # Condition C: concatenate 512 from each of 4 corpora = 2048 total
    refit_C: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_id in TARGET_BLOCKS:
        hs = [block_hidden_C[block_id][c][0] for c in CORPUS_PATHS]
        ys = [block_hidden_C[block_id][c][1] for c in CORPUS_PATHS]
        refit_C[block_id] = (torch.cat(hs, dim=0), torch.cat(ys, dim=0))

    # ── Step 5: Build packs and evaluate ───────────────────────────────────
    for cond_label, refit_data in [("A_shakespeare_512", refit_A), ("B_mixed_512", refit_B), ("C_mixed_2048", refit_C)]:
        print(f"  [seed={seed}] Condition {cond_label}: building pack + evaluating...")
        pack = build_compressed_pack_from_refits(teacher, atlas_shakes, refit_data, cond_label)
        compressed_model = swap_compressed_ffns(teacher, pack, inplace=False)
        metrics = _evaluate_model(teacher, compressed_model, wiki_eval_corpus, batch_size=BATCH_SIZE)
        ppl = metrics["perplexity"]
        rel_degradation = (ppl - baseline_ppl) / baseline_ppl * 100.0
        print(f"    PPL={ppl:.2f}, degradation={rel_degradation:+.1f}% (baseline={baseline_ppl:.2f})")
        all_results.append({
            "seed": seed,
            "condition": cond_label,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "ppl_degradation_pp": ppl - baseline_ppl,
            "ppl_degradation_rel_pct": rel_degradation,
            "eval_loss": metrics["eval_loss"],
            "kl": metrics["teacher_student_kl"],
            "top1_agreement": metrics["top1_agreement"],
        })
        del compressed_model

    del atlas_shakes


# ── Aggregate and verdict ─────────────────────────────────────────────────────
print(f"\n{'='*60}\nAGGREGATE\n{'='*60}")

import statistics
conditions = ["A_shakespeare_512", "B_mixed_512", "C_mixed_2048"]
agg = {}
for cond in conditions:
    rows = [r for r in all_results if r["condition"] == cond]
    ppls = [r["ppl"] for r in rows]
    rels = [r["ppl_degradation_rel_pct"] for r in rows]
    agg[cond] = {
        "mean_ppl": statistics.mean(ppls),
        "std_ppl": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
        "mean_rel_pct": statistics.mean(rels),
        "std_rel_pct": statistics.stdev(rels) if len(rels) > 1 else 0.0,
        "ppls": ppls,
        "rels": rels,
    }
    print(f"  {cond}: PPL={agg[cond]['mean_ppl']:.2f}±{agg[cond]['std_ppl']:.2f}, "
          f"degradation={agg[cond]['mean_rel_pct']:+.1f}±{agg[cond]['std_rel_pct']:.1f}%")

# Verdict: does mixed (B or C) reduce degradation by >= 10pp relative vs A?
deg_A = agg["A_shakespeare_512"]["mean_rel_pct"]
deg_B = agg["B_mixed_512"]["mean_rel_pct"]
deg_C = agg["C_mixed_2048"]["mean_rel_pct"]

improvement_B = deg_A - deg_B   # positive = B is better
improvement_C = deg_A - deg_C   # positive = C is better
best_improvement = max(improvement_B, improvement_C)

print(f"\nImprovement B vs A: {improvement_B:+.1f}pp relative")
print(f"Improvement C vs A: {improvement_C:+.1f}pp relative")

if best_improvement >= 10.0:
    verdict = "confirmed"
    print("Verdict: CONFIRMED — mixed refit reduces degradation by >= 10pp relative")
elif best_improvement < 5.0:
    verdict = "refuted"
    print("Verdict: REFUTED — mixed refit does not improve by >= 5pp relative")
else:
    verdict = "inconclusive"
    print("Verdict: INCONCLUSIVE — improvement < 10pp relative (", best_improvement, "pp)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = range(len(conditions))
colors = ["steelblue", "darkorange", "forestgreen"]
labels = ["A: Shakespeare\n512 tok", "B: Mixed\n512 tok", "C: Mixed\n2048 tok"]

# PPL bar chart per seed + mean
seed_ppls = [[r["ppl"] for r in all_results if r["condition"] == c] for c in conditions]
mean_ppls = [agg[c]["mean_ppl"] for c in conditions]
std_ppls  = [agg[c]["std_ppl"]  for c in conditions]

bars = ax1.bar(x, mean_ppls, color=colors, alpha=0.7, width=0.5, yerr=std_ppls, capsize=5)
# overlay individual seeds
for i, ppls in enumerate(seed_ppls):
    ax1.scatter([i] * len(ppls), ppls, color="black", s=30, zorder=3)
ax1.axhline(baseline_ppl, ls="--", color="red", label=f"Baseline PPL={baseline_ppl:.1f}")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("WikiText-2 PPL")
ax1.set_title("Compressed PPL by Refit Corpus")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Degradation bar chart
mean_rels = [agg[c]["mean_rel_pct"] for c in conditions]
std_rels  = [agg[c]["std_rel_pct"]  for c in conditions]
seed_rels = [[r["ppl_degradation_rel_pct"] for r in all_results if r["condition"] == c] for c in conditions]

ax2.bar(x, mean_rels, color=colors, alpha=0.7, width=0.5, yerr=std_rels, capsize=5)
for i, rels in enumerate(seed_rels):
    ax2.scatter([i] * len(rels), rels, color="black", s=30, zorder=3)
ax2.axhline(0, ls="--", color="red", alpha=0.5)
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("PPL Degradation (% relative)")
ax2.set_title("PPL Degradation vs Baseline")
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Cycle 076: Mixed-Corpus Lstsq Refit — GPT-2 (6x, blocks {TARGET_BLOCKS})\nVerdict: {verdict.upper()}", fontsize=12)
plt.tight_layout()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_png = RESULTS_DIR / "cycle_076_mixed_refit.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved plot: {out_png}")

out_json = RESULTS_DIR / "cycle_076_mixed_refit.json"
with open(out_json, "w") as f:
    json.dump({
        "baseline_ppl": baseline_ppl,
        "results": all_results,
        "aggregate": {c: {k: v for k, v in d.items() if k not in ("ppls", "rels")} for c, d in agg.items()},
        "verdict": verdict,
        "improvement_B_pp": improvement_B,
        "improvement_C_pp": improvement_C,
        "blocks": TARGET_BLOCKS,
        "compression": COMPRESSION,
        "seeds": SEEDS,
    }, f, indent=2)
print(f"Saved JSON: {out_json}")
