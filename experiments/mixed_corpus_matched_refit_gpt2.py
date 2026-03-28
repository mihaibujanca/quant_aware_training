"""Cycle 078: Mixed-corpus matched refit with equal sample sizes — GPT-2 FFN.

Two conditions at n=2048:
  A': Shakespeare-only 2048 tokens (16 sequences × 128 seq_len)
  C:  Mixed 2048 tokens (512 from each of Shakespeare, WikiText-2, OWT, C4)

Channel selection FIXED from Shakespeare for both conditions.
Evaluate on WikiText-2 validation (200 sequences).
Blocks [7, 5, 4, 1] at 6x compression. Seeds [42, 123, 7].
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressionBudgetSpec,
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

# n=2048 for both conditions
# A': 16 seqs × 128 = 2048 tokens from Shakespeare only
# C:  4 corpora × 4 seqs × 128 = 4 × 512 = 2048 tokens
N_TOKENS_REFIT = 2048
N_TOKENS_PER_CORPUS = 512   # for condition C: 512 × 4 corpora = 2048 total

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

    # ── Step 1: Channel selection from Shakespeare (fixed for both conditions) ─
    # Use 2048 tokens from Shakespeare for probing (16 seqs × 128)
    print(f"  [seed={seed}] Building Shakespeare corpus (2048 tokens for channel selection)...")
    shakes_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=SHAKESPEARE_PATH,
        sequence_length=SEQ_LEN,
        probe_train_sequences=64, probe_holdout_sequences=16,  # 16 × 128 = 2048 holdout tokens
        eval_val_sequences=32,
        dataset_name="shakespeare", shuffle_seed=seed,
    )
    shakes_corpus = build_probe_corpus(shakes_spec)

    print(f"  [seed={seed}] Probing FFN geometry on Shakespeare (channel selection)...")
    atlas_shakes = probe_ffn_geometry(
        teacher, shakes_corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=N_TOKENS_REFIT, batch_size=BATCH_SIZE, jacobian_samples=4,
    )

    # Log k values
    for layer in atlas_shakes.layer_geometries:
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)
        print(f"    Block {layer.layer_id}: k={k}, hidden={layer.hidden_dim}")

    # ── Step 2: Condition A' — Shakespeare-only at n=2048 ────────────────────
    # Use activations stored in atlas_shakes (collected from the 2048-token holdout)
    print(f"  [seed={seed}] Building Condition A' (Shakespeare 2048 tokens)...")
    refit_Ap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in atlas_shakes.layer_geometries:
        h = layer.hidden_sample.float()
        y = layer.output_sample.float()
        # Truncate to exactly N_TOKENS_REFIT
        refit_Ap[layer.layer_id] = (h[:N_TOKENS_REFIT], y[:N_TOKENS_REFIT])
        print(f"    Block {layer.layer_id}: A' has {min(h.shape[0], N_TOKENS_REFIT)} tokens")

    # ── Step 3: Condition C — Mixed at n=2048 (512 per corpus) ────────────────
    print(f"  [seed={seed}] Collecting activations for Condition C (mixed 2048: 512 per corpus)...")
    block_hidden_C: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {b: {} for b in TARGET_BLOCKS}

    for cname, cpath in CORPUS_PATHS.items():
        # Build token tensor large enough to get 512 tokens
        c_spec = ProbeCorpusSpec(
            teacher=teacher, text_path=cpath,
            sequence_length=SEQ_LEN,
            probe_train_sequences=16, probe_holdout_sequences=4,  # 4 × 128 = 512 holdout tokens
            eval_val_sequences=8,
            dataset_name=cname, shuffle_seed=seed,
        )
        c_corpus = build_probe_corpus(c_spec)
        tok_tensor = c_corpus.probe_holdout  # [4, 129]

        for block_id in TARGET_BLOCKS:
            h512, y512 = collect_hidden_activations(
                teacher, tok_tensor, block_id, n_tokens=N_TOKENS_PER_CORPUS
            )
            block_hidden_C[block_id][cname] = (h512, y512)
            print(f"    Block {block_id}, {cname}: {h512.shape[0]} tokens")

    # Build refit_C dict: concatenate 512 from each of 4 corpora = 2048 total
    refit_C: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_id in TARGET_BLOCKS:
        hs = [block_hidden_C[block_id][c][0] for c in CORPUS_PATHS]
        ys = [block_hidden_C[block_id][c][1] for c in CORPUS_PATHS]
        h_cat = torch.cat(hs, dim=0)
        y_cat = torch.cat(ys, dim=0)
        refit_C[block_id] = (h_cat, y_cat)
        print(f"    Block {block_id}: C total tokens = {h_cat.shape[0]}")

    # ── Step 4: Build packs and evaluate ────────────────────────────────────
    for cond_label, refit_data in [("Ap_shakespeare_2048", refit_Ap), ("C_mixed_2048", refit_C)]:
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

conditions = ["Ap_shakespeare_2048", "C_mixed_2048"]
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

# Verdict: C improves over A' (lower PPL = better)
mean_Ap = agg["Ap_shakespeare_2048"]["mean_ppl"]
mean_C  = agg["C_mixed_2048"]["mean_ppl"]

# Relative improvement: (A' - C) / A' * 100  (positive = C is better)
rel_improvement_pct = (mean_Ap - mean_C) / mean_Ap * 100.0

print(f"\nMean A' PPL:  {mean_Ap:.2f}")
print(f"Mean C  PPL:  {mean_C:.2f}")
print(f"Relative improvement (A' -> C): {rel_improvement_pct:+.1f}%")

if rel_improvement_pct >= 5.0:
    verdict = "confirmed"
    print("Verdict: CONFIRMED — mixed refit reduces PPL by >= 5% relative vs Shakespeare-only")
elif rel_improvement_pct < 3.0:
    verdict = "refuted"
    print("Verdict: REFUTED — mixed refit does not improve by >= 3% relative (within noise)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — improvement {rel_improvement_pct:.1f}% (3-5% range)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

x = list(range(len(conditions)))
colors = ["steelblue", "forestgreen"]
labels = ["A': Shakespeare\n2048 tok", "C: Mixed\n2048 tok\n(512/corpus)"]

# PPL bar chart per seed + mean
seed_ppls = [[r["ppl"] for r in all_results if r["condition"] == c] for c in conditions]
mean_ppls = [agg[c]["mean_ppl"] for c in conditions]
std_ppls  = [agg[c]["std_ppl"]  for c in conditions]

ax1.bar(x, mean_ppls, color=colors, alpha=0.7, width=0.5, yerr=std_ppls, capsize=5)
for i, ppls in enumerate(seed_ppls):
    ax1.scatter([i] * len(ppls), ppls, color="black", s=40, zorder=3)
ax1.axhline(baseline_ppl, ls="--", color="red", label=f"Baseline PPL={baseline_ppl:.1f}")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("WikiText-2 PPL")
ax1.set_title("Compressed PPL: Matched n=2048")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Degradation bar chart
mean_rels = [agg[c]["mean_rel_pct"] for c in conditions]
std_rels  = [agg[c]["std_rel_pct"]  for c in conditions]
seed_rels = [[r["ppl_degradation_rel_pct"] for r in all_results if r["condition"] == c] for c in conditions]

ax2.bar(x, mean_rels, color=colors, alpha=0.7, width=0.5, yerr=std_rels, capsize=5)
for i, rels in enumerate(seed_rels):
    ax2.scatter([i] * len(rels), rels, color="black", s=40, zorder=3)
ax2.axhline(0, ls="--", color="red", alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("PPL Degradation (% relative)")
ax2.set_title("Degradation vs Baseline (matched n=2048)")
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    f"Cycle 078: Mixed-Corpus Matched Refit — GPT-2 (6x, blocks {TARGET_BLOCKS})\n"
    f"Verdict: {verdict.upper()} (rel improvement = {rel_improvement_pct:+.1f}%)",
    fontsize=11,
)
plt.tight_layout()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_png = RESULTS_DIR / "cycle_078_matched_refit.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"\nSaved plot: {out_png}")

out_json = RESULTS_DIR / "cycle_078_matched_refit.json"
with open(out_json, "w") as f:
    json.dump({
        "baseline_ppl": baseline_ppl,
        "results": all_results,
        "aggregate": {c: {k: v for k, v in d.items() if k not in ("ppls", "rels")} for c, d in agg.items()},
        "verdict": verdict,
        "rel_improvement_pct": rel_improvement_pct,
        "mean_Ap_ppl": mean_Ap,
        "mean_C_ppl": mean_C,
        "blocks": TARGET_BLOCKS,
        "compression": COMPRESSION,
        "seeds": SEEDS,
        "n_tokens_refit": N_TOKENS_REFIT,
    }, f, indent=2)
print(f"Saved JSON: {out_json}")
