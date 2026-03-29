"""Cycle 098: Joint FFN+attention compression — GPT-2 (item 113).

Tests 4 conditions at 3 seeds [42, 123, 7] on OWT 200 eval seqs:
  Baseline: unmodified GPT-2
  A: FFN-only, blocks [7,5,4,1] at 6× topk+lstsq + 500 Adam polish on W_down
  B: Attn-only, blocks [1,4,9] rank-11 per-head SVD on QKV
  C: Joint A+B simultaneously (FFN polish on W_down only, attn SVD fixed)

Also computes k=2 subset: FFN blocks [7,5] + attn blocks [1,4] for refutation check.

Interaction = delta_C - (delta_A + delta_B)
  positive = super-additive (amplifying)
  ~0       = orthogonal
  negative = sub-additive (cancellation)

Verdict:
  CONFIRMED    if delta_joint_k4 <= +20%
  REFUTED      if delta_joint_k2 > +20%
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

# FFN compression settings
FFN_BLOCKS = [7, 5, 4, 1]       # Full k=4 set
FFN_BLOCKS_K2 = [7, 5]          # k=2 subset for refutation check
HIDDEN_FRACTION = 1 / 6          # 6x compression
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

# Attention compression settings
ATTN_BLOCKS = [1, 4, 9]         # Full k=3 viable set (cycle 095)
ATTN_BLOCKS_K2 = [1, 4]         # k=2 subset for refutation check
RANK = 11
N_HEADS = 12
D_HEAD = 64
D_MODEL = 768

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: torch.Tensor | None) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _std_weight(module: nn.Module) -> torch.Tensor:
    """Return (out_features, in_features) weight, handling Conv1D."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _compress_attn_block_inplace(block: nn.Module, rank: int) -> None:
    """Compress all 12 heads' Q, K, V projections in one attention block (rank-r SVD)."""
    attn = block.attn
    W_full = _std_weight(attn.c_attn)  # (2304, 768)
    W_new = W_full.clone()
    for proj_idx in range(3):  # 0=Q, 1=K, 2=V
        base = proj_idx * D_MODEL
        for h in range(N_HEADS):
            row_start = base + h * D_HEAD
            row_end = row_start + D_HEAD
            W_h = W_full[row_start:row_end, :]  # (64, 768)
            U, S, Vt = torch.linalg.svd(W_h, full_matrices=False)
            W_new[row_start:row_end, :] = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    if attn.c_attn.__class__.__name__ == "Conv1D":
        attn.c_attn.weight.data.copy_(W_new.t())
    else:
        attn.c_attn.weight.data.copy_(W_new)


def _derive_block_pack_from_atlas(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_ids: list[int],
) -> CompressedFfnPack:
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)
        student_w_out, student_b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)

        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj)
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out
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
                notes={"selection_mode": budget.selection_mode},
            )
        )
    return CompressedFfnPack(
        method="geometric",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _polish_block(
    student_model: nn.Module,
    block_id: int,
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    if n_steps == 0:
        return []
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        print(f"  WARNING: no c_proj for block {block_id}, skipping polish")
        return []

    student_model.requires_grad_(False)
    w_down = mlp.c_proj.weight
    w_down.requires_grad_(True)
    b_down = getattr(mlp.c_proj, "bias", None)
    if b_down is not None:
        b_down.requires_grad_(True)

    params_to_optimize = [w_down] + ([b_down] if b_down is not None else [])
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


def _extract_polished_weights(student_model: nn.Module, block_id: int) -> dict:
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return {}
    return {
        "c_proj_weight": mlp.c_proj.weight.detach().clone(),
        "c_proj_bias": (
            mlp.c_proj.bias.detach().clone()
            if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None
            else None
        ),
    }


def _inject_polished_weights(student_model: nn.Module, block_id: int, polished: dict) -> None:
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return
    with torch.no_grad():
        mlp.c_proj.weight.copy_(polished["c_proj_weight"])
        if polished["c_proj_bias"] is not None and hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None:
            mlp.c_proj.bias.copy_(polished["c_proj_bias"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")

budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    min_hidden=32,
    selection_mode="topk",
)

# ---------------------------------------------------------------------------
# Per-seed loop: build corpus, probe atlas, polish FFN blocks, run 5 conditions
# ---------------------------------------------------------------------------
# Conditions: baseline, A (FFN k=4), B (attn k=3), C (joint k=4+k=3),
#             A_k2 (FFN k=2), C_k2 (joint k=2)
per_seed: dict[int, dict] = {}

ALL_FFN_BLOCKS = list(set(FFN_BLOCKS))

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    # Baseline
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe FFN geometry
    print(f"  Probing FFN geometry for blocks {sorted(ALL_FFN_BLOCKS)}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(ALL_FFN_BLOCKS)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    polish_tokens = corpus.probe_holdout

    # Per-block FFN polish
    print(f"  Per-block polish ({POLISH_STEPS} steps)...")
    polished_weights: dict[int, dict] = {}
    for block_id in ALL_FFN_BLOCKS:
        _set_seed(seed)
        single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
        single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_student.eval()
        losses = _polish_block(single_student, block_id, polish_tokens, POLISH_STEPS, POLISH_LR, teacher.device)
        if losses:
            print(f"    block {block_id}: polish {losses[0]:.4f} → {losses[-1]:.4f}")
        polished_weights[block_id] = _extract_polished_weights(single_student, block_id)
        del single_student

    def _run_condition(ffn_blocks: list[int], attn_blocks: list[int], label: str) -> float:
        """Build student with given FFN and/or attn compression, eval PPL."""
        _set_seed(seed)
        if ffn_blocks:
            pack = _derive_block_pack_from_atlas(atlas, budget, ffn_blocks)
            student = swap_compressed_ffns(teacher, pack, inplace=False)
            student.eval()
            for bid in ffn_blocks:
                if bid in polished_weights and polished_weights[bid]:
                    _inject_polished_weights(student, bid, polished_weights[bid])
        else:
            student = copy.deepcopy(teacher.model)
            student.eval()

        for bid in attn_blocks:
            _compress_attn_block_inplace(student.transformer.h[bid], rank=RANK)

        metrics = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        ppl = float(metrics["perplexity"])
        delta = (ppl - baseline_ppl) / baseline_ppl * 100.0
        print(f"  {label}: PPL={ppl:.3f} ({delta:+.2f}%)")

        if ppl > 10 * baseline_ppl:
            print(f"ABORT: {label} PPL={ppl:.1f} > 10× baseline")
            sys.exit(1)
        del student
        return ppl

    ppl_A = _run_condition(FFN_BLOCKS, [], "A (FFN k=4)")
    ppl_B = _run_condition([], ATTN_BLOCKS, "B (Attn k=3)")
    ppl_C = _run_condition(FFN_BLOCKS, ATTN_BLOCKS, "C (Joint k=4+k=3)")
    ppl_A_k2 = _run_condition(FFN_BLOCKS_K2, [], "A_k2 (FFN k=2)")
    ppl_C_k2 = _run_condition(FFN_BLOCKS_K2, ATTN_BLOCKS_K2, "C_k2 (Joint k=2+k=2)")

    per_seed[seed] = {
        "baseline": baseline_ppl,
        "A": ppl_A,
        "B": ppl_B,
        "C": ppl_C,
        "A_k2": ppl_A_k2,
        "C_k2": ppl_C_k2,
    }

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

def _agg(key: str) -> tuple[float, float]:
    vals = [per_seed[s][key] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))

mean_baseline, std_baseline = _agg("baseline")
mean_A, std_A = _agg("A")
mean_B, std_B = _agg("B")
mean_C, std_C = _agg("C")
mean_A_k2, std_A_k2 = _agg("A_k2")
mean_C_k2, std_C_k2 = _agg("C_k2")

delta_A = (mean_A - mean_baseline) / mean_baseline * 100.0
delta_B = (mean_B - mean_baseline) / mean_baseline * 100.0
delta_C = (mean_C - mean_baseline) / mean_baseline * 100.0
delta_A_k2 = (mean_A_k2 - mean_baseline) / mean_baseline * 100.0
delta_C_k2 = (mean_C_k2 - mean_baseline) / mean_baseline * 100.0

additive_prediction = mean_A + mean_B - mean_baseline
additive_delta = (additive_prediction - mean_baseline) / mean_baseline * 100.0
interaction = mean_C - additive_prediction
interaction_pct = (interaction / mean_baseline) * 100.0

print(f"Baseline:          PPL={mean_baseline:.3f} ± {std_baseline:.3f}")
print(f"A (FFN k=4):       PPL={mean_A:.3f} ± {std_A:.3f}  delta={delta_A:+.2f}%")
print(f"B (Attn k=3):      PPL={mean_B:.3f} ± {std_B:.3f}  delta={delta_B:+.2f}%")
print(f"C (Joint):         PPL={mean_C:.3f} ± {std_C:.3f}  delta={delta_C:+.2f}%")
print(f"A_k2 (FFN k=2):    PPL={mean_A_k2:.3f} ± {std_A_k2:.3f}  delta={delta_A_k2:+.2f}%")
print(f"C_k2 (Joint k=2):  PPL={mean_C_k2:.3f} ± {std_C_k2:.3f}  delta={delta_C_k2:+.2f}%")
print(f"Additive pred:     PPL={additive_prediction:.3f}  delta={additive_delta:+.2f}%")
print(f"Interaction:       {interaction:+.3f} PPL ({interaction_pct:+.2f}%)")

if interaction_pct > 1.0:
    interaction_type = "super_additive"
elif interaction_pct < -1.0:
    interaction_type = "sub_additive"
else:
    interaction_type = "orthogonal"

# Verdict
if delta_C <= 20.0:
    verdict = "CONFIRMED"
elif delta_C_k2 > 20.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nInteraction type: {interaction_type}")
print(f"Verdict: {verdict}")
print(f"  (delta_joint_k4={delta_C:+.2f}%, delta_joint_k2={delta_C_k2:+.2f}%)")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 98,
    "item": 113,
    "seeds": SEEDS,
    "ffn_blocks_k4": FFN_BLOCKS,
    "ffn_blocks_k2": FFN_BLOCKS_K2,
    "attn_blocks": ATTN_BLOCKS,
    "attn_blocks_k2": ATTN_BLOCKS_K2,
    "attn_rank": RANK,
    "ffn_hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "condition_A_ffn_k4": {"mean": mean_A, "std": std_A, "delta_pct": delta_A},
    "condition_B_attn_k3": {"mean": mean_B, "std": std_B, "delta_pct": delta_B},
    "condition_C_joint_k4": {"mean": mean_C, "std": std_C, "delta_pct": delta_C},
    "condition_A_k2_ffn": {"mean": mean_A_k2, "std": std_A_k2, "delta_pct": delta_A_k2},
    "condition_C_k2_joint": {"mean": mean_C_k2, "std": std_C_k2, "delta_pct": delta_C_k2},
    "additive_prediction_ppl": additive_prediction,
    "additive_prediction_delta_pct": additive_delta,
    "interaction_ppl": interaction,
    "interaction_pct": interaction_pct,
    "interaction_type": interaction_type,
    "verdict": verdict,
    "per_seed": {
        str(s): per_seed[s] for s in SEEDS
    },
}

json_path = RESULTS_DIR / "cycle_098_joint_compression.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plot: grouped bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

labels_plot = ["Baseline", "A: FFN k=4", "B: Attn k=3", "Additive\nprediction", "C: Joint k=4+k=3"]
means_plot = [mean_baseline, mean_A, mean_B, additive_prediction, mean_C]
stds_plot = [std_baseline, std_A, std_B, 0.0, std_C]
colors_plot = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E", "#E91E63"]

x = np.arange(len(labels_plot))
bars = ax.bar(x, means_plot, color=colors_plot, alpha=0.85, width=0.6,
              yerr=stds_plot, capsize=5, error_kw={"elinewidth": 1.5})

# Add delta% annotations
for i, (m, b, std) in enumerate(zip(means_plot, bars, stds_plot)):
    if i == 0:
        label_txt = "ref"
    else:
        d = (m - mean_baseline) / mean_baseline * 100.0
        label_txt = f"{d:+.1f}%"
    ax.text(b.get_x() + b.get_width() / 2, m + std + 0.3,
            label_txt, ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(mean_baseline, color="green", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Baseline {mean_baseline:.2f}")
ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=10)
ax.set_ylabel("Perplexity (OWT, 200 seqs)", fontsize=11)
ax.set_title(
    f"Joint FFN+Attention Compression — GPT-2\n"
    f"Verdict: {verdict} | Interaction: {interaction_pct:+.2f}% ({interaction_type})\n"
    f"FFN blocks {FFN_BLOCKS} (6×) + Attn blocks {ATTN_BLOCKS} (rank-{RANK})",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.set_ylim(bottom=mean_baseline * 0.96)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

png_path = RESULTS_DIR / "cycle_098_joint_compression.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved chart: {png_path}")

print(f"\n=== DONE ===")
print(f"Verdict: {verdict}")
print(f"delta_A (FFN k=4):    {delta_A:+.2f}%")
print(f"delta_B (Attn k=3):   {delta_B:+.2f}%")
print(f"delta_C (Joint k=4):  {delta_C:+.2f}%")
print(f"delta_C_k2 (Joint k=2): {delta_C_k2:+.2f}%")
print(f"Additive prediction:  {additive_delta:+.2f}%")
print(f"Interaction:          {interaction_pct:+.2f}% ({interaction_type})")
