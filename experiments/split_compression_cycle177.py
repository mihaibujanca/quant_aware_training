"""Cycle 177 / Item 187: SoLA-style split compression — GPT-2 FFN k=4.

Instead of topk+lstsq (keep 512 top channels, drop rest), keep 512 top channels
full-rank AND SVD-compress the remaining 2560 channels at rank r. Fit lstsq from
[h_topk, z_tail_r] -> output, where z_tail_r = h_tail @ V_r is the rank-r
projection of tail hidden activations.

At inference, the SplitCompressedMLP keeps the full c_fc (3072x768) to compute
all hidden activations, then splits into topk (full) + tail (projected) before
the output projection.

Sweep: r in {0, 2, 4, 8, 16} where r=0 is the standard topk+lstsq baseline.

Verdict:
  CONFIRMED    if best-r stacked PPL <= 68.7 (>= 2 PPL improvement over ~70.7)
  REFUTED      if all r configs PPL >= 72.7
  INCONCLUSIVE if best-r within +/-2 of topk+lstsq
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _apply_activation,
    _evaluate_model,
    _extract_ffn_handles,
    _fit_output_projection,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
POLISH_STEPS = 500
POLISH_LR = 1e-4

GPT2_FFN_HIDDEN = 3072
GPT2_MODEL_DIM = 768
BLOCKS = [1, 4, 7, 11]
TOPK_HIDDEN = 512  # top-k channels (k=4 groups x 128 channels)

RANK_VALUES = [0, 2, 4, 8, 16]  # r=0 is standard topk+lstsq baseline

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Split Compressed MLP
# ---------------------------------------------------------------------------
class SplitCompressedMLP(nn.Module):
    """Compressed MLP: full c_fc, topk channels + rank-r tail projection."""

    def __init__(
        self,
        c_fc_weight: torch.Tensor,
        c_fc_bias: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        tail_indices: torch.Tensor,
        V_r: torch.Tensor,
        c_proj_weight: torch.Tensor,
        c_proj_bias: Optional[torch.Tensor],
        activation_kind: str = "gelu",
    ):
        super().__init__()
        self.activation_kind = activation_kind
        self.c_fc = nn.Linear(c_fc_weight.shape[1], c_fc_weight.shape[0],
                              bias=c_fc_bias is not None)
        with torch.no_grad():
            self.c_fc.weight.copy_(c_fc_weight)
            if c_fc_bias is not None:
                self.c_fc.bias.copy_(c_fc_bias)

        combined_dim = topk_indices.shape[0] + V_r.shape[1]
        self.c_proj = nn.Linear(combined_dim, c_proj_weight.shape[0],
                                bias=c_proj_bias is not None)
        with torch.no_grad():
            self.c_proj.weight.copy_(c_proj_weight)
            if c_proj_bias is not None:
                self.c_proj.bias.copy_(c_proj_bias)

        self.register_buffer("topk_indices", topk_indices.long())
        self.register_buffer("tail_indices", tail_indices.long())
        self.register_buffer("V_r", V_r.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        z = self.c_fc(x_flat)
        h = _apply_activation(z, self.activation_kind)
        h_topk = h[:, self.topk_indices]
        z_tail = h[:, self.tail_indices] @ self.V_r
        combined = torch.cat([h_topk, z_tail], dim=1)
        y = self.c_proj(combined)
        return y.reshape(*orig_shape, -1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count_split(topk_dim, tail_dim, rank, model_dim=768, hidden_dim=3072):
    c_fc = hidden_dim * model_dim + hidden_dim
    V_r = tail_dim * rank
    c_proj = model_dim * (topk_dim + rank) + model_dim
    return {"c_fc": c_fc, "V_r": V_r, "c_proj": c_proj, "total": c_fc + V_r + c_proj}


def _param_count_topk(topk_dim, model_dim=768):
    c_fc = topk_dim * model_dim + topk_dim
    c_proj = model_dim * topk_dim + model_dim
    return {"c_fc": c_fc, "c_proj": c_proj, "total": c_fc + c_proj}


def _precompute_block_data(atlas, block_ids, topk_dim):
    """Pre-compute per-block data: topk selection, tail SVD, etc. Cached across ranks."""
    budget = CompressionBudgetSpec(
        hidden_fraction=topk_dim / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: topk_dim for bid in block_ids},
    )

    block_cache = {}
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue

        bid = layer.layer_id
        selected = _select_hidden_units(layer, budget, topk_dim)

        all_indices = torch.arange(layer.hidden_dim)
        tail_mask = torch.ones(layer.hidden_dim, dtype=torch.bool)
        tail_mask[selected] = False
        tail_indices = all_indices[tail_mask]

        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        h_topk = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        h_tail = layer.hidden_sample[:, tail_indices].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)

        # Compute SVD once for this block
        h_tail_centered = h_tail - h_tail.mean(dim=0, keepdim=True)
        P(f"    Block {bid}: computing SVD of h_tail ({h_tail.shape})...")
        _, S, Vh = torch.linalg.svd(h_tail_centered, full_matrices=False)
        P(f"    Block {bid}: SVD done. Top-16 singular values: "
          f"{S[:16].tolist()[:4]}...")

        # Total tail variance
        total_var = h_tail_centered.var(dim=0).sum().item()

        block_cache[bid] = {
            "selected": selected,
            "tail_indices": tail_indices,
            "source_w_in": source_w_in,
            "source_b_in": source_b_in,
            "use_output_bias": use_output_bias,
            "h_topk": h_topk,
            "h_tail": h_tail,
            "outputs": outputs,
            "Vh": Vh,  # full Vh, truncate per rank
            "S": S,
            "total_var": total_var,
            "activation_kind": layer.activation_kind,
        }

    return block_cache


def _build_layers_for_rank(block_cache, block_ids, rank):
    """Build layer data for a specific rank, using cached SVDs."""
    layer_data_list = []
    stats = {}

    for bid in block_ids:
        bc = block_cache[bid]
        h_topk = bc["h_topk"]
        h_tail = bc["h_tail"]
        outputs = bc["outputs"]

        if rank == 0:
            student_w_out, student_b_out = _fit_output_projection(
                h_topk, outputs, use_bias=bc["use_output_bias"],
            )
            V_r = torch.zeros(bc["tail_indices"].shape[0], 1)
            combined_features = h_topk
            tail_explained = 0.0
        else:
            V_r = bc["Vh"][:rank, :].T  # [2560, r]
            z_tail = h_tail @ V_r  # [N, r]
            combined_features = torch.cat([h_topk, z_tail], dim=1)
            student_w_out, student_b_out = _fit_output_projection(
                combined_features, outputs, use_bias=bc["use_output_bias"],
            )
            projected_var = z_tail.var(dim=0).sum().item()
            tail_explained = projected_var / max(bc["total_var"], 1e-12)

        # Fit quality
        with torch.no_grad():
            if bc["use_output_bias"] and student_b_out is not None:
                predicted = combined_features @ student_w_out.t() + student_b_out
            else:
                predicted = combined_features @ student_w_out.t()
            mse = F.mse_loss(predicted, outputs).item()
            cosine = F.cosine_similarity(
                predicted.reshape(predicted.shape[0], -1),
                outputs.reshape(outputs.shape[0], -1),
                dim=1,
            ).mean().item()

        layer_data_list.append({
            "layer_id": bid,
            "activation_kind": bc["activation_kind"],
            "source_w_in": bc["source_w_in"],
            "source_b_in": bc["source_b_in"],
            "topk_indices": bc["selected"],
            "tail_indices": bc["tail_indices"],
            "V_r": V_r if rank > 0 else torch.zeros(bc["tail_indices"].shape[0], 1),
            "c_proj_weight": student_w_out,
            "c_proj_bias": student_b_out,
            "rank": rank,
        })

        stats[bid] = {
            "lstsq_mse": round(mse, 6),
            "lstsq_cosine": round(cosine, 6),
            "tail_explained_var": round(tail_explained, 4),
            "n_topk": int(h_topk.shape[1]),
            "n_tail": int(bc["tail_indices"].shape[0]),
        }

    return layer_data_list, stats


def _swap_split_ffns(teacher, layer_data_list, rank):
    """Replace FFN modules with split compressed MLPs."""
    model = copy.deepcopy(teacher.model)
    model.eval()

    for ld in layer_data_list:
        bid = ld["layer_id"]
        block = model.transformer.h[bid]

        if rank == 0:
            new_mlp = CompressedMLP(
                input_dim=ld["source_w_in"].shape[1],
                hidden_dim=ld["topk_indices"].shape[0],
                output_dim=ld["c_proj_weight"].shape[0],
                activation_kind=ld["activation_kind"],
                c_fc_bias=ld["source_b_in"] is not None,
                c_proj_bias=ld["c_proj_bias"] is not None,
            )
            with torch.no_grad():
                new_mlp.c_fc.weight.copy_(ld["source_w_in"][ld["topk_indices"]])
                if new_mlp.c_fc.bias is not None and ld["source_b_in"] is not None:
                    new_mlp.c_fc.bias.copy_(ld["source_b_in"][ld["topk_indices"]])
                new_mlp.c_proj.weight.copy_(ld["c_proj_weight"])
                if new_mlp.c_proj.bias is not None and ld["c_proj_bias"] is not None:
                    new_mlp.c_proj.bias.copy_(ld["c_proj_bias"])
        else:
            new_mlp = SplitCompressedMLP(
                c_fc_weight=ld["source_w_in"],
                c_fc_bias=ld["source_b_in"],
                topk_indices=ld["topk_indices"],
                tail_indices=ld["tail_indices"],
                V_r=ld["V_r"],
                c_proj_weight=ld["c_proj_weight"],
                c_proj_bias=ld["c_proj_bias"],
                activation_kind=ld["activation_kind"],
            )

        device = next(block.mlp.parameters()).device
        new_mlp.to(device)
        block.mlp = new_mlp

    return model


def _find_compressed_mlp(model, block_id):
    try:
        return model.transformer.h[block_id].mlp
    except (AttributeError, IndexError):
        return None


def _joint_polish(student_model, block_ids, polish_tokens, n_steps, lr, device, seed):
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_compressed_mlp(student_model, bid)
        if mlp is None:
            continue
        if hasattr(mlp, "c_proj"):
            w = mlp.c_proj.weight
            w.requires_grad_(True)
            params.append(w)
            b = getattr(mlp.c_proj, "bias", None)
            if b is not None:
                b.requires_grad_(True)
                params.append(b)

    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
P("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
P(f"  Teacher on device={device}")

# Parameter accounting
P("\nParameter accounting per block:")
for r in RANK_VALUES:
    if r == 0:
        pc = _param_count_topk(TOPK_HIDDEN)
        P(f"  r=0 (topk+lstsq): {pc['total']:,} params")
    else:
        pc = _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, r)
        P(f"  r={r} (split): {pc['total']:,} params "
          f"(c_fc={pc['c_fc']:,}, V_r={pc['V_r']:,}, c_proj={pc['c_proj']:,})")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

for seed in SEEDS:
    P(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
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

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    P(f"  Teacher baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        P(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe blocks — one SVD per block, cached
    P(f"  Probing FFN geometry for blocks {BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    P("  Pre-computing SVD for tail channels (once per block)...")
    block_cache = _precompute_block_data(atlas, BLOCKS, TOPK_HIDDEN)

    seed_results: dict = {"baseline_ppl": baseline_ppl, "rank_configs": {}}

    for rank in RANK_VALUES:
        P(f"\n  rank={rank}:")

        layer_data, stats = _build_layers_for_rank(block_cache, BLOCKS, rank)

        for bid in BLOCKS:
            s = stats[bid]
            P(f"    Block {bid}: mse={s['lstsq_mse']:.4f}, "
              f"cos={s['lstsq_cosine']:.4f}, "
              f"tail_var={s['tail_explained_var']:.3f}")

        # Build model, eval unpolished, then polish same model
        P(f"    Building compressed model...")
        student = _swap_split_ffns(teacher, layer_data, rank)
        student.eval()
        m_raw = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        ppl_raw = float(m_raw["perplexity"])
        P(f"    Unpolished PPL: {ppl_raw:.3f}")

        # Polish the same model (mutates in place)
        P(f"    Joint polish ({POLISH_STEPS} steps)...")
        losses = _joint_polish(
            student, BLOCKS, corpus.probe_holdout,
            POLISH_STEPS, POLISH_LR, device, seed,
        )
        if losses:
            P(f"      loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        student.eval()
        m_polished = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        ppl_polished = float(m_polished["perplexity"])
        P(f"    Polished PPL: {ppl_polished:.3f}")

        if rank == 0:
            pc = _param_count_topk(TOPK_HIDDEN)
        else:
            pc = _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, rank)

        seed_results["rank_configs"][str(rank)] = {
            "ppl_unpolished": ppl_raw,
            "ppl_polished": ppl_polished,
            "per_block_params": pc["total"],
            "total_params_4blocks": pc["total"] * len(BLOCKS),
            "per_block_stats": {str(bid): stats[bid] for bid in BLOCKS},
            "kl": float(m_polished.get("teacher_student_kl", 0)),
            "agreement": float(m_polished.get("top1_agreement", 0)),
        }
        del student
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_seed_results[seed] = seed_results

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
P(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}")

aggregated: dict[str, dict] = {}
for rank in RANK_VALUES:
    r_str = str(rank)
    ppls_raw = [all_seed_results[s]["rank_configs"][r_str]["ppl_unpolished"] for s in SEEDS]
    ppls_pol = [all_seed_results[s]["rank_configs"][r_str]["ppl_polished"] for s in SEEDS]
    mean_raw = sum(ppls_raw) / len(ppls_raw)
    std_raw = (sum((p - mean_raw)**2 for p in ppls_raw) / len(ppls_raw)) ** 0.5
    mean_pol = sum(ppls_pol) / len(ppls_pol)
    std_pol = (sum((p - mean_pol)**2 for p in ppls_pol) / len(ppls_pol)) ** 0.5
    aggregated[r_str] = {
        "mean_ppl_unpolished": round(mean_raw, 2),
        "std_ppl_unpolished": round(std_raw, 2),
        "mean_ppl_polished": round(mean_pol, 2),
        "std_ppl_polished": round(std_pol, 2),
        "per_seed_unpolished": [round(p, 2) for p in ppls_raw],
        "per_seed_polished": [round(p, 2) for p in ppls_pol],
    }

# Print summary table
baseline_topk_pol = aggregated["0"]["mean_ppl_polished"]
baseline_topk_raw = aggregated["0"]["mean_ppl_unpolished"]
teacher_ppls = [all_seed_results[s]["baseline_ppl"] for s in SEEDS]
mean_teacher = sum(teacher_ppls) / len(teacher_ppls)

P(f"\nTeacher PPL (mean): {mean_teacher:.2f}")
P(f"\n{'Rank':<8} {'PPL(raw)':<22} {'PPL(polish)':<22} "
  f"{'Delta vs r=0':>12} {'Params/blk':>12}")
P("-" * 80)
for rank in RANK_VALUES:
    r_str = str(rank)
    agg = aggregated[r_str]
    delta = agg["mean_ppl_polished"] - baseline_topk_pol
    if rank == 0:
        pc = _param_count_topk(TOPK_HIDDEN)
    else:
        pc = _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, rank)
    P(f"r={rank:<6} "
      f"{agg['mean_ppl_unpolished']:.2f}+/-{agg['std_ppl_unpolished']:.1f}     "
      f"{agg['mean_ppl_polished']:.2f}+/-{agg['std_ppl_polished']:.1f}     "
      f"{delta:+.2f}       "
      f"{pc['total']:>10,}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
best_rank = None
best_ppl = float("inf")
for rank in RANK_VALUES:
    if rank == 0:
        continue
    ppl = aggregated[str(rank)]["mean_ppl_polished"]
    if ppl < best_ppl:
        best_ppl = ppl
        best_rank = rank

topk_unpolished = aggregated["0"]["mean_ppl_unpolished"]
best_unpolished = aggregated[str(best_rank)]["mean_ppl_unpolished"]
topk_polished = aggregated["0"]["mean_ppl_polished"]
best_polished = aggregated[str(best_rank)]["mean_ppl_polished"]

delta_unpolished = topk_unpolished - best_unpolished
delta_polished = topk_polished - best_polished

if delta_unpolished >= 2.0:
    verdict = "CONFIRMED"
    verdict_note = (f"r={best_rank} unpolished PPL {best_unpolished:.2f} improves over "
                    f"topk+lstsq {topk_unpolished:.2f} by {delta_unpolished:.1f} points (>= 2). "
                    f"Polished: {best_polished:.2f} vs {topk_polished:.2f} ({delta_polished:+.1f})")
elif all(aggregated[str(r)]["mean_ppl_unpolished"] >= topk_unpolished
         for r in RANK_VALUES if r > 0):
    verdict = "REFUTED"
    verdict_note = (f"All split configs >= topk+lstsq baseline {topk_unpolished:.2f}. "
                    f"Best: r={best_rank} at {best_unpolished:.2f} ({delta_unpolished:+.1f}). "
                    f"Extra tail info provides no PPL benefit despite 3.5x more params.")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"Best r={best_rank}: unpolished {best_unpolished:.2f} vs "
                    f"topk+lstsq {topk_unpolished:.2f} ({delta_unpolished:+.1f}). "
                    f"Polished: {best_polished:.2f} vs {topk_polished:.2f} ({delta_polished:+.1f}). "
                    f"Within +/-2 PPL.")

P(f"\nVERDICT: {verdict}")
P(f"  {verdict_note}")

pc_topk = _param_count_topk(TOPK_HIDDEN)
pc_split_best = _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, best_rank)
param_ratio = pc_split_best["total"] / pc_topk["total"]
P(f"  Param ratio: split(r={best_rank}) / topk = {param_ratio:.2f}x "
  f"({pc_split_best['total']:,} vs {pc_topk['total']:,} per block)")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
result = {
    "cycle": 177,
    "item": 187,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "topk_hidden": TOPK_HIDDEN,
    "rank_values": RANK_VALUES,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "eval_sequences": EVAL_SEQS,
    "teacher_ppl_mean": round(mean_teacher, 2),
    "teacher_ppl_per_seed": {str(s): round(all_seed_results[s]["baseline_ppl"], 2) for s in SEEDS},
    "aggregated": aggregated,
    "baseline_topk_unpolished": round(topk_unpolished, 2),
    "baseline_topk_polished": round(topk_polished, 2),
    "best_rank": best_rank,
    "best_ppl_unpolished": round(best_unpolished, 2),
    "best_ppl_polished": round(best_polished, 2),
    "delta_unpolished": round(delta_unpolished, 2),
    "delta_polished": round(delta_polished, 2),
    "param_ratio_best": round(param_ratio, 2),
    "param_counts_per_block": {
        str(r): (_param_count_topk(TOPK_HIDDEN) if r == 0
                 else _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, r))
        for r in RANK_VALUES
    },
    "per_seed_detail": {
        str(s): all_seed_results[s] for s in SEEDS
    },
}

out_path = RESULTS_DIR / "cycle_177_split_compression.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
P(f"\nResults saved to {out_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ranks = RANK_VALUES
    means_pol = [aggregated[str(r)]["mean_ppl_polished"] for r in ranks]
    stds_pol = [aggregated[str(r)]["std_ppl_polished"] for r in ranks]
    means_raw = [aggregated[str(r)]["mean_ppl_unpolished"] for r in ranks]
    stds_raw = [aggregated[str(r)]["std_ppl_unpolished"] for r in ranks]

    ax = axes[0]
    ax.errorbar(ranks, means_pol, yerr=stds_pol, fmt="o-", label="Polished", capsize=4)
    ax.errorbar(ranks, means_raw, yerr=stds_raw, fmt="s--", alpha=0.5,
                label="Unpolished", capsize=4)
    ax.axhline(y=mean_teacher, color="green", linestyle=":", label=f"Teacher ({mean_teacher:.1f})")
    ax.set_xlabel("SVD rank r (0 = topk+lstsq)")
    ax.set_ylabel("Stacked PPL")
    ax.set_title("Split compression: PPL vs tail rank")
    ax.legend(fontsize=8)
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)

    params_by_r = []
    for r in ranks:
        if r == 0:
            params_by_r.append(_param_count_topk(TOPK_HIDDEN)["total"] * len(BLOCKS))
        else:
            params_by_r.append(
                _param_count_split(TOPK_HIDDEN, GPT2_FFN_HIDDEN - TOPK_HIDDEN, r)["total"] * len(BLOCKS))

    ax = axes[1]
    for i, r in enumerate(ranks):
        ax.annotate(f"r={r}", (params_by_r[i], means_pol[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(params_by_r, means_pol, "o-")
    ax.axhline(y=mean_teacher, color="green", linestyle=":", alpha=0.5)
    ax.set_xlabel("Total params (4 blocks)")
    ax.set_ylabel("Stacked PPL (polished)")
    ax.set_title("Params vs PPL")
    ax.grid(True, alpha=0.3)

    avg_cosines = []
    avg_mses = []
    for r in ranks:
        r_str = str(r)
        cos_vals = []
        mse_vals = []
        for s in SEEDS:
            per_block = all_seed_results[s]["rank_configs"][r_str].get("per_block_stats", {})
            for bid_str, st in per_block.items():
                cos_vals.append(st["lstsq_cosine"])
                mse_vals.append(st["lstsq_mse"])
        avg_cosines.append(sum(cos_vals) / max(len(cos_vals), 1))
        avg_mses.append(sum(mse_vals) / max(len(mse_vals), 1))

    ax = axes[2]
    ax2 = ax.twinx()
    l1 = ax.plot(ranks, avg_cosines, "o-b", label="Cosine sim")
    l2 = ax2.plot(ranks, avg_mses, "s--r", alpha=0.7, label="MSE")
    ax.set_xlabel("SVD rank r")
    ax.set_ylabel("lstsq cosine similarity", color="b")
    ax2.set_ylabel("lstsq MSE", color="r")
    ax.set_title("Local fit quality vs rank")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_177_split_compression.png"
    plt.savefig(plot_path, dpi=150)
    P(f"Plot saved to {plot_path}")
    plt.close()
except Exception as e:
    P(f"Plot failed: {e}")
