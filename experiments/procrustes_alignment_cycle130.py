"""Cycle 130: Cross-block Procrustes alignment — GPT-2 FFN k=4 (item 87).

Compute pairwise Procrustes distances on polished W_down (c_proj_weight)
from k=4 compressed blocks [7,5,4,1]. If any pair < 10%, build shared-weight model.

Verdict:
  CONFIRMED    if >=1 pair < 10% AND shared PPL within 2pp of independent
  REFUTED      if ALL pairs > 30%
  INCONCLUSIVE if low residual but shared PPL degrades
"""
from __future__ import annotations

import copy
import itertools
import json
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
from scipy.linalg import orthogonal_procrustes

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
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# Config
SEED = 42
FFN_BLOCKS = [7, 5, 4, 1]
D_MODEL = 768
HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
BATCH_SIZE = 4
SEQ_LEN = 128
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_SEQS = 200

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_count(w, b=None):
    return w.numel() + (0 if b is None else b.numel())


def _derive_block_pack_from_atlas(atlas, budget, block_ids):
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
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out)
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=int(k),
            c_fc_weight=student_w_in, c_fc_bias=student_b_in,
            c_proj_weight=student_w_out, c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_mode": budget.selection_mode},
        ))
    return CompressedFfnPack(
        method="geometric", model_family=atlas.model_family,
        dataset_name=atlas.dataset_name, layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


def _find_compressed_mlp(model, block_id):
    try:
        return model.transformer.h[block_id].mlp
    except (AttributeError, IndexError):
        return None


def _polish_block(student_model, block_id, polish_tokens, n_steps, lr, device):
    if n_steps == 0:
        return []
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return []
    student_model.requires_grad_(False)
    w = mlp.c_proj.weight
    w.requires_grad_(True)
    b = getattr(mlp.c_proj, "bias", None)
    if b is not None:
        b.requires_grad_(True)
    params = [w] + ([b] if b is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    rng = torch.Generator()
    rng.manual_seed(42)
    for step in range(n_steps):
        idx = torch.randperm(polish_tokens.shape[0], generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inp = batch[:, :-1].contiguous()
        tgt = batch[:, 1:].contiguous()
        opt.zero_grad()
        raw = student_model(inp)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    student_model.eval()
    student_model.requires_grad_(False)
    return losses


def _extract_polished_weights(model, block_id):
    mlp = _find_compressed_mlp(model, block_id)
    if mlp is None:
        return {}
    return {
        "c_proj_weight": mlp.c_proj.weight.detach().clone(),
        "c_proj_bias": (mlp.c_proj.bias.detach().clone()
                        if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None else None),
    }


def _inject_polished_weights(model, block_id, polished):
    mlp = _find_compressed_mlp(model, block_id)
    if mlp is None:
        return
    with torch.no_grad():
        mlp.c_proj.weight.copy_(polished["c_proj_weight"])
        if polished["c_proj_bias"] is not None and hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None:
            mlp.c_proj.bias.copy_(polished["c_proj_bias"])


def procrustes_distance(W_i: np.ndarray, W_j: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Procrustes distance: ||W_i - W_j @ R||_F / ||W_i||_F.
    W_i, W_j: (768, 512). R: (512, 512) orthogonal.
    """
    R, scale = orthogonal_procrustes(W_j, W_i)  # Find R s.t. W_j @ R ≈ W_i
    residual = np.linalg.norm(W_i - W_j @ R, 'fro')
    norm_i = np.linalg.norm(W_i, 'fro')
    return residual / norm_i, R


if __name__ == "__main__":
    _set_seed(SEED)

    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher, text_path=str(OWT_PATH), sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS, probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS, dataset_name="openwebtext", shuffle_seed=SEED,
    ))

    # Baseline
    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl['perplexity']:.3f}", flush=True)

    # Compress
    print(f"\nCompressing blocks {FFN_BLOCKS}...", flush=True)
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(sorted(FFN_BLOCKS)),
                                max_tokens_per_layer=MAX_TOKENS_PER_LAYER, batch_size=BATCH_SIZE)

    # Per-block polish
    polished_weights = {}
    for bid in FFN_BLOCKS:
        _set_seed(SEED)
        pack = _derive_block_pack_from_atlas(atlas, budget, [bid])
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.eval()
        losses = _polish_block(student, bid, corpus.probe_holdout, POLISH_STEPS, POLISH_LR, teacher.device)
        if losses:
            print(f"  block {bid}: polish {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        polished_weights[bid] = _extract_polished_weights(student, bid)
        del student

    # Build full compressed model
    _set_seed(SEED)
    full_pack = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
    compressed = swap_compressed_ffns(teacher, full_pack, inplace=False)
    compressed.eval()
    for bid in FFN_BLOCKS:
        if polished_weights[bid]:
            _inject_polished_weights(compressed, bid, polished_weights[bid])

    unc = _evaluate_model(teacher, compressed, corpus, batch_size=BATCH_SIZE)
    print(f"  Independent k=4 PPL: {unc['perplexity']:.3f}", flush=True)

    # Extract W_down matrices
    print(f"\nExtracting W_down matrices...", flush=True)
    w_downs = {}
    for bid in FFN_BLOCKS:
        w = polished_weights[bid]["c_proj_weight"]
        w_downs[bid] = w.numpy()
        print(f"  Block {bid}: W_down shape {w.shape}, ||W||_F = {np.linalg.norm(w.numpy(), 'fro'):.4f}", flush=True)

    # Pairwise Procrustes
    print(f"\n{'='*50}", flush=True)
    print("PAIRWISE PROCRUSTES DISTANCES", flush=True)
    print(f"{'='*50}", flush=True)

    pairs = list(itertools.combinations(sorted(FFN_BLOCKS), 2))
    distances = {}
    rotations = {}
    for i, j in pairs:
        dist, R = procrustes_distance(w_downs[i], w_downs[j])
        distances[(i, j)] = dist
        rotations[(i, j)] = R
        print(f"  ({i}, {j}): {dist*100:.2f}%", flush=True)

    min_pair = min(distances, key=distances.get)
    min_dist = distances[min_pair]
    max_dist = max(distances.values())
    print(f"\n  Min: {min_pair} = {min_dist*100:.2f}%", flush=True)
    print(f"  Max: {max_dist*100:.2f}%", flush=True)

    # Verdict logic
    all_above_30 = all(d > 0.30 for d in distances.values())
    any_below_10 = any(d < 0.10 for d in distances.values())

    shared_ppl = None
    if all_above_30:
        verdict = "REFUTED"
        detail = f"All pairwise residuals > 30% (min={min_dist*100:.1f}%). Blocks are functionally dissimilar."
    elif any_below_10:
        # Build shared-weight variant
        i_star, j_star = min_pair
        R_best = rotations[min_pair]
        print(f"\n  Building shared-weight model ({i_star},{j_star})...", flush=True)

        shared_model = copy.deepcopy(compressed)
        # Use W_i* as shared, rotate for j*
        W_shared = w_downs[i_star]
        W_rotated = (W_shared @ R_best.T).astype(np.float32)

        with torch.no_grad():
            shared_model.transformer.h[j_star].mlp.c_proj.weight.copy_(torch.from_numpy(W_rotated))

        sh = _evaluate_model(teacher, shared_model, corpus, batch_size=BATCH_SIZE)
        shared_ppl = sh["perplexity"]
        ppl_delta = abs(shared_ppl - unc["perplexity"])
        print(f"  Shared PPL: {shared_ppl:.3f} (delta: {ppl_delta:.3f})", flush=True)

        if ppl_delta <= 2.0:
            verdict = "CONFIRMED"
            detail = f"Pair ({i_star},{j_star}) residual={min_dist*100:.1f}%, shared PPL delta={ppl_delta:.2f}pp"
        else:
            verdict = "INCONCLUSIVE"
            detail = f"Pair ({i_star},{j_star}) residual={min_dist*100:.1f}% but shared PPL degrades by {ppl_delta:.2f}pp"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Min residual={min_dist*100:.1f}% (10-30% range). Moderate similarity."

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save JSON
    output = {
        "cycle": 130, "item": 87,
        "float_ppl": bl["perplexity"],
        "independent_ppl": unc["perplexity"],
        "shared_ppl": shared_ppl,
        "distances": {f"{i}-{j}": d for (i,j), d in distances.items()},
        "min_pair": list(min_pair),
        "min_distance": min_dist,
        "max_distance": max_dist,
        "all_above_30": all_above_30,
        "any_below_10": any_below_10,
        "verdict": verdict,
        "verdict_detail": detail,
    }
    with open(RESULTS_DIR / "cycle_130_procrustes.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON", flush=True)

    # Plot: heatmap
    block_ids = sorted(FFN_BLOCKS)
    n = len(block_ids)
    mat = np.zeros((n, n))
    for (i, j), d in distances.items():
        ii, jj = block_ids.index(i), block_ids.index(j)
        mat[ii, jj] = d * 100
        mat[jj, ii] = d * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"B{b}" for b in block_ids])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"B{b}" for b in block_ids])
    for ii in range(n):
        for jj in range(n):
            ax.text(jj, ii, f"{mat[ii,jj]:.1f}%", ha="center", va="center", fontsize=10,
                    color="white" if mat[ii,jj] > 50 else "black")
    ax.set_title(f"Procrustes Residual (%) — k=4 W_down\nVerdict: {verdict}")
    fig.colorbar(im, ax=ax, label="Residual (%)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cycle_130_heatmap.png", dpi=120)
    plt.close()
    print("Saved plot", flush=True)
    print("\nDone.", flush=True)
