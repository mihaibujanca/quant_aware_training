"""Cycle 161 / Item 171: Centroid-affinity channel selection vs activation-norm topk — GPT-2 FFN k=4.

For each block, compute per-channel centroid embeddings in d_model-space:
  c_j = mean_over_tokens(a[:, j].unsqueeze(-1) * h)   shape: (d_model,)
Then use farthest-point sampling to select k=4 channels maximising coverage.
Compare single-block and stacked PPL vs. activation-norm topk baseline.

Criteria:
  CONFIRMED    if centroid PPL < topk PPL by >= 1.5 points stacked
  REFUTED      if centroid within 0.5 of topk, OR Jaccard > 0.75 on all blocks
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _standard_bias,
    _standard_weight,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

SEEDS = [42, 137, 7]
K = 4
BLOCKS = [1, 4, 7, 11]
SEQ_LEN = 128
BATCH_SIZE = 4
POLISH_STEPS = 500
POLISH_LR = 1e-4
N_EVAL_SEQS = 200
D_MODEL = 768

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Centroid affinity channel selection
# ---------------------------------------------------------------------------

def _compute_centroid_embeddings(
    activations: torch.Tensor,   # [N, 3072] post-GELU hidden activations
    hidden_states: torch.Tensor, # [N, 768] input hidden states (x to MLP)
) -> torch.Tensor:
    """Per-channel centroid in d_model-space.

    c_j = mean_over_tokens(a[:, j] * h)   shape: (3072, d_model)
    """
    # activations: [N, C], hidden_states: [N, D]
    # c_j = (1/N) * sum_i a[i,j] * h[i]  = A.T @ H / N
    N = activations.shape[0]
    centroids = (activations.t() @ hidden_states) / N  # [C, D]
    return centroids


def _farthest_point_sampling(vectors: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-point sampling on rows of 'vectors' (shape [C, D]).

    Returns indices of selected k rows.
    Seed = row with largest norm.
    """
    C = vectors.shape[0]
    # Normalise for stable distance computation (keep raw norms for seeding)
    norms = vectors.norm(dim=1)  # [C]
    seed_idx = int(norms.argmax().item())

    selected = [seed_idx]
    # min-distance to selected set, initialised to inf
    min_dists = torch.full((C,), float("inf"), device=vectors.device)

    for _ in range(k - 1):
        last = vectors[selected[-1]].unsqueeze(0)  # [1, D]
        dists_to_last = (vectors - last).norm(dim=1)  # [C]
        min_dists = torch.minimum(min_dists, dists_to_last)
        # Exclude already selected
        for s in selected:
            min_dists[s] = -1.0
        next_idx = int(min_dists.argmax().item())
        selected.append(next_idx)

    return torch.tensor(sorted(selected), dtype=torch.long, device=vectors.device)


def _build_centroid_compression(
    teacher,
    atlas,
    corpus,
    *,
    batch_size: int = 4,
) -> CompressedFfnPack:
    """Build compressed pack using centroid-affinity FPS channel selection."""
    from aleph.lm_geometric_compression import (
        _extract_ffn_handles,
        _apply_activation,
        _sample_tensor,
        _iter_lm_batches,
    )

    # Collect post-GELU activations AND input hidden states per block
    handles_by_id = {
        h.layer_id: h
        for h in _extract_ffn_handles(teacher.model, teacher.activation_kind)
        if h.layer_id in BLOCKS
    }

    max_tokens = 2048
    block_data: dict[int, dict] = {}

    for bid, handle in handles_by_id.items():
        act_samples: list[torch.Tensor] = []
        hid_samples: list[torch.Tensor] = []

        def hook_fn(module, inputs, output,
                    _act_samples=act_samples, _hid_samples=hid_samples):
            x = inputs[0].detach()
            z = module.c_fc(x).detach()
            from aleph.lm_geometric_compression import _apply_activation
            h_act = _apply_activation(z, handle.activation_kind).detach()
            # Flatten [B, S, D] -> [B*S, D]
            x_flat = x.reshape(-1, x.shape[-1])
            h_flat = h_act.reshape(-1, h_act.shape[-1])
            remaining = max_tokens - sum(t.shape[0] for t in _act_samples)
            if remaining <= 0:
                return
            n = min(x_flat.shape[0], remaining)
            _act_samples.append(h_flat[:n].cpu())
            _hid_samples.append(x_flat[:n].cpu())

        hook = handle.mlp.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for input_ids, _ in _iter_lm_batches(
                    corpus.probe_holdout, batch_size, teacher.device
                ):
                    _ = teacher.model(input_ids)
                    if sum(t.shape[0] for t in act_samples) >= max_tokens:
                        break
        finally:
            hook.remove()

        block_data[bid] = {
            "activations": torch.cat(act_samples, dim=0).float(),
            "hidden": torch.cat(hid_samples, dim=0).float(),
        }

    # Build compressed layers
    compressed_layers = []
    atlas_by_id = {g.layer_id: g for g in atlas.layer_geometries}

    for bid in BLOCKS:
        geom = atlas_by_id[bid]
        data = block_data[bid]
        activations = data["activations"]  # [N, 3072]
        hidden = data["hidden"]            # [N, 768]

        centroids = _compute_centroid_embeddings(activations, hidden)  # [3072, 768]
        selected = _farthest_point_sampling(centroids, K)  # [k]
        selected = selected.cpu()

        source_w_in = _standard_weight(geom.handle.c_fc).float()
        source_b_in = _standard_bias(geom.handle.c_fc)
        use_bias_out = _standard_bias(geom.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        # Use hidden_sample from atlas (already collected at inference) for lstsq
        hidden_sub = geom.hidden_sample[:, selected].float()
        student_w_out, student_b_out = _fit_output_projection(
            hidden_sub, geom.output_sample.float(), use_bias=use_bias_out
        )

        compressed_layers.append(
            CompressedFfnLayer(
                layer_id=bid,
                activation_kind=geom.activation_kind,
                source_hidden_dim=geom.hidden_dim,
                compressed_hidden_dim=K,
                c_fc_weight=student_w_in,
                c_fc_bias=student_b_in,
                c_proj_weight=student_w_out,
                c_proj_bias=student_b_out,
                selected_indices=selected,
                source_param_count=0,
                compressed_param_count=0,
                notes={"selection_mode": "centroid_fps"},
            )
        )

    return CompressedFfnPack(
        method="centroid_fps",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


# ---------------------------------------------------------------------------
# Adam polish (only c_proj weights) — matching baseline protocol
# ---------------------------------------------------------------------------

def _polish_c_proj(
    student_model: nn.Module,
    block_ids: list[int],
    tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> None:
    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = student_model.transformer.h[bid].mlp
        if hasattr(mlp, "c_proj"):
            w = mlp.c_proj.weight; w.requires_grad_(True); params.append(w)
            b = getattr(mlp.c_proj, "bias", None)
            if b is not None:
                b.requires_grad_(True); params.append(b)
    if not params:
        return

    optimizer = torch.optim.Adam(params, lr=lr)
    rng = torch.Generator(); rng.manual_seed(seed)
    n_seqs = tokens.shape[0]
    student_model.train()
    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        optimizer.zero_grad()
        raw = student_model(batch[:, :-1])
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"    polish step {step:4d}: loss={loss.item():.4f}", flush=True)
    student_model.eval()
    student_model.requires_grad_(False)


# ---------------------------------------------------------------------------
# Jaccard overlap
# ---------------------------------------------------------------------------

def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(a.tolist())
    sb = set(b.tolist())
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    device = teacher.device

    _set_seed(42)
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=N_EVAL_SEQS,
        probe_holdout_sequences=N_EVAL_SEQS,
        eval_val_sequences=N_EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=42,
    ))

    float_ppl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)["perplexity"]
    print(f"Float PPL: {float_ppl:.3f}", flush=True)

    # Build geometry atlas (shared by both methods)
    print("Building geometry atlas...", flush=True)
    atlas = probe_ffn_geometry(teacher, corpus, layer_ids=tuple(BLOCKS), batch_size=BATCH_SIZE)

    # ---- Topk baseline ----
    print("\nBuilding topk baseline...", flush=True)
    topk_budget = CompressionBudgetSpec(
        per_layer_hidden={bid: K for bid in BLOCKS},
        selection_mode="topk",
    )
    topk_pack = derive_ffn_compression(atlas, topk_budget)
    topk_selected = {layer.layer_id: layer.selected_indices for layer in topk_pack.layers}

    # ---- Centroid-affinity selection ----
    print("\nBuilding centroid-affinity selection...", flush=True)
    centroid_pack = _build_centroid_compression(teacher, atlas, corpus, batch_size=BATCH_SIZE)
    centroid_selected = {layer.layer_id: layer.selected_indices for layer in centroid_pack.layers}

    # Print selected channels and Jaccard
    print("\nSelected channels per block:", flush=True)
    jaccard_scores: dict[int, float] = {}
    for bid in BLOCKS:
        t = topk_selected[bid]; c = centroid_selected[bid]
        j = _jaccard(t, c)
        jaccard_scores[bid] = j
        print(f"  Block {bid:2d}  topk={sorted(t.tolist())}  centroid={sorted(c.tolist())}  Jaccard={j:.3f}", flush=True)

    # ---- Per-seed eval: single-block + stacked ----
    all_results = {}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)
        _set_seed(seed)

        # --- Topk ---
        topk_model = swap_compressed_ffns(teacher, topk_pack, inplace=False)

        # Single-block PPL (block-by-block)
        topk_single_ppl: dict[int, float] = {}
        for bid in BLOCKS:
            single_budget = CompressionBudgetSpec(
                per_layer_hidden={bid: K},
                selection_mode="topk",
            )
            single_pack_layers = [l for l in topk_pack.layers if l.layer_id == bid]
            single_pack = CompressedFfnPack(
                method="geometric",
                model_family=topk_pack.model_family,
                dataset_name=topk_pack.dataset_name,
                layers=tuple(single_pack_layers),
                metadata=topk_pack.metadata,
            )
            m = swap_compressed_ffns(teacher, single_pack, inplace=False)
            _polish_c_proj(m, [bid], corpus.probe_train, POLISH_STEPS, POLISH_LR, device, seed)
            ppl = _evaluate_model(teacher, m, corpus, batch_size=BATCH_SIZE)["perplexity"]
            topk_single_ppl[bid] = ppl
            print(f"  topk single block {bid}: PPL={ppl:.3f}", flush=True)
            del m

        # Stacked PPL with polish
        _polish_c_proj(topk_model, BLOCKS, corpus.probe_train, POLISH_STEPS, POLISH_LR, device, seed)
        topk_stacked = _evaluate_model(teacher, topk_model, corpus, batch_size=BATCH_SIZE)["perplexity"]
        print(f"  topk stacked PPL: {topk_stacked:.3f}", flush=True)
        del topk_model

        # --- Centroid ---
        centroid_single_ppl: dict[int, float] = {}
        for bid in BLOCKS:
            single_pack_layers = [l for l in centroid_pack.layers if l.layer_id == bid]
            single_pack = CompressedFfnPack(
                method="centroid_fps",
                model_family=centroid_pack.model_family,
                dataset_name=centroid_pack.dataset_name,
                layers=tuple(single_pack_layers),
                metadata=centroid_pack.metadata,
            )
            m = swap_compressed_ffns(teacher, single_pack, inplace=False)
            _polish_c_proj(m, [bid], corpus.probe_train, POLISH_STEPS, POLISH_LR, device, seed)
            ppl = _evaluate_model(teacher, m, corpus, batch_size=BATCH_SIZE)["perplexity"]
            centroid_single_ppl[bid] = ppl
            print(f"  centroid single block {bid}: PPL={ppl:.3f}", flush=True)
            del m

        centroid_model = swap_compressed_ffns(teacher, centroid_pack, inplace=False)
        _polish_c_proj(centroid_model, BLOCKS, corpus.probe_train, POLISH_STEPS, POLISH_LR, device, seed)
        centroid_stacked = _evaluate_model(teacher, centroid_model, corpus, batch_size=BATCH_SIZE)["perplexity"]
        print(f"  centroid stacked PPL: {centroid_stacked:.3f}", flush=True)
        del centroid_model

        all_results[seed] = {
            "topk_single_ppl": {str(k): v for k, v in topk_single_ppl.items()},
            "topk_stacked_ppl": topk_stacked,
            "centroid_single_ppl": {str(k): v for k, v in centroid_single_ppl.items()},
            "centroid_stacked_ppl": centroid_stacked,
        }

    # ---- Aggregate ----
    topk_stacked_ppls = [all_results[s]["topk_stacked_ppl"] for s in SEEDS]
    centroid_stacked_ppls = [all_results[s]["centroid_stacked_ppl"] for s in SEEDS]

    mean_topk = float(np.mean(topk_stacked_ppls))
    std_topk = float(np.std(topk_stacked_ppls))
    mean_centroid = float(np.mean(centroid_stacked_ppls))
    std_centroid = float(np.std(centroid_stacked_ppls))

    all_jaccard_all_blocks = all(j > 0.75 for j in jaccard_scores.values())
    ppl_delta = mean_topk - mean_centroid  # positive = centroid better

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"  Float PPL:              {float_ppl:.3f}", flush=True)
    print(f"  Topk stacked PPL:       {mean_topk:.3f} ± {std_topk:.3f}", flush=True)
    print(f"  Centroid stacked PPL:   {mean_centroid:.3f} ± {std_centroid:.3f}", flush=True)
    print(f"  Delta (topk-centroid):  {ppl_delta:.3f}  (+ve = centroid wins)", flush=True)
    print(f"  Jaccard per block:      {jaccard_scores}", flush=True)
    print(f"  All Jaccard > 0.75:     {all_jaccard_all_blocks}", flush=True)

    # Verdict
    if ppl_delta >= 1.5:
        verdict = "CONFIRMED"
        detail = f"Centroid better by {ppl_delta:.2f} PPL (>= 1.5 threshold)"
    elif ppl_delta <= 0.5 or all_jaccard_all_blocks:
        verdict = "REFUTED"
        if all_jaccard_all_blocks:
            detail = f"All Jaccard > 0.75 — methods equivalent; delta={ppl_delta:.2f}"
        else:
            detail = f"Centroid within {abs(ppl_delta):.2f} of topk (threshold 0.5)"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Delta={ppl_delta:.2f}, in [0.5,1.5] range; Jaccard={jaccard_scores}"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail:  {detail}", flush=True)

    # ---- Save JSON ----
    output = {
        "cycle": 161,
        "item": 171,
        "seeds": SEEDS,
        "float_ppl": float_ppl,
        "k": K,
        "blocks": BLOCKS,
        "polish_steps": POLISH_STEPS,
        "jaccard_per_block": {str(k): v for k, v in jaccard_scores.items()},
        "all_jaccard_gt_075": all_jaccard_all_blocks,
        "topk_selected": {str(k): sorted(v.tolist()) for k, v in topk_selected.items()},
        "centroid_selected": {str(k): sorted(v.tolist()) for k, v in centroid_selected.items()},
        "topk_stacked_ppl_mean": mean_topk,
        "topk_stacked_ppl_std": std_topk,
        "centroid_stacked_ppl_mean": mean_centroid,
        "centroid_stacked_ppl_std": std_centroid,
        "ppl_delta_topk_minus_centroid": ppl_delta,
        "per_seed": {
            str(s): all_results[s] for s in SEEDS
        },
        "verdict": verdict,
        "detail": detail,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "cycle_161_centroid_affinity.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}", flush=True)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cycle 161: Centroid-Affinity vs Topk Channel Selection (k=4, GPT-2)")

    # Single-block PPL per block (averaged over seeds)
    topk_single_means = []
    centroid_single_means = []
    for bid in BLOCKS:
        t_vals = [all_results[s]["topk_single_ppl"][str(bid)] for s in SEEDS]
        c_vals = [all_results[s]["centroid_single_ppl"][str(bid)] for s in SEEDS]
        topk_single_means.append(np.mean(t_vals))
        centroid_single_means.append(np.mean(c_vals))

    x = np.arange(len(BLOCKS))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, topk_single_means, w, label="topk", color="steelblue")
    ax.bar(x + w/2, centroid_single_means, w, label="centroid-FPS", color="coral")
    ax.set_xticks(x); ax.set_xticklabels([f"Block {b}" for b in BLOCKS])
    ax.set_ylabel("PPL"); ax.set_title("Single-block PPL (mean over seeds)")
    ax.legend()

    # Stacked PPL per seed
    ax2 = axes[1]
    xs = np.arange(len(SEEDS))
    ax2.bar(xs - w/2, topk_stacked_ppls, w, label="topk", color="steelblue")
    ax2.bar(xs + w/2, centroid_stacked_ppls, w, label="centroid-FPS", color="coral")
    ax2.set_xticks(xs); ax2.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax2.set_ylabel("PPL"); ax2.set_title(f"Stacked PPL — topk={mean_topk:.1f}±{std_topk:.1f}  centroid={mean_centroid:.1f}±{std_centroid:.1f}")
    ax2.legend()

    plt.tight_layout()
    png_path = RESULTS_DIR / "cycle_161_centroid_affinity.png"
    plt.savefig(png_path, dpi=120)
    print(f"Saved: {png_path}", flush=True)

    print("\nDone.", flush=True)
