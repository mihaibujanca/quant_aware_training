"""Cycle 176 / Item 183: Sign-pattern-conditioned lstsq — GPT-2 FFN k=4.

Current lstsq fits ONE linear map per block. The FFN is piecewise-linear
(via GELU): tokens in different sign-pattern regions use different effective
affine maps. This experiment tests whether fitting a SEPARATE lstsq per
k-means cluster of sign patterns improves stacked PPL.

Cycle 167 REFUTED the premise that sign patterns cluster tractably (all 4096
tokens unique, K*>256 in 11/12 blocks). This experiment tests whether forced
k-means clustering helps despite the lack of natural clusters.

Per review R4: use topk-channel sign patterns (512-dim) for clustering,
not full 3072-dim. Cheaper and tests the relevant question.

Configurations:
  K=1   — baseline: single lstsq (standard pipeline)
  K=4   — 4 clusters per block
  K=8   — 8 clusters per block (primary test)
  K=16  — 16 clusters per block

Verdict:
  CONFIRMED    if K=8 piecewise lstsq improves stacked PPL >= 3 points over K=1
  REFUTED      if < 1 PPL improvement
  INCONCLUSIVE if 1-3 PPL improvement
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
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
BLOCKS = [1, 4, 7, 11]
HIDDEN_DIM = 512  # 6x compression (k=4 means 4 blocks)

K_VALUES = [1, 4, 8, 16]
MIN_CLUSTER_SIZE = 64  # merge smaller clusters

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Piecewise MLP module
# ---------------------------------------------------------------------------
class PiecewiseCompressedMLP(nn.Module):
    """Compressed MLP with K piecewise output projections.

    Shared c_fc (topk channels). At forward time:
    1. Compute hidden = activation(c_fc(x))
    2. Compute sign pattern on the topk pre-activations
    3. Route each token to nearest centroid
    4. Apply the corresponding c_proj_k
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation_kind: str,
        n_clusters: int,
        centroids: torch.Tensor,  # [K, hidden_dim] sign-pattern centroids
        *,
        c_fc_bias: bool,
        c_proj_bias: bool,
    ):
        super().__init__()
        self.activation_kind = activation_kind
        self.n_clusters = n_clusters
        self.c_fc = nn.Linear(input_dim, hidden_dim, bias=c_fc_bias)
        # K separate output projections
        self.c_projs = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim, bias=c_proj_bias)
            for _ in range(n_clusters)
        ])
        # Register centroids as buffer (not trainable)
        self.register_buffer("centroids", centroids.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]

        z = self.c_fc(x_flat)  # [N, hidden_dim]
        h = _apply_activation(z, self.activation_kind)  # [N, hidden_dim]

        if self.n_clusters == 1:
            out = self.c_projs[0](h)
        else:
            # Sign pattern routing on topk pre-activations
            signs = (z > 0).float()  # [N, hidden_dim]
            # Assign to nearest centroid (L2 on binary vectors = Hamming)
            # dists[i, k] = ||signs[i] - centroids[k]||^2
            # = ||signs[i]||^2 + ||centroids[k]||^2 - 2 * signs[i] @ centroids[k].T
            # Since signs are binary, ||signs[i]||^2 = signs[i].sum()
            # centroids are cluster means (float in [0,1])
            dists = (
                signs.pow(2).sum(dim=1, keepdim=True)
                + self.centroids.pow(2).sum(dim=1, keepdim=True).t()
                - 2.0 * signs @ self.centroids.t()
            )  # [N, K]
            assignments = dists.argmin(dim=1)  # [N]

            out = torch.zeros(N, self.c_projs[0].out_features,
                              device=x.device, dtype=h.dtype)
            for k in range(self.n_clusters):
                mask = (assignments == k)
                if mask.any():
                    out[mask] = self.c_projs[k](h[mask])

        return out.reshape(*orig_shape, -1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _joint_polish(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    """Joint polish: optimize ALL compressed blocks' c_proj weights simultaneously."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)

    params = []
    for bid in block_ids:
        mlp = _find_compressed_mlp(student_model, bid)
        if mlp is None:
            continue
        # Handle both CompressedMLP (single c_proj) and PiecewiseCompressedMLP (c_projs list)
        if hasattr(mlp, "c_projs"):
            for proj in mlp.c_projs:
                w = proj.weight
                w.requires_grad_(True)
                params.append(w)
                b = getattr(proj, "bias", None)
                if b is not None:
                    b.requires_grad_(True)
                    params.append(b)
        elif hasattr(mlp, "c_proj"):
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


def _kmeans_sign_patterns(
    signs: np.ndarray,
    K: int,
    max_iter: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple k-means on sign patterns (binary vectors cast to float).

    Returns (centroids [K, D], assignments [N]).
    """
    N, D = signs.shape
    rng = np.random.RandomState(seed)

    # Initialize with k-means++
    centroids = np.zeros((K, D), dtype=np.float32)
    idx = rng.randint(N)
    centroids[0] = signs[idx]
    for k in range(1, K):
        dists = np.min(
            np.sum((signs[:, None, :] - centroids[None, :k, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / dists.sum()
        idx = rng.choice(N, p=probs)
        centroids[k] = signs[idx]

    for _ in range(max_iter):
        # Assign
        # dists: [N, K]
        dists = np.sum((signs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = dists.argmin(axis=1)
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0:
                new_centroids[k] = signs[mask].mean(axis=0)
            else:
                # Empty cluster — reinitialize from random point
                new_centroids[k] = signs[rng.randint(N)]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids, assignments


def _build_piecewise_pack(
    atlas: FfnGeometryAtlas,
    block_ids: list[int],
    hidden_dim: int,
    K: int,
    seed: int,
) -> tuple[CompressedFfnPack, dict]:
    """Build compressed pack with piecewise lstsq (K clusters per block).

    For K=1, this is equivalent to the standard single-lstsq pipeline.
    Returns (pack, cluster_stats).
    """
    budget = CompressionBudgetSpec(
        hidden_fraction=hidden_dim / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: hidden_dim for bid in block_ids},
    )
    compressed_layers: list[CompressedFfnLayer] = []
    cluster_stats: dict[int, dict] = {}

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue

        bid = layer.layer_id
        k = hidden_dim
        selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)  # [N, k]
        outputs = layer.output_sample.to(dtype=torch.float32)  # [N, 768]

        # Pre-activations on topk channels for sign patterns
        preact = layer.preactivation_sample[:, selected].to(dtype=torch.float32)  # [N, k]
        signs = (preact > 0).float().numpy()  # [N, k]

        if K == 1:
            # Standard single lstsq
            student_w_out, student_b_out = _fit_output_projection(
                hidden, outputs, use_bias=use_output_bias,
            )
            centroids_t = torch.zeros(1, k, dtype=torch.float32)  # dummy
            cluster_stats[bid] = {"K": 1, "sizes": [int(hidden.shape[0])], "merged": 0}
        else:
            centroids_np, assignments = _kmeans_sign_patterns(signs, K, seed=seed)

            # Merge small clusters
            merged = 0
            cluster_sizes = []
            for c in range(K):
                cluster_sizes.append(int((assignments == c).sum()))

            for c in range(K):
                if cluster_sizes[c] < MIN_CLUSTER_SIZE and cluster_sizes[c] > 0:
                    # Merge into nearest non-tiny cluster
                    dists = np.sum((centroids_np[c] - centroids_np) ** 2, axis=1)
                    dists[c] = np.inf
                    for other in range(K):
                        if cluster_sizes[other] < MIN_CLUSTER_SIZE:
                            dists[other] = np.inf
                    nearest = dists.argmin()
                    assignments[assignments == c] = nearest
                    cluster_sizes[nearest] += cluster_sizes[c]
                    cluster_sizes[c] = 0
                    merged += 1

            # Recompute centroids after merging
            for c in range(K):
                mask = (assignments == c)
                if mask.sum() > 0:
                    centroids_np[c] = signs[mask].mean(axis=0)

            cluster_stats[bid] = {
                "K": K,
                "sizes": [int(s) for s in cluster_sizes],
                "merged": merged,
            }

            # Fit per-cluster lstsq
            # We store all K projections even for empty clusters (will be unused)
            per_cluster_w = []
            per_cluster_b = []
            for c in range(K):
                mask = (assignments == c)
                n_c = mask.sum()
                if n_c == 0:
                    # Empty cluster — use global lstsq as fallback
                    w_c, b_c = _fit_output_projection(
                        hidden, outputs, use_bias=use_output_bias,
                    )
                else:
                    mask_t = torch.from_numpy(mask)
                    h_c = hidden[mask_t]
                    y_c = outputs[mask_t]
                    w_c, b_c = _fit_output_projection(h_c, y_c, use_bias=use_output_bias)
                per_cluster_w.append(w_c)
                per_cluster_b.append(b_c)

            centroids_t = torch.from_numpy(centroids_np).float()

            # For the CompressedFfnLayer, we store the first cluster's projection
            # as c_proj (for compatibility). The real piecewise logic is in the
            # custom module we'll build during swap.
            student_w_out = per_cluster_w[0]
            student_b_out = per_cluster_b[0]

        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj),
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out,
        ) * K  # K projections
        compressed_param_count += centroids_t.numel()  # centroids

        compressed_layers.append(CompressedFfnLayer(
            layer_id=bid,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=k,
            c_fc_weight=student_w_in,
            c_fc_bias=student_b_in,
            c_proj_weight=student_w_out,
            c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={
                "selection_mode": "topk",
                "piecewise_K": K,
                "centroids": centroids_t,
                "per_cluster_w": per_cluster_w if K > 1 else None,
                "per_cluster_b": per_cluster_b if K > 1 else None,
            },
        ))

    return CompressedFfnPack(
        method="geometric",
        model_family="gpt2",
        dataset_name="openwebtext",
        layers=tuple(compressed_layers),
        metadata={"sequence_length": SEQ_LEN, "piecewise_K": K},
    ), cluster_stats


def _swap_piecewise_ffns(
    teacher: TeacherBundle,
    pack: CompressedFfnPack,
) -> nn.Module:
    """Replace FFN modules with piecewise compressed MLPs."""
    model = copy.deepcopy(teacher.model)
    model.eval()

    for layer in pack.layers:
        K = layer.notes.get("piecewise_K", 1)
        block = model.transformer.h[layer.layer_id]

        if K == 1:
            # Standard compressed MLP
            new_mlp = CompressedMLP(
                input_dim=layer.c_fc_weight.shape[1],
                hidden_dim=layer.compressed_hidden_dim,
                output_dim=layer.c_proj_weight.shape[0],
                activation_kind=layer.activation_kind,
                c_fc_bias=layer.c_fc_bias is not None,
                c_proj_bias=layer.c_proj_bias is not None,
            )
            with torch.no_grad():
                new_mlp.c_fc.weight.copy_(layer.c_fc_weight)
                if new_mlp.c_fc.bias is not None and layer.c_fc_bias is not None:
                    new_mlp.c_fc.bias.copy_(layer.c_fc_bias)
                new_mlp.c_proj.weight.copy_(layer.c_proj_weight)
                if new_mlp.c_proj.bias is not None and layer.c_proj_bias is not None:
                    new_mlp.c_proj.bias.copy_(layer.c_proj_bias)
        else:
            centroids = layer.notes["centroids"]
            per_cluster_w = layer.notes["per_cluster_w"]
            per_cluster_b = layer.notes["per_cluster_b"]

            new_mlp = PiecewiseCompressedMLP(
                input_dim=layer.c_fc_weight.shape[1],
                hidden_dim=layer.compressed_hidden_dim,
                output_dim=per_cluster_w[0].shape[0],
                activation_kind=layer.activation_kind,
                n_clusters=K,
                centroids=centroids,
                c_fc_bias=layer.c_fc_bias is not None,
                c_proj_bias=per_cluster_b[0] is not None,
            )
            with torch.no_grad():
                new_mlp.c_fc.weight.copy_(layer.c_fc_weight)
                if new_mlp.c_fc.bias is not None and layer.c_fc_bias is not None:
                    new_mlp.c_fc.bias.copy_(layer.c_fc_bias)
                for c_idx in range(K):
                    new_mlp.c_projs[c_idx].weight.copy_(per_cluster_w[c_idx])
                    if new_mlp.c_projs[c_idx].bias is not None and per_cluster_b[c_idx] is not None:
                        new_mlp.c_projs[c_idx].bias.copy_(per_cluster_b[c_idx])

        device = next(block.mlp.parameters()).device
        new_mlp.to(device)
        block.mlp = new_mlp

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
print(f"  Teacher on device={device}")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

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

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    print(f"  Teacher baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe blocks
    print(f"  Probing FFN geometry for blocks {BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    seed_results: dict = {"baseline_ppl": baseline_ppl, "K_configs": {}}

    for K in K_VALUES:
        print(f"\n  K={K}:")
        _set_seed(seed)

        pack, cluster_stats = _build_piecewise_pack(atlas, BLOCKS, HIDDEN_DIM, K, seed)

        # Print cluster stats
        for bid in BLOCKS:
            cs = cluster_stats[bid]
            nonzero = [s for s in cs["sizes"] if s > 0]
            print(f"    Block {bid}: K={cs['K']}, sizes={nonzero}, merged={cs['merged']}")

        # Unpolished eval
        student_raw = _swap_piecewise_ffns(teacher, pack)
        student_raw.eval()
        m_raw = _evaluate_model(teacher, student_raw, corpus, batch_size=BATCH_SIZE)
        ppl_raw = float(m_raw["perplexity"])
        print(f"    Unpolished PPL: {ppl_raw:.3f}")
        del student_raw

        # Joint polish
        student_polished = _swap_piecewise_ffns(teacher, pack)
        print(f"    Joint polish ({POLISH_STEPS} steps)...")
        losses = _joint_polish(
            student_polished, BLOCKS, corpus.probe_holdout,
            POLISH_STEPS, POLISH_LR, device, seed,
        )
        if losses:
            print(f"      loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        student_polished.eval()
        m_polished = _evaluate_model(teacher, student_polished, corpus, batch_size=BATCH_SIZE)
        ppl_polished = float(m_polished["perplexity"])
        print(f"    Polished PPL: {ppl_polished:.3f}")

        # Param count
        total_params = sum(l.compressed_param_count for l in pack.layers)
        source_params = sum(l.source_param_count for l in pack.layers)

        seed_results["K_configs"][str(K)] = {
            "ppl_unpolished": ppl_raw,
            "ppl_polished": ppl_polished,
            "compressed_params": int(total_params),
            "source_params": int(source_params),
            "compression_ratio": float(source_params / total_params),
            "cluster_stats": {str(bid): cluster_stats[bid] for bid in BLOCKS},
        }
        del student_polished

    all_seed_results[seed] = seed_results


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}")

aggregated: dict[str, dict] = {}
for K in K_VALUES:
    k_str = str(K)
    ppls_raw = [all_seed_results[s]["K_configs"][k_str]["ppl_unpolished"] for s in SEEDS]
    ppls_pol = [all_seed_results[s]["K_configs"][k_str]["ppl_polished"] for s in SEEDS]
    mean_raw = sum(ppls_raw) / len(ppls_raw)
    std_raw = (sum((p - mean_raw)**2 for p in ppls_raw) / len(ppls_raw)) ** 0.5
    mean_pol = sum(ppls_pol) / len(ppls_pol)
    std_pol = (sum((p - mean_pol)**2 for p in ppls_pol) / len(ppls_pol)) ** 0.5
    aggregated[k_str] = {
        "mean_ppl_unpolished": round(mean_raw, 2),
        "std_ppl_unpolished": round(std_raw, 2),
        "mean_ppl_polished": round(mean_pol, 2),
        "std_ppl_polished": round(std_pol, 2),
        "per_seed_unpolished": [round(p, 2) for p in ppls_raw],
        "per_seed_polished": [round(p, 2) for p in ppls_pol],
    }

# Print summary table
baseline_pol = aggregated["1"]["mean_ppl_polished"]
print(f"\n{'K':<6} {'PPL(raw)':<22} {'PPL(polish)':<22} {'Delta vs K=1':>12}")
print("-" * 65)
for K in K_VALUES:
    k_str = str(K)
    agg = aggregated[k_str]
    delta = agg["mean_ppl_polished"] - baseline_pol
    print(f"K={K:<4} "
          f"{agg['mean_ppl_unpolished']:.2f}+/-{agg['std_ppl_unpolished']:.1f}     "
          f"{agg['mean_ppl_polished']:.2f}+/-{agg['std_ppl_polished']:.1f}     "
          f"{delta:+.2f}")

# Param accounting
print(f"\nParam accounting (seed {SEEDS[0]}):")
for K in K_VALUES:
    k_str = str(K)
    info = all_seed_results[SEEDS[0]]["K_configs"][k_str]
    print(f"  K={K}: {info['compressed_params']:,} params "
          f"(source: {info['source_params']:,}, ratio: {info['compression_ratio']:.2f}x)")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
k8_pol = aggregated["8"]["mean_ppl_polished"]
delta_k8 = baseline_pol - k8_pol  # positive = K=8 better

best_K = None
best_delta = 0.0
for K in K_VALUES:
    if K == 1:
        continue
    k_str = str(K)
    delta = baseline_pol - aggregated[k_str]["mean_ppl_polished"]
    if delta > best_delta:
        best_delta = delta
        best_K = K

teacher_ppls = [all_seed_results[s]["baseline_ppl"] for s in SEEDS]
mean_teacher = sum(teacher_ppls) / len(teacher_ppls)

# Verdict based on K=8 specifically (per item criteria)
if delta_k8 >= 3.0:
    verdict = "CONFIRMED"
    verdict_note = f"K=8 PPL {k8_pol:.2f} improves over K=1 PPL {baseline_pol:.2f} by {delta_k8:.1f} points (>= 3)"
elif delta_k8 < 1.0:
    verdict = "REFUTED"
    verdict_note = f"K=8 PPL {k8_pol:.2f} vs K=1 PPL {baseline_pol:.2f}, delta={delta_k8:+.1f} (< 1 improvement)"
else:
    verdict = "INCONCLUSIVE"
    verdict_note = f"K=8 PPL {k8_pol:.2f} vs K=1 PPL {baseline_pol:.2f}, delta={delta_k8:+.1f} (1-3 range)"

if best_K and best_K != 8:
    verdict_note += f" (best K={best_K}, delta={best_delta:+.1f})"

print(f"\nVERDICT: {verdict}")
print(f"  {verdict_note}")
print(f"  Teacher PPL (mean): {mean_teacher:.2f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
result = {
    "cycle": 176,
    "item": 183,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "hidden_dim": HIDDEN_DIM,
    "K_values": K_VALUES,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "eval_sequences": EVAL_SEQS,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "teacher_ppl_mean": round(mean_teacher, 2),
    "teacher_ppl_per_seed": {str(s): round(all_seed_results[s]["baseline_ppl"], 2) for s in SEEDS},
    "aggregated": aggregated,
    "baseline_polished_ppl": round(baseline_pol, 2),
    "k8_polished_ppl": round(k8_pol, 2),
    "k8_delta": round(delta_k8, 2),
    "best_K": best_K,
    "best_delta": round(best_delta, 2),
    "per_seed_detail": {
        str(s): all_seed_results[s] for s in SEEDS
    },
}

out_path = RESULTS_DIR / "cycle_176_sign_lstsq.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: PPL vs K
    ks = K_VALUES
    means_pol = [aggregated[str(k)]["mean_ppl_polished"] for k in ks]
    stds_pol = [aggregated[str(k)]["std_ppl_polished"] for k in ks]
    means_raw = [aggregated[str(k)]["mean_ppl_unpolished"] for k in ks]

    ax = axes[0]
    ax.errorbar(ks, means_pol, yerr=stds_pol, fmt="o-", label="Polished", capsize=4)
    ax.plot(ks, means_raw, "s--", alpha=0.5, label="Unpolished")
    ax.axhline(y=mean_teacher, color="green", linestyle=":", label=f"Teacher ({mean_teacher:.1f})")
    ax.set_xlabel("K (clusters)")
    ax.set_ylabel("Stacked PPL")
    ax.set_title("Sign-pattern piecewise lstsq")
    ax.legend()
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    # Right: Param count vs PPL
    params_by_k = [all_seed_results[SEEDS[0]]["K_configs"][str(k)]["compressed_params"] for k in ks]
    ax = axes[1]
    for i, k in enumerate(ks):
        ax.annotate(f"K={k}", (params_by_k[i], means_pol[i]),
                    textcoords="offset points", xytext=(5, 5))
    ax.plot(params_by_k, means_pol, "o-")
    ax.set_xlabel("Total compressed params")
    ax.set_ylabel("Stacked PPL (polished)")
    ax.set_title("Params vs PPL tradeoff")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_176_sign_lstsq.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()
except Exception as e:
    print(f"Plot failed: {e}")
