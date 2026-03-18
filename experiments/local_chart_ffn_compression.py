"""Cycle 029: Local-chart analytic FFN compression vs global geometric baseline.

For each of 3 representative GPT-2 blocks (2=medium, 5=easy, 11=hard) at 8x compression:
  - Global geometric: top-384 channels by activity score + lstsq refit W_down
  - Local-chart K=4: 4 input-space clusters, 96 channels/chart, chart-local selection+refit
  - Local-chart K=8: 8 input-space clusters, 48 channels/chart, chart-local selection+refit

Confirmed if: local-chart improves PPL over geometric by >=20% on >=3 blocks.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _apply_activation,
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

SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

BLOCKS = (2, 5, 11)
K_GLOBAL = 384          # 8x compression: 3072 -> 384
K_CHART_VALUES = (4, 8) # number of local charts
BATCH_SIZE = 4
MAX_TOKENS = 2048
KMEANS_SEED = 42
KMEANS_INITS = 20


# ---------------------------------------------------------------------------
# LocalChartMLP: mixture-of-experts FFN using chart-specific compressed MLPs
# ---------------------------------------------------------------------------

class LocalChartMLP(nn.Module):
    """Routes each token to its nearest centroid, applies chart-local compressed FFN.

    Budget parity with global: K charts each with k_per_chart = k_global // K channels.
    Total parameters ≈ K * (k_per_chart * input_dim + k_per_chart * output_dim) + K * input_dim
                      = k_global * (input_dim + output_dim) + K * input_dim
    The centroid overhead (K * 768 floats) is negligible at K=8.
    """

    def __init__(
        self,
        centroids: torch.Tensor,               # [K, input_dim]
        w_up_list: list[torch.Tensor],          # K × [k_per_chart, input_dim]
        b_up_list: list[torch.Tensor | None],   # K × [k_per_chart] or None
        w_down_list: list[torch.Tensor],        # K × [output_dim, k_per_chart]
        b_down: torch.Tensor | None,            # [output_dim] shared bias from c_proj
        activation_kind: str,
    ):
        super().__init__()
        self.activation_kind = activation_kind
        K = len(w_up_list)

        self.register_buffer("centroids", centroids)  # [K, 768]

        # Store chart weights as parameters (flattened into parameter lists)
        self.w_up_list = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w in w_up_list])
        self.b_up_list = nn.ParameterList([
            nn.Parameter(b, requires_grad=False) if b is not None else nn.Parameter(torch.zeros(1), requires_grad=False)
            for b in b_up_list
        ])
        self.has_b_up = [b is not None for b in b_up_list]
        self.w_down_list = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w in w_down_list])

        if b_down is not None:
            self.register_buffer("b_down", b_down)
        else:
            self.b_down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, input_dim] or [B, T, input_dim] — handle both
        orig_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x
        N = x_flat.shape[0]
        output = torch.zeros(N, self.w_down_list[0].shape[0], device=x.device, dtype=x.dtype)

        # Assign each token to nearest centroid
        dists = torch.cdist(x_flat.float(), self.centroids.float())  # [N, K]
        chart_ids = dists.argmin(dim=1)  # [N]

        K = len(self.w_up_list)
        for k in range(K):
            mask = chart_ids == k
            if not mask.any():
                continue
            x_k = x_flat[mask]  # [N_k, input_dim]
            z_k = x_k @ self.w_up_list[k].t()
            if self.has_b_up[k]:
                z_k = z_k + self.b_up_list[k]
            h_k = _apply_activation(z_k, self.activation_kind)
            out_k = h_k @ self.w_down_list[k].t()
            output[mask] = out_k.to(output.dtype)

        if self.b_down is not None:
            output = output + self.b_down

        if x.ndim == 3:
            output = output.reshape(B, T, -1)
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_kmeans(
    x: torch.Tensor,
    K: int,
    n_init: int = 20,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (centroids [K, D], labels [N])."""
    torch.manual_seed(seed)
    N, D = x.shape
    best_centroids = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Random init: pick K random points
        idx = torch.randperm(N)[:K]
        centroids = x[idx].clone()

        for _iter in range(max_iter):
            dists = torch.cdist(x.float(), centroids.float())  # [N, K]
            labels = dists.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                members = x[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            if torch.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = 0.0
        dists = torch.cdist(x.float(), centroids.float())
        labels = dists.argmin(dim=1)
        for k in range(K):
            members = x[labels == k]
            if len(members) > 0:
                inertia += float(((members - centroids[k]) ** 2).sum().item())

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.clone()

    dists = torch.cdist(x.float(), best_centroids.float())
    labels = dists.argmin(dim=1)
    return best_centroids, labels


def build_local_chart_module(
    geom: FfnLayerGeometry,
    K: int,
    k_global: int,
) -> tuple[LocalChartMLP, float, int]:
    """Build a LocalChartMLP for one block.

    Returns (module, block_mse, total_params).
    """
    k_per_chart = k_global // K
    if k_per_chart < 1:
        k_per_chart = 1

    x = geom.input_sample.float()   # [N, 768]
    y = geom.output_sample.float()  # [N, 768]
    z = geom.preactivation_sample.float()  # [N, 3072]
    h = geom.hidden_sample.float()  # [N, 3072]

    w_up = _standard_weight(geom.handle.c_fc).float()   # [3072, 768]
    b_up = _standard_bias(geom.handle.c_fc)
    if b_up is not None:
        b_up = b_up.float()
    b_down_orig = _standard_bias(geom.handle.c_proj)
    if b_down_orig is not None:
        b_down_orig = b_down_orig.float().clone()
    output_norms = geom.output_norms.float()  # [3072]

    print(f"    k-means K={K}, k_per_chart={k_per_chart} ...", flush=True)
    centroids, labels = _simple_kmeans(x, K=K, n_init=KMEANS_INITS, seed=KMEANS_SEED)

    w_up_list = []
    b_up_list = []
    w_down_list = []

    for k in range(K):
        mask = labels == k
        n_k = int(mask.sum().item())
        if n_k == 0:
            # Empty chart: use global top channels
            fallback = geom.channel_scores.topk(k_per_chart).indices.sort().values
            w_up_list.append(w_up[fallback].contiguous())
            b_up_list.append(b_up[fallback].contiguous() if b_up is not None else None)
            # Zero output projection (won't be called at inference on real data)
            w_down_list.append(torch.zeros(y.shape[1], k_per_chart))
            continue

        x_k = x[mask]  # [n_k, 768]
        h_k = h[mask]  # [n_k, 3072]
        y_k = y[mask]  # [n_k, 768]

        # Chart-local channel scoring
        mean_h_k = h_k.mean(dim=0)         # [3072]
        std_h_k = h_k.std(dim=0)           # [3072]
        deriv_k = torch.sigmoid(z[mask])   # GELU deriv approx: σ(z) * (1 + z*(1-σ(z)))
        # Simplified: use h_k activations directly as the signal
        dead_k = h_k.abs().amax(dim=0) <= 1e-6
        scores_k = std_h_k * output_norms * (mean_h_k.abs() + 1e-6)
        scores_k[dead_k] = -1.0

        # Top-k_per_chart channels for this chart
        n_live = int((~dead_k).sum().item())
        k_eff = min(k_per_chart, max(n_live, 1))
        selected_k = scores_k.topk(k_eff).indices.sort().values

        # Pad if needed (shouldn't happen with valid probes)
        if k_eff < k_per_chart:
            # Duplicate last selected channel to pad
            pad_count = k_per_chart - k_eff
            selected_k = torch.cat([selected_k, selected_k[-1:].expand(pad_count)])

        w_up_k = w_up[selected_k].contiguous()              # [k_per_chart, 768]
        b_up_k = b_up[selected_k].contiguous() if b_up is not None else None

        # lstsq refit for W_down within this chart
        h_k_sub = h_k[:, selected_k]  # [n_k, k_per_chart]
        # Remove bias contribution from y_k (shared b_down)
        y_k_adj = y_k
        if b_down_orig is not None:
            y_k_adj = y_k - b_down_orig.unsqueeze(0)  # center target

        use_bias = False  # b_down is shared; no per-chart bias in W_down
        w_down_k, _ = _fit_output_projection(h_k_sub, y_k_adj, use_bias=use_bias)
        # w_down_k: [768, k_per_chart]

        w_up_list.append(w_up_k)
        b_up_list.append(b_up_k)
        w_down_list.append(w_down_k)

    module = LocalChartMLP(
        centroids=centroids,
        w_up_list=w_up_list,
        b_up_list=b_up_list,
        w_down_list=w_down_list,
        b_down=b_down_orig,
        activation_kind=geom.activation_kind,
    )

    # Compute block MSE on probe data
    module.eval()
    with torch.no_grad():
        pred = module(x)
        block_mse = float(F.mse_loss(pred, y).item())

    # Count parameters (centroids not counted as "compression params")
    total_params = sum(w.numel() for w in w_up_list)
    total_params += sum(b.numel() for b in b_up_list if b is not None)
    total_params += sum(w.numel() for w in w_down_list)
    if b_down_orig is not None:
        total_params += b_down_orig.numel()

    return module, block_mse, total_params


def swap_local_chart(teacher, geom: FfnLayerGeometry, module: LocalChartMLP) -> nn.Module:
    """Deep copy teacher model and replace block geom.layer_id's mlp with module."""
    model = copy.deepcopy(teacher.model)
    model.eval()

    # Navigate to the right block
    # GPT-2 structure: model.transformer.h[i].mlp
    target_block = None
    for name, m in model.named_modules():
        # Check if this module corresponds to the right block's MLP
        pass

    # Direct navigation
    if hasattr(model, "transformer"):
        block_mlp = model.transformer.h[geom.layer_id]
        block_mlp.mlp = module.to(next(teacher.model.parameters()).device)
    elif hasattr(model, "layers"):
        model.layers[geom.layer_id].mlp = module.to(next(teacher.model.parameters()).device)
    else:
        raise RuntimeError("Unknown model structure for FFN swap")

    return model


# ---------------------------------------------------------------------------
# Global geometric baseline (thin wrapper around existing API)
# ---------------------------------------------------------------------------

def build_global_geometric(geom: FfnLayerGeometry, atlas: FfnGeometryAtlas, k: int) -> tuple[float, int]:
    """Returns (block_mse, param_count) for global geometric at k channels."""
    from aleph.lm_geometric_compression import CompressedMLP, _load_compressed_layer

    budget = CompressionBudgetSpec(
        hidden_fraction=k / geom.hidden_dim,
        min_hidden=k,
        selection_mode="topk",
    )
    pack = derive_ffn_compression(atlas, budget)
    layer = pack.layers[0]

    mlp = CompressedMLP(
        input_dim=layer.c_fc_weight.shape[1],
        hidden_dim=layer.compressed_hidden_dim,
        output_dim=layer.c_proj_weight.shape[0],
        activation_kind=layer.activation_kind,
        c_fc_bias=layer.c_fc_bias is not None,
        c_proj_bias=layer.c_proj_bias is not None,
    )
    _load_compressed_layer(mlp, layer)
    with torch.no_grad():
        pred = mlp(geom.input_sample.float())
        mse = float(F.mse_loss(pred, geom.output_sample.float()).item())

    return mse, layer.compressed_param_count, pack


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    shakes_path = SHAKESPEARE_PATH
    if not Path(shakes_path).exists():
        for candidate in [
            Path(__file__).parent.parent / "data" / "input.txt",
            Path(__file__).parent.parent / "data" / "tinyshakespeare.txt",
        ]:
            if candidate.exists():
                shakes_path = str(candidate)
                break
        else:
            raise FileNotFoundError(f"Shakespeare corpus not found at {shakes_path}")

    print("Loading GPT-2 teacher ...", flush=True)
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="auto",
        )
    )

    print(f"Building probe corpus from {shakes_path} ...", flush=True)
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            sequence_length=128,
            probe_train_sequences=64,
            probe_holdout_sequences=32,
            eval_val_sequences=32,
            text_path=shakes_path,
            teacher=teacher,
        )
    )

    print("Computing baseline PPL ...", flush=True)
    baseline_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.2f}", flush=True)

    results: dict = {
        "baseline_ppl": baseline_ppl,
        "baseline_metrics": baseline_metrics,
        "k_global": K_GLOBAL,
        "blocks": BLOCKS,
        "cells": [],
    }

    for block_id in BLOCKS:
        print(f"\n=== Block {block_id} ===", flush=True)

        atlas = probe_ffn_geometry(
            teacher,
            corpus,
            layer_ids=(block_id,),
            max_tokens_per_layer=MAX_TOKENS,
            batch_size=BATCH_SIZE,
        )
        geom = atlas.layer_geometries[0]
        print(f"  Probe: {geom.input_sample.shape[0]} tokens, hidden_dim={geom.hidden_dim}", flush=True)

        # --- Global geometric baseline ---
        print(f"  [global] top-{K_GLOBAL} channels + lstsq refit ...", flush=True)
        geo_mse, geo_params, geo_pack = build_global_geometric(geom, atlas, K_GLOBAL)
        student_geo = swap_compressed_ffns(teacher, geo_pack, inplace=False)
        geo_eval = _evaluate_model(teacher, student_geo, corpus, batch_size=BATCH_SIZE)
        geo_ppl = geo_eval["perplexity"]
        print(f"  [global] MSE={geo_mse:.4f}  PPL={geo_ppl:.2f}  ΔPPL={geo_ppl-baseline_ppl:+.2f} ({(geo_ppl-baseline_ppl)/baseline_ppl*100:+.1f}%)", flush=True)
        results["cells"].append({
            "block_id": block_id,
            "method": "global_geometric",
            "K": 1,
            "k_per_chart": K_GLOBAL,
            "block_mse": geo_mse,
            "param_count": geo_params,
            "ppl": geo_ppl,
            "delta_ppl": geo_ppl - baseline_ppl,
            "delta_ppl_pct": (geo_ppl - baseline_ppl) / baseline_ppl * 100,
        })

        # --- Local-chart variants ---
        for K in K_CHART_VALUES:
            k_per_chart = K_GLOBAL // K
            print(f"  [local K={K}] {k_per_chart} channels/chart ...", flush=True)

            chart_module, chart_mse, chart_params = build_local_chart_module(geom, K, K_GLOBAL)
            student_chart = swap_local_chart(teacher, geom, chart_module)
            chart_eval = _evaluate_model(teacher, student_chart, corpus, batch_size=BATCH_SIZE)
            chart_ppl = chart_eval["perplexity"]

            # Improvement over global geometric (positive = local is better)
            improvement_pct = (geo_ppl - chart_ppl) / geo_ppl * 100

            print(
                f"  [local K={K}] MSE={chart_mse:.4f}  PPL={chart_ppl:.2f}  "
                f"ΔPPL={chart_ppl-baseline_ppl:+.2f} ({(chart_ppl-baseline_ppl)/baseline_ppl*100:+.1f}%)  "
                f"vs_geo={improvement_pct:+.1f}%",
                flush=True,
            )
            results["cells"].append({
                "block_id": block_id,
                "method": f"local_chart_K{K}",
                "K": K,
                "k_per_chart": k_per_chart,
                "block_mse": chart_mse,
                "param_count": chart_params,
                "ppl": chart_ppl,
                "delta_ppl": chart_ppl - baseline_ppl,
                "delta_ppl_pct": (chart_ppl - baseline_ppl) / baseline_ppl * 100,
                "improvement_vs_geo_pct": improvement_pct,
            })

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cycle_029_local_chart.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print("\n=== Summary: improvement of local-chart over global geometric ===")
    print(f"{'Block':>6}  {'Method':>18}  {'BlockMSE':>10}  {'PPL':>8}  {'ΔPPL%':>8}  {'vs_geo%':>8}")
    for cell in results["cells"]:
        vs_geo = cell.get("improvement_vs_geo_pct", 0.0)
        print(
            f"{cell['block_id']:>6}  {cell['method']:>18}  "
            f"{cell['block_mse']:>10.4f}  {cell['ppl']:>8.2f}  "
            f"{cell['delta_ppl_pct']:>+8.1f}%  {vs_geo:>+8.1f}%"
        )

    # Verdict
    confirmed_blocks = set()
    for cell in results["cells"]:
        if "improvement_vs_geo_pct" in cell and cell["improvement_vs_geo_pct"] >= 20.0:
            confirmed_blocks.add(cell["block_id"])

    print(f"\nBlocks with >=20% PPL improvement over geometric: {sorted(confirmed_blocks)}")
    n_confirmed = len(confirmed_blocks)
    if n_confirmed >= 3:
        print("VERDICT: CONFIRMED — local-chart improves >=20% on >=3 blocks")
    elif n_confirmed > 0:
        print(f"VERDICT: INCONCLUSIVE — improved >=20% on only {n_confirmed}/3 blocks")
    else:
        print("VERDICT: REFUTED — local-chart fails to improve >=20% on any block")


if __name__ == "__main__":
    main()
