"""Cycle 026: Rank-only (SVD) baseline vs geometric vs distillation on GPT-2 FFN blocks.

For each (block, compression_ratio) pair, compare three methods at equal parameter budget:
  1. SVD: truncated SVD on composed linear map W_down @ W_up
  2. Geometric: channel-score top-k selection + least-squares output refit (existing)
  3. Distillation: random init compressed FFN trained 200 steps with KL+CE+MSE

Key question: does the geometric method beat SVD on full-model perplexity?
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressionBudgetSpec,
    DistillationSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    _pack_layer_metrics,
    _param_count,
    _standard_bias,
    _standard_weight,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    run_ffn_distillation_baseline,
    swap_compressed_ffns,
)

SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"

BLOCKS = (2, 5, 8, 11)
HIDDEN_FRACTIONS = (0.25, 0.125, 0.0625)
COMPRESSION_RATIOS = (4, 8, 16)
DISTILL_STEPS = 200
BATCH_SIZE = 4


def make_svd_composed_pack(
    atlas: FfnGeometryAtlas,
    k: int,
    layer_id: int,
) -> CompressedFfnPack:
    """SVD on composed linear map W_down @ W_up; no data required.

    NOTE: This is the naive rank baseline. It is broken for nonlinear FFNs because
    GELU does not commute with the change-of-basis, and it drops b_fc entirely.
    Included for documentation of why naive SVD fails on GELU networks.
    """
    geom_layers = {g.layer_id: g for g in atlas.layer_geometries}
    geom = geom_layers[layer_id]

    w_up = _standard_weight(geom.handle.c_fc).to(torch.float32)   # [3072, 768]
    w_down = _standard_weight(geom.handle.c_proj).to(torch.float32)  # [768, 3072]
    b_down = _standard_bias(geom.handle.c_proj)

    # Composed map: R^768 -> R^768 (ignoring nonlinearity)
    w_product = w_down @ w_up  # [768, 768]
    U, S, Vh = torch.linalg.svd(w_product, full_matrices=False)

    # Truncate to rank k
    U_k = U[:, :k]        # [768, k]
    S_k = S[:k]           # [k]
    Vh_k = Vh[:k, :]      # [k, 768]

    sqrt_S = S_k.sqrt()
    c_fc_w = (sqrt_S.unsqueeze(1) * Vh_k).contiguous()  # [k, 768]
    c_proj_w = (U_k * sqrt_S.unsqueeze(0)).contiguous()  # [768, k]

    src_param = (
        _param_count(w_up, _standard_bias(geom.handle.c_fc))
        + _param_count(w_down, b_down)
    )
    comp_param = _param_count(c_fc_w, None) + _param_count(c_proj_w, b_down)

    compressed_layer = CompressedFfnLayer(
        layer_id=layer_id,
        activation_kind=geom.activation_kind,
        source_hidden_dim=geom.hidden_dim,
        compressed_hidden_dim=k,
        c_fc_weight=c_fc_w,
        c_fc_bias=None,
        c_proj_weight=c_proj_w,
        c_proj_bias=b_down.clone() if b_down is not None else None,
        selected_indices=None,
        source_param_count=src_param,
        compressed_param_count=comp_param,
        notes={"method": "svd_composed", "rank": k},
    )
    return CompressedFfnPack(
        method="svd_composed",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=(compressed_layer,),
        metadata={"sequence_length": atlas.sequence_length},
    )


def make_svd_norm_pack(
    atlas: FfnGeometryAtlas,
    k: int,
    layer_id: int,
) -> CompressedFfnPack:
    """Correct SVD-inspired baseline: select k channels by W_up row-norms + lstsq refit.

    This is the proper rank baseline for nonlinear FFNs:
    - Selects k hidden units by the L2 norm of their input weight vector (row of W_up)
    - Keeps original W_up rows and b_fc slice for selected units
    - Refits W_down via least-squares on probe activations (same as geometric method)
    The only difference from geometric: channel selection criterion (norm vs activity score).
    """
    from aleph.lm_geometric_compression import _fit_output_projection

    geom_layers = {g.layer_id: g for g in atlas.layer_geometries}
    geom = geom_layers[layer_id]

    w_up = _standard_weight(geom.handle.c_fc).to(torch.float32)   # [3072, 768]
    w_down = _standard_weight(geom.handle.c_proj).to(torch.float32)
    b_fc = _standard_bias(geom.handle.c_fc)
    b_proj = _standard_bias(geom.handle.c_proj)

    # Select k channels by W_up row norms (SVD importance proxy)
    row_norms = w_up.norm(dim=1)   # [3072]
    selected = torch.topk(row_norms, k=k).indices.sort().values

    student_w_in = w_up[selected].contiguous()
    student_b_in = b_fc[selected].contiguous() if b_fc is not None else None

    # Refit output projection via lstsq (same as geometric method)
    hidden = geom.hidden_sample[:, selected].to(torch.float32)
    use_output_bias = b_proj is not None
    student_w_out, student_b_out = _fit_output_projection(
        hidden, geom.output_sample.to(torch.float32), use_bias=use_output_bias
    )

    src_param = (
        _param_count(w_up, b_fc) + _param_count(w_down, b_proj)
    )
    comp_param = (
        _param_count(student_w_in, student_b_in) + _param_count(student_w_out, student_b_out)
    )

    compressed_layer = CompressedFfnLayer(
        layer_id=layer_id,
        activation_kind=geom.activation_kind,
        source_hidden_dim=geom.hidden_dim,
        compressed_hidden_dim=k,
        c_fc_weight=student_w_in,
        c_fc_bias=student_b_in,
        c_proj_weight=student_w_out,
        c_proj_bias=student_b_out,
        selected_indices=selected,
        source_param_count=src_param,
        compressed_param_count=comp_param,
        notes={"method": "svd_norm", "rank": k},
    )
    return CompressedFfnPack(
        method="svd_norm",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=(compressed_layer,),
        metadata={"sequence_length": atlas.sequence_length},
    )


def evaluate_pack(teacher, pack, corpus) -> dict:
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    metrics = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
    layer = pack.layers[0]
    metrics["ffn_param_count"] = float(layer.compressed_param_count)
    metrics["source_ffn_param_count"] = float(layer.source_param_count)
    metrics["compression_ratio"] = float(layer.source_param_count) / max(layer.compressed_param_count, 1)
    return metrics


def get_block_mse(teacher, pack, atlas) -> float:
    """Compute local block MSE on probe inputs."""
    from aleph.lm_geometric_compression import CompressedMLP, _load_compressed_layer, _apply_activation
    layer = pack.layers[0]
    geom = {g.layer_id: g for g in atlas.layer_geometries}[layer.layer_id]

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
        pred = mlp(geom.input_sample)
        mse = F.mse_loss(pred, geom.output_sample).item()
    return mse


def main() -> None:
    device_str = "auto"

    # Check Shakespeare file
    shakes_path = SHAKESPEARE_PATH
    if not Path(shakes_path).exists():
        # Try alternate common locations
        for candidate in [
            Path(__file__).parent.parent / "data" / "input.txt",
            Path(__file__).parent.parent / "data" / "tinyshakespeare.txt",
        ]:
            if candidate.exists():
                shakes_path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"Shakespeare corpus not found at {shakes_path}. "
                "Place tinyshakespeare.txt at data/shakespeare.txt"
            )

    print(f"Loading teacher GPT-2 ...", flush=True)
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device=device_str,
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

    # Baseline perplexity (uncompressed)
    print("Computing baseline perplexity ...", flush=True)
    baseline_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline PPL: {baseline_ppl:.2f}", flush=True)

    results = {
        "baseline_ppl": baseline_ppl,
        "baseline_metrics": baseline_metrics,
        "cells": [],
    }

    for block_id in BLOCKS:
        print(f"\n=== Block {block_id} ===", flush=True)

        # Probe geometry once per block (reuse across methods and budgets)
        print(f"  Probing block {block_id} geometry ...", flush=True)
        atlas = probe_ffn_geometry(
            teacher,
            corpus,
            layer_ids=(block_id,),
            max_tokens_per_layer=2048,
            batch_size=BATCH_SIZE,
        )

        for frac, ratio in zip(HIDDEN_FRACTIONS, COMPRESSION_RATIOS):
            k = max(8, int(3072 * frac))
            budget = CompressionBudgetSpec(
                hidden_fraction=frac,
                min_hidden=8,
                selection_mode="topk",
            )

            print(f"  Budget {ratio}× (k={k}) ...", flush=True)
            cell_base = {
                "block_id": block_id,
                "compression_ratio_target": ratio,
                "hidden_fraction": frac,
                "k": k,
                "baseline_ppl": baseline_ppl,
            }

            # --- Method 1a: SVD composed (naive, no data) ---
            print(f"    SVD-composed ...", flush=True)
            try:
                svd_pack = make_svd_composed_pack(atlas, k, block_id)
                svd_metrics = evaluate_pack(teacher, svd_pack, corpus)
                svd_mse = get_block_mse(teacher, svd_pack, atlas)
                results["cells"].append({
                    **cell_base,
                    "method": "svd_composed",
                    "metrics": svd_metrics,
                    "block_mse": svd_mse,
                    "ppl": svd_metrics["perplexity"],
                    "ppl_increase_pct": 100.0 * (svd_metrics["perplexity"] - baseline_ppl) / baseline_ppl,
                })
                print(f"    SVD-composed ppl={svd_metrics['perplexity']:.2f} mse={svd_mse:.4f}", flush=True)
            except Exception as e:
                print(f"    SVD-composed FAILED: {e}", flush=True)
                results["cells"].append({**cell_base, "method": "svd_composed", "error": str(e)})

            # --- Method 1b: SVD-norm (weight row-norm selection + lstsq refit) ---
            print(f"    SVD-norm ...", flush=True)
            try:
                svd_norm_pack = make_svd_norm_pack(atlas, k, block_id)
                svd_norm_metrics = evaluate_pack(teacher, svd_norm_pack, corpus)
                svd_norm_mse = get_block_mse(teacher, svd_norm_pack, atlas)
                results["cells"].append({
                    **cell_base,
                    "method": "svd_norm",
                    "metrics": svd_norm_metrics,
                    "block_mse": svd_norm_mse,
                    "ppl": svd_norm_metrics["perplexity"],
                    "ppl_increase_pct": 100.0 * (svd_norm_metrics["perplexity"] - baseline_ppl) / baseline_ppl,
                })
                print(f"    SVD-norm ppl={svd_norm_metrics['perplexity']:.2f} mse={svd_norm_mse:.4f}", flush=True)
            except Exception as e:
                print(f"    SVD-norm FAILED: {e}", flush=True)
                results["cells"].append({**cell_base, "method": "svd_norm", "error": str(e)})

            # --- Method 2: Geometric ---
            print(f"    Geometric ...", flush=True)
            try:
                geo_pack = derive_ffn_compression(atlas, budget)
                geo_metrics = evaluate_pack(teacher, geo_pack, corpus)
                geo_mse = get_block_mse(teacher, geo_pack, atlas)
                results["cells"].append({
                    **cell_base,
                    "method": "geometric",
                    "metrics": geo_metrics,
                    "block_mse": geo_mse,
                    "ppl": geo_metrics["perplexity"],
                    "ppl_increase_pct": 100.0 * (geo_metrics["perplexity"] - baseline_ppl) / baseline_ppl,
                })
                print(f"    Geo ppl={geo_metrics['perplexity']:.2f} mse={geo_mse:.4f}", flush=True)
            except Exception as e:
                print(f"    Geometric FAILED: {e}", flush=True)
                results["cells"].append({**cell_base, "method": "geometric", "error": str(e)})

            # --- Method 3: Distillation ---
            print(f"    Distillation ({DISTILL_STEPS} steps) ...", flush=True)
            try:
                distill_spec = DistillationSpec(
                    teacher=teacher,
                    corpus=corpus,
                    budget=budget,
                    layer_ids=(block_id,),
                    batch_size=BATCH_SIZE,
                    steps=DISTILL_STEPS,
                    lr=1e-3,
                    kl_weight=1.0,
                    target_ce_weight=0.25,
                    layer_output_weight=0.2,
                    seed=42,
                )
                distill_result = run_ffn_distillation_baseline(distill_spec)
                distill_ppl = distill_result.metrics["perplexity"]
                # Get block MSE from layer_metrics
                distill_mse = distill_result.layer_metrics[0].get("layer_output_mse", float("nan")) if distill_result.layer_metrics else float("nan")
                results["cells"].append({
                    **cell_base,
                    "method": "distillation",
                    "metrics": distill_result.metrics,
                    "block_mse": distill_mse,
                    "ppl": distill_ppl,
                    "ppl_increase_pct": 100.0 * (distill_ppl - baseline_ppl) / baseline_ppl,
                })
                print(f"    Distill ppl={distill_ppl:.2f} mse={distill_mse:.4f}", flush=True)
            except Exception as e:
                print(f"    Distillation FAILED: {e}", flush=True)
                results["cells"].append({**cell_base, "method": "distillation", "error": str(e)})

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "cycle_026_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)

    # Build summary table
    print_summary(results)
    write_summary(results, RESULTS_DIR / "cycle_026_summary.md", baseline_ppl)


def print_summary(results: dict) -> None:
    print(f"\n{'Block':>6} {'Ratio':>6} {'Method':>12} {'PPL':>8} {'ΔPPL%':>8} {'BlockMSE':>10}")
    print("-" * 60)
    for cell in results["cells"]:
        if "error" in cell:
            print(f"{cell['block_id']:>6} {cell['compression_ratio_target']:>6}× {cell['method']:>12}  ERROR: {cell['error'][:30]}")
            continue
        print(
            f"{cell['block_id']:>6} {cell['compression_ratio_target']:>6}× "
            f"{cell['method']:>12} {cell['ppl']:>8.2f} {cell['ppl_increase_pct']:>+8.1f}% "
            f"{cell.get('block_mse', float('nan')):>10.4f}"
        )

    # Win/loss table (geometric vs svd_norm — the legitimate baseline)
    print("\n--- Win/Loss: geometric vs svd_norm (lower PPL = win) ---")
    cells_by_key = {}
    for cell in results["cells"]:
        if "error" in cell:
            continue
        key = (cell["block_id"], cell["compression_ratio_target"])
        cells_by_key.setdefault(key, {})[cell["method"]] = cell["ppl"]

    geo_wins, svd_wins, ties = 0, 0, 0
    for key, methods in sorted(cells_by_key.items()):
        if "svd_norm" in methods and "geometric" in methods:
            g, s = methods["geometric"], methods["svd_norm"]
            if g < s - 0.05:
                result = "geo"
                geo_wins += 1
            elif s < g - 0.05:
                result = "svd_norm"
                svd_wins += 1
            else:
                result = "tie"
                ties += 1
            print(f"  Block {key[0]} {key[1]}×: geo={g:.2f} svd_norm={s:.2f} → {result}")

    total = geo_wins + svd_wins + ties
    print(f"\nGeometric wins: {geo_wins}/{total} ({100*geo_wins/max(total,1):.0f}%)")
    print(f"SVD-norm wins:  {svd_wins}/{total} ({100*svd_wins/max(total,1):.0f}%)")
    print(f"Ties:           {ties}/{total}")

    # Also report svd_composed (broken baseline)
    print("\n--- SVD-composed (naive, broken): PPL summary ---")
    for cell in results["cells"]:
        if "error" in cell or cell["method"] != "svd_composed":
            continue
        print(f"  Block {cell['block_id']} {cell['compression_ratio_target']}×: ppl={cell['ppl']:.1f} (+{cell['ppl_increase_pct']:.0f}%)")


def write_summary(results: dict, path: Path, baseline_ppl: float) -> None:
    cells_by_key = {}
    for cell in results["cells"]:
        if "error" in cell:
            continue
        key = (cell["block_id"], cell["compression_ratio_target"])
        cells_by_key.setdefault(key, {})[cell["method"]] = cell

    geo_wins = sum(
        1 for methods in cells_by_key.values()
        if "svd_norm" in methods and "geometric" in methods
        and methods["geometric"]["ppl"] < methods["svd_norm"]["ppl"] - 0.05
    )
    svd_wins = sum(
        1 for methods in cells_by_key.values()
        if "svd_norm" in methods and "geometric" in methods
        and methods["svd_norm"]["ppl"] < methods["geometric"]["ppl"] - 0.05
    )
    total_pairs = sum(1 for m in cells_by_key.values() if "svd_norm" in m and "geometric" in m)

    geo_win_pct = 100 * geo_wins / max(total_pairs, 1)
    svd_win_pct = 100 * svd_wins / max(total_pairs, 1)

    if geo_wins > total_pairs * 0.5:
        verdict = "CONFIRMED"
        detail = f"Geometric beats SVD in {geo_wins}/{total_pairs} ({geo_win_pct:.0f}%) pairs."
    elif svd_wins > total_pairs * 0.8:
        verdict = "REFUTED"
        detail = f"SVD beats geometric in {svd_wins}/{total_pairs} ({svd_win_pct:.0f}%) pairs."
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Geometric wins {geo_wins}/{total_pairs}, SVD wins {svd_wins}/{total_pairs}. No consistent edge."

    lines = [
        "# Cycle 026 Summary — Rank vs Geometry vs Distillation",
        "",
        f"**Baseline PPL (uncompressed)**: {baseline_ppl:.2f}",
        f"**Verdict**: {verdict}",
        f"**Detail**: {detail}",
        "",
        "## Per-cell table",
        "",
        "| Block | Ratio | Method | PPL | ΔPPL% | BlockMSE |",
        "|-------|-------|--------|-----|-------|----------|",
    ]
    for (block, ratio), methods in sorted(cells_by_key.items()):
        for method in ("svd_composed", "svd_norm", "geometric", "distillation"):
            if method not in methods:
                continue
            c = methods[method]
            lines.append(
                f"| {block} | {ratio}× | {method} | {c['ppl']:.2f} | "
                f"{c['ppl_increase_pct']:+.1f}% | {c.get('block_mse', float('nan')):.4f} |"
            )

    lines += [
        "",
        "## Win/Loss: geometric vs SVD",
        "",
    ]
    for (block, ratio), methods in sorted(cells_by_key.items()):
        if "svd_norm" not in methods or "geometric" not in methods:
            continue
        g = methods["geometric"]["ppl"]
        s = methods["svd_norm"]["ppl"]
        winner = "geometric" if g < s - 0.05 else ("svd_norm" if s < g - 0.05 else "tie")
        lines.append(f"- Block {block}, {ratio}×: geo={g:.2f} svd_norm={s:.2f} → **{winner}**")

    path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {path}", flush=True)


if __name__ == "__main__":
    main()
