"""Cycle 092: Adam polish dissection — GPT-2 FFN (item 107).

Analyzes what Adam polish actually learns when applied to compressed W_down:
- SVD rank of the weight update delta (vs random null baseline)
- Projection onto lstsq residual subspace vs orthogonal complement
- PPL convergence over 500 steps

Criterion:
  CONFIRMED  if rank95(delta) < 0.5 * null_rank95 AND residual_frac > 0.70
             → polish is low-rank correction in lstsq residual subspace
  REFUTED    if rank95(delta) >= null_rank95 OR residual_frac < 0.50
             → full-rank / orthogonal structure discovery
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

sys.path.insert(0, str(Path(__file__).parent.parent))

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
TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6   # 6× compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER_RESIDUAL = 8192   # for residual subspace (N/d ≈ 10.7)
MAX_TOKENS_PER_LAYER_ATLAS = 4096     # for atlas probe
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

POLISH_STEPS = 500
EVAL_CHECKPOINTS = [0, 10, 50, 100, 200, 500]
POLISH_LR = 1e-4
NULL_REPEATS = 10

DATA_DIR = Path(__file__).parent.parent / "data"
OWT_PATH = DATA_DIR / "openwebtext_sample.txt"

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _collect_polish_params(student_model: nn.Module, block_ids: list[int]) -> list[torch.Tensor]:
    params = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(student_model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            print(f"  WARNING: no c_proj for block {block_id}")
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)
    return params


def _get_w_down(student_model: nn.Module, block_id: int) -> torch.Tensor | None:
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return None
    return mlp.c_proj.weight.detach().clone()


def _polish_joint_with_snapshots(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    teacher: TeacherBundle,
    owt_corpus,
    device: torch.device,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, float]]:
    """Run POLISH_STEPS of Adam on all W_down jointly.

    Returns:
        snapshots: {block_id: {step: W_tensor}} at EVAL_CHECKPOINTS
        ppls: {step: perplexity}
    """
    student_model.requires_grad_(False)
    params = _collect_polish_params(student_model, block_ids)
    if not params:
        return {}, {}

    optimizer = torch.optim.Adam(params, lr=POLISH_LR)
    student_model.train()

    snapshots: dict[int, dict[int, torch.Tensor]] = {b: {} for b in block_ids}
    ppls: dict[int, float] = {}
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    # Step 0 snapshot + eval
    student_model.eval()
    for b in block_ids:
        w = _get_w_down(student_model, b)
        if w is not None:
            snapshots[b][0] = w
    m0 = _evaluate_model(teacher, student_model, owt_corpus, batch_size=BATCH_SIZE)
    ppls[0] = m0["perplexity"]
    print(f"    step 0: PPL={ppls[0]:.3f}")
    student_model.train()

    for step in range(1, POLISH_STEPS + 1):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        if step in EVAL_CHECKPOINTS:
            student_model.eval()
            for b in block_ids:
                w = _get_w_down(student_model, b)
                if w is not None:
                    snapshots[b][step] = w
            m = _evaluate_model(teacher, student_model, owt_corpus, batch_size=BATCH_SIZE)
            ppls[step] = m["perplexity"]
            print(f"    step {step}: PPL={ppls[step]:.3f}, loss={float(loss.item()):.4f}")
            student_model.train()

    student_model.eval()
    student_model.requires_grad_(False)
    return snapshots, ppls


def _compute_rank95(S: torch.Tensor) -> int:
    cum = S.cumsum(0) / S.sum()
    r = int((cum < 0.95).sum().item()) + 1
    return min(r, len(S))


def _null_rank95(shape: tuple[int, int], delta_norm: float, n_rep: int = NULL_REPEATS) -> tuple[float, float]:
    ranks = []
    for _ in range(n_rep):
        nd = torch.randn(*shape)
        nd = nd * (delta_norm / nd.norm().item())
        _, S_n, _ = torch.linalg.svd(nd, full_matrices=False)
        ranks.append(_compute_rank95(S_n))
    t = torch.tensor(ranks, dtype=torch.float32)
    return float(t.mean().item()), float(t.std().item())


def _delta_svd_analysis(W_before: torch.Tensor, W_after: torch.Tensor) -> dict:
    delta = W_after - W_before
    U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    rank95 = _compute_rank95(S)
    frob_ratio = float(delta.norm().item() / (W_before.norm().item() + 1e-12))
    null_mean, null_std = _null_rank95(delta.shape, float(delta.norm().item()))
    return {
        "rank95": rank95,
        "null_rank95_mean": null_mean,
        "null_rank95_std": null_std,
        "rank_ratio": rank95 / (null_mean + 1e-9),
        "top5_singular": S[:5].tolist(),
        "frob_ratio": frob_ratio,
        "delta_shape": list(delta.shape),
        # keep U, delta for residual projection (returned separately)
        "_delta": delta,
    }


def _residual_projection(
    atlas: FfnGeometryAtlas,
    block_idx_in_atlas: int,
    block_id: int,
    delta: torch.Tensor,
    selected: torch.Tensor,
    rank95: int,
) -> float:
    """Project delta onto the lstsq residual subspace.

    Returns residual_frac = energy of delta in residual subspace / total energy.
    """
    geom = atlas.layer_geometries[block_idx_in_atlas]
    if geom.layer_id != block_id:
        # search by layer_id
        for g in atlas.layer_geometries:
            if g.layer_id == block_id:
                geom = g
                break

    # hidden activations for selected neurons: [N, k]
    hidden = geom.hidden_sample[:, selected].to(dtype=torch.float32)
    outputs = geom.output_sample.to(dtype=torch.float32)  # [N, d_model]

    # lstsq fit
    W_lstsq, _ = torch.linalg.lstsq(hidden, outputs, rcond=None)[:2]
    predicted = hidden @ W_lstsq  # [N, d_model]
    residual = outputs - predicted  # [N, d_model]

    # SVD of residual (column space)
    U_res, _, _ = torch.linalg.svd(residual.t(), full_matrices=False)
    U_res_k = U_res[:, :rank95]  # [d_model, rank95]

    # delta: [d_model, k] — project rows (d_model dimension)
    proj_energy = float((U_res_k.t() @ delta).norm().item() ** 2)
    total_energy = float(delta.norm().item() ** 2 + 1e-12)
    return proj_energy / total_energy


def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="cpu",
        )
    )
    print(f"Teacher loaded: {teacher.model_name}")

    # Teacher baseline
    owt_eval_corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQUENCE_LENGTH,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_eval_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Teacher PPL: {baseline_ppl:.3f}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    all_results: dict = {
        "teacher_ppl": baseline_ppl,
        "seeds": {},
    }

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        _set_seed(seed)

        owt_corpus = build_probe_corpus(
            ProbeCorpusSpec(
                teacher=teacher,
                text_path=str(OWT_PATH),
                sequence_length=SEQUENCE_LENGTH,
                probe_train_sequences=PROBE_TRAIN_SEQS,
                probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
                eval_val_sequences=EVAL_VAL_SEQS,
                shuffle_seed=seed,
            )
        )

        # Probe FFN geometry — use larger token budget for residual subspace
        print(f"  Probing FFN geometry (max_tokens={MAX_TOKENS_PER_LAYER_RESIDUAL})...")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER_RESIDUAL,
            batch_size=BATCH_SIZE,
        )

        # Compression
        _set_seed(seed)
        pack = derive_ffn_compression(atlas, budget)
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.eval()

        # Build selected-index map: block_id -> selected indices
        selected_map: dict[int, torch.Tensor] = {}
        for layer in pack.layers:
            if layer.selected_indices is not None:
                selected_map[layer.layer_id] = layer.selected_indices
            else:
                # fallback: topk from channel_scores
                for geom in atlas.layer_geometries:
                    if geom.layer_id == layer.layer_id:
                        k = layer.compressed_hidden_dim
                        selected_map[layer.layer_id] = torch.topk(
                            geom.channel_scores, k
                        ).indices
                        break

        polish_tokens = owt_corpus.probe_holdout
        print(f"  Polish tokens shape: {polish_tokens.shape}")

        print(f"  Running {POLISH_STEPS}-step joint Adam polish...")
        _set_seed(seed)
        snapshots, ppls = _polish_joint_with_snapshots(
            student,
            TARGET_BLOCKS,
            polish_tokens,
            teacher,
            owt_corpus,
            teacher.device,
        )

        # Per-block delta analysis at each checkpoint
        seed_results: dict = {"ppls": ppls, "blocks": {}}
        for block_id in TARGET_BLOCKS:
            W_before = snapshots[block_id][0]
            block_res: dict = {}
            selected = selected_map.get(block_id)

            # atlas index
            atlas_idx = next(
                (i for i, g in enumerate(atlas.layer_geometries) if g.layer_id == block_id),
                None,
            )

            for step in EVAL_CHECKPOINTS[1:]:   # skip step 0 (no delta)
                if step not in snapshots[block_id]:
                    continue
                W_after = snapshots[block_id][step]
                analysis = _delta_svd_analysis(W_before, W_after)
                delta = analysis.pop("_delta")

                # Residual projection
                if selected is not None and atlas_idx is not None:
                    res_frac = _residual_projection(
                        atlas, atlas_idx, block_id, delta, selected, analysis["rank95"]
                    )
                else:
                    res_frac = float("nan")
                    print(f"  WARNING: no selected indices or atlas for block {block_id}")

                analysis["residual_frac"] = res_frac
                block_res[step] = analysis

                print(
                    f"  block {block_id} step {step}: "
                    f"rank95={analysis['rank95']} "
                    f"(null={analysis['null_rank95_mean']:.1f}±{analysis['null_rank95_std']:.1f}, "
                    f"ratio={analysis['rank_ratio']:.2f}), "
                    f"frob_ratio={analysis['frob_ratio']:.4f}, "
                    f"res_frac={res_frac:.3f}"
                )

            seed_results["blocks"][block_id] = block_res

        all_results["seeds"][seed] = seed_results

    # ---- Aggregate over seeds at step 500 ----
    agg_500: dict[int, dict] = {}
    for block_id in TARGET_BLOCKS:
        rank95_vals = [
            all_results["seeds"][s]["blocks"][block_id][500]["rank95"]
            for s in SEEDS
            if 500 in all_results["seeds"][s]["blocks"].get(block_id, {})
        ]
        null_vals = [
            all_results["seeds"][s]["blocks"][block_id][500]["null_rank95_mean"]
            for s in SEEDS
            if 500 in all_results["seeds"][s]["blocks"].get(block_id, {})
        ]
        res_vals = [
            all_results["seeds"][s]["blocks"][block_id][500]["residual_frac"]
            for s in SEEDS
            if 500 in all_results["seeds"][s]["blocks"].get(block_id, {})
        ]
        r95_t = torch.tensor(rank95_vals, dtype=torch.float32)
        null_t = torch.tensor(null_vals, dtype=torch.float32)
        res_t = torch.tensor([v for v in res_vals if not math.isnan(v)], dtype=torch.float32)
        agg_500[block_id] = {
            "rank95_mean": float(r95_t.mean()),
            "rank95_std": float(r95_t.std()) if len(rank95_vals) > 1 else 0.0,
            "null_rank95_mean": float(null_t.mean()),
            "rank_ratio": float((r95_t / null_t).mean()),
            "residual_frac_mean": float(res_t.mean()) if len(res_t) > 0 else float("nan"),
            "residual_frac_std": float(res_t.std()) if len(res_t) > 1 else 0.0,
        }

    all_results["aggregate_step500"] = agg_500

    # ---- Verdict ----
    confirmed_count = 0
    refuted_count = 0
    inconclusive_count = 0
    for block_id, agg in agg_500.items():
        rr = agg["rank_ratio"]
        rf = agg["residual_frac_mean"]
        if rr < 0.5 and rf > 0.70:
            verdict = "CONFIRMED"
            confirmed_count += 1
        elif rr >= 1.0 or rf < 0.50:
            verdict = "REFUTED"
            refuted_count += 1
        else:
            verdict = "INCONCLUSIVE"
            inconclusive_count += 1
        agg["verdict"] = verdict
        print(
            f"Block {block_id}: rank_ratio={rr:.2f}, res_frac={rf:.3f} → {verdict}"
        )

    if confirmed_count >= 3:
        overall_verdict = "CONFIRMED"
    elif refuted_count >= 3:
        overall_verdict = "REFUTED"
    else:
        overall_verdict = "INCONCLUSIVE"

    all_results["overall_verdict"] = overall_verdict
    print(f"\nOverall verdict: {overall_verdict}")

    # ---- Convergence plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PPL convergence across seeds
    ax = axes[0]
    steps_list = sorted(EVAL_CHECKPOINTS)
    for seed in SEEDS:
        ppls_s = all_results["seeds"][seed]["ppls"]
        xs = [s for s in steps_list if s in ppls_s]
        ys = [ppls_s[s] for s in xs]
        ax.plot(xs, ys, marker="o", label=f"seed {seed}", alpha=0.7)
    ax.axhline(baseline_ppl, color="k", linestyle="--", label="Teacher PPL")
    ax.set_xlabel("Adam steps")
    ax.set_ylabel("PPL")
    ax.set_title("PPL vs Adam Steps (joint polish, blocks [7,5,4,1])")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rank ratio and residual_frac at step 500
    ax2 = axes[1]
    x_pos = list(range(len(TARGET_BLOCKS)))
    labels = [f"Block {b}" for b in TARGET_BLOCKS]
    rr_vals = [agg_500[b]["rank_ratio"] for b in TARGET_BLOCKS]
    rf_vals = [agg_500[b]["residual_frac_mean"] for b in TARGET_BLOCKS]
    width = 0.35
    bars1 = ax2.bar([x - width/2 for x in x_pos], rr_vals, width, label="rank_ratio (rank95/null)", color="steelblue")
    bars2 = ax2.bar([x + width/2 for x in x_pos], rf_vals, width, label="residual_frac", color="darkorange")
    ax2.axhline(0.5, color="steelblue", linestyle="--", alpha=0.5, label="rank_ratio=0.5 threshold")
    ax2.axhline(0.7, color="darkorange", linestyle="--", alpha=0.5, label="res_frac=0.7 threshold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Value")
    ax2.set_title("Delta Structure at Step 500")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_092_convergence.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    # ---- Save JSON ----
    def _jsonify(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    json_path = RESULTS_DIR / "cycle_092_polish_dissection.json"
    with open(json_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)
    print(f"JSON saved: {json_path}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== SUMMARY ===")
    print(f"Overall verdict: {results['overall_verdict']}")
    for b, agg in results["aggregate_step500"].items():
        print(
            f"  Block {b}: rank95={agg['rank95_mean']:.1f} "
            f"null={agg['null_rank95_mean']:.1f} "
            f"ratio={agg['rank_ratio']:.2f} "
            f"res_frac={agg['residual_frac_mean']:.3f} "
            f"→ {agg['verdict']}"
        )
