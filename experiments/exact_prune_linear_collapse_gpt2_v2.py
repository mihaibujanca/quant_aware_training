"""
Cycle 028 — Dead Channel Census & Prune+Collapse — GPT-2 (redesigned).

Original experiment used DEAD_THRESHOLD=1e-5 on post-GELU activations.
GELU is never truly zero (gelu(-3)≈-0.004), so the threshold caught nothing.

Redesigned with correct definitions for GELU networks:
  1. Pre-GELU dead: max(h_j) < -T → channel always in GELU's saturated tail
  2. Output-contribution dead: ||W_down[:, j]|| * rms(act_j) is negligible
  3. Near-linear: min(h_j) > T → channel always in GELU's linear regime
  4. Smooth pruning curve: sort by contribution, remove bottom K%

GPT-2 FFN: W_up (768→3072), gelu, W_down (3072→768)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
)

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-GELU dead thresholds (max(h_j) < T means always in saturated regime)
DEAD_THRESHOLDS = [-2.0, -3.0, -4.0, -5.0]

# Contribution-based dead thresholds (fraction of total output contribution)
CONTRIB_THRESHOLDS = [1e-4, 5e-4, 1e-3]  # 0.01%, 0.05%, 0.1%

# Near-linear thresholds (min(h_j) > T means always in linear regime)
LINEAR_THRESHOLDS = [2.0, 3.0, 4.0]

# Smooth pruning curve: remove bottom K% by contribution
PRUNE_FRACS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

BATCH_SIZE = 4
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50


# ── GELU (NewGELU tanh approximation, matching GPT-2) ─────────────────────────
def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


# ── Perplexity ─────────────────────────────────────────────────────────────────
def compute_perplexity(model: nn.Module, tokens: torch.Tensor,
                       device: torch.device, batch_size: int = BATCH_SIZE) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start:start + batch_size].to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            out = model(input_ids, use_cache=False)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


# ── Activation collection ──────────────────────────────────────────────────────
def collect_ffn_activations(model: nn.Module, tokens: torch.Tensor,
                             device: torch.device) -> list[dict]:
    """Collect pre-GELU (h), post-GELU (act), input (x_in), output (out) per block."""
    n_blocks = len(model.transformer.h)
    all_data: list[dict] = [{"h": [], "act": [], "x_in": [], "out": []} for _ in range(n_blocks)]

    hooks = []
    def make_hook(block_idx: int):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            h = module.c_fc(x).detach()
            act = new_gelu(h)
            out = output.detach()
            B, T, D = x.shape
            all_data[block_idx]["x_in"].append(x.reshape(-1, D).cpu())
            all_data[block_idx]["h"].append(h.reshape(-1, h.shape[-1]).cpu())
            all_data[block_idx]["act"].append(act.reshape(-1, act.shape[-1]).cpu())
            all_data[block_idx]["out"].append(out.reshape(-1, out.shape[-1]).cpu())
        return hook

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.mlp.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            batch = tokens[start:start + BATCH_SIZE].to(device)
            model(batch[:, :-1], use_cache=False)

    for h in hooks:
        h.remove()

    return [{k: torch.cat(v, dim=0) for k, v in d.items()} for d in all_data]


# ── Channel analysis (fixed for GELU) ─────────────────────────────────────────

class ChannelAnalysis(NamedTuple):
    # Per-threshold dead counts
    dead_by_threshold: dict[float, torch.Tensor]     # T → (3072,) bool
    # Contribution-based
    contribution: torch.Tensor                         # (3072,) float
    contribution_frac: torch.Tensor                    # (3072,) float (fraction of total)
    dead_by_contrib: dict[float, torch.Tensor]         # threshold → (3072,) bool
    # Near-linear
    linear_by_threshold: dict[float, torch.Tensor]     # T → (3072,) bool
    # Pre-GELU stats
    h_max: torch.Tensor                                # (3072,) max pre-GELU per channel
    h_min: torch.Tensor                                # (3072,) min pre-GELU per channel


def analyze_channels(h: torch.Tensor, act: torch.Tensor,
                     W_down: torch.Tensor) -> ChannelAnalysis:
    """
    h: (N, 3072) pre-GELU activations
    act: (N, 3072) post-GELU activations
    W_down: (768, 3072) c_proj weight matrix
    """
    h_max = h.max(dim=0).values    # (3072,)
    h_min = h.min(dim=0).values    # (3072,)

    # 1. Pre-GELU dead: max(h_j) < T → always in saturated negative tail
    dead_by_threshold = {}
    for T in DEAD_THRESHOLDS:
        dead_by_threshold[T] = h_max < T

    # 2. Output-contribution: ||W_down[:, j]|| * rms(act_j)
    # GPT-2 Conv1D stores weight as (n_intermediate, n_embed) = (3072, 768),
    # so row j maps intermediate channel j → output. norm(dim=1) gives (3072,).
    if W_down.shape[0] == W_down.shape[1]:
        w_col_norm = W_down.norm(dim=0)               # square case
    elif W_down.shape[0] > W_down.shape[1]:
        # Conv1D convention: (3072, 768) → norm each row → (3072,)
        w_col_norm = W_down.norm(dim=1)               # (3072,)
    else:
        # Linear convention: (768, 3072) → norm each column → (3072,)
        w_col_norm = W_down.norm(dim=0)               # (3072,)
    act_rms = act.pow(2).mean(dim=0).sqrt()           # (3072,)
    contribution = w_col_norm * act_rms                # (3072,)
    total = contribution.sum() + 1e-12
    contribution_frac = contribution / total           # (3072,)

    dead_by_contrib = {}
    for thresh in CONTRIB_THRESHOLDS:
        dead_by_contrib[thresh] = contribution_frac < thresh

    # 3. Near-linear: min(h_j) > T → always in identity regime of GELU
    linear_by_threshold = {}
    for T in LINEAR_THRESHOLDS:
        linear_by_threshold[T] = h_min > T

    return ChannelAnalysis(
        dead_by_threshold=dead_by_threshold,
        contribution=contribution,
        contribution_frac=contribution_frac,
        dead_by_contrib=dead_by_contrib,
        linear_by_threshold=linear_by_threshold,
        h_max=h_max,
        h_min=h_min,
    )


# ── Contribution-based pruning via patched forward ────────────────────────────

class ContribPrunedMLP(nn.Module):
    """Zero out the bottom-K channels by output contribution."""
    def __init__(self, original_mlp: nn.Module, prune_mask: torch.Tensor):
        super().__init__()
        self.mlp = original_mlp
        self.register_buffer("keep_mask", (~prune_mask).float())  # 1 for kept, 0 for pruned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp.c_fc(x)
        act = new_gelu(h)
        act = act * self.keep_mask.to(act.device)
        return self.mlp.c_proj(act)


def patch_all_contrib(model, all_prune_masks):
    """Patch all blocks with contribution-based pruning. Return originals."""
    originals = []
    for i, mask in enumerate(all_prune_masks):
        orig = model.transformer.h[i].mlp
        model.transformer.h[i].mlp = ContribPrunedMLP(orig, mask)
        originals.append(orig)
    return originals


def restore_all(model, originals):
    for i, orig in enumerate(originals):
        model.transformer.h[i].mlp = orig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading GPT-2...")
    load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
    bundle = load_nanochat_teacher(load_spec)
    model = bundle.model
    device = bundle.device
    print(f"  device: {device}")

    print("Building probe corpus...")
    corpus_spec = ProbeCorpusSpec(
        teacher=bundle,
        text_path="data/shakespeare.txt",
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
        eval_val_sequences=PROBE_HOLDOUT_SEQ,
        dataset_name="proxy_shakespeare",
    )
    corpus = build_probe_corpus(corpus_spec)
    probe_tokens = corpus.probe_train
    val_tokens = corpus.probe_holdout
    print(f"  probe: {probe_tokens.shape}, val: {val_tokens.shape}")

    # Baseline
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, val_tokens, device)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Collect activations
    print("Collecting FFN activations...")
    block_data = collect_ffn_activations(model, probe_tokens, device)
    n_tokens = block_data[0]["h"].shape[0]
    print(f"  {n_tokens} tokens per block")

    # ── Per-block analysis ─────────────────────────────────────────────────
    n_blocks = len(block_data)
    n_hidden = 3072
    all_analyses: list[ChannelAnalysis] = []

    print(f"\n{'='*80}")
    print("PRE-GELU DEAD CHANNELS (max(h_j) < threshold)")
    print(f"{'Block':>5}", end="")
    for T in DEAD_THRESHOLDS:
        print(f"  T={T:+.0f}%", end="")
    print()

    for i, data in enumerate(block_data):
        W_down = model.transformer.h[i].mlp.c_proj.weight.detach().cpu()
        analysis = analyze_channels(data["h"], data["act"], W_down)
        all_analyses.append(analysis)

        print(f"  {i:>3}", end="")
        for T in DEAD_THRESHOLDS:
            n = analysis.dead_by_threshold[T].sum().item()
            pct = 100.0 * n / n_hidden
            print(f"  {pct:6.2f}", end="")
        print()

    # Summary
    print("  AVG", end="")
    for T in DEAD_THRESHOLDS:
        avg = np.mean([a.dead_by_threshold[T].sum().item() / n_hidden * 100
                       for a in all_analyses])
        print(f"  {avg:6.2f}", end="")
    print()

    print(f"\n{'='*80}")
    print("CONTRIBUTION-BASED DEAD CHANNELS (||W_down[:,j]|| * rms(act_j) < frac of total)")
    print(f"{'Block':>5}", end="")
    for ct in CONTRIB_THRESHOLDS:
        print(f"  <{ct:.0e}%", end="")
    print()

    for i, analysis in enumerate(all_analyses):
        print(f"  {i:>3}", end="")
        for ct in CONTRIB_THRESHOLDS:
            n = analysis.dead_by_contrib[ct].sum().item()
            pct = 100.0 * n / n_hidden
            print(f"  {pct:7.2f}", end="")
        print()

    print("  AVG", end="")
    for ct in CONTRIB_THRESHOLDS:
        avg = np.mean([a.dead_by_contrib[ct].sum().item() / n_hidden * 100
                       for a in all_analyses])
        print(f"  {avg:7.2f}", end="")
    print()

    print(f"\n{'='*80}")
    print("NEAR-LINEAR CHANNELS (min(h_j) > threshold, GELU ≈ identity)")
    print(f"{'Block':>5}", end="")
    for T in LINEAR_THRESHOLDS:
        print(f"  T={T:+.0f}%", end="")
    print()

    for i, analysis in enumerate(all_analyses):
        print(f"  {i:>3}", end="")
        for T in LINEAR_THRESHOLDS:
            n = analysis.linear_by_threshold[T].sum().item()
            pct = 100.0 * n / n_hidden
            print(f"  {pct:6.2f}", end="")
        print()

    print("  AVG", end="")
    for T in LINEAR_THRESHOLDS:
        avg = np.mean([a.linear_by_threshold[T].sum().item() / n_hidden * 100
                       for a in all_analyses])
        print(f"  {avg:6.2f}", end="")
    print()

    # ── Smooth pruning curve ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SMOOTH PRUNING CURVE (remove bottom K% by output contribution)")

    pruning_results: list[dict] = []
    for frac in PRUNE_FRACS:
        k = int(n_hidden * frac)
        # Build prune masks: for each block, sort by contribution, mask bottom k
        all_prune_masks = []
        for analysis in all_analyses:
            sorted_idx = torch.argsort(analysis.contribution)  # ascending
            mask = torch.zeros(n_hidden, dtype=torch.bool)
            mask[sorted_idx[:k]] = True
            all_prune_masks.append(mask)

        originals = patch_all_contrib(model, all_prune_masks)
        ppl = compute_perplexity(model, val_tokens, device)
        restore_all(model, originals)

        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        param_saved = frac  # fraction of FFN intermediate channels removed
        print(f"  {frac*100:5.1f}% pruned ({k:4d} ch/block): "
              f"PPL={ppl:.2f} ({rel:+.2f}%)")
        pruning_results.append({
            "frac": frac, "k": k, "ppl": ppl, "rel_pct": rel,
        })

    # ── Outcome ────────────────────────────────────────────────────────────
    # Use T=-3 dead + contribution-based analysis for primary outcome
    avg_dead_t3 = np.mean([a.dead_by_threshold[-3.0].sum().item() / n_hidden * 100
                           for a in all_analyses])
    avg_dead_contrib = np.mean([a.dead_by_contrib[1e-3].sum().item() / n_hidden * 100
                                for a in all_analyses])
    avg_linear_t3 = np.mean([a.linear_by_threshold[3.0].sum().item() / n_hidden * 100
                             for a in all_analyses])
    total_removable = avg_dead_t3 + avg_linear_t3

    # Find max pruning fraction with <1% PPL increase
    max_safe_frac = 0.0
    for pr in pruning_results:
        if pr["rel_pct"] <= 1.0:
            max_safe_frac = pr["frac"]

    if total_removable >= 5.0 and max_safe_frac >= 0.05:
        outcome = "CONFIRMED"
    elif total_removable < 1.0 and max_safe_frac < 0.02:
        outcome = "REFUTED"
    else:
        outcome = "INCONCLUSIVE"

    print(f"\n=== Outcome: {outcome} ===")
    print(f"  Avg dead (pre-GELU, T=-3): {avg_dead_t3:.2f}%")
    print(f"  Avg dead (contribution <0.1%): {avg_dead_contrib:.2f}%")
    print(f"  Avg near-linear (T=+3): {avg_linear_t3:.2f}%")
    print(f"  Total removable (dead+linear, T=±3): {total_removable:.2f}%")
    print(f"  Max safe pruning (<1% PPL): {max_safe_frac*100:.1f}%")

    # ── Save results ──────────────────────────────────────────────────────
    per_block_stats = []
    for i, analysis in enumerate(all_analyses):
        stats: dict = {"block": i}
        for T in DEAD_THRESHOLDS:
            stats[f"dead_T{abs(T):.0f}"] = int(analysis.dead_by_threshold[T].sum().item())
            stats[f"pct_dead_T{abs(T):.0f}"] = float(
                analysis.dead_by_threshold[T].sum().item() / n_hidden * 100)
        for ct in CONTRIB_THRESHOLDS:
            key = f"contrib_{ct:.0e}"
            stats[f"dead_{key}"] = int(analysis.dead_by_contrib[ct].sum().item())
            stats[f"pct_dead_{key}"] = float(
                analysis.dead_by_contrib[ct].sum().item() / n_hidden * 100)
        for T in LINEAR_THRESHOLDS:
            stats[f"linear_T{T:.0f}"] = int(analysis.linear_by_threshold[T].sum().item())
            stats[f"pct_linear_T{T:.0f}"] = float(
                analysis.linear_by_threshold[T].sum().item() / n_hidden * 100)
        # Pre-GELU range stats
        stats["h_max_median"] = float(analysis.h_max.median().item())
        stats["h_min_median"] = float(analysis.h_min.median().item())
        per_block_stats.append(stats)

    results = {
        "cycle": 32,
        "item": 47,
        "outcome": outcome,
        "baseline_ppl": baseline_ppl,
        "summary": {
            "avg_dead_T3_pct": avg_dead_t3,
            "avg_dead_contrib_0.1pct": avg_dead_contrib,
            "avg_linear_T3_pct": avg_linear_t3,
            "total_removable_pct": total_removable,
            "max_safe_prune_frac": max_safe_frac,
        },
        "pruning_curve": pruning_results,
        "per_block": per_block_stats,
        "thresholds": {
            "dead_pregelu": DEAD_THRESHOLDS,
            "contrib_frac": CONTRIB_THRESHOLDS,
            "linear_pregelu": LINEAR_THRESHOLDS,
        },
    }

    out_json = RESULTS_DIR / "cycle_032_prune_collapse.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Per-block dead channel counts at different thresholds
    ax = axes[0, 0]
    blocks = list(range(n_blocks))
    for T in DEAD_THRESHOLDS:
        pcts = [a.dead_by_threshold[T].sum().item() / n_hidden * 100
                for a in all_analyses]
        ax.plot(blocks, pcts, marker="o", ms=4, label=f"T={T:.0f}")
    ax.set_xlabel("Block"); ax.set_ylabel("Dead channels (%)")
    ax.set_title("Pre-GELU Dead Channels per Block")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(blocks)

    # Panel 2: Contribution-based dead
    ax = axes[0, 1]
    for ct in CONTRIB_THRESHOLDS:
        pcts = [a.dead_by_contrib[ct].sum().item() / n_hidden * 100
                for a in all_analyses]
        ax.plot(blocks, pcts, marker="o", ms=4, label=f"<{ct:.0e}")
    ax.set_xlabel("Block"); ax.set_ylabel("Dead channels (%)")
    ax.set_title("Contribution-Based Dead Channels per Block")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(blocks)

    # Panel 3: Smooth pruning curve
    ax = axes[1, 0]
    fracs_pct = [pr["frac"] * 100 for pr in pruning_results]
    rels = [pr["rel_pct"] for pr in pruning_results]
    ax.plot(fracs_pct, rels, marker="o", color="steelblue", ms=5)
    ax.axhline(1.0, color="red", ls="--", alpha=0.7, label="1% threshold")
    ax.axhline(0, color="black", ls="-", alpha=0.3)
    ax.set_xlabel("Channels pruned (%)"); ax.set_ylabel("Relative PPL change (%)")
    ax.set_title("Smooth Pruning Curve (by output contribution)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: Cumulative contribution curve (average across blocks)
    ax = axes[1, 1]
    # Average the sorted contribution fractions across blocks
    all_sorted_fracs = []
    for analysis in all_analyses:
        sorted_frac, _ = torch.sort(analysis.contribution_frac)
        all_sorted_fracs.append(sorted_frac.numpy())
    avg_sorted = np.mean(all_sorted_fracs, axis=0)
    cumulative = np.cumsum(avg_sorted)
    x_pct = np.arange(1, n_hidden + 1) / n_hidden * 100
    ax.plot(x_pct, cumulative * 100, color="teal")
    ax.axhline(1, color="red", ls="--", alpha=0.5, label="1% contribution")
    ax.axhline(5, color="orange", ls="--", alpha=0.5, label="5% contribution")
    ax.set_xlabel("Channels (sorted by contribution, %)")
    ax.set_ylabel("Cumulative contribution (%)")
    ax.set_title("Cumulative Output Contribution (avg across blocks)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)  # zoom into bottom half
    ax.set_ylim(0, 20)

    plt.suptitle(f"Cycle 032: Dead Channel Census v2 — GPT-2 (GELU-aware)\n"
                 f"Outcome: {outcome} | Dead(T=-3)={avg_dead_t3:.1f}% | "
                 f"Linear(T=+3)={avg_linear_t3:.1f}% | "
                 f"Safe prune={max_safe_frac*100:.0f}%",
                 fontsize=11)
    plt.tight_layout()
    out_png = RESULTS_DIR / "cycle_032_prune_collapse.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
