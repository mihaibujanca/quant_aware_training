"""
Cycle 011 — Item 21: Single-affine collapse quality on GPT-2 FFN blocks.

For each of 12 GPT-2 FFN blocks, fit a single affine map A, b from probe tokens,
measure normalized residual r on held-out tokens, then swap in the affine
approximation for the 2 best blocks and measure perplexity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _extract_ffn_handles,
    build_probe_corpus,
    load_nanochat_teacher,
)


# ── Cycle 005 single-block swap damage (relative PPL increase) ──
CYCLE_005_DAMAGE = {
    0: 0.3312, 1: 0.0074, 2: 0.0114, 3: 0.0100,
    4: 0.0064, 5: 0.0012, 6: 0.0100, 7: -0.0063,
    8: 0.0195, 9: 0.0163, 10: 0.0181, 11: 0.0652,
}


def collect_ffn_io(
    model: nn.Module,
    mlp_module: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    max_tokens: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect FFN (h_in, h_out) pairs via forward hook.

    Returns:
        h_in:  [N, 768] - input to MLP block
        h_out: [N, 768] - output of MLP block
    """
    chunks_in: list[torch.Tensor] = []
    chunks_out: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):
        x_in = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
        x_out = output.detach().reshape(-1, output.shape[-1]).cpu()
        remaining = max_tokens - sum(c.shape[0] for c in chunks_in)
        if remaining > 0:
            take = min(remaining, x_in.shape[0])
            chunks_in.append(x_in[:take])
            chunks_out.append(x_out[:take])

    hook = mlp_module.register_forward_hook(hook_fn)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, input_ids.shape[0], 4):
                batch = input_ids[start:start + 4].to(device)
                model(batch[:, :-1])
                if sum(c.shape[0] for c in chunks_in) >= max_tokens:
                    break
    finally:
        hook.remove()

    h_in = torch.cat(chunks_in, dim=0).float()
    h_out = torch.cat(chunks_out, dim=0).float()
    return h_in, h_out


def fit_affine(h_in: torch.Tensor, h_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit h_out ≈ h_in @ A + b via least squares.

    Returns:
        A: [d_in, d_out]
        b: [d_out]
    """
    n = h_in.shape[0]
    H_aug = torch.cat([h_in, torch.ones(n, 1)], dim=1)  # [N, d+1]
    try:
        sol = torch.linalg.lstsq(H_aug, h_out, driver="gelsd")
        coef = sol.solution  # [d+1, d_out]
    except Exception:
        coef = torch.linalg.pinv(H_aug) @ h_out
    A = coef[:-1]   # [d_in, d_out]
    b = coef[-1]    # [d_out]
    return A, b


def normalized_residual(h_in: torch.Tensor, h_out: torch.Tensor,
                         A: torch.Tensor, b: torch.Tensor) -> float:
    """r = ||h_out - h_in @ A - b||_F / ||h_out - mean(h_out)||_F"""
    pred = h_in @ A + b
    resid_norm = (h_out - pred).norm()
    denom = (h_out - h_out.mean(dim=0, keepdim=True)).norm()
    if denom < 1e-10:
        return float("nan")
    return (resid_norm / denom).item()


def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute cross-entropy perplexity on sequences."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], 4):
            batch = input_ids[start:start + 4].to(device)
            # Causal LM: predict next token
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            out = model(inputs)
            logits = out.logits  # [B, T, V]
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += B * T
    return math.exp(total_nll / total_tokens)


def main():
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model and corpus ──
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device=str(device))
    )
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path="data/shakespeare.txt",
            sequence_length=128,
            probe_train_sequences=64,
            probe_holdout_sequences=32,
            eval_val_sequences=32,
            dataset_name="proxy_shakespeare",
        )
    )

    model = teacher.model
    model.eval()
    handles = _extract_ffn_handles(model, teacher.activation_kind)
    print(f"Loaded GPT-2 with {len(handles)} FFN blocks")
    print(f"Probe train seqs: {corpus.probe_train.shape}, holdout: {corpus.probe_holdout.shape}")

    # ── Baseline perplexity ──
    baseline_ppl = compute_perplexity(model, corpus.eval_val, device)
    print(f"\nBaseline perplexity (eval_val): {baseline_ppl:.2f}")
    if not (20 <= baseline_ppl <= 60):
        print(f"WARNING: baseline ppl={baseline_ppl:.2f} outside expected 20-60 range")

    # ── Per-block affine fit ──
    block_results = {}

    print(f"\n{'Block':>5}  {'r_fit':>8}  {'r_held':>8}  {'L_cycle009':>10}")
    print("=" * 45)

    # Need > 769 tokens for overdetermined system (768-dim input + bias column).
    # Use 1024 for fit (overdetermined with margin) and 512 for held-out.
    FIT_TOKENS = 1024
    HELD_TOKENS = 512

    for handle in handles:
        lid = handle.layer_id
        mlp = handle.mlp

        # Collect fit split (1024 tokens from probe_train)
        h_in_fit, h_out_fit = collect_ffn_io(
            model, mlp, corpus.probe_train, device, max_tokens=FIT_TOKENS
        )
        # Collect held-out split (512 tokens from probe_holdout)
        h_in_held, h_out_held = collect_ffn_io(
            model, mlp, corpus.probe_holdout, device, max_tokens=HELD_TOKENS
        )
        print(f"  Block {lid}: fit={h_in_fit.shape[0]} tokens, held={h_in_held.shape[0]} tokens")

        A, b = fit_affine(h_in_fit, h_out_fit)
        r_fit = normalized_residual(h_in_fit, h_out_fit, A, b)
        r_held = normalized_residual(h_in_held, h_out_held, A, b)

        # Sanity: r_fit should be <= r_held (no overfitting if data >> params)
        if r_fit > r_held + 0.05:
            print(f"  Block {lid}: WARNING r_fit={r_fit:.4f} > r_held={r_held:.4f}")

        block_results[lid] = {
            "r_fit": r_fit,
            "r_held": r_held,
            "A": A,
            "b": b,
            "damage_cycle005": CYCLE_005_DAMAGE.get(lid, None),
        }

        print(f"  {lid:3d}   {r_fit:8.4f}  {r_held:8.4f}  {CYCLE_005_DAMAGE.get(lid, 0.0):10.4f}")

    # ── Identify 2 best blocks ──
    sorted_blocks = sorted(block_results.keys(), key=lambda l: block_results[l]["r_held"])
    best_two = sorted_blocks[:2]
    print(f"\nBest 2 blocks by r_held: {best_two}")
    print(f"  Block {best_two[0]}: r_held={block_results[best_two[0]]['r_held']:.4f}")
    print(f"  Block {best_two[1]}: r_held={block_results[best_two[1]]['r_held']:.4f}")

    # ── Perplexity swap test ──
    def make_affine_forward(A: torch.Tensor, b: torch.Tensor, dev: torch.device) -> Callable:
        A_dev = A.to(dev)
        b_dev = b.to(dev)
        def affine_forward(x):
            return x @ A_dev + b_dev
        return affine_forward

    swap_results = {}

    # Swap individual best blocks
    for lid in best_two:
        A = block_results[lid]["A"]
        b = block_results[lid]["b"]
        mlp_module = handles[lid].mlp

        # Monkey-patch forward
        original_forward = mlp_module.forward
        mlp_module.forward = make_affine_forward(A, b, device)
        try:
            ppl_swapped = compute_perplexity(model, corpus.eval_val, device)
        finally:
            mlp_module.forward = original_forward

        rel_increase = (ppl_swapped - baseline_ppl) / baseline_ppl
        swap_results[lid] = {
            "ppl_swapped": ppl_swapped,
            "ppl_baseline": baseline_ppl,
            "rel_increase": rel_increase,
        }
        print(f"  Block {lid}: ppl_swapped={ppl_swapped:.2f}, rel_increase={rel_increase:.4f} ({rel_increase*100:.1f}%)")

    # Swap both best blocks simultaneously
    original_fwds = {}
    for lid in best_two:
        A = block_results[lid]["A"]
        b = block_results[lid]["b"]
        mlp_module = handles[lid].mlp
        original_fwds[lid] = mlp_module.forward
        mlp_module.forward = make_affine_forward(A, b, device)

    try:
        ppl_both = compute_perplexity(model, corpus.eval_val, device)
    finally:
        for lid in best_two:
            handles[lid].mlp.forward = original_fwds[lid]

    rel_both = (ppl_both - baseline_ppl) / baseline_ppl
    print(f"\n  Both blocks {best_two}: ppl={ppl_both:.2f}, rel_increase={rel_both:.4f} ({rel_both*100:.1f}%)")

    # ── Verdict ──
    min_r_held = min(block_results[l]["r_held"] for l in block_results)
    max_r_held = max(block_results[l]["r_held"] for l in block_results)
    blocks_below_015 = [l for l in block_results if block_results[l]["r_held"] < 0.15]
    max_rel_increase = max(swap_results[l]["rel_increase"] for l in swap_results)

    if len(blocks_below_015) >= 2 and max_rel_increase <= 0.05:
        verdict = "confirmed"
        verdict_reason = (
            f"{len(blocks_below_015)} blocks have r_held < 0.15 "
            f"(min={min_r_held:.4f}) AND max ppl increase={max_rel_increase*100:.1f}% <= 5%."
        )
    elif min_r_held > 0.5 or max_rel_increase > 0.20:
        verdict = "refuted"
        verdict_reason = (
            f"min r_held={min_r_held:.4f} > 0.5 OR "
            f"max ppl increase={max_rel_increase*100:.1f}% > 20%."
        )
    else:
        verdict = "inconclusive"
        verdict_reason = (
            f"r values moderate (range {min_r_held:.4f}–{max_r_held:.4f}), "
            f"blocks_below_0.15={len(blocks_below_015)}, "
            f"max ppl increase={max_rel_increase*100:.1f}%."
        )

    print(f"\nVerdict: {verdict.upper()}")
    print(f"Reason: {verdict_reason}")

    # ── Save JSON ──
    out_data = {
        "experiment": "single_affine_collapse_gpt2",
        "cycle": 11,
        "item": 21,
        "baseline_ppl": baseline_ppl,
        "per_block": {
            str(l): {
                "r_fit": block_results[l]["r_fit"],
                "r_held": block_results[l]["r_held"],
                "damage_cycle005": block_results[l]["damage_cycle005"],
            }
            for l in sorted(block_results.keys())
        },
        "best_two_blocks": best_two,
        "swap_results": {
            str(l): {
                "ppl_swapped": swap_results[l]["ppl_swapped"],
                "ppl_baseline": swap_results[l]["ppl_baseline"],
                "rel_increase": swap_results[l]["rel_increase"],
            }
            for l in swap_results
        },
        "both_blocks_swap": {
            "blocks": best_two,
            "ppl_swapped": ppl_both,
            "ppl_baseline": baseline_ppl,
            "rel_increase": rel_both,
        },
        "min_r_held": min_r_held,
        "max_r_held": max_r_held,
        "blocks_below_015": blocks_below_015,
        "max_rel_ppl_increase": max_rel_increase,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    out_json = Path("autoresearch/results/cycle_011_single_affine_gpt2.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # ── Plot ──
    layers = sorted(block_results.keys())
    r_fits = [block_results[l]["r_fit"] for l in layers]
    r_helds = [block_results[l]["r_held"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cycle 011: Single-Affine Collapse Quality — GPT-2 FFN Blocks", fontsize=13)

    ax = axes[0]
    x = range(len(layers))
    ax.bar([i - 0.2 for i in x], r_fits, 0.35, label="r_fit", color="steelblue", alpha=0.8)
    ax.bar([i + 0.2 for i in x], r_helds, 0.35, label="r_held", color="tomato", alpha=0.8)
    ax.axhline(0.15, color="green", linestyle="--", label="r=0.15 (confirmed)")
    ax.axhline(0.5, color="red", linestyle="--", label="r=0.5 (refuted)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("FFN Block")
    ax.set_ylabel("Normalized Residual r")
    ax.set_title("Affine Fit Residual per Block")
    ax.legend(fontsize=8)

    # Annotate best two
    for lid in best_two:
        idx = layers.index(lid)
        ax.annotate(f"best\nr={block_results[lid]['r_held']:.3f}",
                    (idx + 0.2, r_helds[idx] + 0.01), ha="center", fontsize=7, color="darkred")

    ax2 = axes[1]
    swap_blocks = list(swap_results.keys())
    swap_rels = [swap_results[l]["rel_increase"] * 100 for l in swap_blocks]
    colors = ["green" if r <= 5 else "orange" if r <= 20 else "red" for r in swap_rels]
    ax2.bar(range(len(swap_blocks)), swap_rels, color=colors, alpha=0.8)
    ax2.axhline(5, color="green", linestyle="--", label="5% threshold (confirmed)")
    ax2.axhline(20, color="red", linestyle="--", label="20% threshold (refuted)")
    ax2.set_xticks(range(len(swap_blocks)))
    ax2.set_xticklabels([f"Block {l}\nr={block_results[l]['r_held']:.3f}" for l in swap_blocks], fontsize=9)
    ax2.set_xlabel("Swapped Block")
    ax2.set_ylabel("Relative PPL Increase (%)")
    ax2.set_title(f"PPL Swap Test (baseline={baseline_ppl:.1f})")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_png = Path("autoresearch/results/cycle_011_single_affine_gpt2.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved plot: {out_png}")

    print(f"\nDone. Verdict={verdict}")
    return verdict, out_data


if __name__ == "__main__":
    main()
