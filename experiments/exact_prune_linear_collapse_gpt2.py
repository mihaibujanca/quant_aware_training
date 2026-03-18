"""
Cycle 028 — Exact Prune Plus Linear Collapse Baseline — GPT-2
Item 16: Test how much "free" (no-SGD) FFN compression is achievable via:
  (a) dead channel pruning (max|act| < threshold)
  (b) always-active / near-linear channel collapse (GELU → identity)
  (c) lstsq refit of remaining weights on probe data

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

DEAD_THRESHOLD = 1e-5          # max|act| < this → dead
NEAR_LINEAR_RATIO = 0.05       # |gelu(h)-h|/|h| threshold for linearity
NEAR_LINEAR_FRAC = 0.95        # fraction of tokens that must be near-linear
BATCH_SIZE = 4
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50


# ── GELU (NewGELU tanh approximation) ─────────────────────────────────────────
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
    """
    Collect FFN pre-activations (h = W_up @ x) and post-activations (act = gelu(h))
    for all 12 blocks. Returns list of dicts with keys 'h', 'act', 'x_in', 'out'.
    """
    n_blocks = len(model.transformer.h)
    all_data: list[dict] = [{"h": [], "act": [], "x_in": [], "out": []} for _ in range(n_blocks)]

    hooks = []
    def make_hook(block_idx: int):
        def hook(module, inputs, output):
            # inputs[0] is the hidden state fed into MLP
            x = inputs[0].detach()  # (B, T, 768)
            h = module.c_fc(x).detach()  # (B, T, 3072) before activation
            act = new_gelu(h)            # (B, T, 3072)
            out = output.detach()        # (B, T, 768)
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

    # Concatenate all collected data
    result = []
    for d in all_data:
        result.append({k: torch.cat(v, dim=0) for k, v in d.items()})
    return result


# ── Channel analysis ───────────────────────────────────────────────────────────
class ChannelMasks(NamedTuple):
    dead: torch.Tensor        # (3072,) bool
    always_active: torch.Tensor   # (3072,) bool — positive AND near-linear
    near_linear: torch.Tensor     # (3072,) bool — near-linear only
    live_gelu: torch.Tensor       # (3072,) bool — everything else (true nonlinear)


def analyze_channels(h: torch.Tensor, act: torch.Tensor) -> ChannelMasks:
    """
    h: (N, 3072) pre-GELU activations
    act: (N, 3072) post-GELU activations
    """
    # Dead: max absolute activation below threshold
    max_abs = act.abs().max(dim=0).values  # (3072,)
    dead = max_abs < DEAD_THRESHOLD

    # Near-linear ratio: |gelu(h) - h| / (|h| + eps)
    ratio = (act - h).abs() / (h.abs() + 1e-8)  # (N, 3072)
    frac_near_linear = (ratio < NEAR_LINEAR_RATIO).float().mean(dim=0)  # (3072,)
    near_linear_mask = frac_near_linear > NEAR_LINEAR_FRAC

    # Always-active: near-linear AND all activations > 0 (channel is in linear regime)
    all_positive = (act.min(dim=0).values > 0)
    always_active = near_linear_mask & all_positive & ~dead

    # Near-linear (broader): includes channels that are near-linear but may have neg values
    near_linear_only = near_linear_mask & ~dead & ~always_active

    # Live GELU: truly nonlinear
    live_gelu = ~dead & ~always_active & ~near_linear_only

    return ChannelMasks(dead=dead, always_active=always_active,
                        near_linear=near_linear_only, live_gelu=live_gelu)


# ── Patched MLP module ─────────────────────────────────────────────────────────
class PatchedMLP(nn.Module):
    """
    Wraps an existing GPT-2 MLP block with channel masking.
    Modes:
      'original': no change
      'prune': zero dead channels
      'collapse': zero dead + replace near-linear/always-active with identity
      'refit': zero dead + collapse + use refitted W_down
    """

    def __init__(self, original_mlp: nn.Module, masks: ChannelMasks,
                 mode: str = "prune", W_down_refit: torch.Tensor | None = None):
        super().__init__()
        self.mlp = original_mlp
        self.masks = masks
        self.mode = mode
        # Register W_down_refit as buffer if provided
        if W_down_refit is not None:
            self.register_buffer("W_down_refit", W_down_refit)
        else:
            self.W_down_refit = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # c_fc: linear 768→3072, c_proj: linear 3072→768
        h = self.mlp.c_fc(x)         # (..., 3072)
        act = new_gelu(h)            # (..., 3072)

        if self.mode == "original":
            return self.mlp.c_proj(act)

        dev = h.device
        dead = self.masks.dead.to(dev)
        always_active = self.masks.always_active.to(dev)
        near_linear = self.masks.near_linear.to(dev)

        if self.mode in ("prune", "collapse", "refit"):
            # Zero out dead channels
            act = act * (~dead).float()

        if self.mode in ("collapse", "refit"):
            # Replace always-active and near-linear channels with linear (h instead of gelu(h))
            linear_mask = always_active | near_linear
            act = torch.where(linear_mask, h, act)
            # Dead channels already zeroed above

        if self.mode == "refit" and self.W_down_refit is not None:
            # Use refitted W_down instead of original c_proj weights
            # W_down_refit: (768, 3072)
            out = act @ self.W_down_refit.T
            return out

        return self.mlp.c_proj(act)


# ── Refit W_down via least squares ────────────────────────────────────────────
def refit_w_down(act_probe: torch.Tensor, out_probe: torch.Tensor,
                 masks: ChannelMasks, device: torch.device) -> torch.Tensor:
    """
    Solve: W_down_new @ act_probe_patched.T ≈ out_probe.T
    where act_probe_patched has dead channels zeroed and linear channels unmodified.

    act_probe: (N, 3072) — activations AFTER patching (dead zeroed, linear channels = h)
    out_probe: (N, 768)  — original block output (target)
    Returns W_down_new: (768, 3072)
    """
    # Identify live channels (not dead)
    live = ~masks.dead  # (3072,)
    act_live = act_probe[:, live].to(device)    # (N, n_live)
    out = out_probe.to(device)                  # (N, 768)

    # lstsq: find X such that act_live @ X ≈ out  → X: (n_live, 768)
    result = torch.linalg.lstsq(act_live, out)
    X = result.solution  # (n_live, 768)

    # Reconstruct full W_down (768, 3072) with zeros for dead channels
    W_orig = next(p for name, p in torch.nn.utils.parametrize.parameters_and_names
                  if name == "c_proj.weight") if False else None

    # Build W_down_refit: (768, 3072)
    W_refit = torch.zeros(3072, 768, device=device, dtype=act_live.dtype)
    W_refit[live] = X  # X is (n_live, 768), W_refit indexed by channel
    return W_refit.T  # (768, 3072) to match c_proj.weight shape convention


# ── Patch/restore helpers ──────────────────────────────────────────────────────
def patch_block(model: nn.Module, block_idx: int, masks: ChannelMasks,
                mode: str, W_down_refit: torch.Tensor | None = None):
    """Replace block_idx's MLP with PatchedMLP. Returns original MLP."""
    orig_mlp = model.transformer.h[block_idx].mlp
    patched = PatchedMLP(orig_mlp, masks, mode=mode, W_down_refit=W_down_refit)
    model.transformer.h[block_idx].mlp = patched
    return orig_mlp


def restore_block(model: nn.Module, block_idx: int, orig_mlp: nn.Module):
    model.transformer.h[block_idx].mlp = orig_mlp


def patch_all_blocks(model: nn.Module, all_masks: list[ChannelMasks], mode: str,
                     all_refit: list[torch.Tensor | None] | None = None):
    """Patch all 12 blocks. Returns list of originals."""
    originals = []
    for i, masks in enumerate(all_masks):
        refit = all_refit[i] if all_refit else None
        orig = patch_block(model, i, masks, mode=mode, W_down_refit=refit)
        originals.append(orig)
    return originals


def restore_all_blocks(model: nn.Module, originals: list[nn.Module]):
    for i, orig in enumerate(originals):
        restore_block(model, i, orig)


# ── Compute patched activations for refit ────────────────────────────────────
def get_patched_activations(h: torch.Tensor, act: torch.Tensor,
                             masks: ChannelMasks) -> torch.Tensor:
    """Apply dead-zero + linear-collapse to activations for lstsq refit."""
    dead = masks.dead
    linear_mask = masks.always_active | masks.near_linear
    patched_act = act.clone()
    patched_act[:, dead] = 0.0
    patched_act[:, linear_mask] = h[:, linear_mask]
    return patched_act


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
    probe_tokens = corpus.probe_train   # (200, 129)
    val_tokens = corpus.probe_holdout   # (50, 129)
    print(f"  probe tokens: {probe_tokens.shape}, val tokens: {val_tokens.shape}")

    # Baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, val_tokens, device)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ── Collect activations on probe set ─────────────────────────────────────
    print("Collecting FFN activations on probe set...")
    block_data = collect_ffn_activations(model, probe_tokens, device)
    print(f"  Collected {block_data[0]['h'].shape[0]} token activations per block")

    # ── Analyze channels per block ────────────────────────────────────────────
    print("\nChannel analysis per block:")
    print(f"  {'Block':>5}  {'Dead%':>6}  {'AlwAct%':>8}  {'NearLin%':>9}  {'LiveGELU%':>10}  {'Total_saved%':>13}")

    all_masks: list[ChannelMasks] = []
    block_stats = []
    n_hidden = 3072

    for i, data in enumerate(block_data):
        masks = analyze_channels(data["h"], data["act"])
        all_masks.append(masks)

        n_dead = masks.dead.sum().item()
        n_aa = masks.always_active.sum().item()
        n_nl = masks.near_linear.sum().item()
        n_live = masks.live_gelu.sum().item()

        dead_pct = 100 * n_dead / n_hidden
        aa_pct = 100 * n_aa / n_hidden
        nl_pct = 100 * n_nl / n_hidden
        live_pct = 100 * n_live / n_hidden
        saved_pct = dead_pct + aa_pct + nl_pct

        print(f"  {i:>5}  {dead_pct:>6.2f}  {aa_pct:>8.2f}  {nl_pct:>9.2f}  {live_pct:>10.2f}  {saved_pct:>13.2f}")
        block_stats.append({
            "block": i,
            "n_dead": int(n_dead),
            "n_always_active": int(n_aa),
            "n_near_linear": int(n_nl),
            "n_live_gelu": int(n_live),
            "pct_dead": dead_pct,
            "pct_always_active": aa_pct,
            "pct_near_linear": nl_pct,
            "pct_saved": saved_pct,
        })

    # Summary stats
    total_saved = sum(s["pct_saved"] for s in block_stats) / len(block_stats)
    total_dead = sum(s["pct_dead"] for s in block_stats) / len(block_stats)
    print(f"\n  Mean dead%: {total_dead:.2f}%, mean saved (dead+collapse)%: {total_saved:.2f}%")

    # Compute parameter reduction ratio for pruning
    # Each block has W_up (768×3072) + W_down (3072×768) = 2 × 768 × 3072 params
    # Pruning n channels removes 2 × 768 × n params from effective computation
    avg_pct_removed = total_dead  # only dead channels can be truly removed
    param_reduction = 1.0 / (1.0 - avg_pct_removed / 100 + 1e-8)
    print(f"  Param reduction (prune-only): {param_reduction:.3f}x")

    # ── Experiment 1: Prune-only (all blocks) ────────────────────────────────
    print("\n--- Prune-only (all 12 blocks) ---")
    originals = patch_all_blocks(model, all_masks, mode="prune")
    ppl_prune = compute_perplexity(model, val_tokens, device)
    restore_all_blocks(model, originals)
    rel_prune = (ppl_prune - baseline_ppl) / baseline_ppl * 100
    print(f"  PPL: {ppl_prune:.2f}  ({rel_prune:+.2f}%)")

    # ── Experiment 2: Prune + Collapse (all blocks) ───────────────────────────
    print("\n--- Prune + Collapse (all 12 blocks) ---")
    originals = patch_all_blocks(model, all_masks, mode="collapse")
    ppl_collapse = compute_perplexity(model, val_tokens, device)
    restore_all_blocks(model, originals)
    rel_collapse = (ppl_collapse - baseline_ppl) / baseline_ppl * 100
    print(f"  PPL: {ppl_collapse:.2f}  ({rel_collapse:+.2f}%)")

    # ── Experiment 3: Prune + Collapse + Refit (all blocks) ──────────────────
    print("\n--- Prune + Collapse + Refit (all 12 blocks) ---")
    all_refit: list[torch.Tensor | None] = []
    for i, (masks, data) in enumerate(zip(all_masks, block_data)):
        patched_act = get_patched_activations(data["h"], data["act"], masks)
        out = data["out"]
        # lstsq: act_patched @ X ≈ out
        live = ~masks.dead
        act_live = patched_act[:, live].to(device)
        out_dev = out.to(device)
        try:
            result = torch.linalg.lstsq(act_live, out_dev, driver="gelsd")
            X = result.solution  # (n_live, 768)
            # Build W_down_refit (768, 3072)
            W_refit = torch.zeros(3072, 768, device=device, dtype=act_live.dtype)
            W_refit[live] = X
            all_refit.append(W_refit.T.contiguous())  # (768, 3072)
            if i % 4 == 0:
                print(f"  Block {i}: lstsq solved, live channels={live.sum().item()}")
        except Exception as e:
            print(f"  Block {i}: lstsq failed ({e}), using original W_down")
            all_refit.append(None)

    originals = patch_all_blocks(model, all_masks, mode="refit", all_refit=all_refit)
    ppl_refit = compute_perplexity(model, val_tokens, device)
    restore_all_blocks(model, originals)
    rel_refit = (ppl_refit - baseline_ppl) / baseline_ppl * 100
    print(f"  PPL: {ppl_refit:.2f}  ({rel_refit:+.2f}%)")

    # ── Outcome determination ─────────────────────────────────────────────────
    # Confirmed: ≥1.1x param reduction with ≤1% ppl increase
    # Refuted: param_reduction < 1.05 or best ppl increase > 1% at meaningful budget
    best_ppl_rel = min(abs(rel_prune), abs(rel_collapse), abs(rel_refit))
    best_compression = param_reduction  # based on dead channels only

    if best_compression >= 1.1 and min(rel_prune, rel_collapse, rel_refit) <= 1.0:
        outcome = "CONFIRMED"
    elif best_compression < 1.05 or (rel_prune > 1.0 and rel_collapse > 1.0):
        outcome = "REFUTED"
    else:
        outcome = "INCONCLUSIVE"

    print(f"\n=== Outcome: {outcome} ===")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  Prune-only PPL: {ppl_prune:.2f} ({rel_prune:+.2f}%)")
    print(f"  Prune+Collapse PPL: {ppl_collapse:.2f} ({rel_collapse:+.2f}%)")
    print(f"  Prune+Collapse+Refit PPL: {ppl_refit:.2f} ({rel_refit:+.2f}%)")
    print(f"  Param reduction (prune-only): {param_reduction:.4f}x")
    print(f"  Mean dead channels: {total_dead:.2f}%")
    print(f"  Mean saved channels (dead+collapse): {total_saved:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "cycle": 28,
        "item": 16,
        "outcome": outcome,
        "baseline_ppl": baseline_ppl,
        "ppl_prune_only": ppl_prune,
        "ppl_prune_collapse": ppl_collapse,
        "ppl_prune_collapse_refit": ppl_refit,
        "rel_prune_pct": rel_prune,
        "rel_collapse_pct": rel_collapse,
        "rel_refit_pct": rel_refit,
        "param_reduction_prune_only": param_reduction,
        "mean_pct_dead": total_dead,
        "mean_pct_saved_collapse": total_saved,
        "thresholds": {
            "dead": DEAD_THRESHOLD,
            "near_linear_ratio": NEAR_LINEAR_RATIO,
            "near_linear_frac": NEAR_LINEAR_FRAC,
        },
        "per_block": block_stats,
    }

    out_json = RESULTS_DIR / "cycle_028_prune_collapse.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-block channel fractions stacked bar
    ax = axes[0]
    blocks = [s["block"] for s in block_stats]
    dead_pcts = [s["pct_dead"] for s in block_stats]
    aa_pcts = [s["pct_always_active"] for s in block_stats]
    nl_pcts = [s["pct_near_linear"] for s in block_stats]
    live_pcts = [100.0 - s["pct_dead"] - s["pct_always_active"] - s["pct_near_linear"] for s in block_stats]

    ax.bar(blocks, dead_pcts, label="Dead", color="red", alpha=0.8)
    ax.bar(blocks, aa_pcts, bottom=dead_pcts, label="Always-active (linear regime)", color="orange", alpha=0.8)
    ax.bar(blocks, nl_pcts, bottom=[d + a for d, a in zip(dead_pcts, aa_pcts)],
           label="Near-linear only", color="yellow", alpha=0.8)
    ax.bar(blocks, live_pcts, bottom=[d + a + n for d, a, n in zip(dead_pcts, aa_pcts, nl_pcts)],
           label="Live GELU", color="steelblue", alpha=0.8)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Channel fraction (%)")
    ax.set_title("GPT-2 FFN Channel Classification per Block")
    ax.legend(fontsize=8)
    ax.set_xticks(blocks)

    # Right: PPL comparison
    ax = axes[1]
    labels = ["Baseline", "Prune-only", "Prune+Collapse", "Prune+Collapse\n+Refit"]
    ppls = [baseline_ppl, ppl_prune, ppl_collapse, ppl_refit]
    colors = ["steelblue", "orange", "red", "green"]
    bars = ax.bar(labels, ppls, color=colors, alpha=0.8)
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity by Compression Method")
    ax.axhline(baseline_ppl * 1.01, color="black", linestyle="--", linewidth=1, label="1% above baseline")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)

    plt.suptitle(f"Cycle 028: Exact Prune + Linear Collapse — GPT-2\n"
                 f"Outcome: {outcome} | Dead={total_dead:.2f}% | Saved={total_saved:.2f}%",
                 fontsize=11)
    plt.tight_layout()
    out_png = RESULTS_DIR / "cycle_028_prune_collapse.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
