"""
Cycle 025 — GELU PWL Approximation Fidelity — GPT-2
Item 33: Replace all GELU activations in GPT-2 with K-piece PWL, no retraining.
Measure perplexity (full-model sweep for K in [3,5,10,20] and block-by-block for K=3).

GPT-2 uses NewGELUActivation (tanh approximation of GELU). We approximate it with
a K-piece piecewise-linear function over [-5, 5] with:
  - x < -5: output = 0  (GELU asymptote)
  - x > 5:  output = x  (GELU asymptote, linear)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LO = -5.0
HI = 5.0
K_VALUES = [3, 5, 10, 20]
BATCH_SIZE = 4


# ── PWL activation ────────────────────────────────────────────────────────────
class PiecewiseLinearGELU(nn.Module):
    """K-piece PWL approximation of NewGELU (tanh variant) over [lo, hi].

    Outside range: 0 for x < lo, x for x > hi.
    Breakpoints uniformly spaced in [lo, hi].
    """

    def __init__(self, n_pieces: int, lo: float = LO, hi: float = HI):
        super().__init__()
        bps = torch.linspace(lo, hi, n_pieces + 1)
        # Evaluate NewGELU at breakpoints
        vals = 0.5 * bps * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (bps + 0.044715 * bps ** 3)))
        self.register_buffer("bps", bps)
        self.register_buffer("vals", vals)
        self.lo = lo
        self.hi = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bps = self.bps  # (n_pieces+1,)
        vals = self.vals

        # searchsorted: for each element find interval index
        idx = torch.searchsorted(bps.contiguous(), x.contiguous()) - 1
        idx = idx.clamp(0, len(bps) - 2)

        x0 = bps[idx]
        x1 = bps[idx + 1]
        y0 = vals[idx]
        y1 = vals[idx + 1]

        t = (x - x0) / (x1 - x0 + 1e-12)
        interp = y0 + t * (y1 - y0)

        result = torch.where(x < self.lo, torch.zeros_like(x),
                    torch.where(x > self.hi, x, interp))
        return result


# ── Perplexity ────────────────────────────────────────────────────────────────
def compute_perplexity(model, tokens: torch.Tensor, device: torch.device,
                        batch_size: int = BATCH_SIZE) -> float:
    """Compute average per-token cross-entropy loss, return perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start:start + batch_size].to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            out = model(input_ids, use_cache=False)
            logits = out.logits  # [B, T-1, V]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)).item())


# ── Activation replacement helpers ───────────────────────────────────────────
def replace_all_gelu(model, n_pieces: int) -> list[nn.Module]:
    """Replace all 12 blocks' .mlp.act with PiecewiseLinearGELU(n_pieces).
    Returns list of original act modules for restoration."""
    originals = []
    for block in model.transformer.h:
        originals.append(block.mlp.act)
        block.mlp.act = PiecewiseLinearGELU(n_pieces).to(next(model.parameters()).device)
    return originals


def restore_all_gelu(model, originals: list[nn.Module]) -> None:
    """Restore original act modules."""
    for block, orig in zip(model.transformer.h, originals):
        block.mlp.act = orig


def replace_single_gelu(model, block_idx: int, n_pieces: int) -> nn.Module:
    """Replace only block block_idx's GELU. Returns original."""
    orig = model.transformer.h[block_idx].mlp.act
    model.transformer.h[block_idx].mlp.act = PiecewiseLinearGELU(n_pieces).to(
        next(model.parameters()).device
    )
    return orig


def restore_single_gelu(model, block_idx: int, orig: nn.Module) -> None:
    model.transformer.h[block_idx].mlp.act = orig


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Load model
    print("Loading GPT-2...")
    load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
    bundle = load_nanochat_teacher(load_spec)
    model = bundle.model
    device = bundle.device
    print(f"  device: {device}")

    # Load probe corpus (validation tokens)
    print("Building probe corpus...")
    corpus_spec = ProbeCorpusSpec(
        teacher=bundle,
        text_path="data/shakespeare.txt",
        sequence_length=128,
        probe_train_sequences=64,
        probe_holdout_sequences=32,
        eval_val_sequences=32,
        dataset_name="proxy_shakespeare",
    )
    corpus = build_probe_corpus(corpus_spec)
    val_tokens = corpus.probe_holdout  # [32, 129]
    print(f"  val tokens shape: {val_tokens.shape}")

    # Baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, val_tokens, device)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ── Full-model sweep ──────────────────────────────────────────────────────
    print("\nFull-model GELU replacement sweep:")
    full_sweep = {}
    for K in K_VALUES:
        originals = replace_all_gelu(model, K)
        ppl = compute_perplexity(model, val_tokens, device)
        restore_all_gelu(model, originals)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        full_sweep[K] = {"ppl": ppl, "rel_increase_pct": rel}
        print(f"  K={K:2d}: PPL={ppl:.2f}  ({rel:+.1f}%)")

    # ── Block-by-block sweep for K=3 ─────────────────────────────────────────
    print("\nBlock-by-block sweep (K=3):")
    block_sweep = {}
    n_blocks = len(model.transformer.h)
    for i in range(n_blocks):
        orig = replace_single_gelu(model, i, n_pieces=3)
        ppl = compute_perplexity(model, val_tokens, device)
        restore_single_gelu(model, i, orig)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        block_sweep[i] = {"ppl": ppl, "rel_increase_pct": rel}
        print(f"  block {i:2d}: PPL={ppl:.2f}  ({rel:+.1f}%)")

    # ── Determine outcome ─────────────────────────────────────────────────────
    rel_3piece = full_sweep[3]["rel_increase_pct"]
    if rel_3piece < 5.0:
        outcome = "CONFIRMED"
    elif rel_3piece > 20.0:
        outcome = "REFUTED"
    else:
        outcome = "INCONCLUSIVE"

    print(f"\nOutcome: {outcome}")
    print(f"  3-piece PPL={full_sweep[3]['ppl']:.2f}, relative increase={rel_3piece:+.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "cycle": 25,
        "item": 33,
        "outcome": outcome,
        "baseline_ppl": baseline_ppl,
        "full_sweep": {str(k): v for k, v in full_sweep.items()},
        "block_sweep_k3": {str(k): v for k, v in block_sweep.items()},
        "range": [LO, HI],
    }
    json_path = RESULTS_DIR / "cycle_025_gelu_pwl_gpt2.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Top: full-model sweep
    ks = K_VALUES
    ppls = [full_sweep[k]["ppl"] for k in ks]
    rels = [full_sweep[k]["rel_increase_pct"] for k in ks]
    ax1.axhline(baseline_ppl, color="green", linestyle="--", label=f"Baseline ({baseline_ppl:.1f})")
    ax1.plot(ks, ppls, "o-", color="blue", label="PWL PPL")
    ax1.set_xlabel("Number of PWL pieces (K)")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("GPT-2: Full GELU→PWL replacement (all 12 blocks)")
    ax1.legend()
    ax1.set_xticks(ks)
    # Add relative % labels
    for k, ppl, rel in zip(ks, ppls, rels):
        ax1.annotate(f"{rel:+.1f}%", (k, ppl), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    # Bottom: block-by-block K=3
    block_ids = list(range(n_blocks))
    block_rels = [block_sweep[i]["rel_increase_pct"] for i in block_ids]
    colors = ["red" if r > 5 else "orange" if r > 1 else "steelblue" for r in block_rels]
    ax2.bar(block_ids, block_rels, color=colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("GPT-2 Block Index")
    ax2.set_ylabel("Relative PPL increase (%)")
    ax2.set_title("Per-block GELU→3-piece PWL (one block at a time)")
    ax2.set_xticks(block_ids)

    plt.tight_layout()
    png_path = RESULTS_DIR / "cycle_025_gelu_pwl_gpt2.png"
    plt.savefig(png_path, dpi=120)
    print(f"Saved: {png_path}")

    return result


if __name__ == "__main__":
    main()
