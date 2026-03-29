"""Cycle 107: Output-space direct regression bypass — GPT-2 FFN (item 117).

For each block, bypasses the hidden GELU layer entirely with a rank-r linear map
learned directly from FFN input X → FFN output Y. Compared to topk+lstsq at
matched param count (2×768×r params each).

Methods:
  direct:    SVD(X) top-r basis → lstsq Y = X @ V_r @ A^T + b
  topk:      select top-k channels → lstsq W_out

Blocks: [1, 4, 7, 11], r/k in {32, 64, 128, 256, 384}, seeds [42, 123, 7].

Optimized: hook-based block replacement avoids deepcopy overhead.
Reduced eval to 50 sequences for speed (all comparisons use same eval).

Verdict:
  CONFIRMED   if direct beats topk in >=2/4 blocks (any rank)
  REFUTED     if topk wins by >=5% PPL everywhere
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from contextlib import contextmanager
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
TARGET_BLOCKS = [1, 4, 7, 11]
RANKS = [32, 64, 128, 256, 384]
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
EVAL_VAL_SEQS = 100  # balanced: fewer than 200 for speed, enough for stable comparisons
SEQ_LEN = 128
BATCH_SIZE = 8       # larger batch for faster throughput

# Abort bounds: relaxed upper bound for smaller eval set (higher variance)
PPL_MIN, PPL_MAX = 25.0, 80.0

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_ppl(model: nn.Module, eval_tokens: torch.Tensor,
              device: torch.device, batch_size: int = BATCH_SIZE) -> float:
    """Evaluate PPL on pre-tokenized eval_tokens (n_seqs, seq_len)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, eval_tokens.shape[0], batch_size):
            batch = eval_tokens[start:start + batch_size].to(device)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            raw = model(inputs)
            logits = raw[0] if isinstance(raw, tuple) else (
                raw.logits if hasattr(raw, "logits") else raw)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int(targets.numel())
    return float(math.exp(min(total_loss / max(total_tokens, 1), 20.0)))


@contextmanager
def _patch_mlp(model: nn.Module, block_id: int, replacement: nn.Module):
    """Temporarily replace block's MLP with replacement, restore on exit."""
    block = model.transformer.h[block_id]
    original = block.mlp
    block.mlp = replacement
    try:
        yield
    finally:
        block.mlp = original


def _collect_ffn_io(model: nn.Module, block_ids: list[int], tokens: torch.Tensor,
                    device: torch.device, max_tokens: int = MAX_TOKENS_PER_LAYER
                    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Collect FFN (input X, output Y) for given blocks via hooks."""
    io: dict[int, list] = {b: [[], []] for b in block_ids}
    hooks = []

    for block_id in block_ids:
        block = model.transformer.h[block_id]
        mlp = block.mlp

        def make_hook(bid):
            def hook_fn(module, inputs, output):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu()
                y = output.detach().reshape(-1, output.shape[-1]).float().cpu()
                n_have = sum(t.shape[0] for t in io[bid][0])
                remaining = max_tokens - n_have
                if remaining <= 0:
                    return
                n_take = min(remaining, x.shape[0])
                io[bid][0].append(x[:n_take])
                io[bid][1].append(y[:n_take])
            return hook_fn

        h = mlp.register_forward_hook(make_hook(block_id))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        n_seqs = tokens.shape[0]
        for start in range(0, n_seqs, BATCH_SIZE):
            batch = tokens[start:start + BATCH_SIZE].to(device)
            model(batch)
            if all(sum(t.shape[0] for t in io[b][0]) >= max_tokens for b in block_ids):
                break

    for h in hooks:
        h.remove()

    return {
        b: (
            torch.cat(io[b][0], dim=0)[:max_tokens],
            torch.cat(io[b][1], dim=0)[:max_tokens],
        )
        for b in block_ids
    }


def _direct_regression_mlp(X: torch.Tensor, Y: torch.Tensor, r: int) -> nn.Module:
    """Fit direct rank-r map Y ≈ X @ V_r @ A^T + b.
    Returns an nn.Linear(768, 768) with rank-r weight."""
    # SVD of input data
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    V_r = Vt[:r].t()  # (768, r)

    # Project inputs and add bias column
    X_r = X @ V_r
    ones = torch.ones(X_r.shape[0], 1, dtype=X_r.dtype)
    X_aug = torch.cat([X_r, ones], dim=1)

    sol = torch.linalg.lstsq(X_aug, Y, driver="gelsd")
    C = sol.solution.t()  # (768, r+1)
    A = C[:, :r]  # (768, r)
    b = C[:, r]   # (768,)

    # Full weight W = A @ V_r^T  (768×768, rank r)
    W = (A @ V_r.t()).contiguous()

    lin = nn.Linear(W.shape[1], W.shape[0], bias=True)
    lin.weight.data.copy_(W)
    lin.bias.data.copy_(b)
    return lin, A, V_r, b


def _recon_error(X: torch.Tensor, Y: torch.Tensor, mlp: nn.Module) -> float:
    """||Y - mlp(X)||_F / ||Y||_F"""
    with torch.no_grad():
        Y_hat = mlp(X)
    return float((Y - Y_hat).norm("fro") / (Y.norm("fro") + 1e-8))


def _topk_mlp_and_recon(X: torch.Tensor, Y: torch.Tensor, atlas: FfnGeometryAtlas,
                         block_id: int, k: int) -> tuple[nn.Module, float]:
    """Build topk+lstsq MLP for block_id at k hidden units and compute recon error."""
    layer = next(lg for lg in atlas.layer_geometries if lg.layer_id == block_id)
    budget = CompressionBudgetSpec(per_layer_hidden={block_id: k}, selection_mode="topk")
    selected = _select_hidden_units(layer, budget, k)

    W_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
    b_in = _standard_bias(layer.handle.c_fc)
    W_out, b_out = _fit_output_projection(
        layer.hidden_sample[:, selected].to(dtype=torch.float32),
        layer.output_sample.to(dtype=torch.float32),
        use_bias=True,
    )

    # Build a small 2-layer MLP: Linear(768→k) -> GELU -> Linear(k→768)
    # W_in has shape (hidden_dim=3072, in_dim=768); win[sel] has shape (k, 768)
    # W_out has shape (out_dim=768, k)
    class TopkMLP(nn.Module):
        def __init__(self, win, bin_, wout, bout, sel):
            super().__init__()
            k_ = len(sel)
            in_dim = win.shape[1]  # 768
            out_dim = wout.shape[0]  # 768
            self.fc = nn.Linear(in_dim, k_, bias=bin_ is not None)
            self.fc.weight.data.copy_(win[sel])  # (k, 768)
            if bin_ is not None:
                self.fc.bias.data.copy_(bin_[sel])
            self.proj = nn.Linear(k_, out_dim, bias=bout is not None)
            self.proj.weight.data.copy_(wout)  # (768, k)
            if bout is not None:
                self.proj.bias.data.copy_(bout)

        def forward(self, x):
            return self.proj(F.gelu(self.fc(x)))

    mlp = TopkMLP(W_in, b_in, W_out, b_out, selected)

    # Recon error on probe data
    with torch.no_grad():
        Y_hat = mlp(X)
    recon = float((Y - Y_hat).norm("fro") / (Y.norm("fro") + 1e-8))

    return mlp, recon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cpu")

    print("Loading GPT-2...")
    teacher_spec = TeacherLoadSpec(
        source="hf_pretrained",
        hf_model_name="gpt2",
        device="cpu",
        dataset_name="openwebtext",
    )
    teacher = load_nanochat_teacher(teacher_spec)
    model = teacher.model
    model.eval()

    print("Building probe corpus...")
    corpus_spec = ProbeCorpusSpec(
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_VAL_SEQS,
        text_path=str(OWT_PATH),
        teacher=teacher,
        shuffle_seed=42,
    )
    corpus = build_probe_corpus(corpus_spec)

    print("Baseline PPL...")
    baseline_ppl = _eval_ppl(model, corpus.eval_val, device)
    print(f"  Baseline PPL: {baseline_ppl:.3f}")
    if not (PPL_MIN <= baseline_ppl <= PPL_MAX):
        raise RuntimeError(f"PPL {baseline_ppl:.1f} outside abort bounds [{PPL_MIN},{PPL_MAX}]")

    print("Probing FFN geometry (for topk)...")
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    results: dict = {}  # (block, method, rank, seed) -> {ppl, recon_err}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        _set_seed(seed)

        # Re-sample probe tokens for this seed
        n_seqs = corpus.probe_holdout.shape[0]
        idx = torch.randperm(n_seqs)
        probe_subset = corpus.probe_holdout[idx]

        io_data = _collect_ffn_io(model, TARGET_BLOCKS, probe_subset, device)

        for block_id in TARGET_BLOCKS:
            X, Y = io_data[block_id]
            print(f"  Block {block_id}: X={X.shape}, Y={Y.shape}")

            for r in RANKS:
                # ---- Direct regression ----
                lin_direct, A, V_r, b = _direct_regression_mlp(X, Y, r)
                recon_direct = _recon_error(X, Y, lin_direct)

                with _patch_mlp(model, block_id, lin_direct):
                    ppl_direct = _eval_ppl(model, corpus.eval_val, device)

                results[(block_id, "direct", r, seed)] = {
                    "ppl": ppl_direct, "recon_err": recon_direct
                }
                print(f"    direct r={r:3d}: PPL={ppl_direct:.3f}  recon={recon_direct:.4f}")

                # ---- topk + lstsq ----
                mlp_topk, recon_topk = _topk_mlp_and_recon(X, Y, atlas, block_id, r)

                with _patch_mlp(model, block_id, mlp_topk):
                    ppl_topk = _eval_ppl(model, corpus.eval_val, device)

                results[(block_id, "topk", r, seed)] = {
                    "ppl": ppl_topk, "recon_err": recon_topk
                }
                print(f"    topk   k={r:3d}: PPL={ppl_topk:.3f}  recon={recon_topk:.4f}")

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    print("\n=== Verdict ===")
    block_wins_direct = 0
    all_deltas = []

    for block_id in TARGET_BLOCKS:
        block_direct_wins = 0
        for r in RANKS:
            ppl_d_seeds = [results[(block_id, "direct", r, s)]["ppl"] for s in SEEDS]
            ppl_t_seeds = [results[(block_id, "topk", r, s)]["ppl"] for s in SEEDS]
            mean_d = sum(ppl_d_seeds) / len(ppl_d_seeds)
            mean_t = sum(ppl_t_seeds) / len(ppl_t_seeds)
            delta_pct = (mean_d - mean_t) / mean_t * 100
            all_deltas.append(delta_pct)
            if delta_pct <= 0:
                block_direct_wins += 1
                print(f"  Block {block_id} r={r}: direct WINS (Δ={delta_pct:+.2f}%)")
            else:
                print(f"  Block {block_id} r={r}: topk wins  (Δ={delta_pct:+.2f}%)")
        if block_direct_wins > 0:
            block_wins_direct += 1

    if block_wins_direct >= 2:
        verdict = "CONFIRMED"
    elif all(d > 5.0 for d in all_deltas):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVerdict: {verdict} ({block_wins_direct}/4 blocks won by direct)")
    print(f"Delta range: [{min(all_deltas):+.2f}%, {max(all_deltas):+.2f}%]")

    # ---------------------------------------------------------------------------
    # Serialize
    # ---------------------------------------------------------------------------
    serializable = {
        "baseline_ppl": baseline_ppl,
        "verdict": verdict,
        "block_wins_direct": block_wins_direct,
        "n_blocks": len(TARGET_BLOCKS),
        "delta_range": [float(min(all_deltas)), float(max(all_deltas))],
        "eval_val_seqs": EVAL_VAL_SEQS,
        "runs": {
            f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in results.items()
        },
    }
    json_path = RESULTS_DIR / "cycle_107_direct_regression.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"JSON: {json_path}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, len(TARGET_BLOCKS), figsize=(16, 8))
    param_counts = [2 * 768 * r for r in RANKS]

    for col, block_id in enumerate(TARGET_BLOCKS):
        ax_ppl = axes[0, col]
        ax_rec = axes[1, col]

        ppl_d_means, ppl_d_stds = [], []
        ppl_t_means, ppl_t_stds = [], []
        rec_d_means, rec_d_stds = [], []
        rec_t_means, rec_t_stds = [], []

        for r in RANKS:
            ppl_d = [results[(block_id, "direct", r, s)]["ppl"] for s in SEEDS]
            ppl_t = [results[(block_id, "topk", r, s)]["ppl"] for s in SEEDS]
            rec_d = [results[(block_id, "direct", r, s)]["recon_err"] for s in SEEDS]
            rec_t = [results[(block_id, "topk", r, s)]["recon_err"] for s in SEEDS]
            ppl_d_means.append(float(np.mean(ppl_d))); ppl_d_stds.append(float(np.std(ppl_d)))
            ppl_t_means.append(float(np.mean(ppl_t))); ppl_t_stds.append(float(np.std(ppl_t)))
            rec_d_means.append(float(np.mean(rec_d))); rec_d_stds.append(float(np.std(rec_d)))
            rec_t_means.append(float(np.mean(rec_t))); rec_t_stds.append(float(np.std(rec_t)))

        pc = np.array(param_counts)
        ax_ppl.errorbar(pc, ppl_d_means, yerr=ppl_d_stds, label="direct", marker="o", color="tab:blue")
        ax_ppl.errorbar(pc, ppl_t_means, yerr=ppl_t_stds, label="topk", marker="s", color="tab:orange")
        ax_ppl.axhline(baseline_ppl, color="gray", ls="--", label=f"base ({baseline_ppl:.1f})")
        ax_ppl.set_title(f"Block {block_id} — PPL")
        ax_ppl.set_xlabel("Params (2×768×r)")
        ax_ppl.set_ylabel("PPL")
        ax_ppl.legend(fontsize=7)

        ax_rec.errorbar(pc, rec_d_means, yerr=rec_d_stds, label="direct", marker="o", color="tab:blue")
        ax_rec.errorbar(pc, rec_t_means, yerr=rec_t_stds, label="topk", marker="s", color="tab:orange")
        ax_rec.set_title(f"Block {block_id} — Recon Error")
        ax_rec.set_xlabel("Params (2×768×r)")
        ax_rec.set_ylabel("||Y-Ŷ||_F / ||Y||_F")
        ax_rec.legend(fontsize=7)

    fig.suptitle(
        f"Cycle 107: Direct Regression vs topk+lstsq — GPT-2 FFN\n"
        f"Verdict: {verdict} | Baseline PPL={baseline_ppl:.2f} | {block_wins_direct}/4 blocks",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_107_direct_regression.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Plot: {plot_path}")

    return verdict, baseline_ppl


if __name__ == "__main__":
    main()
