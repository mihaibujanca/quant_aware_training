"""Cycle 162: GELU-aware dead/near-saturated channel census — GPT-2 FFN (item 172).

For each of GPT-2's 12 FFN blocks, compute per-channel pre-GELU activation
statistics over calibration data (3 seeds × 256 seqs × 256 tokens).

This cycle adds:
  - T=2 threshold (not tested in cycle 128)
  - Output-contribution score: ‖W_down[:,j]‖₂ × rms(GELU(z_j))
  - 3-panel plot
  - Single-block PPL validation for top-candidate block

Categories (item 172 definition):
  dead (saturated tail): max(z_j) < -T for T ∈ {2, 3, 4}
  near-linear:           min(z_j) > T  for T ∈ {2, 3, 4}

Verdict (from cycle_162.md):
  CONFIRMED    if ≥5% qualifying AND single-block PPL delta <0.5%
  REFUTED      if <1% at ALL thresholds tested
  INCONCLUSIVE if 1-5% but PPL delta >0.5%
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _forward_logits,
    build_probe_corpus,
    load_nanochat_teacher,
)


@torch.no_grad()
def compute_ppl(model: nn.Module, tokens: torch.Tensor, device: torch.device, batch_size: int = 8) -> float:
    """Compute perplexity on token sequences (input: [N, seq_len])."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = tokens.shape[0]
    for i in range(0, n, batch_size):
        batch = tokens[i:i + batch_size].to(device)
        inp = batch[:, :-1].contiguous()
        tgt = batch[:, 1:].contiguous()
        logits = _forward_logits(model, inp)  # (B, T, vocab)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
N_SEQS = 256
SEQ_LEN = 256
BATCH_SIZE = 8

N_BLOCKS = 12
D_FFN = 3072
D_MODEL = 768

THRESHOLDS = [2, 3, 4]

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def census_one_seed(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run census + contribution scores for all 12 blocks.

    Returns per-block dict with:
      - min_z, max_z per channel
      - dead/near-linear counts per threshold
      - contribution scores
      - raw gelu_rms for contribution score computation
    """
    results = {}
    for block_id in range(N_BLOCKS):
        block = model.transformer.h[block_id]
        fc = block.mlp.c_fc   # output is pre-GELU
        proj = block.mlp.c_proj  # W_down, shape (D_MODEL, D_FFN)

        pre_acts: list[torch.Tensor] = []

        def _hook(module, inp, out):
            pre_acts.append(out.detach().cpu())

        h = fc.register_forward_hook(_hook)

        n_seqs = tokens.shape[0]
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = tokens[i:i + BATCH_SIZE].to(device)
            inputs = batch[:, :-1].contiguous()
            model(inputs)

        h.remove()

        # z_all: (total_tokens, D_FFN)
        z_all = torch.cat(pre_acts, dim=0).reshape(-1, D_FFN)
        n_tokens = z_all.shape[0]

        min_z = z_all.min(dim=0).values   # (D_FFN,)
        max_z = z_all.max(dim=0).values

        # Census at each threshold
        census = {}
        for T in THRESHOLDS:
            dead = (max_z < -T).sum().item()
            near_linear = (min_z > T).sum().item()
            census[T] = {
                "dead": dead,
                "near_linear": near_linear,
                "qualifying": dead + near_linear,
                "pct": (dead + near_linear) / D_FFN * 100,
            }

        # Output-contribution score: ‖W_down[:,j]‖₂ × rms(GELU(z_j))
        # W_down (proj) weight: shape (D_MODEL, D_FFN) in GPT-2 (Conv1D stored transposed)
        # GPT-2 uses Conv1D where weight is (in_features, out_features)
        # c_proj.weight shape: (D_FFN, D_MODEL) — column j is W_down[:,j]
        W_down = proj.weight.detach().cpu()  # (D_FFN, D_MODEL)
        w_norm = W_down.norm(dim=1)          # (D_FFN,) — ‖W_down[:,j]‖₂

        gelu_z = F.gelu(z_all)              # (n_tokens, D_FFN)
        gelu_rms = gelu_z.pow(2).mean(dim=0).sqrt()  # (D_FFN,)

        score = w_norm * gelu_rms           # (D_FFN,)
        score_norm = score / score.sum()    # normalized so sum=1

        # Store per-channel scores as list (for JSON)
        results[block_id] = {
            "n_tokens": n_tokens,
            "census": {str(T): census[T] for T in THRESHOLDS},
            "score_norm": score_norm.tolist(),
            "gelu_rms": gelu_rms.tolist(),
            "w_norm": w_norm.tolist(),
            # For cross-seed aggregation keep min/max as lists
            "_min_z": min_z.tolist(),
            "_max_z": max_z.tolist(),
        }

        print(
            f"  Block {block_id:2d}: "
            + "  ".join(
                f"T={T}: dead={census[T]['dead']:4d} near_lin={census[T]['near_linear']:4d} ({census[T]['pct']:.2f}%)"
                for T in THRESHOLDS
            ),
            flush=True,
        )

    return results


@torch.no_grad()
def ppl_block_prune(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    block_id: int,
    channel_mask: torch.Tensor,
    orig_ppl: float,
) -> float:
    """Zero out channels in channel_mask for block_id, measure PPL, restore."""
    block = model.transformer.h[block_id]
    fc = block.mlp.c_fc
    proj = block.mlp.c_proj

    # Zero out channels: set rows of W_up (c_fc.weight) and cols of W_down
    # GPT-2 Conv1D: c_fc.weight shape (D_MODEL, D_FFN), bias (D_FFN,)
    # c_proj.weight shape (D_FFN, D_MODEL)
    orig_fc_w = fc.weight.data.clone()
    orig_fc_b = fc.bias.data.clone()
    orig_proj_w = proj.weight.data.clone()

    # Zero the output channels of fc (rows of fc.weight.T = cols of fc.weight)
    fc.weight.data[:, channel_mask] = 0.0
    fc.bias.data[channel_mask] = 0.0
    # Zero corresponding rows of proj.weight
    proj.weight.data[channel_mask, :] = 0.0

    new_ppl = compute_ppl(model, tokens, device, batch_size=BATCH_SIZE)

    # Restore
    fc.weight.data.copy_(orig_fc_w)
    fc.bias.data.copy_(orig_fc_b)
    proj.weight.data.copy_(orig_proj_w)

    return new_ppl


# ===========================================================================
if __name__ == "__main__":
    print("Loading GPT-2...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    model = teacher.model
    model.eval()
    device = teacher.device
    print(f"  Device: {device}", flush=True)

    # -----------------------------------------------------------------------
    # Census per seed
    # -----------------------------------------------------------------------
    all_seed_results: dict[int, dict] = {}
    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}: {N_SEQS} seqs × {SEQ_LEN} tokens", flush=True)
        print(f"{'='*60}", flush=True)

        _set_seed(seed)
        corpus = build_probe_corpus(ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=N_SEQS,
            probe_holdout_sequences=256,
            eval_val_sequences=256,
            dataset_name="openwebtext",
            shuffle_seed=seed,
        ))
        tokens = corpus.probe_train  # (N_SEQS, SEQ_LEN)
        print(f"  Calibration tokens: {tokens.shape}", flush=True)

        results = census_one_seed(model, tokens, device)
        all_seed_results[seed] = results

    # -----------------------------------------------------------------------
    # Cross-seed summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print("CROSS-SEED SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    summary: dict[int, dict] = {}
    for block_id in range(N_BLOCKS):
        block_summary: dict = {"census": {}}
        for T in THRESHOLDS:
            dead_vals = [all_seed_results[s][block_id]["census"][str(T)]["dead"] for s in SEEDS]
            nl_vals = [all_seed_results[s][block_id]["census"][str(T)]["near_linear"] for s in SEEDS]
            q_vals = [all_seed_results[s][block_id]["census"][str(T)]["qualifying"] for s in SEEDS]
            block_summary["census"][T] = {
                "dead_mean": float(sum(dead_vals) / 3),
                "near_linear_mean": float(sum(nl_vals) / 3),
                "qualifying_mean": float(sum(q_vals) / 3),
                "qualifying_pct": float(sum(q_vals) / 3) / D_FFN * 100,
            }

        # Average contribution scores across seeds (element-wise)
        scores = [torch.tensor(all_seed_results[s][block_id]["score_norm"]) for s in SEEDS]
        mean_score = torch.stack(scores).mean(dim=0)
        block_summary["score_norm_mean"] = mean_score.tolist()

        summary[block_id] = block_summary

        line = f"  Block {block_id:2d}:"
        for T in THRESHOLDS:
            d = block_summary["census"][T]
            line += f"  T={T}: {d['qualifying_mean']:.0f} ({d['qualifying_pct']:.2f}%)"
        print(line, flush=True)

    # -----------------------------------------------------------------------
    # Verdict (pre-PPL)
    # -----------------------------------------------------------------------
    # Best T = lowest threshold (T=2) gives most qualifying channels
    total_at_best_T: dict[int, float] = {}
    for T in THRESHOLDS:
        total_at_best_T[T] = sum(summary[b]["census"][T]["qualifying_mean"] for b in range(N_BLOCKS))

    best_T = 2  # smallest T → most inclusive
    total_qualifying = total_at_best_T[best_T]
    total_neurons = N_BLOCKS * D_FFN
    pct_qualifying = total_qualifying / total_neurons * 100

    print(f"\nTotal qualifying at T=2: {total_qualifying:.0f}/{total_neurons} ({pct_qualifying:.2f}%)", flush=True)
    for T in THRESHOLDS:
        print(f"  T={T}: {total_at_best_T[T]:.0f} ({total_at_best_T[T]/total_neurons*100:.2f}%)", flush=True)

    null_share = 1.0 / D_FFN * 100  # 0.0326%
    print(f"\nNull baseline (equal share): {null_share:.4f}%", flush=True)

    # -----------------------------------------------------------------------
    # Single-block PPL validation (best candidate block, T=2)
    # -----------------------------------------------------------------------
    # Load held-out sequences (use first seed's holdout)
    print(f"\nLoading held-out sequences for PPL validation...", flush=True)
    _set_seed(SEEDS[0])
    corpus_eval = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=N_SEQS,
        probe_holdout_sequences=256,
        eval_val_sequences=256,
        dataset_name="openwebtext",
        shuffle_seed=SEEDS[0],
    ))
    eval_tokens = corpus_eval.probe_holdout  # (256, SEQ_LEN)
    print(f"  Eval tokens shape: {eval_tokens.shape}", flush=True)

    orig_ppl = compute_ppl(model, eval_tokens, device, batch_size=BATCH_SIZE)
    print(f"  Original PPL: {orig_ppl:.4f}", flush=True)

    # Find top block (most qualifying channels at T=2)
    top_block = max(range(N_BLOCKS), key=lambda b: summary[b]["census"][2]["qualifying_mean"])
    print(f"  Top block for PPL test: Block {top_block}", flush=True)

    ppl_results: dict[str, float | int] = {"orig_ppl": orig_ppl}

    # Test 1: zero out dead+near-linear channels at T=2
    min_z_top = torch.tensor(all_seed_results[SEEDS[0]][top_block]["_min_z"])
    max_z_top = torch.tensor(all_seed_results[SEEDS[0]][top_block]["_max_z"])
    dead_mask_T2 = max_z_top < -2.0
    nl_mask_T2 = min_z_top > 2.0
    qual_mask_T2 = dead_mask_T2 | nl_mask_T2
    n_qual = qual_mask_T2.sum().item()

    if n_qual > 0:
        ppl_qual = ppl_block_prune(model, eval_tokens, device, top_block, qual_mask_T2, orig_ppl)
        ppl_delta_pct = (ppl_qual - orig_ppl) / orig_ppl * 100
        ppl_results["top_block"] = top_block
        ppl_results["n_pruned_T2"] = int(n_qual)
        ppl_results["ppl_after_T2"] = float(ppl_qual)
        ppl_results["ppl_delta_pct"] = float(ppl_delta_pct)
        print(f"  Block {top_block} T=2 prune ({n_qual} channels): PPL {orig_ppl:.3f} -> {ppl_qual:.3f} ({ppl_delta_pct:+.2f}%)", flush=True)
    else:
        ppl_results["top_block"] = top_block
        ppl_results["n_pruned_T2"] = 0
        ppl_results["ppl_after_T2"] = orig_ppl
        ppl_results["ppl_delta_pct"] = 0.0
        print(f"  Block {top_block}: no qualifying channels at T=2", flush=True)

    # Test 2: bottom 1% by contribution score
    score_top = torch.tensor(summary[top_block]["score_norm_mean"])
    n_bottom1pct = max(1, int(0.01 * D_FFN))
    _, bottom_idx = score_top.topk(n_bottom1pct, largest=False)
    bottom_mask = torch.zeros(D_FFN, dtype=torch.bool)
    bottom_mask[bottom_idx] = True
    ppl_bottom1 = ppl_block_prune(model, eval_tokens, device, top_block, bottom_mask, orig_ppl)
    ppl_delta_bottom1 = (ppl_bottom1 - orig_ppl) / orig_ppl * 100
    ppl_results["ppl_bottom1pct"] = float(ppl_bottom1)
    ppl_results["ppl_delta_bottom1pct"] = float(ppl_delta_bottom1)
    print(f"  Block {top_block} bottom 1% score ({n_bottom1pct} channels): PPL -> {ppl_bottom1:.3f} ({ppl_delta_bottom1:+.2f}%)", flush=True)

    # Test 3: bottom 5% by contribution score
    n_bottom5pct = max(1, int(0.05 * D_FFN))
    _, bottom5_idx = score_top.topk(n_bottom5pct, largest=False)
    bottom5_mask = torch.zeros(D_FFN, dtype=torch.bool)
    bottom5_mask[bottom5_idx] = True
    ppl_bottom5 = ppl_block_prune(model, eval_tokens, device, top_block, bottom5_mask, orig_ppl)
    ppl_delta_bottom5 = (ppl_bottom5 - orig_ppl) / orig_ppl * 100
    ppl_results["ppl_bottom5pct"] = float(ppl_bottom5)
    ppl_results["ppl_delta_bottom5pct"] = float(ppl_delta_bottom5)
    print(f"  Block {top_block} bottom 5% score ({n_bottom5pct} channels): PPL -> {ppl_bottom5:.3f} ({ppl_delta_bottom5:+.2f}%)", flush=True)

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    ppl_ok = (n_qual > 0) and abs(ppl_results.get("ppl_delta_pct", 99)) < 0.5

    if pct_qualifying >= 5.0 and ppl_ok:
        verdict = "CONFIRMED"
        detail = f"≥5% qualifying at T=2 ({pct_qualifying:.1f}%) AND PPL delta {ppl_results.get('ppl_delta_pct', 0):+.2f}% <0.5%"
    elif pct_qualifying >= 5.0 and not ppl_ok:
        verdict = "INCONCLUSIVE"
        detail = f"≥5% qualifying at T=2 ({pct_qualifying:.1f}%) BUT PPL delta {ppl_results.get('ppl_delta_pct', 99):+.2f}% ≥0.5%"
    elif all(total_at_best_T[T] / total_neurons * 100 < 1.0 for T in THRESHOLDS):
        verdict = "REFUTED"
        detail = f"<1% qualifying at ALL thresholds (T=2: {pct_qualifying:.2f}%)"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"1-5% qualifying at T=2 ({pct_qualifying:.2f}%), PPL delta {ppl_results.get('ppl_delta_pct', 99):+.2f}%"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    # Strip raw per-channel lists from per-seed results to keep JSON small
    per_seed_clean: dict[str, dict] = {}
    for s in SEEDS:
        per_seed_clean[str(s)] = {}
        for b in range(N_BLOCKS):
            bdata = dict(all_seed_results[s][b])
            del bdata["score_norm"]
            del bdata["gelu_rms"]
            del bdata["w_norm"]
            del bdata["_min_z"]
            del bdata["_max_z"]
            per_seed_clean[str(s)][str(b)] = bdata

    output = {
        "cycle": 162,
        "item": 172,
        "seeds": SEEDS,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "thresholds": THRESHOLDS,
        "total_neurons": total_neurons,
        "total_qualifying_by_T": {str(T): float(total_at_best_T[T]) for T in THRESHOLDS},
        "pct_qualifying_by_T": {str(T): float(total_at_best_T[T] / total_neurons * 100) for T in THRESHOLDS},
        "null_share_pct": null_share,
        "ppl_results": ppl_results,
        "verdict": verdict,
        "verdict_detail": detail,
        "per_seed": per_seed_clean,
        "cross_seed_summary": {
            str(b): {
                "census": {str(T): summary[b]["census"][T] for T in THRESHOLDS},
            }
            for b in range(N_BLOCKS)
        },
    }

    json_path = RESULTS_DIR / "cycle_162_census.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}", flush=True)

    # -----------------------------------------------------------------------
    # Plots — 3 panels
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    import numpy as np

    blocks = list(range(N_BLOCKS))
    x = np.arange(N_BLOCKS)
    w = 0.25
    colors = {2: "#e74c3c", 3: "#e67e22", 4: "#2ecc71"}

    # Panel 1: per-block dead+near-linear counts by threshold
    ax = axes[0]
    for idx, T in enumerate(THRESHOLDS):
        vals = [summary[b]["census"][T]["qualifying_mean"] for b in blocks]
        ax.bar(x + (idx - 1) * w, vals, w, label=f"T={T}", color=colors[T], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i}" for i in blocks])
    ax.set_ylabel("Dead + near-linear channels (mean across 3 seeds)")
    ax.set_title("GELU Dead / Near-Saturated Channels per Block — GPT-2 FFN")
    ax.axhline(0.05 * D_FFN, color="k", ls="--", lw=1, label="5% threshold (=154)")
    ax.axhline(0.01 * D_FFN, color="gray", ls=":", lw=1, label="1% threshold (=31)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: contribution score distribution (log-x histogram, all blocks pooled)
    ax2 = axes[1]
    all_scores = []
    for b in blocks:
        all_scores.extend(summary[b]["score_norm_mean"])
    all_scores_t = np.array(all_scores)
    # Log-spaced bins
    eps = all_scores_t[all_scores_t > 0].min() / 2
    log_bins = np.logspace(
        np.log10(max(all_scores_t.min(), eps)),
        np.log10(all_scores_t.max()),
        60,
    )
    ax2.hist(all_scores_t, bins=log_bins, color="#3498db", alpha=0.7)
    ax2.axvline(null_share / 100, color="k", ls="--", lw=1.5, label=f"Equal share ({null_share:.4f}%)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Normalized contribution score")
    ax2.set_ylabel("Channel count")
    ax2.set_title("Output-Contribution Score Distribution (all blocks pooled, log-x)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: contribution score vs channel index, colored by block
    ax3 = axes[2]
    cmap = plt.get_cmap("tab20", N_BLOCKS)
    for b in blocks:
        s_vals = np.array(summary[b]["score_norm_mean"])
        ch_idx = np.arange(D_FFN)
        ax3.scatter(ch_idx, s_vals, s=0.5, color=cmap(b), alpha=0.4, label=f"B{b}")
    ax3.set_xlabel("Channel index (0-3071)")
    ax3.set_ylabel("Normalized contribution score")
    ax3.set_title("Contribution Score by Channel and Block")
    ax3.legend(loc="upper right", fontsize=5, ncol=4, markerscale=5)
    ax3.grid(alpha=0.2)

    plt.suptitle(
        f"Cycle 162 — GELU Channel Census | Verdict: {verdict}\n"
        f"T=2: {pct_qualifying:.2f}% qualifying | PPL delta: {ppl_results.get('ppl_delta_pct', 'N/A'):.2f}%",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_162_census.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved plot: {plot_path}", flush=True)

    print("\nDone.", flush=True)
