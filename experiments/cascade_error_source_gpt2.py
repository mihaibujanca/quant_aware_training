"""Cycle 104: Cascade error source attribution — GPT-2 FFN (item 115).

Decomposes c≈0.41 cascade amplification into:
  - Attention sub-layer contribution
  - GELU/FFN sub-layer contribution
  - Interaction term

For each of 11 block transitions (L → L+1, L in 0..10), compress block L FFN at 6x.
Four variants measure how error propagates through block L+1:
  Full (F):        normal forward — attn + FFN both see perturbed input
  Attn-frozen (A): attn sub-layer gets FLOAT LN output; FFN sees perturbed residual
  FFN-frozen (B):  FFN sub-layer gets float post-attn state; attn sees perturbed input
  Both-frozen (D): both sub-layers get float inputs (pure residual stream passthrough)

Decomposition (excess over passthrough):
  amp_full = ||δ_F|| - ||δ_D||
  amp_attn = ||δ_B|| - ||δ_D||   (attn's contribution to excess)
  amp_gelu = ||δ_A|| - ||δ_D||   (FFN/GELU contribution to excess)
  amp_inter = amp_full - amp_attn - amp_gelu
  fracs: attn_frac = amp_attn / amp_full, gelu_frac = amp_gelu / amp_full

Verdict:
  CONFIRMED  if one source ≥70% in ≥8/11 transitions (same source)
  REFUTED    if all transitions show 40-60% split
  INCONCLUSIVE otherwise

3 seeds [42, 123, 7], 200 OWT eval seqs.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_bias,
    _standard_weight,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
HIDDEN_FRACTION = 1 / 6         # 6x compression
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
N_TRANSITIONS = 11              # blocks 0..10 → 1..11
PPL_ABORT_MIN = 25.0
PPL_ABORT_MAX = 65.0

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tok_norm(x: torch.Tensor) -> float:
    """Mean L2 norm over all tokens in the batch."""
    return x.norm(dim=-1).mean().item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two [B, T, D] tensors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


# ---------------------------------------------------------------------------
# GPT-2 block 4-variant forward implementations
# ---------------------------------------------------------------------------

def _block_attn(block, x: torch.Tensor) -> torch.Tensor:
    """Run only the attention sub-layer on x, return post-attn residual."""
    residual = x
    h = block.ln_1(x)
    attn_out, _ = block.attn(h)
    return attn_out + residual


def _block_ffn(block, x: torch.Tensor) -> torch.Tensor:
    """Run only the FFN sub-layer on x (given post-attn residual), return output."""
    residual = x
    h = block.ln_2(x)
    ffn_out = block.mlp(h)
    return ffn_out + residual


def variant_full(block, x_perturbed: torch.Tensor) -> torch.Tensor:
    """Full (F): normal block forward with perturbed input."""
    x = _block_attn(block, x_perturbed)
    x = _block_ffn(block, x)
    return x


def variant_D(block, x_perturbed: torch.Tensor, x_float: torch.Tensor) -> torch.Tensor:
    """Both-frozen (D): both sub-layers receive float inputs; residual stream carries error.

    Attn output = float_attn_output + x_perturbed (perturb in residual stream only)
    FFN output = float_ffn_output + x_after_attn_perturbed_residual
    """
    # Attention: float LN input, add to perturbed residual
    attn_out_float, _ = block.attn(block.ln_1(x_float))
    x_after_attn = attn_out_float + x_perturbed

    # FFN: float LN input (post-attn float state), add to perturbed residual stream
    x_after_attn_float = attn_out_float + x_float
    ffn_out_float = block.mlp(block.ln_2(x_after_attn_float))
    x_out = ffn_out_float + x_after_attn
    return x_out


def variant_A(block, x_perturbed: torch.Tensor, x_float: torch.Tensor) -> torch.Tensor:
    """Attn-frozen (A): attn uses float LN input; FFN sees perturbed residual.

    Measures how much FFN/GELU amplifies error (attn contribution zeroed out).
    """
    # Attention: float LN input, add to perturbed residual
    attn_out_float, _ = block.attn(block.ln_1(x_float))
    x_after_attn = attn_out_float + x_perturbed

    # FFN: sees perturbed post-attn state (LN of perturbed stream)
    ffn_out = block.mlp(block.ln_2(x_after_attn))
    x_out = ffn_out + x_after_attn
    return x_out


def variant_B(block, x_perturbed: torch.Tensor, x_float: torch.Tensor) -> torch.Tensor:
    """FFN-frozen (B): FFN uses float post-attn LN input; attn sees perturbed input.

    Measures how much attention amplifies error (FFN contribution zeroed out).
    """
    # Attention: perturbed input
    attn_out_perturbed, _ = block.attn(block.ln_1(x_perturbed))
    x_after_attn_perturbed = attn_out_perturbed + x_perturbed

    # FFN: float post-attn state
    attn_out_float, _ = block.attn(block.ln_1(x_float))
    x_after_attn_float = attn_out_float + x_float

    ffn_out_float = block.mlp(block.ln_2(x_after_attn_float))
    x_out = ffn_out_float + x_after_attn_perturbed
    return x_out


POLISH_STEPS = 500
POLISH_LR = 1e-4


def _find_mlp(model: nn.Module, block_id: int):
    try:
        return model.transformer.h[block_id].mlp
    except (AttributeError, IndexError):
        return None


def _polish_ffn(student: nn.Module, block_id: int, tokens: torch.Tensor, device: torch.device) -> None:
    """500-step Adam polish on W_down of block_id."""
    mlp = _find_mlp(student, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return
    student.requires_grad_(False)
    w = mlp.c_proj.weight
    w.requires_grad_(True)
    b = getattr(mlp.c_proj, "bias", None)
    if b is not None:
        b.requires_grad_(True)
    params = [w] + ([b] if b is not None else [])
    opt = torch.optim.Adam(params, lr=POLISH_LR)
    student.train()
    n = tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)
    for _ in range(POLISH_STEPS):
        idx = torch.randperm(n, generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        inputs, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        opt.zero_grad()
        raw = student(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).backward()
        opt.step()
    student.eval()
    student.requires_grad_(False)


# ---------------------------------------------------------------------------
# Compress single block FFN
# ---------------------------------------------------------------------------
def _compress_single_block(teacher: TeacherBundle, corpus, block_id: int, seed: int) -> nn.Module:
    """Return a student model with only block `block_id` FFN compressed (6x topk+lstsq+500 Adam)."""
    _set_seed(seed)
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=(block_id,),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    _polish_ffn(student, block_id, corpus.probe_holdout, DEVICE)
    return student


# ---------------------------------------------------------------------------
# Run one transition L → L+1
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_transition(
    float_model: nn.Module,
    student_model: nn.Module,
    corpus,
    source_block: int,
) -> dict:
    """4-variant error decomposition for transition source_block → source_block+1."""
    target_block = source_block + 1

    norms_F, norms_A, norms_B, norms_D, norms_L = [], [], [], [], []
    cosines_FL = []

    for ids, _ in _iter_lm_batches(corpus.eval_val, batch_size=BATCH_SIZE, device=DEVICE):
        # Use input_ids without last token for hidden state computation
        input_ids = ids[:, :-1]
        B, T = input_ids.shape

        # ---- Forward through shared prefix (embedding + blocks 0..source_block-1) ----
        x = float_model.transformer.wte(input_ids) + float_model.transformer.wpe(
            torch.arange(T, device=DEVICE)
        )
        x = float_model.transformer.drop(x)

        for blk_idx in range(source_block):
            x = float_model.transformer.h[blk_idx](x)[0]

        x_pre_src = x  # input to source block

        # ---- Float and compressed forward through source_block ----
        x_float_after_src = float_model.transformer.h[source_block](x_pre_src)[0]
        x_comp_after_src = student_model.transformer.h[source_block](x_pre_src)[0]

        delta_L = x_comp_after_src - x_float_after_src
        norms_L.append(_tok_norm(delta_L))

        # ---- Four variants through target_block ----
        blk_next = float_model.transformer.h[target_block]

        # Full (F): normal forward with compressed input
        x_F = variant_full(blk_next, x_comp_after_src)

        # Both-frozen (D): pure residual passthrough
        x_D = variant_D(blk_next, x_comp_after_src, x_float_after_src)

        # Attn-frozen (A): FFN sees perturbed; attn uses float
        x_A = variant_A(blk_next, x_comp_after_src, x_float_after_src)

        # FFN-frozen (B): attn sees perturbed; FFN uses float
        x_B = variant_B(blk_next, x_comp_after_src, x_float_after_src)

        # Float reference through target_block
        x_float_tgt = variant_full(blk_next, x_float_after_src)

        # δs relative to float reference
        norms_F.append(_tok_norm(x_F - x_float_tgt))
        norms_A.append(_tok_norm(x_A - x_float_tgt))
        norms_B.append(_tok_norm(x_B - x_float_tgt))
        norms_D.append(_tok_norm(x_D - x_float_tgt))
        cosines_FL.append(_cosine(x_F - x_float_tgt, delta_L))

    n_F = float(np.mean(norms_F))
    n_A = float(np.mean(norms_A))
    n_B = float(np.mean(norms_B))
    n_D = float(np.mean(norms_D))
    n_L = float(np.mean(norms_L))

    ln_ratio = n_D / n_L if n_L > 1e-10 else float("nan")

    amp_full = n_F - n_D
    amp_attn = n_B - n_D   # attn contribution
    amp_gelu = n_A - n_D   # FFN/GELU contribution
    amp_inter = amp_full - amp_attn - amp_gelu

    if abs(amp_full) > 1e-8:
        attn_frac = amp_attn / amp_full
        gelu_frac = amp_gelu / amp_full
        inter_frac = amp_inter / amp_full
        attenuation = False
    else:
        attn_frac = gelu_frac = inter_frac = float("nan")
        attenuation = True

    return {
        "source_block": source_block,
        "target_block": target_block,
        "norm_L": n_L,
        "norm_F": n_F,
        "norm_A": n_A,
        "norm_B": n_B,
        "norm_D": n_D,
        "amp_full": amp_full,
        "amp_attn": amp_attn,
        "amp_gelu": amp_gelu,
        "amp_inter": amp_inter,
        "attn_frac": attn_frac,
        "gelu_frac": gelu_frac,
        "inter_frac": inter_frac,
        "ln_ratio": ln_ratio,
        "cosine_F_L": float(np.mean(cosines_FL)),
        "attenuation": attenuation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Loading teacher...")
    load_spec = TeacherLoadSpec(
        source="hf_pretrained",
        hf_model_name="gpt2",
        device=str(DEVICE),
    )
    teacher = load_nanochat_teacher(load_spec)
    float_model = teacher.model
    float_model.eval()

    print("Building probe corpus...")
    probe_spec = ProbeCorpusSpec(
        probe_train_sequences=PROBE_TRAIN_SEQS,
        eval_val_sequences=EVAL_SEQS,
        sequence_length=SEQ_LEN,
        text_path=str(OWT_PATH),
        teacher=teacher,
    )
    corpus = build_probe_corpus(probe_spec)

    # Abort check
    base_metrics = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
    base_ppl = base_metrics["perplexity"]
    print(f"Baseline PPL: {base_ppl:.2f}")
    if not (PPL_ABORT_MIN <= base_ppl <= PPL_ABORT_MAX):
        print(f"ABORT: baseline PPL {base_ppl:.2f} outside [{PPL_ABORT_MIN}, {PPL_ABORT_MAX}]")
        sys.exit(1)

    all_results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        for src_block in range(N_TRANSITIONS):
            print(f"  Block {src_block}→{src_block+1}: ", end="", flush=True)
            student = _compress_single_block(teacher, corpus, src_block, seed)
            student.eval()
            row = run_transition(float_model, student, corpus, src_block)
            row["seed"] = seed
            all_results.append(row)
            if row["attenuation"]:
                print("ATTENUATION")
            else:
                print(
                    f"attn={row['attn_frac']:.3f} gelu={row['gelu_frac']:.3f} "
                    f"inter={row['inter_frac']:.3f} ln_ratio={row['ln_ratio']:.3f}"
                )

    # Average fractions across seeds per transition
    avg_by_trans: dict[int, dict] = {}
    for src in range(N_TRANSITIONS):
        rows = [r for r in all_results if r["source_block"] == src and not r["attenuation"]]
        if not rows:
            avg_by_trans[src] = {"attenuation": True}
            continue
        avg_by_trans[src] = {
            "attn_frac": float(np.mean([r["attn_frac"] for r in rows])),
            "gelu_frac": float(np.mean([r["gelu_frac"] for r in rows])),
            "inter_frac": float(np.mean([r["inter_frac"] for r in rows])),
            "ln_ratio": float(np.mean([r["ln_ratio"] for r in rows])),
            "cosine_F_L": float(np.mean([r["cosine_F_L"] for r in rows])),
            "norm_L": float(np.mean([r["norm_L"] for r in rows])),
            "norm_F": float(np.mean([r["norm_F"] for r in rows])),
            "norm_D": float(np.mean([r["norm_D"] for r in rows])),
            "amp_full": float(np.mean([r["amp_full"] for r in rows])),
            "attenuation": False,
            "n_seeds": len(rows),
        }

    # Verdict
    valid_trans = [t for t in range(N_TRANSITIONS) if not avg_by_trans[t].get("attenuation", True)]
    n_valid = len(valid_trans)

    dominant_attn = sum(1 for t in valid_trans if avg_by_trans[t]["attn_frac"] >= 0.70)
    dominant_gelu = sum(1 for t in valid_trans if avg_by_trans[t]["gelu_frac"] >= 0.70)
    all_split = all(0.40 <= avg_by_trans[t]["attn_frac"] <= 0.60 for t in valid_trans)

    if dominant_attn >= 8:
        verdict = "CONFIRMED_ATTN"
        verdict_note = f"Attention dominates (≥70%) in {dominant_attn}/{n_valid} transitions"
    elif dominant_gelu >= 8:
        verdict = "CONFIRMED_GELU"
        verdict_note = f"GELU/FFN dominates (≥70%) in {dominant_gelu}/{n_valid} transitions"
    elif all_split and n_valid > 0:
        verdict = "REFUTED"
        verdict_note = "All transitions show 40-60% split between attn and GELU"
    else:
        verdict = "INCONCLUSIVE"
        verdict_note = f"attn dominant: {dominant_attn}/{n_valid}, gelu dominant: {dominant_gelu}/{n_valid}"

    print(f"\nVerdict: {verdict}")
    print(f"  {verdict_note}")

    print("\nTransition | attn_frac | gelu_frac | inter_frac | ln_ratio | cosine")
    for t in range(N_TRANSITIONS):
        d = avg_by_trans[t]
        if d.get("attenuation"):
            print(f"  {t:2d}→{t+1:2d}   | ATTENUATION")
        else:
            print(
                f"  {t:2d}→{t+1:2d}   | {d['attn_frac']:7.3f}   | {d['gelu_frac']:7.3f}   | "
                f"{d['inter_frac']:8.3f}   | {d['ln_ratio']:6.3f}   | {d['cosine_F_L']:.3f}"
            )

    # Save JSON
    output = {
        "verdict": verdict,
        "verdict_note": verdict_note,
        "base_ppl": base_ppl,
        "seeds": SEEDS,
        "n_transitions": N_TRANSITIONS,
        "n_valid_transitions": n_valid,
        "dominant_attn_count": dominant_attn,
        "dominant_gelu_count": dominant_gelu,
        "avg_by_transition": {str(k): v for k, v in avg_by_trans.items()},
        "all_results": all_results,
    }
    json_path = RESULTS_DIR / "cycle_104_cascade_source.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot: stacked bar chart
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(N_TRANSITIONS)
    w = 0.6

    attn_fracs = [avg_by_trans[t].get("attn_frac", 0.0) or 0.0 for t in range(N_TRANSITIONS)]
    gelu_fracs = [avg_by_trans[t].get("gelu_frac", 0.0) or 0.0 for t in range(N_TRANSITIONS)]
    inter_fracs = [avg_by_trans[t].get("inter_frac", 0.0) or 0.0 for t in range(N_TRANSITIONS)]
    ln_ratios = [avg_by_trans[t].get("ln_ratio", 1.0) or 1.0 for t in range(N_TRANSITIONS)]

    ax1.bar(x, attn_fracs, w, label="Attention", color="#2196F3")
    ax1.bar(x, gelu_fracs, w, bottom=attn_fracs, label="GELU/FFN", color="#FF9800")
    inter_bottoms = [a + g for a, g in zip(attn_fracs, gelu_fracs)]
    ax1.bar(x, inter_fracs, w, bottom=inter_bottoms, label="Interaction", color="#9E9E9E")
    ax1.axhline(0.7, color="red", linestyle="--", linewidth=1, label="70% threshold")
    ax1.set_xlabel("Source block L (transition L→L+1)")
    ax1.set_ylabel("Fraction of excess amplification")
    ax1.set_title(f"Cascade Error Source Attribution — GPT-2 FFN\n{verdict}: {verdict_note}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}→{t+1}" for t in range(N_TRANSITIONS)], rotation=45, ha="right")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, ln_ratios, "k--o", linewidth=1.5, markersize=4, label="LN passthrough ratio")
    ax2.set_ylabel("||δ_D|| / ||δ_L|| (LN passthrough)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_104_cascade_source.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

    return verdict


if __name__ == "__main__":
    main()
